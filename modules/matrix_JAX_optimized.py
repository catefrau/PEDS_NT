"""
Optimized matrix_JAX.py
Key changes vs original:
  1. New Aphi_Fphi_scan(): computes A@phi and F@phi directly via lax.scan
     - NO full N×N matrix ever materialized
     - NO Python loop unrolling (single compiled XLA kernel)
     - Computes both products in ONE pass over cells
  2. Original diffusion_setup_jax() kept for use in _NTdiff_fwd (forward only).

In _NTdiff_bwd (NTadaptation-2.py), replace the two separate vjp calls with:
    def AF_phi(xs):
        return Aphi_Fphi_scan(xs, geo_data, SLAY, phi_fwd)

    _, vjp_fn = jax.vjp(AF_phi, xs_tensor)
    numerator, = vjp_fn((phi_adj, -(1.0 / k) * phi_adj))
    # ^ computes dA_dp_phi - (1/k)*dF_dp_phi in a SINGLE backward pass
"""

import jax
import jax.numpy as jnp


# ──────────────────────────────────────────────────────────────────────────────
# ORIGINAL (kept for forward pass in _NTdiff_fwd — output saved as residual)
# ──────────────────────────────────────────────────────────────────────────────

def diffusion_setup_jax(xs_tensor, geo_data, lay):
    """
    JAX-traceable diffusion matrix assembly.
    xs_tensor: [num_regions, M]
    Returns: A [G*(I+1), G*(I+1)], F [G*(I+1), G*(I+1)]
    """
    G = geo_data['G']
    I = geo_data['I']
    Delta_r = geo_data['Delta_r']
    S = geo_data['S']
    V = geo_data['V']
    BC = geo_data['BC']
    region_of_cell = geo_data['region_of_cell']

    tot_size = G * (I + 1)

    def idx(g, i):
        return g * (I + 1) + i

    A = jnp.zeros((tot_size, tot_size))
    B = jnp.zeros((tot_size, tot_size))

    for g in range(G):
        A = A.at[idx(g, I), idx(g, I  )].set(BC[0]/2 + BC[1]/Delta_r)
        A = A.at[idx(g, I), idx(g, I-1)].set(BC[0]/2 - BC[1]/Delta_r)

    Dplus_prev = 0.0

    for i in range(I):
        reg      = region_of_cell[i]
        reg_next = region_of_cell[min(i+1, I-1)]

        D_g      = xs_tensor[reg, lay['D']]
        D_g_next = xs_tensor[reg_next, lay['D']]
        Sig_a    = xs_tensor[reg, lay['Sigma_a']]
        nuSigf   = xs_tensor[reg, lay['nuSigma_f']]
        chi_g    = xs_tensor[reg, lay['chi']]
        Sig_s    = xs_tensor[reg, lay['Sigma_s']]
        Sig_s_mat = Sig_s.reshape(G, G)

        for g in range(G):
            Dplus  = 2*D_g[g]*D_g_next[g] / (D_g[g] + D_g_next[g] + 1e-30)
            Dminus = Dplus_prev if i > 0 else 0.0

            Sig_s_out = sum(Sig_s_mat[g, gp] for gp in range(G) if gp != g)
            diag = (1/(Delta_r * V[i])) * Dplus * S[i+1] + Sig_a[g] + Sig_s_out
            A = A.at[idx(g,i), idx(g,i)].add(diag)

            if i > 0:
                A = A.at[idx(g,i), idx(g,i-1)].add(-Dminus/(Delta_r * V[i]) * S[i])
                A = A.at[idx(g,i), idx(g,i  )].add( Dminus/(Delta_r * V[i]) * S[i])

            A = A.at[idx(g,i), idx(g,i+1)].add(-Dplus/(Delta_r * V[i]) * S[i+1])

            for gp in range(G):
                if gp != g:
                    A = A.at[idx(g,i), idx(gp,i)].add(-Sig_s_mat[gp, g])

            for gp in range(G):
                val = chi_g[g] * nuSigf[gp]
                B = B.at[idx(g,i), idx(gp,i)].add(val)

        Dplus_prev = Dplus

    return A, B


# ──────────────────────────────────────────────────────────────────────────────
# OPTIMIZED: direct mat-vec with lax.scan — use this in _NTdiff_bwd
# ──────────────────────────────────────────────────────────────────────────────

def Aphi_Fphi_scan(xs_tensor, geo_data, lay, phi):
    """
    Directly computes (A(xs) @ phi, F(xs) @ phi) using jax.lax.scan.

    WHY this is faster than building A, F first:
      - Never allocates a G*(I+1) × G*(I+1) dense matrix
      - lax.scan compiles the loop body ONCE → tight XLA while-loop instead
        of unrolling I Python iterations into I copies of the loop body
      - Both products computed in a single forward pass over cells

    xs_tensor : [num_regions, M]  — JAX array (NN output)
    geo_data  : dict from precompute_geometry
    lay       : dict from xs_layout(G)
    phi       : [G*(I+1)]         — flux vector (constant w.r.t. xs in backward)

    Returns: (Aphi, Fphi), each [G*(I+1)]
    """
    G       = geo_data['G']
    I       = geo_data['I']
    Delta_r = geo_data['Delta_r']
    S       = jnp.array(geo_data['S'], dtype=jnp.float32)            # [I+1]
    V       = jnp.array(geo_data['V'], dtype=jnp.float32)            # [I]
    BC      = geo_data['BC']
    region_of_cell = jnp.array(geo_data['region_of_cell'],
                                dtype=jnp.int32)                      # [I]

    # phi2d[g, i] == phi[g*(I+1) + i]  (same layout as idx(g,i) in original)
    phi2d = phi.reshape(G, I + 1)                                     # [G, I+1]

    # ── Boundary condition rows for cell I (independent of xs_tensor) ──
    Aphi_bc = (BC[0]/2 + BC[1]/Delta_r) * phi2d[:, I] + \
              (BC[0]/2 - BC[1]/Delta_r) * phi2d[:, I - 1]            # [G]

    # ── Interior cells: one compiled XLA loop via lax.scan ─────────────
    def body(Dplus_prev, i):
        """
        carry : Dplus_prev [G]  — harmonic-mean D from previous cell
        x     : i (scan counter)
        out   : (Aphi_i [G], Fphi_i [G]) — contribution at cell i
        """
        reg      = region_of_cell[i]
        reg_next = jax.lax.cond(
            i < I - 1,
            lambda: region_of_cell[i + 1],
            lambda: region_of_cell[i],      # last cell: use same region
        )

        D_g      = xs_tensor[reg, lay['D']]                          # [G]
        D_g_next = xs_tensor[reg_next, lay['D']]                     # [G]
        Sig_a    = xs_tensor[reg, lay['Sigma_a']]                    # [G]
        nuSigf   = xs_tensor[reg, lay['nuSigma_f']]                  # [G]
        chi_g    = xs_tensor[reg, lay['chi']]                        # [G]
        Sig_s    = xs_tensor[reg, lay['Sigma_s']].reshape(G, G)      # [G, G]

        Dplus  = 2 * D_g * D_g_next / (D_g + D_g_next + 1e-30)      # [G]
        Dminus = jnp.where(i > 0, Dplus_prev, jnp.zeros(G))         # [G]

        phi_i   = phi2d[:, i]                                        # [G]
        phi_im1 = phi2d[:, jnp.maximum(i - 1, 0)]                   # safe at i=0
        phi_ip1 = phi2d[:, jnp.minimum(i + 1, I)]                   # safe at i=I-1

        # ── A @ phi contribution at row (g, i) ─────────────────────────
        # Σ_s_out[g] = sum_{g'≠g} Σ_s[g, g']  (scatter out of g)
        Sig_s_out = jnp.sum(Sig_s, axis=0) - jnp.diag(Sig_s)        # [G]
        diag_coef = Dplus * S[i+1] / (Delta_r * V[i]) + Sig_a + Sig_s_out

        Aphi_i  = diag_coef * phi_i                                  # diagonal
        Aphi_i -= Dplus * S[i+1] / (Delta_r * V[i]) * phi_ip1       # right off-diag
        Aphi_i += jnp.where(                                         # left off-diag
            i > 0,
            Dminus * S[i] / (Delta_r * V[i]) * (phi_i - phi_im1),
            jnp.zeros(G)
        )
        # Scatter-in: A.at[idx(g,i), idx(gp,i)].add(-Sig_s[gp, g])
        # → (A@phi)[g,i] += sum_{g'≠g} -Sig_s[g', g]*phi[g', i]
        #                 = -(Sig_s.T @ phi_i)[g] + Sig_s[g,g]*phi_i[g]

        Aphi_i -= Sig_s.T @ phi_i - jnp.diag(Sig_s) * phi_i  # exclude self

        # ── F @ phi contribution at row (g, i) ─────────────────────────
        # F[idx(g,i), idx(g',i)] = chi[g] * nuSigf[g']
        # → (F@phi)[g,i] = chi[g] * Σ_{g'} nuSigf[g'] * phi[g', i]
        Fphi_i = chi_g * (nuSigf @ phi_i)                           # [G]

        return Dplus, (Aphi_i, Fphi_i)

    _, (Aphi_cells, Fphi_cells) = jax.lax.scan(
        body, jnp.zeros(G), jnp.arange(I)
    )
    # Aphi_cells: [I, G]  (one row per cell, all groups)
    # Fphi_cells: [I, G]

    # ── Assemble into flat layout [G*(I+1)] matching idx(g,i) = g*(I+1)+i ──
    Aphi_2d = jnp.concatenate(
        [Aphi_cells.T, Aphi_bc[:, None]], axis=1)                    # [G, I+1]
    Fphi_2d = jnp.concatenate(
        [Fphi_cells.T, jnp.zeros((G, 1))], axis=1)                  # [G, I+1]

    return Aphi_2d.ravel(), Fphi_2d.ravel()
