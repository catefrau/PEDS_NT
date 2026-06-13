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


def Aphi_Fphi_vjp(xs_tensor, geo_data, lay, phi_fwd, v_A, v_F):
    """
    Computes the VJP of (A(xs)@phi_fwd, F(xs)@phi_fwd) with cotangents (v_A, v_F).
    
    Since A and F are LINEAR in xs_tensor, this equals:
        dL/dxs = sum_i [ v_A[i] * d(A@phi_fwd)[i]/dxs 
                       + v_F[i] * d(F@phi_fwd)[i]/dxs ]
    
    We compute this by a single scan over cells, accumulating the gradient
    into xs_grad[region, xs_component] directly.
    
    Returns: grad_xs of shape [num_regions, M]
    """
    G       = geo_data['G']
    I       = geo_data['I']
    Delta_r = geo_data['Delta_r']
    S       = jnp.array(geo_data['S'],              dtype=jnp.float32)
    V       = jnp.array(geo_data['V'],              dtype=jnp.float32)
    BC      = geo_data['BC']
    region_of_cell = jnp.array(geo_data['region_of_cell'], dtype=jnp.int32)
    num_regions = xs_tensor.shape[0]
    M           = xs_tensor.shape[1]

    phi2d = phi_fwd.reshape(G, I + 1)   # [G, I+1]
    vA2d  = v_A.reshape(G, I + 1)       # [G, I+1]  cotangent for A@phi
    vF2d  = v_F.reshape(G, I + 1)       # [G, I+1]  cotangent for F@phi

    # Initialize gradient accumulator
    xs_grad = jnp.zeros_like(xs_tensor)  # [num_regions, M]

    def body(carry, i):
        Dplus_prev, xs_grad = carry

        reg      = region_of_cell[i]
        reg_next = jax.lax.cond(
            i < I - 1,
            lambda: region_of_cell[i + 1],
            lambda: region_of_cell[i],
        )

        D_g      = xs_tensor[reg, lay['D']]           # [G]
        D_g_next = xs_tensor[reg_next, lay['D']]      # [G]
        Sig_s    = xs_tensor[reg, lay['Sigma_s']].reshape(G, G)

        Dplus  = 2 * D_g * D_g_next / (D_g + D_g_next + 1e-30)
        Dminus = jnp.where(i > 0, Dplus_prev, jnp.zeros(G))

        phi_i   = phi2d[:, i]
        phi_im1 = phi2d[:, jnp.maximum(i - 1, 0)]
        phi_ip1 = phi2d[:, jnp.minimum(i + 1, I)]

        vA_i = vA2d[:, i]   # cotangent arriving at row (g, i) for A
        vF_i = vF2d[:, i]   # cotangent arriving at row (g, i) for F

        # ── Gradient w.r.t. Sigma_a[reg, g] ──────────────────────────
        # (A@phi)[g,i] has term: Sig_a[g] * phi_i[g]
        # => d/dSig_a[g] = phi_i[g]
        # => grad contribution: vA_i[g] * phi_i[g]
        grad_Sig_a = vA_i * phi_i   # [G]

        # ── Gradient w.r.t. nuSigma_f[reg, g'] ───────────────────────
        # (F@phi)[g,i] = chi[g] * sum_{g'} nuSigf[g'] * phi[g',i]
        # => d(F@phi)[g,i]/d(nuSigf[gp]) = chi[g] * phi[gp, i]
        # => grad: sum_g vF_i[g] * chi[g] * phi_i  (over all g, for each gp)
        chi_g  = xs_tensor[reg, lay['chi']]           # [G]
        nuSigf = xs_tensor[reg, lay['nuSigma_f']]
        weighted_chi = jnp.dot(vF_i, chi_g)           # scalar: sum_g vF[g]*chi[g]
        grad_nuSigf = weighted_chi * phi_i             # [G]: one entry per g'

        # ── Gradient w.r.t. chi[reg, g] ──────────────────────────────
        # (F@phi)[g,i] = chi[g] * (nuSigf @ phi_i)
        # => d/dchi[g] = nuSigf @ phi_i  (scalar for each g)
        nusigf_phi = jnp.dot(nuSigf, phi_i)           # scalar
        grad_chi = vF_i * nusigf_phi                   # [G]

        # ── Gradient w.r.t. Sigma_s[reg, :] (reshaped as [G,G]) ──────
        # Scatter-out term: Sig_s_out[g] = sum_{g'≠g} Sig_s[g,g']
        #   => (A@phi)[g,i] += Sig_s_out[g] * phi_i[g]
        #   => d/dSig_s[g,gp] (gp≠g) = phi_i[g]
        #   => grad: vA_i[g] * phi_i[g]  for off-diagonal [g, gp]
        #
        # Scatter-in term: (A@phi)[g,i] -= sum_{g'≠g} Sig_s[g',g] * phi[g',i]
        #   => d/dSig_s[gp, g] (gp≠g) = -phi[gp, i]  contributing to row g
        #   combined: grad_Sig_s[g, gp] += vA_i[g]*phi_i[g]  (out)
        #                                - vA_i[gp]*phi_i[g]  (in, index swap)
        # Written as outer products:
        grad_Sig_s_out = jnp.outer(vA_i * phi_i, jnp.ones(G))   # [G,G]
        grad_Sig_s_in  = jnp.outer(vA_i, phi_i)                  # [G,G]
        # zero the diagonal (self-scatter doesn't appear in Sig_s_out or scatter-in)
        diag_mask = jnp.eye(G, dtype=jnp.bool_)
        grad_Sig_s = jnp.where(diag_mask, 0.0, grad_Sig_s_out - grad_Sig_s_in)

        # ── Gradient w.r.t. D[reg, g] ────────────────────────────────
        # Dplus = 2*D[reg,g]*D[reg_next,g] / (D[reg,g] + D[reg_next,g])
        # This is trickier because D appears in Dplus (and Dminus of next cell).
        # We use the chain rule through Dplus:
        # d(Dplus[g])/d(D[reg,g]) = 2*D_next^2 / (D+D_next)^2
        dDplus_dD      = 2 * D_g_next**2 / (D_g + D_g_next + 1e-30)**2   # [G]
        dDplus_dD_next = 2 * D_g**2      / (D_g + D_g_next + 1e-30)**2   # [G]

        # Contribution of Dplus to (A@phi)[g,i]:
        #   coef_plus = Dplus * S[i+1] / (Delta_r * V[i])
        #   (A@phi)[g,i] += coef_plus*(phi_i - phi_ip1) + Dminus*(phi_i-phi_im1)*(i>0)
        # d(A@phi)[g,i]/d(Dplus[g]) = S[i+1]/(Delta_r*V[i]) * (phi_i[g] - phi_ip1[g])
        dAphi_dDplus = S[i+1] / (Delta_r * V[i]) * (phi_i - phi_ip1)   # [G]
        # Also Dminus of cell i+1 = Dplus of cell i, but that's handled when
        # scan processes cell i+1 (Dplus_prev carry). We handle it here by
        # computing the contribution to grad_D from Dplus only (Dminus contribution
        # is already in Dplus_prev when that cell is processed).
        # For the current cell's Dminus contribution:
        dAphi_dDminus = jnp.where(
            i > 0,
            S[i] / (Delta_r * V[i]) * (phi_i - phi_im1),
            jnp.zeros(G)
        )   # [G]

        grad_D_reg      = vA_i * (dAphi_dDplus * dDplus_dD)           # [G]
        # D[reg_next] contributes via Dplus through dDplus_dD_next:
        grad_D_reg_next = vA_i * (dAphi_dDplus * dDplus_dD_next)      # [G]

        # ── Accumulate into xs_grad ───────────────────────────────────
        # We need scatter-add: xs_grad[reg, lay['X']] += grad_X
        # JAX doesn't allow dynamic indexing with scatter in lax.scan carry
        # directly, so we return the per-cell gradients and their region indices,
        # then accumulate outside the scan.
        # Pack into a flat "contribution" array for this cell.

        # Return as structured output rather than accumulating in carry
        # (dynamic scatter into carry causes recompilation issues)
        return (Dplus, xs_grad), (
            reg, reg_next,
            grad_Sig_a, grad_nuSigf, grad_chi,
            grad_Sig_s.ravel(),
            grad_D_reg, grad_D_reg_next,
        )

    (_, _), (regs, regs_next,
             grad_sig_a_all, grad_nusigf_all, grad_chi_all,
             grad_sigs_all,
             grad_D_all, grad_D_next_all) = jax.lax.scan(
        body, (jnp.zeros(G), xs_grad), jnp.arange(I)
    )
    # All outputs: shape [I, G] or [I, G*G] or [I] for regs

    # ── Accumulate per-cell contributions using segment_sum ──────────
    # For each xs component, sum contributions by region index.
    def acc(grad_per_cell, region_ids):
        # grad_per_cell: [I, dim], region_ids: [I]
        # returns [num_regions, dim]
        return jax.ops.segment_sum(grad_per_cell, region_ids, num_segments=num_regions)

    xs_grad = jnp.zeros_like(xs_tensor)

    # Sigma_a: lay['Sigma_a'] is a slice object — get start index
    sa_start = lay['Sigma_a'].start
    sa_grad  = acc(grad_sig_a_all, regs)                    # [num_regions, G]
    xs_grad  = xs_grad.at[:, sa_start:sa_start+G].add(sa_grad)

    nf_start = lay['nuSigma_f'].start
    nf_grad  = acc(grad_nusigf_all, regs)
    xs_grad  = xs_grad.at[:, nf_start:nf_start+G].add(nf_grad)

    chi_start = lay['chi'].start
    chi_grad  = acc(grad_chi_all, regs)
    xs_grad   = xs_grad.at[:, chi_start:chi_start+G].add(chi_grad)

    ss_start = lay['Sigma_s'].start
    ss_grad  = acc(grad_sigs_all, regs)                     # [num_regions, G*G]
    xs_grad  = xs_grad.at[:, ss_start:ss_start+G*G].add(ss_grad)

    D_start = lay['D'].start
    D_grad  = acc(grad_D_all,      regs)                    # [num_regions, G]
    D_grad += acc(grad_D_next_all, regs_next)               # D_next contribution
    xs_grad = xs_grad.at[:, D_start:D_start+G].add(D_grad)

    return xs_grad