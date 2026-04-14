

import jax.numpy as jnp

def diffusion_setup_jax(xs_tensor, geo_data, lay):
    """
    JAX-traceable version of diffusion_setup.
    xs_tensor: shape [num_regions, M]  — JAX array (NN output)
    geo_data:  dict from precompute_geometry (all plain Python/NumPy)
    lay:       dict from xs_layout(G), maps XS names to indices in xs_tensor

    Returns:
        A: jnp array shape [G*(I+1), G*(I+1)]
        B: jnp array shape [G*(I+1), G*(I+1)]
    """
    G              = geo_data['G']
    I              = geo_data['I']
    Delta_r        = geo_data['Delta_r']
    S              = geo_data['S']          # shape [I+1], plain numpy
    V              = geo_data['V']          # shape [I],   plain numpy
    BC             = geo_data['BC']
    region_of_cell = geo_data['region_of_cell']  # Python list of ints

    tot_size = G * (I + 1)

    def idx(g, i):
        return g * (I + 1) + i

    A = jnp.zeros((tot_size, tot_size))
    B = jnp.zeros((tot_size, tot_size))

    # ── Boundary condition rows ───────────────────────────────────────
    for g in range(G):
        A = A.at[idx(g, I), idx(g, I  )].set(BC[0]/2 + BC[1]/Delta_r)
        A = A.at[idx(g, I), idx(g, I-1)].set(BC[0]/2 - BC[1]/Delta_r)

    # ── Interior rows ─────────────────────────────────────────────────
    Dplus_prev = 0.0

    for i in range(I):
        reg      = region_of_cell[i]          # Python int — safe for tracing
        reg_next = region_of_cell[min(i+1, I-1)]

        # --- pull XS values from xs_tensor (these are JAX ops) ---
        D_g      = xs_tensor[reg,      lay['D']]           # shape [G]
        D_g_next = xs_tensor[reg_next, lay['D']]           # shape [G]
        Sig_a    = xs_tensor[reg,      lay['Sigma_a']]     # shape [G]
        nuSigf   = xs_tensor[reg,      lay['nuSigma_f']]   # shape [G]
        chi_g    = xs_tensor[reg,      lay['chi']]         # shape [G]
        Sig_s    = xs_tensor[reg,      lay['Sigma_s']]     # shape [G*G] → reshape below
        Sig_s_mat = Sig_s.reshape(G, G)                    # shape [G, G]

        for g in range(G):
            # harmonic mean diffusion coefficients
            Dplus  = 2*D_g[g]*D_g_next[g] / (D_g[g] + D_g_next[g] + 1e-30)
            Dminus = Dplus_prev if i > 0 else 0.0   # ← careful: use stored value

            # scattering out of group g
            Sig_s_out = sum(Sig_s_mat[g, gp] for gp in range(G) if gp != g)

            # diagonal entry
            diag = (1/(Delta_r * V[i])) * Dplus * S[i+1] + Sig_a[g] + Sig_s_out
            A = A.at[idx(g,i), idx(g,i)].add(diag)

            # leakage to left neighbor
            if i > 0:
                A = A.at[idx(g,i), idx(g,i-1)].add(-Dminus/(Delta_r * V[i]) * S[i])
                A = A.at[idx(g,i), idx(g,i  )].add( Dminus/(Delta_r * V[i]) * S[i])

            # leakage to right neighbor
            A = A.at[idx(g,i), idx(g,i+1)].add(-Dplus/(Delta_r * V[i]) * S[i+1])

            # scattering in from other groups
            for gp in range(G):
                if gp != g:
                    A = A.at[idx(g,i), idx(gp,i)].add(-Sig_s_mat[gp, g])

            # fission matrix B
            for gp in range(G):
                val = chi_g[g] * nuSigf[gp]
                B = B.at[idx(g,i), idx(gp,i)].add(val)

        Dplus_prev = Dplus   # carry forward for next cell

    return A, B