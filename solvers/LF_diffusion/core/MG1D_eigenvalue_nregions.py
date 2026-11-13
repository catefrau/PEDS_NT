from .inverse_power import inverse_power, inverse_power_adjoint
import numpy as np


def create_grid(R, I):
    """
    Create cell edges and centers for a uniform mesh over [0, R].

    Args:
        R : size of domain
        I : number of cells

    Returns:
        Delta_r : width of each cell
        centers : cell centers array of size I
        edges   : cell edges array of size I+1
    """
    Delta_r = float(R) / I
    centers = np.arange(I) * Delta_r + 0.5 * Delta_r
    edges   = np.arange(I + 1) * Delta_r
    return Delta_r, centers, edges


def region_index(r, r_divisions):
    """
    Return which region the position r belongs to, given a sorted list
    of N-1 internal boundary positions r_divisions.

    The domain is split into N regions:
        Region 0 : [0,          r_divisions[0])
        Region 1 : [r_divisions[0], r_divisions[1])
        ...
        Region N-1: [r_divisions[N-2], R]

    This is a convenience helper you can use inside your material
    functions instead of writing if/elif chains.

    Args:
        r            : radial (or axial) position
        r_divisions  : list or array of N-1 internal boundaries, sorted
                       ascending. For a single-region problem pass [].
                       For the original 2-region problem pass [r_div].

    Returns:
        idx : integer region index (0-based)

    Example
    -------
    # 3-region fuel / gap / reflector sphere
    r_divisions = [2.0, 2.2]   # fuel up to 2.0, gap 2.0-2.2, refl beyond

    def D(r, r_divisions):
        n = region_index(r, r_divisions)
        return [D_fuel, D_gap, D_refl][n]          # pick the right value
    """
    return int(np.searchsorted(r_divisions, r, side='right'))


def diffusion_setup(R, I, G, r_divisions,
                    D, Sig_a, nuSig_f, Sigma_s, chi,
                    BC, geometry):
    """
    Multigroup 1D diffusion eigenvalue matrix assembly (cell-averaged).

    The geometry is divided into N regions by the list r_divisions,
    which contains N-1 internal boundaries (sorted, ascending).
    Material functions receive the position r and the full r_divisions
    list and must return the correct cross-section for that position.

    Args:
        R           : outer boundary of domain
        I           : total number of uniform cells
        G           : number of energy groups
        r_divisions : list of N-1 internal region boundaries (sorted).
                      Use [] for a single-region problem.
                      Use [r1] for the original 2-region problem.
                      Use [r1, r2, ..., r_{N-1}] for N regions.
        D           : function D(r, r_divisions)       -> array size G
        Sig_a       : function Sig_a(r, r_divisions)   -> array size G
        nuSig_f     : function nuSig_f(r, r_divisions) -> array size G
        Sigma_s     : function Sigma_s(r, r_divisions) -> array size (G, G)
        chi         : function chi(r, r_divisions)     -> array size G
        BC          : outer boundary condition [A, B, C]
        geometry    : 0 = slab, 1 = cylinder, 2 = sphere

    Returns:
        centers : cell-center positions, array size I
        A       : loss matrix,   size G*(I+1) x G*(I+1)
        B       : source matrix, size G*(I+1) x G*(I+1)
    """
    Delta_R, centers, edges = create_grid(R, I)

    def idx(g, i):
        return g * (I + 1) + i

    tot_size = G * (I + 1)
    A = np.zeros((tot_size, tot_size))
    B = np.zeros((tot_size, tot_size))

    # --- geometry-dependent surface areas and volumes ---
    if geometry == 0:   # slab
        S = np.ones_like(edges)
        S[0] = 0.        # reflective centre
        V = np.full(I, Delta_R)

    elif geometry == 1:  # cylinder
        S = 2.0 * np.pi * edges
        V = np.pi * (edges[1:I+1]**2 - edges[0:I]**2)

    elif geometry == 2:  # sphere
        S = 4.0 * np.pi * edges**2
        V = (4.0/3.0) * np.pi * (edges[1:I+1]**3 - edges[0:I]**3)

    # --- fill A and B group by group ---
    for g in range(G):

        # outer boundary condition row
        A[idx(g, I), idx(g, I)]   =  BC[0]/2.0 + BC[1]/Delta_R
        A[idx(g, I), idx(g, I-1)] =  BC[0]/2.0 - BC[1]/Delta_R

        Dplus = 0.0
        for i in range(I):
            r = centers[i]
            Dminus = Dplus

            D_g      = D(r,            r_divisions)[g]
            D_g_next = D(r + Delta_R,  r_divisions)[g]
            Dplus    = 2.0 * D_g * D_g_next / (D_g + D_g_next)   # harmonic mean

            # scattering out of group g (sum over all other groups g')
            Sig_s_out = sum(
                Sigma_s(r, r_divisions)[g, gp]
                for gp in range(G) if gp != g
            )

            # diagonal: leakage + absorption + out-scatter
            A[idx(g, i), idx(g, i)] = (
                  Dplus / (Delta_R * V[i]) * S[i+1]
                + Sig_a(r, r_divisions)[g]
                + Sig_s_out
            )

            # off-diagonal: in-scatter from group gp -> g  &  fission
            for gp in range(G):
                if gp != g:
                    A[idx(g, i), idx(gp, i)] -= Sigma_s(r, r_divisions)[gp, g]

                # fission: neutrons born in group g from fission in group gp
                B[idx(g, i), idx(gp, i)] = (
                    chi(r, r_divisions)[g] * nuSig_f(r, r_divisions)[gp]
                )

            if i > 0:
                A[idx(g, i), idx(g, i-1)] -= Dminus / (Delta_R * V[i]) * S[i]
                A[idx(g, i), idx(g, i)]   += Dminus / (Delta_R * V[i]) * S[i]

            A[idx(g, i), idx(g, i+1)] = -Dplus / (Delta_R * V[i]) * S[i+1]

    return centers, A, B


def DiffusionEigenvalue_MG(R, I, G, r_divisions,
                           D, Sig_a, nuSig_f, Sigma_s, chi,
                           BC, geometry, epsilon=1e-8):
    """
    Solve the forward multigroup diffusion eigenvalue problem.

    Returns:
        k      : effective multiplication factor k_eff
        phi    : flux array of shape (G, I)
        centers: cell-center positions
    """
    centers, A, B = diffusion_setup(R, I, G, r_divisions,
                                    D, Sig_a, nuSig_f, Sigma_s, chi,
                                    BC, geometry)
    l, phi_values = inverse_power(A, B, epsilon)
    k = 1.0 / l

    phi = np.zeros((G, I))
    for g in range(G):
        phi[g, :] = phi_values[g*(I+1) : g*(I+1) + I]

    return k, phi, centers


def DiffusionEigenvalue_MG_adjoint(R, I, G, r_divisions,
                                   D, Sig_a, nuSig_f, Sigma_s, chi,
                                   BC, geometry, epsilon=1e-8):
    """
    Solve the adjoint multigroup diffusion eigenvalue problem.

    Returns:
        k      : effective multiplication factor k_eff (same as forward)
        phi    : adjoint flux array of shape (G, I)
        centers: cell-center positions
    """
    centers, A, B = diffusion_setup(R, I, G, r_divisions,
                                    D, Sig_a, nuSig_f, Sigma_s, chi,
                                    BC, geometry)

    Delta_r = R / I
    edges   = np.arange(I + 1) * Delta_r

    if geometry == 0:
        V_cell = np.ones(I) * Delta_r
    elif geometry == 1:
        V_cell = np.pi * (edges[1:]**2 - edges[:-1]**2)
    elif geometry == 2:
        V_cell = (4.0/3.0) * np.pi * (edges[1:]**3 - edges[:-1]**3)

    V_1g   = np.append(V_cell, 1.0)   # I cells + 1 BC row
    V_full = np.tile(V_1g, G)          # repeat for all groups

    l, phi_values = inverse_power_adjoint(A, B, epsilon, V=V_full)
    k = 1.0 / l

    phi = np.zeros((G, I))
    for g in range(G):
        phi[g, :] = phi_values[g*(I+1) : g*(I+1) + I]

    return k, phi, centers
