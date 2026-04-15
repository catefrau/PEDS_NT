from .inverse_power import inverse_power, inverse_power_adjoint
import numpy as np


def create_grid(R,I):
    
    '''
    Create cell edges and centers for a domain of
    size R and for I cells
    
    Args:
        R: size of domain
        I: number of cells
        
    Returns:
        Delta_r: width of each cell
        centers: cell centers of the grid
        edges: cell edges of the grid
    '''
    
    Delta_r = float(R)/I # divide size by # of cells
    # calculate the centers by getting each Delta_r
    # and adding 0.5 Delta_R
    centers = np.arange(I)*Delta_r + 0.5*Delta_r
    # Get edges by going beyond one cell
    edges = np.arange(I+1)*Delta_r
    
    return Delta_r, centers, edges


def diffusion_setup(R, I, G, r_division, D, Sig_a, nuSig_f, Sigma_s, chi, BC, geometry):
    
    '''
    Multigroup 1D geometry diffusion eigenvalue setup:
    use cell-averaged quantities
    
    Args:
        R: size of domain
        I: number of cells
        D: function, D(r), returns array (list) of size G of diffusion coefficents at r
        Sig_a: function, Sig_a(r), returns array of size G of macroscopic abs xs at r
        nuSig_f: function, nuSig_f(r), return array of size G of nu*macroscopic fission xs at r
        BC: boundary conditions at r=R in form [A,B,C]
        geometry:
            0 = slab
            1 = cylindrical
            2 = spherical
    
    Returns:
        centers: cell centers of grid
        phi: cell-averaged value of scalar flux
    '''
    
    Delta_R, centers, edges = create_grid(R,I)

    # Helper: flat index from group g and spatial node i
    def idx(g, i):
        return g * (I+1) + i

    # Matrices for A and B containing loss and source terms, respectively
    # for MG formulation, these are (G*I+G) x G(I+1) matrices
    tot_size = G * (I+1)
    A = np.zeros((tot_size, tot_size)) 
    B = np.zeros((tot_size, tot_size)) 
    
    # define vectors of the surface areas S at cell edges
    # and  volumes V at dr
    if geometry == 0:  # slab
        S = np.zeros_like(edges)+1 # 1-D slab surface area
        S[0] = 0. # at the center make it zero, forces reflective BCs. Then define the BC in outer region r=R via the A, B, C coefficients
        # in slab it's dV = dr
        V = np.zeros_like(edges)+ + Delta_R
    
    elif geometry == 1: # cylinder
        # cylinder surface area is 4pi*r^2
        S = 2.*np.pi*edges
        # cylinder differential volume is 4pi*r^2
        # must substract inner cylinder from next outer cylinder
        V = np.pi * (edges[1:I+1]**2 - edges[0:I]**2)
    
    elif geometry == 2: # sphere
        # sphere surface area = 4 pi r^2
        S = 4*np.pi*edges**2
        # volume is 4/3 pi r^3, must subtract inner sphere from outer sphere
        V = 4/3 * np.pi * (edges[1:I+1]**3 - edges[0:I]**3)
    
     # ---- Fill matrix group by group, they will follow vertically 
    for g in range(G):

        # Boundary condition row for this group (last row of each group block)
        A[idx(g, I), idx(g, I)]   = (BC[0]/2 + BC[1]/Delta_R)
        A[idx(g, I), idx(g, I-1)] = (BC[0]/2 - BC[1]/Delta_R)

        # fill A matrix
        Dplus = 0
        for i in range(I): # fill spatial nodes for group g
            r = centers[i]
            Dminus = Dplus
            D_g = D(r, r_division)[g]
            D_g_next = D(r + Delta_R, r_division)[g]
            Dplus = 2 * D_g * D_g_next / (D_g + D_g_next)  # harmonic mean
            # Total removal = absorption + scattering OUT of group g
            # sum only OFF-diagonal: scattering OUT to other groups
            Sig_s_out = sum(
                Sigma_s(r, r_division)[g, gp]
                for gp in range(G) if gp != g
            ) 
            
            # ---- Diagonal: leakage + removal 
            A[idx(g,i), idx(g,i)] = (
                1/(Delta_R * V[i]) * Dplus * S[i+1] 
                + Sig_a(r, r_division)[g] + Sig_s_out
            )   
            # ---- Off-diagonal blocks: scattering IN from other groups g' -> g 
            for gp in range(G):
                if gp != g:
                    # Scattering from group gp into group g at node i
                    A[idx(g,i), idx(gp,i)] -= Sigma_s(r, r_division)[gp, g]   

                # ---- Fission matrix B 
                # Fission in group g' produces neutrons in group g via chi[g]
                B[idx(g,i), idx(gp,i)] =  chi(r, r_division)[g] * nuSig_f(r, r_division)[gp] # fission matrix
            
            if i > 0:
                A[idx(g,i), idx(g,i-1)] = -1 * Dminus/(Delta_R * V[i]) * S[i]
                A[idx(g,i), idx(g,i)] += 1. * Dminus/(Delta_R * V[i]) * S[i]
            
            A[idx(g,i), idx(g,i+1)] = -Dplus/(Delta_R * V[i]) * S[i+1]
     
    return centers, A, B
        

def DiffusionEigenvalue_MG(R, I, G, r_division,
                        D, Sig_a, nuSig_f,  Sigma_s, chi,
                        BC, geometry, epsilon=1e-8):
    
    
    centers, A, B = diffusion_setup(R, I, G, r_division, D, Sig_a, nuSig_f, Sigma_s, chi, BC, geometry)
    l, phi_values = inverse_power(A,B,epsilon)
    # phi_values is the eigenvector of size G(I+1), a 1D array containing the fluxes for all groups and spatial nodes
    # transform back from standard eigenvalue problem to generalized
    k = 1/l 
    # Reshape phi into (G, I): one flux vector per group
    phi = np.zeros((G, I))
    for g in range(G): # unpacking the eivenvector (solved of size G(I+1)) into group fluxes
        phi[g, :] = phi_values[g*(I+1) : g*(I+1) + I] # takes I entries 
        
    return k, phi, centers

def DiffusionEigenvalue_MG_adjoint(R, I, G, r_division,
                        D, Sig_a, nuSig_f, Sigma_s, chi,
                        BC, geometry, epsilon=1e-8):
    
    
    centers, A, B = diffusion_setup(R, I, G, r_division, D, Sig_a, nuSig_f, Sigma_s, chi, BC, geometry)
    """
     print("what is the shape of A?", A.shape)
    print("what is the shape of B?", B.shape)
    print("how does B look like?", B) """

     # --- build cell volumes ---
    Delta_r = R / I
    edges   = np.arange(I + 1) * Delta_r

    if geometry == 0:    # slab — uniform, correction cancels out
        V_cell = np.ones(I) * Delta_r
    elif geometry == 1:  # cylinder
        V_cell = np.pi * (edges[1:]**2 - edges[:-1]**2)
    elif geometry == 2:  # sphere
        V_cell = (4/3) * np.pi * (edges[1:]**3 - edges[:-1]**3)

    # Each group block has I cells + 1 BC row; set BC row weight = 1
    V_1g   = np.append(V_cell, 1.0)   # size I+1
    V_full = np.tile(V_1g, G)          # repeat for all G groups

    l, phi_values = inverse_power_adjoint(A,B,epsilon,V=V_full)
    # transform back from standard eigenvalue problem to generalized
    k = 1/l 
    # Reshape phi into (G, I): one flux vector per group
    phi = np.zeros((G, I))
    for g in range(G): # unpacking the eivenvector (solved of size G(I+1)) into group fluxes
        phi[g, :] = phi_values[g*(I+1) : g*(I+1) + I] # takes I entries 
        
    return k, phi, centers