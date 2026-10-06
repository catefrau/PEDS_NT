import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import jax
import matplotlib.pyplot as plt
import time


import sparse 

def laplacian(conductivity, h=1.0):
    """
    Computes the discrete Laplacian using the Kronecker product.

    Args:
        conductivity (np.array): (N-1)x(N-1) array with mean conductivity for every cell of the grid.
        h (float): Grid spacing.

    Returns:
        scipy.sparse.csr_matrix: Sparse Laplacian matrix.
    """
   
    batch_size, N_cells, _ = conductivity.shape 
    N = N_cells + 1  # Number of points (including boundaries)

    # Define the step sizes (dx1 for forward and dx2 for backward)
    dx1 = np.ones(N - 1)  # Forward step
    dx1[-1] = 0
    dx2 = -np.ones(N-2) # Backward step

    # Create sparse matrix for Laplacian
    D = sp.diags([dx1, dx2], [0, 1], shape=(N - 1, N - 1), format="csr")

    # Scale by h^2 for the Laplacian (if required)
    D = D / (h**2) # can we delete it??


    # Create identity matrix for the size of the Laplacian matrix
    I = sp.eye(N - 1, format="csr")

    Kx = sp.kron(D, I, format="csr")
    Ky = sp.kron(I, D, format="csr")

    Ky = Ky.tolil()

    # Set the first and last N_cells rows to 0, except for the diagonals
    Ky[:N_cells, :] = 0
    Ky[-N_cells:, :] = 0

    # Set the diagonal elements to 1 (where necessary)
    Ky[np.arange(N_cells), np.arange(N_cells)] = 1
    Ky[np.arange(N_cells*(N_cells-1), N_cells**2), np.arange(N_cells*(N_cells-1), N_cells**2)] = 1


    # Convert back to CSR format for efficient computation
    Ky = Ky.tocsr()

    # Stack Kx and Ky vertically to form the 2D Laplacian
    L = sp.vstack([Kx, Ky])


    # repeat L batch_size times to have a final sparse array of size (batch_size, L.shape[0], L.shape[1])
    L_batch = sp.block_diag([L] * batch_size, format="csr")

    # Flatten
    conductivity_flat = conductivity.reshape(batch_size, N_cells ** 2)
    
    # tile to make it double 
    conductivity_flat = np.concatenate([conductivity_flat, conductivity_flat], axis=1).flatten()
    
    C_batch = sp.diags(conductivity_flat, 0, format="csr")

    L_final = -(L_batch.T @ C_batch @ L_batch)


    return L_final, L

@jax.custom_vjp
def fd_diffusion(conductivity):

    batch_size, N_cells, _ = conductivity.shape 

    L, _ = laplacian(conductivity)

    #assert np.allclose(L[0].toarray(), L[0].toarray().transpose()), "L is not symmetric!"I 
    #print("Assertion passed! L is Self-Adjoint!!")
    
    # Source term calculation
    S = np.zeros((batch_size, N_cells, N_cells))
    S[:, 0, :] = -0.5 
    S[:, -1, :] = +0.5

    # Print Source term
    S_flat = S.flatten()
   
    T_flat = spsolve(L, S_flat)
    
    # Reshape the solution to 2D
    T = T_flat.reshape((batch_size, N_cells, N_cells))

    return 150.0 * T


def fd_fwd(conductivity):
    T = fd_diffusion(conductivity)
    return T, (conductivity, T)


def fd_bwd(res, dl_dT):

    K, T = res

    batch_size, N_cells, _ = T.shape
    
    L, L_small = laplacian(K)

    L_repeated_list = [L_small] * batch_size

    L_batch_sparse= sp.vstack(L_repeated_list)

    L_batch_sparse = sparse.COO(L_batch_sparse)

    N_cells_sq = N_cells**2

    L_batch_sparse = L_batch_sparse.reshape((batch_size, 2 * N_cells_sq, N_cells_sq))

    dL_dk_tiled_sparse = -sparse.einsum('bri,brj->bijr', L_batch_sparse, L_batch_sparse)
   
    dL_dk_sparse = dL_dk_tiled_sparse.reshape(( batch_size, N_cells_sq, N_cells_sq, 2, N_cells, N_cells))

    dL_dk_sparse = dL_dk_sparse.sum(axis=3)

    dL_dk_sparse = dL_dk_sparse.reshape((batch_size, N_cells_sq,N_cells_sq, N_cells, N_cells))

    lambd = spsolve(L, dl_dT.flatten())

    lambd = lambd.reshape((batch_size, N_cells_sq))
    T = T.reshape((batch_size, N_cells_sq))

    dfdK_sparse = sparse.einsum('hi,hijab,hj->hab', lambd, dL_dk_sparse, T)
    dfdK_sparse = dfdK_sparse.reshape(K.shape)

    return (-dfdK_sparse.todense(),)




fd_diffusion.defvjp(fd_fwd, fd_bwd)


if __name__ == "__main__":

    from utilities_lowfid import test_solver

    test_solver(fd_diffusion, 200, "laplacian", fd_check = True)