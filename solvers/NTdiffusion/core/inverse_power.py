import numpy as np
from scipy.linalg import lu_factor, lu_solve

# from  LU_factor import LU_factor, LU_solve

def inverse_power(A, B, epsilon=1e-6):
    
    '''
    Solve the generalized eigenvalue problem Ax = lB
    using the inverse power iteration algorithm
    
    Args:
        A: LHS matrix (cannot be singular)
        B: The RHS matrix
        epsilon: tolerance on eigenvalue (inverse of keff)
        x: associated eigenvector (the flux)
    Output:
        l: smallest eigenvalue of the problem
        x: associated eigenvector of smallest eigenvalue
    '''
    # print("--- Solving for the forward of the flux ---")
    Nrows, Ncols = A.shape
    # Generate guess
    x = np.random.random((Nrows)) # random initial guess for the eigenvector (the flux),
    x = x / np.linalg.norm(x)  # make norm(x) = 1
    
    l_old = 0 # initialise previous eigenvalue estimate for convergence check
    converged = 0
    # compute LU factorization of A
    A_lu, A_piv = lu_factor(A)
    iteration = 0
    b_0s = []
    while not(converged):
        
        iteration += 1
        
        b = lu_solve((A_lu, A_piv), np.dot(B,x)) # computes the product Bx and then solves Ab = Bx for b = A^-1 Bx, where A = LU
        b_0s.append(b[0]) # store the first element of b for sign check at the end
        l = np.linalg.norm(b) # estimate of 1/lambda, so keff
        x = b/l  # normalise the new eigenvector (flux shape) estimate
        
        converged = (np.fabs(l-l_old) < epsilon)
        #print(f"Iteration: {iteration}, magnitude of l: {1/l:.4f}, epsilon:{np.fabs(l-l_old):.3e}")
        l_old = l
    
    sign = b_0s[iteration-1]/b_0s[iteration-2]

    return sign/l, x

def inverse_power_adjoint_old(A, B, epsilon=1e-6):
    '''
    Solve the ADJOINT generalized eigenvalue problem A^T x = l B^T x
    using the inverse power iteration algorithm.
    
    The eigenvalue k is the same as the forward problem.
    The eigenvector is the ADJOINT flux (neutron importance).
    '''
    print("--- Going for the adjoint this time!!! ---")
    Nrows, Ncols = A.shape

    x = np.random.random((Nrows))
    x = x / np.linalg.norm(x)

    l_old = 0
    converged = False

    # Only change: transpose A and B before factorizing
    A_lu, A_piv = lu_factor(A.T)
    B_T = B.T

    iteration = 0
    b_0s = []

    while not converged:
        iteration += 1

        b = lu_solve((A_lu, A_piv), np.dot(B_T, x))
        b_0s.append(b[0])
        l = np.linalg.norm(b)
        x = b / l

        converged = (np.fabs(l-l_old) < epsilon)
        #print(f"Iteration: {iteration}, magnitude of l: {1/l:.4f}, epsilon:{np.fabs(l-l_old):.3e}")
        l_old = l

    sign = b_0s[iteration-1] / b_0s[iteration-2]
    return sign/l, x

def inverse_power_adjoint(A, B, epsilon=1e-6, V=None):
    '''
    Solve the ADJOINT generalized eigenvalue problem.
    For curvilinear geometries (sphere/cylinder), pass V (volume array)
    to apply the correct volume-weighted adjoint: V^{-1} A^T V
    '''
    # print("--- Going for the adjoint this time!!! ---")
    Nrows, Ncols = A.shape

    if V is not None:
        V_inv  = np.diag(1.0 / V)
        V_diag = np.diag(V)
        A_T = V_inv @ A.T @ V_diag   # correct adjoint for cylindrical or spherical geometry
        B_T = V_inv @ B.T @ V_diag
    else:
        A_T = A.T   # slab: volumes are uniform, so this still works
        B_T = B.T

    x = np.random.random((Nrows))
    x = x / np.linalg.norm(x)

    l_old = 0
    converged = False
    A_lu, A_piv = lu_factor(A_T)   # use corrected matrix
    iteration = 0
    b_0s = []

    while not converged:
        iteration += 1
        b = lu_solve((A_lu, A_piv), np.dot(B_T, x))  
        b_0s.append(b[0])
        l = np.linalg.norm(b)
        x = b / l
        converged = (np.fabs(l - l_old) < epsilon)
        l_old = l

    sign = b_0s[iteration-1] / b_0s[iteration-2]
    return sign/l, x