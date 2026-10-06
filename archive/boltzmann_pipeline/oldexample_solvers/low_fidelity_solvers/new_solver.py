import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

def harmonic_mean(a, b):
    """Compute harmonic mean for flux continuity."""
    return 2 * a * b / (a + b + 1e-8)  # Small epsilon to avoid division by zero

def build_sparse_laplacian(N, K):
    """
    Constructs the sparse Laplacian matrix with:
    - Periodic BCs on left/right.
    - Dirichlet BCs at top (u=0.5) and bottom (u=-0.5).
    """
    h = 1.0 / N  # Assume domain [0,1] x [0,1]

    # Compute harmonic averages of K at interfaces
    Kx = harmonic_mean(K, np.roll(K, -1, axis=0))  # x-direction
    Ky = harmonic_mean(K, np.roll(K, -1, axis=1))  # y-direction

    # Flatten for sparse matrix construction
    Kx = Kx.flatten()
    Ky = Ky.flatten()
    K  = K.flatten()

    # Main diagonal (- sum of neighboring fluxes)
    diag = -(Kx + np.roll(Kx, 1) + Ky + np.roll(Ky, N)) / h**2

    # X-direction (Periodic BCs)
    off_x = Kx / h**2
    off_xm = np.roll(off_x, 1)
    off_x[N-1::N] = 0  # Prevents cross-row connections
    off_xm[::N] = 0

    # Y-direction (Dirichlet BCs will modify this)
    off_y = Ky / h**2
    off_ym = np.roll(off_y, N)

    # Create matrix with periodic BCs already applied
    L = sp.diags([diag, off_x, off_xm, off_y, off_ym],
                 [0, 1, -1, N, -N], shape=(N**2, N**2), format="csr")

    # Apply periodic BCs for left-right direction (wrap around)

    LL = L.toarray()  # Convert CSR to a dense NumPy array
    print(LL)

    # Apply periodic BCs for left-right direction (wrap around)

        # Connect the right column to the left (periodically)
        #L[idx_left_next_row, idx_right_prev_row] = Kx[idx_right] / h**2
        #L[idx_right_prev_row, idx_left_next_row] = Kx[idx_right] / h**2

    # Apply Dirichlet BCs in the sparse matrix during construction
    for i in range(N):
        # Top boundary: u = 0.5
        idx_top = i  # Indices for top boundary (row 0)
        L[idx_top, :] = 0
        L[idx_top, idx_top] = 1  # Set diagonal to 1 (Dirichlet BC)

        # Bottom boundary: u = -0.5
        idx_bottom = (N - 1) * N + i  # Indices for bottom boundary (row N-1)
        L[idx_bottom, :] = 0
        L[idx_bottom, idx_bottom] = 1  # Set diagonal to 1 (Dirichlet BC)"""
    
    # Apply periodic BCs for left-right direction (wrap around)"""

    for j in range(N):
        # Left-right periodic connection (column 0 <-> column N-1)
        idx_left = j * N
        idx_right = (j + 1) * N - 1
        #idx_left_next_row = ((j + 1) % N) * N  # Wrap to the next row for left
        #idx_right_prev_row = ((j - 1) % N) * N + N - 1  # Wrap to the previous row for right

        # Connect the left column to the right (periodically)
        L[idx_left, idx_right] = Kx[idx_left] / h**2
        L[idx_right, idx_left] = Kx[idx_right] / h**2 #


    return L


# Grid size
N = 50

# Generate a random heterogeneous conductivity field (positive values)
#np.random.seed(42)
#K = np.exp(np.random.randn(N, N))  # Log-normal distributed K

K = np.ones((N,N)) * 0.1

# Construct Laplacian matrix
L = build_sparse_laplacian(N, K)

# Right-hand side (forcing term)
f = np.ones(N * N)

# Apply Dirichlet BCs in f
f[np.arange(N * (N-1), N**2)] = 0.5  # Top boundary
f[np.arange(0, N)] = -0.5  # Bottom boundary

# Solve the sparse system
u = spla.spsolve(L, f)

# Reshape solution back to 2D grid
u = u.reshape(N, N)

# Plot solution
plt.imshow(u, cmap='hot', interpolation='nearest')
plt.colorbar(label='Solution u')
plt.title("Solution to 2D Poisson Equation (Periodic LR, Dirichlet TB)")
plt.show()

