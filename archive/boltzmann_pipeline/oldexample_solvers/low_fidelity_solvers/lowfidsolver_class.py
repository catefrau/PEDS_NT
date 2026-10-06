from flax import nnx
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax


from solvers.low_fidelity_solvers.gauss_diffusion import gauss_solver
from solvers.low_fidelity_solvers.fd_diffusion import fd_diffusion

class lowfid(nnx.Module):
   
    def __init__(self, solver="gauss", iterations=5000):
        self.iterations = iterations
        self.solver = solver

    #@nnx.vmap(in_axes=0, out_axes=0)
    def __call__(self, conductivity):
        if self.solver == "gauss":
            T = gauss_solver(conductivity)
            """if jnp.isinf(T).any() or jnp.isnan(T).any():
                # Find indices where values are infinity or NaN
                inf_indices = jnp.where(jnp.isinf(T))
                nan_indices = jnp.where(jnp.isnan(T))



                # If there are any inf or NaN values, raise an error and print their indices
                if len(inf_indices[0]) > 0 or len(nan_indices[0]) > 0:

                    # Stop gradient for plotting
                    conductivity_plot = jax.lax.stop_gradient(conductivity)
                    temperature_plot = jax.lax.stop_gradient(T)

                    # Iteration None, First NaN at (Array([-1], dtype=int32), Array([-1], dtype=int32), Array([-1], dtype=int32)) in u_new
                    # Iteration None, First NaN at (Array([23], dtype=int32), Array([12], dtype=int32), Array([6], dtype=int32)) in u_new
                    # Iteration None, First NaN at (Array([23], dtype=int32), Array([11], dtype=int32), Array([6], dtype=int32)) in u_new

                    # [[137.0327    116.898224  113.38044  ]
                    # [ 56.11926   131.1144     79.34289  ]
                    # [ 36.57471    -0.2092512  54.503277 ]]

                    # Iteration None, First NaN at (Array([-1], dtype=int32), Array([-1], dtype=int32), Array([-1], dtype=int32)) in u_new
                    # Iteration None, First NaN at (Array([23], dtype=int32), Array([13], dtype=int32), Array([6], dtype=int32)) in u_new
                    # Iteration None, First NaN at (Array([23], dtype=int32), Array([12], dtype=int32), Array([6], dtype=int32)) in u_new

                    # [[  57.37656    131.86316     79.64146 ]
                    # [  33.720364     1.1465117   55.24927  ]
                    # [-126.8251     -73.9943     121.52798  ]]

                    print(conductivity_plot[23, 11:15, 4:8])

                    # Set up the figure and axes grid
                    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

                    # Plot conductivity heatmap for Batch 23
                    im1 = axes[0, 0].imshow(conductivity_plot[23, :, :], cmap='viridis', origin='lower')
                    axes[0, 0].set_title('Conductivity Heatmap (Batch 23)')
                    axes[0, 0].set_xlabel('X-axis')
                    axes[0, 0].set_ylabel('Y-axis')
                    fig.colorbar(im1, ax=axes[0, 0], label='Conductivity')

                    # Plot temperature heatmap for Batch 23
                    im2 = axes[0, 1].imshow(temperature_plot[23, :, :], cmap='coolwarm', origin='lower')
                    axes[0, 1].set_title('Temperature Heatmap (Batch 23)')
                    axes[0, 1].set_xlabel('X-axis')
                    axes[0, 1].set_ylabel('Y-axis')
                    fig.colorbar(im2, ax=axes[0, 1], label='Temperature')

                    # Plot conductivity heatmap for Batch 0
                    im3 = axes[1, 0].imshow(conductivity_plot[0, :, :], cmap='viridis', origin='lower')
                    axes[1, 0].set_title('Conductivity Heatmap (Batch 0)')
                    axes[1, 0].set_xlabel('X-axis')
                    axes[1, 0].set_ylabel('Y-axis')
                    fig.colorbar(im3, ax=axes[1, 0], label='Conductivity')

                    # Plot temperature heatmap for Batch 0
                    im4 = axes[1, 1].imshow(temperature_plot[0, :, :], cmap='coolwarm', origin='lower')
                    axes[1, 1].set_title('Temperature Heatmap (Batch 0)')
                    axes[1, 1].set_xlabel('X-axis')
                    axes[1, 1].set_ylabel('Y-axis')
                    fig.colorbar(im4, ax=axes[1, 1], label='Temperature')

                    # Adjust layout to prevent overlapping
                    plt.tight_layout()
                    plt.show()

                    print("Infinity values found at indices:", inf_indices)
                    print("NaN values found at indices:", nan_indices)
                    raise ValueError("The computed T contains infinity or nan values.")"""
        
        if self.solver == "direct":
            T = fd_diffusion(conductivity)
        return flux_kappa(conductivity, T)

#@nnx.jit
def flux_kappa(conductivity, Ts):
    # Initialize arrays for fluxes
    Jy = jnp.zeros_like(Ts)
    
    Jy = -conductivity[:, :-1, :] * (Ts[:, 1:, :] - Ts[:, :-1, :]) / 1.0
    Jy = jnp.pad(Jy, ((0, 0), (0, 1), (0, 0)), mode='constant', constant_values=0)

    kappas = jnp.sum(Jy[:,   conductivity.shape[1] // 2, :], axis=-1)

    return kappas





    
    