import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import nnx
from jax import random, lax
import time
import jax
import numpy as np

step_size = 5
N = int(100 / step_size)


def optimized_conductivity_grid_jax(pores, N):
    step_size = 100 / N
    size_square = int(10 * 1 / step_size)
    half_size_square = size_square // 2
    subgrid = jnp.ones((size_square, size_square)) * 1e-9
    indices = jnp.stack(jnp.meshgrid(jnp.arange(5), jnp.arange(5)), axis=-1).reshape(-1, 2) # pairs (x,y) of the grid

    batch_size = pores.shape[0]
    pores = jnp.reshape(pores, [batch_size, 5, 5])
    
    conductivity = jnp.ones((batch_size, N, N)) * 150.0
    
    for idx in indices:
        x_idx, y_idx = idx

        # Compute the start and end physical positions for the slice
        start_x = half_size_square + x_idx * size_square * 2
        start_y = half_size_square + y_idx * size_square * 2

        # Identify which batches to update
        mask = pores[:, x_idx, y_idx]  # Shape: (batch_size,)  — 1 if pore exists, 0 if not
        
        # Generate a full grid of subgrid positions for all batches
        update = mask[:, None, None] * subgrid[None, :, :]  # Shape: (batch_size, size_square, size_square)

        # Vectorized update of the conductivity grid
        conductivity = conductivity.at[
            :, start_x : start_x + size_square, start_y : start_y + size_square
        ].set(jnp.where(mask[:, None, None], update, conductivity[:, start_x : start_x + size_square, start_y : start_y + size_square]))
    #print("For Loop generation time:", time.time() - start_time)
    
    return conductivity

#nnx.jit(fun=optimized_conductivity_grid_jax, static_argnums=(1,2,3,4,5))

def conductivity_grid_5by5(pores):
    return jnp.reshape(jnp.where(pores, 1e-7, 160), (pores.shape[0], 5, 5))

   
def conductivity_original_wrapper(pores, N):

    if N >= 20: # size of grid for PEDS example is 25x25 

        return optimized_conductivity_grid_jax(pores, N)
    

    if N == 10:

        pass

    if N ==5:  # this is for the 'code_check' model where the conductivity grid is 5x5 and directly corresponds to the pore configuration.

        return conductivity_grid_5by5(pores)
    

if __name__ == "__main__":


    # Ingest data <- Here we will do active learning
    full_data = jnp.load("data/highfidelity/high_fidelity_2_13000.npz", allow_pickle=True)


    pores_data = full_data['pores'][:10000]  # Slice the first 10,000 entries

    # Reshape the data to have the shape (N, 5, 5), where N is the number of samples
    pores_reshaped = pores_data.reshape(pores_data.shape[0], 5, 5)

    # Now convert to a JAX array
    pores = jnp.asarray(pores_reshaped, dtype=jnp.float32)

    # Similarly, for kappas, if needed
    kappas = jnp.asarray(full_data['kappas'][:10000], dtype=jnp.float32)

    print(pores.shape)

    N = 5

    if N > 10:
        check_speed = time.time()
        conductivity_50 = optimized_conductivity_grid_jax(pores, N)
        print(f"Speed: {time.time()-check_speed} for {conductivity_50.shape} size")

        # Add the new representation in a file called "high_fidelity_10012_with_base.npz"
        # Save new representation
        """output_file = f"data/highfidelity/high_fidelity_10012_{N}steps.npz"
        jnp.savez(output_file, pores=pores, kappas=kappas, conductivity=conductivity_50)
        print(f"Saved extended dataset to {output_file}")"""

    if N== 10:
        pass

    if N == 5:

        check_speed = time.time()
        conductivity_50 = conductivity_grid_5by5(pores)
        print(f"Speed: {time.time()-check_speed} for {conductivity_50.shape} size")
