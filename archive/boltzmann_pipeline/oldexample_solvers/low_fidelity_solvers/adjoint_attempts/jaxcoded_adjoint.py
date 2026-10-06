import jax
import jax.random as random
import jax.numpy as jnp
import jax.lax as lax
from jax import vmap
from flax import nnx
from jax.lib import xla_bridge

from jax.profiler import start_trace, stop_trace

@nnx.jit
def flux_kappa(diffusivity, Ts):
    # Initialize arrays for fluxes
    Jy = jnp.zeros_like(Ts)
    
    Jy = -diffusivity[:, :-1, :] * (Ts[:, 1:, :] - Ts[:, :-1, :]) / 1.0
    Jy = jnp.pad(Jy, ((0, 0), (0, 1), (0, 0)), mode='constant', constant_values=0)

    kappas = jnp.sum(Jy[:,   diffusivity.shape[1] // 2, :], axis=-1)

    return kappas

def count_copies(u_r):
    # Count the number of occurrences of the array with shape (16, 16, 3, 768), which appears exactly once in the
    # parameters to understand how many copies of the parameters currently exist
    return len([None for e in xla_bridge.get_backend().live_arrays() if e.shape == u_r.shape])

@nnx.jit
def fourier_solver_finalstep(diffusivity, u):

    batch_size = diffusivity.shape[0]
    N = diffusivity.shape[1]

    kappa_r = jnp.roll(diffusivity, shift=1, axis=1)
    kappa_l = jnp.roll(diffusivity, shift=-1, axis=1)
    kappa_d = jnp.roll(diffusivity, shift=-1, axis=2)
    kappa_u = jnp.roll(diffusivity, shift=1, axis=2)

    kappa_sum = kappa_r+kappa_l+kappa_d+kappa_u

    u_new = ((
        jnp.roll(u, shift=1, axis=1) * kappa_r +
        jnp.roll(u, shift=-1, axis=1) * kappa_l +
        jnp.roll(u, shift=-1, axis=2) * kappa_d +
        jnp.roll(u, shift=1, axis=2) * kappa_u
    ) / kappa_sum)
    

    # Apply boundary conditions
    u_new = u_new.at[:, 0, :].set(1 / 2)  # Top row
    u_new = u_new.at[:, -1, :].set(-1 / 2)  # Bottom row

    kappas = flux_kappa(diffusivity, u_new)

    return kappas


@jax.custom_vjp
def fourier_solver(diffusivity):
    batch_size = diffusivity.shape[0]
    N = diffusivity.shape[1]

    u = jnp.zeros((batch_size, N, N))

    kappa_r = jnp.roll(diffusivity, shift=1, axis=1)
    kappa_l = jnp.roll(diffusivity, shift=-1, axis=1)
    kappa_d = jnp.roll(diffusivity, shift=-1, axis=2)
    kappa_u = jnp.roll(diffusivity, shift=1, axis=2)

    kappa_sum = kappa_r+kappa_l+kappa_d+kappa_u

    def body_fn(u, _):
        # Shift arrays for each iteration (only u is updated)
        u_r = jnp.roll(u, shift=1, axis=1)
        u_l = jnp.roll(u, shift=-1, axis=1)
        u_d = jnp.roll(u, shift=-1, axis=2)
        u_u = jnp.roll(u, shift=1, axis=2)

        # Perform the update step
        u_new = update_step(u, kappa_sum, kappa_r, kappa_l, kappa_u, kappa_d, u_r, u_l, u_u, u_d)

        # Count the number of copies of u_r (and other temporary arrays if necessary)
        copies = count_copies(u_r)
        print(f"Number of copies of u_r: {copies}")

        return u_new, None

    # Use lax.scan to run iterations
    Ts, _ = lax.scan(body_fn, u, None, length=5000)

    count_copies(u)

    dx, dy = 100 / N, 100 / N

    kappas = flux_kappa(diffusivity, Ts)


    return kappas, Ts

nnx.jit(fourier_solver)

@nnx.jit
def update_step(u, kappa_sum, kappa_r, kappa_l, kappa_u, kappa_d, u_r, u_l, u_u, u_d):
    # Perform the update step using the shifted arrays
    u_new = ((
        u_r * kappa_r +
        u_l * kappa_l +
        u_d * kappa_d +
        u_u * kappa_u
    ) / kappa_sum)

    # Apply boundary conditions
    u_new = u_new.at[:, 0, :].set(1 / 2)  # Top row
    u_new = u_new.at[:, -1, :].set(-1 / 2)  # Bottom row

    return u_new


def fourier_fwd(diffusivity):
    res = fourier_solver(diffusivity)
    kappas, Ts = res
    return (kappas, Ts), (diffusivity, Ts)

@nnx.jit
def fourier_bwd(res, grads):
    """
    Backward function for fourier_solver.
    """
    diffusivity, T_final = res
    dL_dkappas, _ = grads

   
    # Compute VJP for the final step
    _, vjp_final_step = jax.vjp(fourier_solver_finalstep, diffusivity, T_final)
    
    # Use the gradient from the output of the final step (dL_dkappas)
    dL_ddiff, _ = vjp_final_step(dL_dkappas)

    return (dL_ddiff,) 

fourier_solver.defvjp(fourier_fwd, fourier_bwd)







import jax
import jax.numpy as jnp
import time

def main():
    # Define the test case
    batch_size = 500
    grid_size = 100
    iterations = 5000

    # Example diffusivity matrix
    diffusivity = jnp.ones((batch_size, grid_size, grid_size)) * 150.0
    #diffusivity = diffusivity.at[:, 1, 1].set(0.5)  # Add some variation for testing
    key = random.PRNGKey(0)  # Replace 0 with any seed for reproducibility

    """# Random diffusivity matrix
    diffusivity = random.uniform(
        key, shape=(batch_size, grid_size, grid_size), minval=0.5, maxval=150.0
    )"""

    kappas_real = random.uniform(key, shape=(batch_size,), minval=10, maxval=150.0)

    
    def loss_fn(diffusivity):
        res = fourier_solver(diffusivity)
        kappas, T = res
        #return jnp.mean(kappas)
        return 3* jnp.mean((kappas - kappas_real)**2)  # Example loss: sum of output kappas

    #loss, grads = jax.value_and_grad(loss_fn)(diffusivity)
    loss, grads = jax.value_and_grad(loss_fn)(diffusivity)
    
       



    
    
    #print(grads_custom[0:2])
    

if __name__ == "__main__":
    main()


