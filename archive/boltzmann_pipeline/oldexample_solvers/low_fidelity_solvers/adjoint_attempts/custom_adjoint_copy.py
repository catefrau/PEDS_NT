import jax
import jax.random as random
import jax.numpy as jnp
import jax.lax as lax
from jax import vmap
from flax import nnx

from utilities_lowfid import plot_temperature, flux_kappa, flux_kappa_single_batch


def fourier_solver_other(diffusivity, iterations=5000, display = False):
    batch_size = diffusivity.shape[0]
    N = diffusivity.shape[1]

    u = jnp.zeros((batch_size, N, N))

    kappa_r = jnp.roll(diffusivity, shift=1, axis=1)
    kappa_l = jnp.roll(diffusivity, shift=-1, axis=1)
    kappa_d = jnp.roll(diffusivity, shift=-1, axis=2)
    kappa_u = jnp.roll(diffusivity, shift=1, axis=2)

    kappa_sum = kappa_r+kappa_l+kappa_d+kappa_u

    def body_fn(u, _):
        u_new = update_step(u, kappa_sum, kappa_r, kappa_l, kappa_u, kappa_d)
        return u_new, None

    # Use lax.scan to run iterations
    Ts, _ = lax.scan(body_fn, u, None, length=iterations)

    #kappas = flux_kappa(diffusivity, Ts) # Ts, Jxs, Jys

    dx, dy = 100 / N, 100 / N

    # Initialize arrays for fluxes
    Jy = jnp.zeros_like(Ts)
    
    Jy = -diffusivity[:, :-1, :] * (Ts[:, 1:, :] - Ts[:, :-1, :]) / dy
    Jy = jnp.pad(Jy, ((0, 0), (0, 1), (0, 0)), mode='constant', constant_values=0)

    
    kappas = jnp.sum(Jy[:, N // 2, :], axis=-1)

    return kappas, Ts

#@nnx.jit
@jax.custom_vjp
def fourier_solver(diffusivity, iterations=5000):
    batch_size = diffusivity.shape[0]
    N = diffusivity.shape[1]

    u = jnp.zeros((batch_size, N, N))

    kappa_r = jnp.roll(diffusivity, shift=1, axis=1)
    kappa_l = jnp.roll(diffusivity, shift=-1, axis=1)
    kappa_d = jnp.roll(diffusivity, shift=-1, axis=2)
    kappa_u = jnp.roll(diffusivity, shift=1, axis=2)

    kappa_sum = kappa_r+kappa_l+kappa_d+kappa_u

    def body_fn(u, _):
        u_new = update_step(u, kappa_sum, kappa_r, kappa_l, kappa_u, kappa_d)
        return u_new, None

    # Use lax.scan to run iterations
    Ts, _ = lax.scan(body_fn, u, None, length=iterations)

    #kappas = flux_kappa(diffusivity, Ts) # Ts, Jxs, Jys

    dx, dy = 100 / N, 100 / N

    # Initialize arrays for fluxes
    Jy = jnp.zeros_like(Ts)
    
    Jy = -diffusivity[:, :-1, :] * (Ts[:, 1:, :] - Ts[:, :-1, :]) / dy
    Jy = jnp.pad(Jy, ((0, 0), (0, 1), (0, 0)), mode='constant', constant_values=0)

    
    kappas = jnp.sum(Jy[:,   N // 2, :], axis=-1)

    return kappas, Ts

@nnx.jit
def update_step(u, kappa_sum, kappa_r, kappa_l, kappa_u, kappa_d):

    u_new = ((
        jnp.roll(u, shift=1, axis=1) * kappa_r +
        jnp.roll(u, shift=-1, axis=1) * kappa_l +
        jnp.roll(u, shift=-1, axis=2) * kappa_d +
        jnp.roll(u, shift=1, axis=2) * kappa_u
    ) / kappa_sum)

    # Apply boundary conditions
    u_new = u_new.at[:, 0, :].set(1 / 2)  # Top row
    u_new = u_new.at[:, -1, :].set(-1 / 2)  # Bottom row

    return u_new

def fourier_fwd(diffusivity, iterations=5000):
    res = fourier_solver(diffusivity, iterations=iterations)
    kappas, Ts = res
    return (kappas, Ts), (diffusivity, Ts)


def fourier_bwd(res, grads):
    """
    Backward function for fourier_solver.
    """
    diffusivity, T = res
    dL_dkappas, _ = grads

    # Compute dL/d(diffusivity) via adjoint method

    # We want to compute dkappas_d(diffusivity)

    # We first compute dL/dk * dk/dJ (1) * dJ/d(diffusivity) (2)
    batch_size, N, _ = diffusivity.shape
    dx, dy = 100 / N, 100 / N

    # (1) Flux is just integral of the midline (so just those points influence the final result)
    dkappas_dJ = jnp.zeros_like(diffusivity) 
    dkappas_dJ = dkappas_dJ.at[:, N // 2 , :].set(1.0) 

    # (2) Remove the padded column in the gradient
    dJ_ddiff = jnp.zeros_like(diffusivity)
    
    # Gradient of J_y with respect to diffusivity
    dJ_ddiff = dJ_ddiff.at[:, :-1, :].add(- jnp.ones_like(diffusivity)[:, :-1, :] * (T[:, 1:, :] - T[:, :-1, :]) / dy)

    #print(f"(2) dJ/ddiff: (shape: {dJ_ddiff.shape}) \n", dJ_ddiff)

    # Then we iterate: dL/dk * dk/dJ * dJ/dT_5000 (3) * dT_5000/d(diffusivity) (4)
    # NOT IMPLEMENTED: and again: dL/dk * dk/dJ * dJ/dT_5000 * [ dT_5000/dT_4999 * dT_4999/d(diffusivity) ] (5)

    # (3)
    dJ_dT = jnp.zeros_like(diffusivity)
    dJ_dT = dJ_dT.at[:, 1:, :].add(-jnp.ones_like(diffusivity)[:, :-1, :] * diffusivity[:, :-1, :] / dy)
    
    # Gradients with respect to T[:, :-1, :]
    dJ_dT = dJ_dT.at[:, :-1, :].add(jnp.ones_like(diffusivity)[:, :-1, :] * diffusivity[:, :-1, :] / dy)

    #print(f"(3) dJ/dT: (shape: {dJ_dT.shape}) \n", dJ_dT)

    # (4) dT_ddiff

    kappa_r = jnp.roll(diffusivity, shift=1, axis=1)
    kappa_l = jnp.roll(diffusivity, shift=-1, axis=1)
    kappa_d = jnp.roll(diffusivity, shift=-1, axis=2)
    kappa_u = jnp.roll(diffusivity, shift=1, axis=2)

    kappa_sum = kappa_r+kappa_l+kappa_d+kappa_u

    # Calculate the rolled components of `u`
    T_r = jnp.roll(T, shift=1, axis=1)
    T_l = jnp.roll(T, shift=-1, axis=1)
    T_u = jnp.roll(T, shift=1, axis=2)
    T_d = jnp.roll(T, shift=-1, axis=2)

    tot = T_r * kappa_r + T_l * kappa_l + T_u * kappa_u + T_d * kappa_d

    # Gradients for kappa_r
    dT100_dkappa_r = (T_r * kappa_sum - tot) / (kappa_sum**2) # 
    # Mask boundary rows
    dT100_dkappa_r = dT100_dkappa_r.at[:, 0, :].set(0)  # Top row
    dT100_dkappa_r = dT100_dkappa_r.at[:, -1, :].set(0)  # Bottom row

    dT100_dkappa_l = (T_l * kappa_sum  -tot ) / (kappa_sum**2)
    dT100_dkappa_l = dT100_dkappa_l.at[:, 0, :].set(0)  # Top row
    dT100_dkappa_l = dT100_dkappa_l.at[:, -1, :].set(0)  # Bottom row

    dT100_dkappa_u = (T_u * kappa_sum -tot) / (kappa_sum**2)
    dT100_dkappa_u = dT100_dkappa_u.at[:, 0, :].set(0)  # Top row
    dT100_dkappa_u = dT100_dkappa_u.at[:, -1, :].set(0)  # Bottom row

    dT100_dkappa_d = (T_d * kappa_sum - tot) / (kappa_sum**2)
    dT100_dkappa_d = dT100_dkappa_d.at[:, 0, :].set(0)  # Top row
    dT100_dkappa_d = dT100_dkappa_d.at[:, -1, :].set(0)  # Bottom row

    # shift them back?? and sum?

    grad_kappa_r = jnp.roll(dT100_dkappa_r, shift=-1, axis=1)
    grad_kappa_l = jnp.roll(dT100_dkappa_l, shift=1, axis=1)
    grad_kappa_u = jnp.roll(dT100_dkappa_u, shift=-1, axis=2)
    grad_kappa_d = jnp.roll(dT100_dkappa_d, shift=1, axis=2)

    # Combine all gradients
    dT_dK = grad_kappa_r + grad_kappa_l + grad_kappa_u + grad_kappa_d

    #print(f"(4) dT/d(diff): (shape:{dT_dK.shape})\n", dT_dK)


    

    #dL_ddiff = dL_dkappas[:, None, None] * dkappas_dJ * (dJ_ddiff + dJ_dT * dT_dK)
    dL_ddiff = (dL_dkappas.T @ dkappas_dJ) * (dJ_ddiff + dJ_dT * dT_dK)

    
    return (dL_ddiff, None)

fourier_solver.defvjp(fourier_fwd, fourier_bwd)

import jax
import jax.numpy as jnp
import time

from Codes.solvers.low_fidelity_solvers.adjoint_attempts.jaxcoded_adjoint import fourier_solver as fourier_solver_jaxcoded

def main():
    # Define the test case
    batch_size = 1
    grid_size = 5
    iterations = 5000

    # Example diffusivity matrix
    diffusivity = jnp.ones((batch_size, grid_size, grid_size)) * 150.0
    diffusivity = diffusivity.at[:, 1, 1].set(0.5)  # Add some variation for testing
    key = random.PRNGKey(0)  # Replace 0 with any seed for reproducibility

    # Random diffusivity matrix
    """diffusivity = random.uniform(
        key, shape=(batch_size, grid_size, grid_size), minval=0.5, maxval=150.0
    )"""

    kappas_real = random.uniform(key, shape=(batch_size,), minval=10, maxval=150.0)
    # Define a loss function to compute gradients
    def loss_fn_default(diffusivity):
        res = fourier_solver_other(diffusivity)
        kappas, Ts = res
        return jnp.mean(kappas)
        #return 3* jnp.mean((kappas-kappas_real)**2)  # Example loss: sum of output kappas

    # Compute gradients using `jax.grad` (fallback to nnx.grad)
    start_time = time.time()
    grads_default = jax.grad(loss_fn_default)(diffusivity)
    default_time = time.time() - start_time
    print(f"Default grad computation time: {default_time:.4f}s")

    # Define a loss function to compute gradients
    def loss_fn_jaxcoded(diffusivity):
        res = fourier_solver_jaxcoded(diffusivity)
        kappas, Ts = res
        return jnp.mean(kappas)
        #return 3* jnp.mean((kappas-kappas_real)**2)  # Example loss: sum of output kappas

    # Compute gradients using `jax.grad` (fallback to nnx.grad)
    start_time = time.time()
    grads_jaxfinal = jax.grad(loss_fn_jaxcoded)(diffusivity)
    default_jaxfinal = time.time() - start_time
    print(f"Jax Only Final computation time: {default_jaxfinal:.4f}s")


    def loss_fn(diffusivity):
        res = fourier_solver(diffusivity)
        kappas, T = res
        return jnp.mean(kappas)
        #return 3* jnp.mean((kappas - kappas_real)**2)  # Example loss: sum of output kappas

    # Compute gradients using the custom VJP
    start_time = time.time()
    with jax.checking_leaks():  # Ensures no intermediate state leakage
        grads_custom = jax.grad(loss_fn)(diffusivity)
    custom_time = time.time() - start_time
    print(f"Custom VJP grad computation time: {custom_time:.4f}s")

    order_of_magnitude_default = jnp.log10(jnp.abs(grads_default) + 1e-8)
    order_of_magnitude_custom = jnp.log10(jnp.abs(grads_custom) + 1e-8)
    magnitude_diff = jnp.abs(order_of_magnitude_default - order_of_magnitude_custom)

    # Allow up to 1 order of magnitude difference
    same_order = jnp.all(magnitude_diff <= 1)

    # Check signs
    same_sign = jnp.all(jnp.sign(grads_default) == jnp.sign(grads_custom))

    # Print results
    #print(f"Same order of magnitude (within 1 order): {same_order}")
    #print(f"Same sign: {same_sign}")
    print(grads_custom[0])
    print(grads_default[0])

if __name__ == "__main__":
    main()






