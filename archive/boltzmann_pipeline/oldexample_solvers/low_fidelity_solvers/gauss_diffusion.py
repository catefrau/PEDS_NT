import jax.numpy as jnp
import jax.lax as lax
from flax import nnx
from jax import debug, checkpoint
import jax


#@nnx.jit
@jax.custom_vjp
def gauss_solver(conductivity, iterations=1000):
    batch_size = conductivity.shape[0]
    N = conductivity.shape[1]

    u = jnp.zeros((batch_size, N, N))

    for i in range(N):
        gradient = jnp.linspace(0.5, -0.5, N)
        u = u.at[:, :, i].set(gradient)


    kappa_r = jnp.roll(conductivity, shift=1, axis=1)
    kappa_l = jnp.roll(conductivity, shift=-1, axis=1)
    kappa_d = jnp.roll(conductivity, shift=-1, axis=2)
    kappa_u = jnp.roll(conductivity, shift=1, axis=2)

    epsilon = 1e-6  # Small value to prevent division by zero
    kappa_sum = kappa_r + kappa_l + kappa_d + kappa_u + epsilon

    def body_fn(u, _):  # 2D temperature field u updated at each iteration for the 4 neighbours
        
        u_r = jnp.roll(u, shift=1, axis=1)
        u_l = jnp.roll(u, shift=-1, axis=1)
        u_d = jnp.roll(u, shift=-1, axis=2)
        u_u = jnp.roll(u, shift=1, axis=2)

        
        u_new = update_step(kappa_sum, kappa_r, kappa_l, kappa_u, kappa_d, u_r, u_l, u_u, u_d)

        # Debug print to identify NaN values

        
        return u_new, None

    # Use lax.scan to run iterations
    Ts, _ = lax.scan(checkpoint(body_fn), u, None, length=iterations) 

    return Ts


#@nnx.jit
def update_step(kappa_sum, kappa_r, kappa_l, kappa_u, kappa_d, u_r, u_l, u_u, u_d):
    # Perform the update step using the shifted arrays
    u_new = ((
        u_r * kappa_r +
        u_l * kappa_l +
        u_d * kappa_d +
        u_u * kappa_u
    ) / kappa_sum)

    # Apply boundary conditions
    u_new = u_new.at[:,0,:].set(0.5)
    u_new = u_new.at[:, -1, :].set(u_new[:,0,:]-1.0)  # Top row
    
    

    return u_new


#@nnx.jit
def gauss_solver_finalstep(diffusivity, u):

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
    u_new = u_new.at[:, -1, :].set(u_new[:,0,:]-1.0)

   
    return u_new

def gauss_fwd(diffusivity, iterations=1000):
    Ts = gauss_solver(diffusivity, iterations=iterations)
    return Ts, (diffusivity, Ts)

#@nnx.jit
def gauss_bwd(res, grads):
    """
    Backward function for fourier_solver.
    """
    diffusivity, T_final = res
    dL_dTs = grads
   
    # Compute VJP for the final step
    _, vjp_final_step = jax.vjp(gauss_solver_finalstep, diffusivity, T_final)
    
    # Use the gradient from the output of the final step (dL_dkappas)
    dL_ddiff, _ = vjp_final_step(dL_dTs)

    return (dL_ddiff, None) 

gauss_solver.defvjp(gauss_fwd, gauss_bwd)





if __name__ == "__main__":

    from utilities_lowfid import test_solver
    test_solver(gauss_solver, num_obs=100, name_solver='gausseidel', fd_check=True)
