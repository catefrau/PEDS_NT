import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(42)

B, N, M = 4, 10, 10

diffusivity = jax.random.uniform(key, shape=(B, N, M), dtype=jnp.float32)
Ts = jax.random.uniform(key, shape=(B, N, M), dtype=jnp.float32)

dy = 1.0

@jax.custom_vjp
def compute_Jy(diffusivity, Ts, dy):
    Jy = -diffusivity[:, :-1, :] * (Ts[:, 1:, :] - Ts[:, :-1, :]) / dy
    Jy = jnp.pad(Jy, ((0, 0), (0, 1), (0, 0)), mode='constant', constant_values=0)
    return Jy

def compute_Jy_fwd(diffusivity, Ts, dy):
    Jy = -diffusivity[:, :-1, :] * (Ts[:, 1:, :] - Ts[:, :-1, :]) / dy
    Jy = jnp.pad(Jy, ((0, 0), (0, 1), (0, 0)), mode='constant', constant_values=0)
    return Jy, (diffusivity, Ts, dy)

def compute_Jy_bwd(inputs, g):
    diffusivity, Ts, dy = inputs
    grad = jnp.zeros_like(Ts, dtype=jnp.float32)
    g = g[:, :-1, :]
    
    # Gradients with respect to Ts[:, 1:, :]
    grad = grad.at[:, 1:, :].add(-g[:, :, :] * diffusivity[:, :-1, :] / dy)
    
    # Gradients with respect to Ts[:, :-1, :]
    grad = grad.at[:, :-1, :].add(g[:, :, :] * diffusivity[:, :-1, :] / dy)
    
    return (None, grad, None)

compute_Jy.defvjp(compute_Jy_fwd, compute_Jy_bwd)

grad_fn_custom = jax.grad(lambda Ts: jnp.sum(compute_Jy(diffusivity, Ts, dy)), argnums=0)
dkappas_dTs_custom = grad_fn_custom(Ts)

def compute_Jy_default(diffusivity, Ts, dy):
    Jy = -diffusivity[:, :-1, :] * (Ts[:, 1:, :] - Ts[:, :-1, :]) / dy
    Jy = jnp.pad(Jy, ((0, 0), (0, 1), (0, 0)), mode='constant', constant_values=0)
    return Jy

grad_fn_default = jax.grad(lambda Ts: jnp.sum(compute_Jy_default(diffusivity, Ts, dy)), argnums=0)
dkappas_dTs_default = grad_fn_default(Ts)

gradients_are_equal = jnp.allclose(dkappas_dTs_custom, dkappas_dTs_default)
print("Are the gradients equal? ", gradients_are_equal)
