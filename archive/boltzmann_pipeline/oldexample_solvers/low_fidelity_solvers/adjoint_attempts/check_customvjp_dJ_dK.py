import jax
import jax.numpy as jnp

# Example inputs
B, N, M = 3, 100, 100  # Batch size, N-dimension, M-dimension
key = jax.random.PRNGKey(42)

# Generate a non-trivial random diffusivity matrix (uniform distribution between 0 and 1)
diffusivity = jax.random.uniform(key, shape=(B, N, M), dtype=jnp.float32)

Ts = jax.random.uniform(key, shape=(B, N, M), dtype=jnp.float32)
dy = 1.0  # Grid spacing

# Define the forward function for J_y computation with padding
@jax.custom_vjp
def compute_Jy(diffusivity, Ts, dy):
    Jy = -diffusivity[:, :-1, :] * (Ts[:, 1:, :] - Ts[:, :-1, :]) / dy
    Jy = jnp.pad(Jy, ((0, 0), (0, 1), (0, 0)), mode='constant', constant_values=0)
    return Jy

# Forward function for custom VJP
def compute_Jy_fwd(diffusivity, Ts, dy):
    Jy = -diffusivity[:, :-1, :] * (Ts[:, 1:, :] - Ts[:, :-1, :]) / dy
    Jy = jnp.pad(Jy, ((0, 0), (0, 1), (0, 0)), mode='constant', constant_values=0)
    return Jy, (diffusivity, Ts, dy)  # Pass inputs to the backward function

# Backward function for custom VJP
def compute_Jy_bwd(inputs, g):
    diffusivity, Ts, dy = inputs
    grad = jnp.zeros_like(diffusivity)  # Initialize gradient

    # Remove the padded column in the gradient
    g = g[:, :-1, :]  # Remove the last column (which was padded)



    # Gradient of J_y with respect to diffusivity
    grad = grad.at[:, :-1, :].add(- g[:, :, :] * (Ts[:, 1:, :] - Ts[:, :-1, :]) / dy)

    return (grad, None, None)  # No gradient w.r.t. Ts and dy

# Associate forward and backward functions with custom VJP
compute_Jy.defvjp(compute_Jy_fwd, compute_Jy_bwd)

# Compute adjoint using custom VJP
grad_fn_custom = jax.grad(lambda diffusivity: jnp.sum(compute_Jy(diffusivity, Ts, dy)), argnums=0)
dkappas_ddiffusivity_custom = grad_fn_custom(diffusivity)

# Define the same function using JAX's default gradient
def compute_Jy_default(diffusivity, Ts, dy):
    Jy = -diffusivity[:, :-1, :] * (Ts[:, 1:, :] - Ts[:, :-1, :]) / dy
    Jy = jnp.pad(Jy, ((0, 0), (0, 1), (0, 0)), mode='constant', constant_values=0)
    return Jy

# Compute adjoint using JAX's default gradient
grad_fn_default = jax.grad(lambda diffusivity: jnp.sum(compute_Jy_default(diffusivity, Ts, dy)), argnums=0)
dkappas_ddiffusivity_default = grad_fn_default(diffusivity)

# Check that the gradients are equal
print("Shape of diffusivity:", diffusivity.shape)
print("Shape of dkappas/ddiffusivity (custom VJP):", dkappas_ddiffusivity_custom.shape)
print("Shape of dkappas/ddiffusivity (default grad):", dkappas_ddiffusivity_default.shape)

# Verify the values are the same
print("dkappas_ddiffusivity (custom VJP):")
print(dkappas_ddiffusivity_custom)
print("dkappas_ddiffusivity (default grad):")
print(dkappas_ddiffusivity_default)

# Check if gradients are the same
gradients_are_equal = jnp.allclose(dkappas_ddiffusivity_custom, dkappas_ddiffusivity_default)
print("Are the gradients equal? ", gradients_are_equal)
