import jax
import jax.numpy as jnp

# Example inputs
B, N, M = 3, 10, 10  # Batch size, N-dimension, M-dimension

key = jax.random.PRNGKey(42)



J_y = jax.random.uniform(key, shape=(B, N, M), dtype=jnp.float32)


# Define compute_kappas with custom_vjp
@jax.custom_vjp
def compute_kappas_custom(Jy):
    return jnp.sum(Jy[:, N // 2, :], axis=-1)  # kappas of shape (B,)

# Forward function for custom VJP
def compute_kappas_custom_fwd(Jy):
    kappas = jnp.sum(Jy[:, N // 2, :], axis=-1)  # Forward computation
    return kappas, Jy  # Pass Jy to the backward function

# Backward function for custom VJP
def compute_kappas_custom_bwd(Jy, g):
    grad = jnp.zeros_like(Jy)
    grad = grad.at[:, N // 2, :].add(g[:, None])  # Only affect N//2 slice
    return (grad,)

# Associate forward and backward functions
compute_kappas_custom.defvjp(compute_kappas_custom_fwd, compute_kappas_custom_bwd)

# Compute adjoint using custom VJP
kappas_custom = compute_kappas_custom(J_y)
grad_fn_custom = jax.grad(lambda Jy: jnp.sum(compute_kappas_custom(Jy)), argnums=0)  # Scalar output
dkappas_dJy_custom = grad_fn_custom(J_y)

# Define the same function using JAX's default gradient
def compute_kappas_default(Jy):
    return jnp.sum(Jy[:, N // 2, :], axis=-1)

# Compute adjoint using JAX's default gradient
grad_fn_default = jax.grad(lambda Jy: jnp.sum(compute_kappas_default(Jy)), argnums=0)
dkappas_dJy_default = grad_fn_default(J_y)

# Verify the values are the same
print("dkappas_dJy (custom VJP):")
print(dkappas_dJy_custom)
print("dkappas_dJy (default grad):")
print(dkappas_dJy_default)

# Check if gradients are the same
gradients_are_equal = jnp.allclose(dkappas_dJy_custom, dkappas_dJy_default)
print("Are the gradients equal? ", gradients_are_equal)