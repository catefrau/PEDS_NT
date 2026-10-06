import jax
import jax.numpy as jnp

# Set random seed for reproducibility
key = jax.random.PRNGKey(42)

# Set dimensions
B, N, M = 4, 100, 100

# Generate random u, kappa coefficients, and kappa_sum
u = jax.random.uniform(key, shape=(B, N, M), dtype=jnp.float32)
kappa_r = jax.random.uniform(key, shape=(B, N, M), dtype=jnp.float32)
kappa_l = jax.random.uniform(key, shape=(B, N, M), dtype=jnp.float32)
kappa_u = jax.random.uniform(key, shape=(B, N, M), dtype=jnp.float32)
kappa_d = jax.random.uniform(key, shape=(B, N, M), dtype=jnp.float32)
kappa_sum = kappa_r + kappa_l + kappa_u + kappa_d

# Define the custom VJP function for u_new computation
@jax.custom_vjp
def compute_u_new(u, kappa_r, kappa_l, kappa_u, kappa_d, kappa_sum):
    u_new = (
        jnp.roll(u, shift=1, axis=1) * kappa_r +
        jnp.roll(u, shift=-1, axis=1) * kappa_l +
        jnp.roll(u, shift=-1, axis=2) * kappa_d +
        jnp.roll(u, shift=1, axis=2) * kappa_u
    ) / kappa_sum

    # Apply boundary conditions
    u_new = u_new.at[:, 0, :].set(1 / 2)  # Top row
    u_new = u_new.at[:, -1, :].set(-1 / 2)  # Bottom row

    return u_new

# Forward pass for custom VJP
def compute_u_new_fwd(u, kappa_r, kappa_l, kappa_u, kappa_d, kappa_sum):
    u_new = (
        jnp.roll(u, shift=1, axis=1) * kappa_r +
        jnp.roll(u, shift=-1, axis=1) * kappa_l +
        jnp.roll(u, shift=-1, axis=2) * kappa_d +
        jnp.roll(u, shift=1, axis=2) * kappa_u
    ) / kappa_sum

    u_new = u_new.at[:, 0, :].set(1 / 2)  # Top row
    u_new = u_new.at[:, -1, :].set(-1 / 2)  # Bottom row
    return u_new, (u, kappa_r, kappa_l, kappa_u, kappa_d, kappa_sum)

# Backward pass for custom VJP
def compute_u_new_bwd(inputs, g):
    u, kappa_r, kappa_l, kappa_u, kappa_d, kappa_sum = inputs
    grad = jnp.ones_like(u, dtype=jnp.float32)

    grad = grad.at[:, 0, :].set(0.25)
    grad = grad.at[:, -1, :].set(0.25)
    grad = grad.at[:, 1, :].set(0.75)
    grad = grad.at[:, -2, :].set(0.75)

    
    return (grad, None, None, None, None, None)

# Register the forward and backward pass for custom VJP
compute_u_new.defvjp(compute_u_new_fwd, compute_u_new_bwd)

# Compute gradients using the custom VJP
grad_fn_custom = jax.grad(lambda u: jnp.sum(compute_u_new(u, kappa_r, kappa_l, kappa_u, kappa_d, kappa_sum)), argnums=0)
du_new_du_custom = grad_fn_custom(u)

# Compute gradients using the default JAX grad
def compute_u_new_default(u, kappa_r, kappa_l, kappa_u, kappa_d, kappa_sum):
    u_new = (
        jnp.roll(u, shift=1, axis=1) * kappa_r +
        jnp.roll(u, shift=-1, axis=1) * kappa_l +
        jnp.roll(u, shift=-1, axis=2) * kappa_d +
        jnp.roll(u, shift=1, axis=2) * kappa_u
    ) / kappa_sum

    u_new = u_new.at[:, 0, :].set(1 / 2)  # Top row
    u_new = u_new.at[:, -1, :].set(-1 / 2)  # Bottom row
    return u_new

grad_fn_default = jax.grad(lambda u: jnp.sum(compute_u_new_default(u, kappa_r, kappa_l, kappa_u, kappa_d, kappa_sum)), argnums=0)
du_new_du_default = grad_fn_default(u)

# Print the gradients
print(du_new_du_custom)
print(du_new_du_default)

# Check if the gradients are equal
gradients_are_equal = jnp.allclose(du_new_du_custom, du_new_du_default)
print("Are the gradients equal? ", gradients_are_equal)