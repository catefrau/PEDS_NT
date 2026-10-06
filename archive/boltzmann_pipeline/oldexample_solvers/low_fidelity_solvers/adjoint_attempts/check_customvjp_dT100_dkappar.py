import jax
import jax.numpy as jnp

@jax.custom_vjp
def compute_dT100_kappa_r(u, kappa_r, kappa_l, kappa_u, kappa_d, kappa_sum):
    # Compute u_new using only kappa_r
    u_new = (
        jnp.roll(u, shift=1, axis=1) * kappa_r +
        jnp.roll(u, shift=-1, axis=1) * kappa_l +
        jnp.roll(u, shift=1, axis=2) * kappa_u +
        jnp.roll(u, shift=-1, axis=2) * kappa_d
    ) / kappa_sum

    # Apply boundary conditions
    u_new = u_new.at[:, 0, :].set(1 / 2)  # Top row
    u_new = u_new.at[:, -1, :].set(-1 / 2)  # Bottom row
    return u_new

def compute_dT100_kappa_r_fwd(u, kappa_r, kappa_l, kappa_u, kappa_d, kappa_sum):
    u_new = compute_dT100_kappa_r(u, kappa_r, kappa_l, kappa_u, kappa_d, kappa_sum)
    return u_new, (u, kappa_r, kappa_l, kappa_u, kappa_d, kappa_sum)

def compute_dT100_kappa_r_bwd(primals, cotangents):
    u, kappa_r, kappa_l, kappa_u, kappa_d, kappa_sum = primals
    cot_u_new = cotangents  # Adjoint (gradient w.r.t. u_new)

    # Calculate the rolled components of `u`
    u_r = jnp.roll(u, shift=1, axis=1)
    u_l = jnp.roll(u, shift=-1, axis=1)
    u_u = jnp.roll(u, shift=1, axis=2)
    u_d = jnp.roll(u, shift=-1, axis=2)

    tot = u_r * kappa_r + u_l * kappa_l + u_u * kappa_u + u_d * kappa_d

    # Gradients for kappa_r
    dT100_dkappa_u = (u_u * kappa_sum ) / (kappa_sum**2) # + tot ????

    # Mask boundary rows
    dT100_dkappa_u = dT100_dkappa_u.at[:, 0, :].set(0)  # Top row
    dT100_dkappa_u = dT100_dkappa_u.at[:, -1, :].set(0)  # Bottom row

    # Multiply with cotangent
    grad_kappa_u = cot_u_new * dT100_dkappa_u

    # Return gradients
    return (None, None, None, grad_kappa_u, None, None)

compute_dT100_kappa_r.defvjp(compute_dT100_kappa_r_fwd, compute_dT100_kappa_r_bwd)

# Example test
B, N, M = 1, 5, 5
key = jax.random.PRNGKey(42)
u = jnp.ones((B, N, M)) * 150.0
diffusivity = jax.random.uniform(key, shape=(B, N, M), dtype=jnp.float32)

# Precompute shifted diffusivity components
kappa_r = jnp.roll(diffusivity, shift=1, axis=1)
kappa_l = jnp.roll(diffusivity, shift=-1, axis=1)
kappa_u = jnp.roll(diffusivity, shift=1, axis=2)
kappa_d = jnp.roll(diffusivity, shift=-1, axis=2)
kappa_sum = kappa_r + kappa_l + kappa_u + kappa_d

# Compute gradients using custom VJP
grad_fn_custom = jax.grad(
    lambda kappa_u: jnp.sum(compute_dT100_kappa_r(u, kappa_r, kappa_l, kappa_u, kappa_d, kappa_sum))
)
gradients_custom = grad_fn_custom(kappa_u)

# Compute gradients using default JAX grad
def compute_u_new_default(u, kappa_r, kappa_l, kappa_u, kappa_d, kappa_sum):
    u_new = (
        jnp.roll(u, shift=1, axis=1) * kappa_r +
        jnp.roll(u, shift=-1, axis=1) * kappa_l +
        jnp.roll(u, shift=1, axis=2) * kappa_u +
        jnp.roll(u, shift=-1, axis=2) * kappa_d
    ) / kappa_sum

    u_new = u_new.at[:, 0, :].set(1 / 2)  # Top row
    u_new = u_new.at[:, -1, :].set(-1 / 2)  # Bottom row
    return u_new

grad_fn_default = jax.grad(
    lambda kappa_u: jnp.sum(compute_u_new_default(u, kappa_r, kappa_l, kappa_u, kappa_d, kappa_sum))
)
gradients_default = grad_fn_default(kappa_u)

# Print gradients
print("Custom adjoint gradients for kappa_r:")
print(gradients_custom)

print("\nDefault JAX gradients for kappa_r:")
print(gradients_default)

# Check if gradients are equal
gradients_are_equal = jnp.allclose(gradients_custom[:, 1:-1, :], gradients_default[:, 1:-1, :])
print("Are the gradients equal? ", gradients_are_equal)
