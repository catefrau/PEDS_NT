import jax
import jax.numpy as jnp

@jax.custom_vjp
def compute_dT100_dK(u, diffusivity):

    kappa_r = jnp.roll(diffusivity, shift=1, axis=1)
    kappa_l = jnp.roll(diffusivity, shift=-1, axis=1)
    kappa_d = jnp.roll(diffusivity, shift=-1, axis=2)
    kappa_u = jnp.roll(diffusivity, shift=1, axis=2)

    kappa_sum = kappa_r+kappa_l+kappa_d+kappa_u

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
    return u_new, kappa_r, kappa_l, kappa_u, kappa_d, kappa_sum

def compute_dT100_K_fwd(u, diffusivity):
    u_new, kappa_r, kappa_l, kappa_u, kappa_d, kappa_sum = compute_dT100_dK(u, diffusivity)
    return u_new, (u, diffusivity, kappa_r, kappa_l, kappa_u, kappa_d, kappa_sum)

def compute_dT100_K_bwd(primals, cotangents):
    u, diffusivity, kappa_r, kappa_l, kappa_u, kappa_d, kappa_sum = primals
    cot_u_new = cotangents  # Adjoint (gradient w.r.t. u_new)

    # Calculate the rolled components of `u`
    u_r = jnp.roll(u, shift=1, axis=1)
    u_l = jnp.roll(u, shift=-1, axis=1)
    u_u = jnp.roll(u, shift=1, axis=2)
    u_d = jnp.roll(u, shift=-1, axis=2)

    tot = u_r * kappa_r + u_l * kappa_l + u_u * kappa_u + u_d * kappa_d

    # Gradients for kappa_r
    dT100_dkappa_r = (u_r * kappa_sum - tot) / (kappa_sum**2) # - tot ???? WHY NOT??
    # Mask boundary rows
    dT100_dkappa_r = dT100_dkappa_r.at[:, 0, :].set(0)  # Top row
    dT100_dkappa_r = dT100_dkappa_r.at[:, -1, :].set(0)  # Bottom row

    dT100_dkappa_l = (u_l * kappa_sum  -tot ) / (kappa_sum**2)
    dT100_dkappa_l = dT100_dkappa_l.at[:, 0, :].set(0)  # Top row
    dT100_dkappa_l = dT100_dkappa_l.at[:, -1, :].set(0)  # Bottom row

    dT100_dkappa_u = (u_u * kappa_sum -tot) / (kappa_sum**2)
    dT100_dkappa_u = dT100_dkappa_u.at[:, 0, :].set(0)  # Top row
    dT100_dkappa_u = dT100_dkappa_u.at[:, -1, :].set(0)  # Bottom row

    dT100_dkappa_d = (u_d * kappa_sum - tot) / (kappa_sum**2)
    dT100_dkappa_d = dT100_dkappa_d.at[:, 0, :].set(0)  # Top row
    dT100_dkappa_d = dT100_dkappa_d.at[:, -1, :].set(0)  # Bottom row

    # shift them back?? and sum?

    grad_kappa_r = jnp.roll(dT100_dkappa_r, shift=-1, axis=1)
    grad_kappa_l = jnp.roll(dT100_dkappa_l, shift=1, axis=1)
    grad_kappa_u = jnp.roll(dT100_dkappa_u, shift=-1, axis=2)
    grad_kappa_d = jnp.roll(dT100_dkappa_d, shift=1, axis=2)

    # Combine all gradients
    grad_diff = grad_kappa_r + grad_kappa_l + grad_kappa_u + grad_kappa_d
    #grad_diff = dT100_dkappa_d + dT100_dkappa_l + dT100_dkappa_r + dT100_dkappa_u

    # Multiply with cotangent
    grad_diff = cot_u_new * grad_diff

    # Return gradients
    return (None, grad_diff)

compute_dT100_dK.defvjp(compute_dT100_K_fwd, compute_dT100_K_bwd)

# Example test
B, N, M = 1, 5, 5
key = jax.random.PRNGKey(42)
u = jnp.ones((B, N, M)) * 5000.0
diffusivity = jax.random.uniform(key, shape=(B, N, M), dtype=jnp.float32)
#diffusivity = jnp.ones((B,N,M))

# Compute gradients using custom VJP
grad_fn_custom = jax.grad(
    lambda diffusivity: jnp.sum(compute_dT100_dK(u, diffusivity))
)
gradients_custom = grad_fn_custom(diffusivity)

# Compute gradients using default JAX grad
def compute_u_new_default(u, diffusivity):

    kappa_r = jnp.roll(diffusivity, shift=1, axis=1)
    kappa_l = jnp.roll(diffusivity, shift=-1, axis=1)
    kappa_d = jnp.roll(diffusivity, shift=-1, axis=2)
    kappa_u = jnp.roll(diffusivity, shift=1, axis=2)

    kappa_sum = kappa_r+kappa_l+kappa_d+kappa_u

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
    lambda diffusivity: jnp.sum(compute_u_new_default(u, diffusivity))
)
gradients_default = grad_fn_default(diffusivity)

# Print gradients
print("Custom adjoint gradients for kappa_r:")
print(gradients_custom)

print("\nDefault JAX gradients for kappa_r:")
print(gradients_default)

# Check if gradients are equal
gradients_are_equal = jnp.allclose(gradients_custom[:, 1:-1, :], gradients_default[:, 1:-1, :])
print("Are the gradients equal? ", gradients_are_equal)
