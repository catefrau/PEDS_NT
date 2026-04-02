# ============================================================
# STEP 0 — Imports
# ============================================================
import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as lax     # lax.scan = efficient for-loop that JAX can compile
import optax
from flax import nnx
import matplotlib.pyplot as plt

# ============================================================
# STEP 1 — The Differentiable Fourier Solver
#
# This solves the 2D steady-state heat equation:
#    -∇·(κ∇T) = 0
# on a 5×5 grid using Jacobi iterations.
#
# BCs:  T[top row]    = +0.5 K  (hot side)
#       T[bottom row] = -0.5 K  (cold side)
#       Left/right:    periodic (wrap around)
#
# KEY JAX IDEA: Because this entire function uses jnp operations,
# JAX can automatically compute d(kappa_eff)/d(conductivity_field).
# This is how the gradient flows from the loss back into the NN!
# ============================================================
def fourier_solver(conductivity_field, n_iterations=500):
    """
    conductivity_field : (batch, 5, 5)  values in W/m·K
    returns            : (batch,)       effective thermal conductivity κ_eff
    """
    N = conductivity_field.shape[1]
    T = jnp.zeros_like(conductivity_field)   # start from T=0 everywhere

    # Precompute neighbor conductivities (they don't change during iteration)
    kappa_r = jnp.roll(conductivity_field, shift=-1, axis=2)  # right (periodic x)
    kappa_l = jnp.roll(conductivity_field, shift= 1, axis=2)  # left  (periodic x)
    kappa_d = jnp.roll(conductivity_field, shift=-1, axis=1)  # down
    kappa_u = jnp.roll(conductivity_field, shift= 1, axis=1)  # up
    kappa_sum = kappa_r + kappa_l + kappa_d + kappa_u + 1e-10  # avoid div by zero

    # One Jacobi step: update T using weighted average of neighbors
    # Each neighbor's temperature is weighted by the conductivity in that direction.
    # Think of it like: if a wall has high κ, its neighbor temperature matters more.
    def jacobi_step(T, _):
        T_new = (
            jnp.roll(T, shift=-1, axis=2) * kappa_r +
            jnp.roll(T, shift= 1, axis=2) * kappa_l +
            jnp.roll(T, shift=-1, axis=1) * kappa_d +
            jnp.roll(T, shift= 1, axis=1) * kappa_u
        ) / kappa_sum

        # Re-impose boundary conditions after every step
        T_new = T_new.at[:, 0,  :].set( 0.5)   # hot top
        T_new = T_new.at[:, -1, :].set(-0.5)   # cold bottom
        return T_new, None  # None = no output to accumulate (lax.scan requires this)

    # lax.scan runs jacobi_step n_iterations times — like a for-loop but JAX-compiled!
    # This is how the original code runs 5000 iterations efficiently.
    T_final, _ = lax.scan(jacobi_step, T, None, length=n_iterations)

    # --- Compute effective κ from Fourier's law: q = κ · ∇T ---
    # flux_y = κ * (T[i] - T[i+1]) — heat flowing downward (from hot to cold)
    # κ_eff  = N * mean(flux_y)     — N comes from normalizing by cell size (h = L/N)
    flux_y    = conductivity_field * (T_final - jnp.roll(T_final, shift=-1, axis=1))
    kappa_eff = jnp.mean(flux_y[:, :-1, :], axis=(1, 2)) * N   # shape: (batch,)

    return kappa_eff


# ============================================================
# STEP 2 — The Generator Neural Network
#
# Input:  pores          (batch, 25) — binary: 1=pore, 0=Si
# Output: cond_field     (batch, 5, 5) — conductivity field for the solver
#
# The NN answers the question:
# "Given this pore geometry, what conductivity field should I
#  feed the cheap Fourier solver so it matches the expensive BTE?"
#
# The solver's systematic errors are absorbed into the NN's output.
# ============================================================
class GeneratorNN(nnx.Module):
    def __init__(self, hidden_sizes: list, rngs: nnx.Rngs):
        kernel_init = nnx.initializers.kaiming_normal()
        bias_init   = nnx.initializers.constant(0.0)

        # Layer sizes: 25 inputs → hidden layers → 25 outputs (the 5×5 field)
        layer_sizes = [25] + hidden_sizes + [25]
        self.layers = [
            nnx.Linear(i, o, kernel_init=kernel_init, bias_init=bias_init, rngs=rngs)
            for i, o in zip(layer_sizes[:-1], layer_sizes[1:])
        ]

    def __call__(self, pores, training=False):
        x = pores
        for layer in self.layers[:-1]:    # hidden layers with ReLU
            x = nnx.relu(layer(x))
        x = self.layers[-1](x)            # final layer — no activation yet

        # Constrain output to valid conductivity range [ε, 150] W/m·K
        # softplus(x) = log(1 + e^x) — a smooth version of ReLU, always positive
        x = jax.nn.softplus(x)
        x = jnp.clip(x, 1e-4, 150.0)

        return x.reshape(-1, 5, 5)        # shape: (batch, 5, 5)


# ============================================================
# STEP 3 — Combine: Generator → Solver
#
# This is the full PEDS forward pass (simplified, no mixing coeff).
# The model's only learnable parameters are in GeneratorNN.
# The solver has NO parameters — it's pure physics.
# ============================================================
class GeneratorWithSolver(nnx.Module):
    def __init__(self, hidden_sizes: list, rngs: nnx.Rngs):
        self.generator = GeneratorNN(hidden_sizes, rngs)

    def __call__(self, pores, training=False):
        cond_field = self.generator(pores, training)    # NN part: (batch, 5, 5)
        kappa_pred = fourier_solver(cond_field)          # Physics part: (batch,)
        return kappa_pred, cond_field                    # return both for inspection


# ============================================================
# STEP 4 — Load Data
# ============================================================
data   = np.load("./data/highfidelity/high_fidelity_2_20000.npz")
pores  = jnp.array(data['pores'],  dtype=jnp.float32)
kappas = jnp.array(data['kappas'], dtype=jnp.float32)

TRAIN_SIZE = 200
TEST_SIZE  = 200
pores_train,  kappas_train  = pores[:TRAIN_SIZE],                           kappas[:TRAIN_SIZE]
pores_test,   kappas_test   = pores[TRAIN_SIZE:TRAIN_SIZE + TEST_SIZE],     kappas[TRAIN_SIZE:TRAIN_SIZE + TEST_SIZE]

def dataloader(arrays, batch_size):
    n = arrays[0].shape[0]
    for start in range(0, n, batch_size):
        yield tuple(arr[start : start + batch_size] for arr in arrays)


# ============================================================
# STEP 5 — Model, Optimizer, Schedule
# ============================================================
rngs  = nnx.Rngs(42)
model = GeneratorWithSolver(hidden_sizes=[32, 64], rngs=rngs)

EPOCHS     = 100
LR_MAX     = 5e-3
LR_MIN     = 5e-4
BATCH_SIZE = 200

lr_schedule = optax.cosine_decay_schedule(init_value=LR_MAX, decay_steps=EPOCHS, alpha=LR_MIN/LR_MAX)
optimizer   = nnx.Optimizer(model, optax.adam(lr_schedule))


# ============================================================
# STEP 6 — Training Step (JIT-compiled for speed)
#
# JAX CONCEPT: @nnx.jit compiles this function the first time
# it runs, making every subsequent call much faster.
# The gradient flows: loss → kappa_pred → fourier_solver → cond_field → NN weights
# ALL automatically, because every step is a JAX operation!
# ============================================================
@nnx.jit
def train_step(model, optimizer, pores_batch, kappas_batch):
    def loss_fn(model):
        kappa_pred, _ = model(pores_batch, True)
        residuals     = kappa_pred - kappas_batch
        return jnp.sum(residuals ** 2)               # matches training.py exactly

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)
    return loss


# ============================================================
# STEP 7 — Training Loop
# ============================================================
train_losses = []
val_errors   = []

for epoch in range(EPOCHS):

    # Training
    total_loss = 0.0
    for pores_b, kappas_b in dataloader((pores_train, kappas_train), BATCH_SIZE):
        loss = train_step(model, optimizer, pores_b, kappas_b)
        total_loss += float(loss)
    train_losses.append(total_loss / TRAIN_SIZE)

    # Validation
    total_error = 0.0
    for pores_b, kappas_b in dataloader((pores_test, kappas_test), BATCH_SIZE):
        kappa_pred, cond_field = model(pores_b, False)
        total_error += float(
            jnp.sum(jnp.abs(kappa_pred - kappas_b) * 100.0 / jnp.abs(kappas_b))
        )
    val_errors.append(total_error / TEST_SIZE)

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1:4d}/{EPOCHS} | "
              f"Train MSE: {train_losses[-1]:8.2f} | "
              f"Val Error: {val_errors[-1]:.2f}%")

# ============================================================
# STEP 8 — Inspect what the NN learned
# Pick one test sample and visualize the generated conductivity field
# ============================================================
sample_pores  = pores_test[:1]   # shape: (1, 25)
kappa_true    = float(kappas_test[0])
kappa_pred, cond_field = model(sample_pores, False)

fig, axes = plt.subplots(1, 3, figsize=(12, 3))

axes[0].imshow(np.array(sample_pores).reshape(5, 5), cmap='gray_r', vmin=0, vmax=1)
axes[0].set_title("Input: pore geometry\n(black=pore, white=Si)")

axes[1].imshow(np.array(cond_field[0]), cmap='hot')
axes[1].set_title("NN output: conductivity field\n(what goes into solver)")
plt.colorbar(axes[1].images[0], ax=axes[1])

axes[2].bar(['True BTE κ', 'Predicted κ'], [kappa_true, float(kappa_pred[0])], color=['blue', 'orange'])
axes[2].set_ylabel('W/m·K')
axes[2].set_title(f"Result\nError: {abs(kappa_true - float(kappa_pred[0]))/kappa_true*100:.1f}%")

plt.tight_layout()
plt.savefig('peds_results.png', dpi=150)

print(f"\nFinal val error : {val_errors[-1]:.2f}%")
print(f"Paper MLP baseline (200 samples) : ~12.2%")
print(f"Paper PEDS        (200 samples)  : ~7.2%")