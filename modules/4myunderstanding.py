"""
=============================================================================
PEDS Educational Training Script
=============================================================================

Architecture overview:
  pore geometry (5x5 binary) 
        │
        ▼
  GeneratorNN  ──►  conductivity field (5x5 float)
        │
        ▼
  GaussSolver  ──►  effective thermal conductivity κ (scalar per sample)
        │
        ▼
  MSE loss vs κ_true
        │
        ▼
  Backprop through solver into NN weights

Libraries used: JAX, Flax (nnx API), Optax
No MPI, no checkpointing — single-device educational version.
=============================================================================
"""

import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np
import optax
from flax import nnx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ─────────────────────────────────────────────
# SECTION 1: Physics Solver
# ─────────────────────────────────────────────

def _update_step(u, kappa_sum, kappa_r, kappa_l, kappa_u, kappa_d):
    """One Gauss-Seidel sweep on the temperature field u.
    
    This is the core finite-difference stencil: each interior cell's temperature
    is updated as a conductivity-weighted average of its four neighbours.
    The update is vectorised over the whole batch simultaneously by JAX.
    """
    # jnp.roll shifts the array cyclically along an axis.
    # Here it efficiently gathers each cell's four spatial neighbours.
    u_r = jnp.roll(u, shift=1,  axis=1)   # left neighbour (roll right → look left)
    u_l = jnp.roll(u, shift=-1, axis=1)   # right neighbour
    u_d = jnp.roll(u, shift=-1, axis=2)   # upper neighbour (image convention)
    u_u = jnp.roll(u, shift=1,  axis=2)   # lower neighbour

    u_new = (u_r * kappa_r + u_l * kappa_l + u_d * kappa_d + u_u * kappa_u) / kappa_sum

    # Hard Dirichlet boundary conditions: top row = +0.5, bottom row = -0.5
    # .at[...].set(...) is JAX's functional (immutable) array mutation
    u_new = u_new.at[:, 0,  :].set( 0.5)
    u_new = u_new.at[:, -1, :].set(-0.5)
    return u_new


def _flux_kappa(conductivity, T):
    """Compute effective thermal conductivity κ from the temperature field.

    Uses Fourier's law: J = -κ ∇T.
    We integrate the y-flux across the midplane to get the effective κ.
    Shape: conductivity, T → [batch]
    """
    # Finite-difference heat flux between rows (central row of the domain)
    Jy = -conductivity[:, :-1, :] * (T[:, 1:, :] - T[:, :-1, :])  # [batch, N-1, N]
    # Pad back to [batch, N, N] so indexing is consistent
    Jy = jnp.pad(Jy, ((0, 0), (0, 1), (0, 0)))
    # Sum flux at the mid-plane row (integrating over the x-direction)
    kappas = jnp.sum(Jy[:, conductivity.shape[1] // 2, :], axis=-1)   # [batch]
    return kappas


# Mark gauss_solver with @jax.custom_vjp so we can provide an efficient
# hand-written backward pass instead of unrolling 1000 lax.scan iterations.
@jax.custom_vjp
def gauss_solver(conductivity, iterations=1000):
    """Iterative Gauss-Seidel solver for the steady-state heat equation.
    Args:
        conductivity: [batch, N, N]  — spatially varying thermal conductivity
        iterations:   number of relaxation sweeps
    Returns:
        T: [batch, N, N]  — steady-state temperature field
    """
    batch_size, N, _ = conductivity.shape

    # Initialise temperature with a linear gradient (good starting point)
    u = jnp.zeros((batch_size, N, N))
    for i in range(N):
        u = u.at[:, :, i].set(jnp.linspace(0.5, -0.5, N))

    # Pre-compute neighbour conductivities (constant across iterations)
    kappa_r = jnp.roll(conductivity, shift=1,  axis=1)
    kappa_l = jnp.roll(conductivity, shift=-1, axis=1)
    kappa_d = jnp.roll(conductivity, shift=-1, axis=2)
    kappa_u = jnp.roll(conductivity, shift=1,  axis=2)
    # Small epsilon avoids division by zero when conductivity is near 0
    kappa_sum = kappa_r + kappa_l + kappa_d + kappa_u + 1e-6

    def body_fn(u, _):
        # lax.scan calls body_fn repeatedly, threading `u` as the carry.
        # The `_` is the "xs" element (None here — we just want N iterations).
        # This is much more memory-efficient than a Python for-loop because
        # JAX traces body_fn once and compiles a single XLA while-loop.
        u_new = _update_step(u, kappa_sum, kappa_r, kappa_l, kappa_u, kappa_d)
        return u_new, None  # (new carry, scanned output — None means discard)

    # lax.scan: functional fixed-point iteration compiled to a single XLA op.
    # Returns (final_carry, stacked_outputs) — we only need the final T.
    T, _ = lax.scan(body_fn, u, None, length=iterations)
    return T


def _gauss_solver_finalstep(conductivity, T):
    """One additional relaxation step used only in the backward pass.

    By differentiating a single step (cheap) rather than the full scan (expensive),
    we get an approximate but stable gradient through the solver — this is the
    key trick that makes differentiable physics feasible here.
    """
    kappa_r = jnp.roll(conductivity, shift=1,  axis=1)
    kappa_l = jnp.roll(conductivity, shift=-1, axis=1)
    kappa_d = jnp.roll(conductivity, shift=-1, axis=2)
    kappa_u = jnp.roll(conductivity, shift=1,  axis=2)
    kappa_sum = kappa_r + kappa_l + kappa_d + kappa_u + 1e-6
    return _update_step(T, kappa_sum, kappa_r, kappa_l, kappa_u, kappa_d)


def _gauss_fwd(conductivity, iterations=1000):
    """Forward pass: run the solver and save (conductivity, T) for the backward."""
    T = gauss_solver(conductivity, iterations)
    # The second return value is the "residuals" passed to the backward function.
    return T, (conductivity, T)


def _gauss_bwd(res, g):
    """Custom backward pass (VJP) for the Gauss-Seidel solver.

    Instead of differentiating through all `iterations` steps (which would
    require storing every intermediate u — memory O(iterations × N²)), we
    differentiate through ONE additional final step evaluated at the converged T.
    This is valid at a fixed point because dT/dκ can be recovered from the
    Jacobian of a single relaxation step applied at convergence.
    """
    conductivity, T_final = res
    dL_dT = g  # upstream gradient of the loss w.r.t. the output T field

    # jax.vjp computes the vector-Jacobian product (VJP) for an arbitrary function.
    # Here we differentiate `_gauss_solver_finalstep` w.r.t. its first argument
    # (conductivity), passing dL_dT as the cotangent of the output.
    _, vjp_fn = jax.vjp(_gauss_solver_finalstep, conductivity, T_final)
    dL_dconductivity, _ = vjp_fn(dL_dT)

    return dL_dconductivity, None  # None for the non-differentiable `iterations` arg


# Register the custom forward/backward with JAX's autodiff engine.
gauss_solver.defvjp(_gauss_fwd, _gauss_bwd)


# ─────────────────────────────────────────────
# SECTION 2: Neural Network (Generator)
# ─────────────────────────────────────────────

def _hardtanh_positive(x):
    """Clip activations to (1e-16, 160.0) — ensures conductivity stays physical.
    This replaces the final activation layer so the network cannot output
    negative conductivities (which would break the physics solver).
    """
    return jnp.clip(x, 1e-16, 160.0)


class GeneratorNN(nnx.Module):
    """MLP that maps a flat pore-geometry vector → a 2D conductivity field.

    Architecture (matching PEDS config m1):
        Input  : [batch, 25]    — flattened 5×5 binary pore mask
        Hidden : [32, 32]       — fully-connected ReLU layers  
        Output : [batch, 25]    → reshaped to [batch, 5, 5]
                                  with hardtanh_positive activation
                                  so all values ∈ (1e-16, 160)
    """

    def __init__(self, layer_sizes: list, rngs: nnx.Rngs):
        """
        Args:
            layer_sizes: e.g. [25, 32, 32, 25]  (input → hiddens → output)
            rngs: nnx.Rngs wraps a JAX PRNGKey and hands out sub-keys on demand.
                  Every nnx module that needs randomness (Linear init, Dropout …)
                  asks `rngs` for a fresh key — no manual key splitting needed.
        """
        super().__init__()

        # He (Kaiming) initialisation is recommended for ReLU networks because
        # it keeps activation variance constant across layers.
        he_init = nnx.initializers.kaiming_normal()
        bias_init = nnx.initializers.constant(0.0)

        # Build layers as a plain Python list — nnx.Module discovers them
        # automatically via its attribute-scanning mechanism.
        self.layers = [
            nnx.Linear(
                in_features=layer_sizes[i],
                out_features=layer_sizes[i + 1],
                kernel_init=he_init,
                bias_init=bias_init,
                rngs=rngs,          # rngs is passed here so each Linear gets
            )                        # its own unique initialisation key
            for i in range(len(layer_sizes) - 1)
        ]
        self.resolution = int(layer_sizes[-1] ** 0.5)  # output side-length (5)

    def __call__(self, pores: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """
        Args:
            pores:    [batch, 25]  binary pore mask
            training: flag for layers like Dropout (unused here, kept for API parity)

        Returns:
            conductivity_field: [batch, 5, 5]  physical conductivity values
        """
        x = pores
        # Apply ReLU on all but the final layer
        for layer in self.layers[:-1]:
            x = nnx.relu(layer(x))        # nnx.relu is just jax.nn.relu, re-exported
        x = self.layers[-1](x)  # Final layer: linear projection → clip to physical range
        x = _hardtanh_positive(x)  # ensures κ > 0 everywhere; could have used softplus(x)

        # Reshape flat vector [batch, 25] → spatial grid [batch, 5, 5]
        batch_size = pores.shape[0]
        # print(f"The last layer is {x}, the size of the batch being {batch_size} with a resolution of {self.resolution}")
        return jnp.reshape(x, (batch_size, self.resolution, self.resolution))


# ─────────────────────────────────────────────
# SECTION 3: Combined PEDS Model
# ─────────────────────────────────────────────

class PEDSModel(nnx.Module):
    """Wraps GeneratorNN + GaussSolver into a single differentiable forward pass.

    Gradients from the scalar κ loss flow back through the solver (via the
    custom VJP) and then into the NN weights.
    """

    def __init__(self, hidden_sizes: list, resolution: int, rngs: nnx.Rngs):
        super().__init__()
        layer_sizes = [25] + hidden_sizes + [resolution ** 2] #  final size of layers: [25, 32, 32, 25]
        # rngs is stored by nnx and thread through all sub-modules that need it
        self.generator = GeneratorNN(layer_sizes=layer_sizes, rngs=rngs)
        self.resolution = resolution

    def __call__(self, pores: jnp.ndarray, training: bool = False):
        """
        Args:
            pores: [batch, 5, 5] binary pore geometry (flattened inside)

        Returns:
            kappa:               [batch]    effective thermal conductivity
            conductivity_field:  [batch, N, N]  intermediate field (for visualisation)
        """
        batch_size = pores.shape[0]
        pores_flat = jnp.reshape(pores, (batch_size, 25))  # flatten spatial dims

        # 1. NN generates a plausible conductivity field
        conductivity_field = self.generator(pores_flat, training)  # [batch, 5, 5]

        # 2. Physics solver maps conductivity field → κ
        #    Gradients propagate back through this call via the custom VJP
        T = gauss_solver(conductivity_field, iterations=1000)       # [batch, 5, 5]
        kappa = _flux_kappa(conductivity_field, T)                  # [batch]

        return kappa, conductivity_field


# ─────────────────────────────────────────────
# SECTION 4: Data Loading
# ─────────────────────────────────────────────

def data_loader(*arrays, batch_size: int):
    """Yield mini-batches from multiple arrays in lock-step.
    Pure-Python generator that slices NumPy arrays and feeds them one batch at a time.
    """
    n_samples = arrays[0].shape[0]
    for start in range(0, n_samples, batch_size):
        yield tuple(arr[start:start + batch_size] for arr in arrays)


def load_data(filepath: str, train_size: int, test_size: int, seed: int = 42):
    """Load pore geometries and κ labels from the .npz dataset. Returns NumPy arrays 
    """
    data = np.load(filepath, allow_pickle=True)
    print(f"Dataset keys: {list(data.keys())}")
    pores  = np.array(data['pores'],  dtype=np.float32)   # [N, 5, 5]
    kappas = np.array(data['kappas'], dtype=np.float32)   # [N]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pores)) # randomly shuffle the indices of the dataset to ensure random sampling of training and test sets
    train_idx = idx[:train_size] # index to save array from location 0 to location of training set dimension
    test_idx  = idx[train_size:train_size + test_size] # index for the array that takes the following chunk of data, of dimension test set

    return (pores[train_idx], kappas[train_idx]), (pores[test_idx], kappas[test_idx])


# ─────────────────────────────────────────────
# SECTION 5: Training Loop
# ─────────────────────────────────────────────

def train(filepath, train_size, test_size, batch_size, epochs, lr_max, lr_min,
    hidden_sizes, resolution, seed):
    if hidden_sizes is None:
        hidden_sizes = [32, 32]   # matches config m1 in the original codebase
    # ── 5.1  Data ──────────────────────────────────────────────────────────
    print("Loading data …")
    (train_pores, train_kappas), (test_pores, test_kappas) = load_data(
        filepath, train_size, test_size, seed
    )   # train_pores is a numpy array of shape (train_size, 25), while train_kappas is a numpy array of shape (train_size,) 
    print(f"  Train: {train_pores.shape}  Test: {test_pores.shape}")
    
    # ── 5.2  Model & Optimiser ─────────────────────────────────────────────
    # nnx.Rngs(seed) creates a named-key container to get a fresh, unique PRNGKey.
    rngs  = nnx.Rngs(seed)
    model = PEDSModel(hidden_sizes=hidden_sizes, resolution=resolution, rngs=rngs)
    # Cosine decay schedule: learning rate anneals smoothly from lr_max → lr_min.
    # `alpha` is the ratio min/max, so the schedule bottoms out at lr_min.
    lr_schedule = optax.cosine_decay_schedule(
        init_value=lr_max,
        decay_steps=epochs,
        alpha=lr_min / lr_max,   # final lr = lr_max * alpha = lr_min
    )
    optimizer = nnx.Optimizer(model, optax.adam(lr_schedule))     # nnx.Optimizer couples the model's parameters with an Optax update rule.

    # ── 5.3  Loss / gradient function ─────────────────────────────────────
    def loss_fn(model, pores, kappas_true):
        # model is passed explicitly so nnx.value_and_grad knows which pytree to differentiate with respect to.
        kappa_pred, _ = model(pores, training=True) # kappa predicted by PEDS model
        residuals = kappa_pred - kappas_true # kappa from the data batch
        return jnp.sum(residuals ** 2)   # Sum (not mean) MSE — consistent with original training.py
    # nnx.value_and_grad is the Flax-nnx analogue of jax.value_and_grad.
    # It differentiates `loss_fn` with respect to its FIRST argument (the model),
    grad_fn = nnx.value_and_grad(loss_fn)  # returning (loss_value, grad_pytree_of_model_params)

    # ── 5.4  Validation helper ─────────────────────────────────────────────
    def validation_step(pores_np, kappas_np):
        """Compute mean squared loss and mean percentage error over the test set."""
        total_sq  = 0.0
        total_pct = 0.0
        # UNDERSTAND HOW IS THIS LOOP REPEATED
        for batch_idx, (batch_pores, batch_kappas) in enumerate(data_loader(pores_np, kappas_np, batch_size=batch_size)):
            # the batch index is always 0 because the test set is smaller than the batch size, so we only have one batch containing the whole test set.
            kappa_pred, _ = model(jnp.array(batch_pores))   # no training=True → no stochasticity
            sq_err  = jnp.sum((kappa_pred - batch_kappas) ** 2) # squared error
            pct_err = jnp.sum(jnp.abs(kappa_pred - batch_kappas) / jnp.abs(batch_kappas) * 100.0) # percent error
            total_sq  += float(sq_err)
            total_pct += float(pct_err)
        n = len(kappas_np)
        return total_sq / n, total_pct / n   # mean over all test samples

    # ── 5.5  Epoch loop ────────────────────────────────────────────────────
    train_losses   = []
    val_losses     = []
    val_pct_errors = []
    iterations = 0 
    print(f"\nTraining for {epochs} epochs …")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_pores, batch_kappas in data_loader(train_pores, train_kappas, batch_size=batch_size):
            bp = jnp.array(batch_pores) # Convert NumPy → JAX arrays once per batch 
            bk = jnp.array(batch_kappas)
            print("Running forward pass + AD \n ")
            # Part of training: forward pass + automatic differentiation in one call.
            # `loss` is a scalar; `grads` is a pytree mirroring model's parameter tree.
            loss, grads = grad_fn(model, bp, bk) # model is the call to PEDS 
            if epoch % 20 == 0:
                print(f"for this epoch {epoch}, the loss is {loss}, while the grads pytree structure is presented as follow:")
                for i, leaf in enumerate(jax.tree.leaves(grads)):
                    print(
                        f"Param {i:2d} | shape: {leaf.shape} | "
                        f"mean grad: {float(jnp.mean(jnp.abs(leaf))):.2e} | "
                        f"max grad : {float(jnp.max(jnp.abs(leaf))):.2e}"
                    ) # this pytree is where the weights and biases are stored
            grad_state = nnx.state(grads) # print this to visualise the full dict
            epoch_loss += float(loss)
            # Apply Adam update: grads → moment estimates → parameter delta → model update
            #   1. Passes grads through the Optax chain (Adam moment updates, LR scaling)
            #   2. Applies the resulting parameter updates in-place on the model.
            optimizer.update(grads)

        # Normalise by dataset size (matches `avg_loss` in training.py)
        avg_train_loss = epoch_loss / train_size
        # Validation (no gradient tracking needed)
        avg_val_loss, avg_pct_err = validation_step(test_pores, test_kappas)
        print(f"this is batch iteration {iterations + 1} with the average train loss {avg_train_loss}, the average validation loss {avg_val_loss} and the average percentage error {avg_pct_err} \n")
        iterations = iterations+1
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_pct_errors.append(avg_pct_err)
        if (epoch + 1) % 50 == 0:
            current_lr = float(lr_schedule(epoch))
            print(
                f" this message is printed every 50 epochs! \n"
                f"Epoch {epoch+1:4d}/{epochs} \n "
                f"Train MSE: {avg_train_loss:8.3f} \n "
                f"Val MSE: {avg_val_loss:8.3f} \n "
                f"Val%: {avg_pct_err:6.2f}% \n "
                f"LR: {current_lr:.2e}"
            )
        # Periodically clear JAX's compilation cache to avoid memory creep
        if (epoch + 1) % 50 == 0:
            jax.clear_caches()
    
    print(f"the total number of iterations is {iterations}.")
    return model, train_losses, val_losses, val_pct_errors


# ─────────────────────────────────────────────
# SECTION 6: Visualisation
# ─────────────────────────────────────────────

def visualise_results(
    model,
    test_pores:   np.ndarray,
    test_kappas:  np.ndarray,
    train_losses: list,
    val_losses:   list,
    val_pct_errs: list,
    sample_idx:   int = 0,
    save_path:    str = "./experiments/coding/figures/MYCODE_results.png",
):
    """Four-panel figure:
       (a) pore geometry for one test sample
       (b) generated conductivity field for that sample
       (c) training vs validation MSE loss curve
       (d) validation percentage-error curve
    """
    # ── Run one sample through the model ──────────────────────────────────
    pore_sample   = jnp.array(test_pores[sample_idx:sample_idx + 1])   # [1, 5, 5]
    kappa_true    = float(test_kappas[sample_idx])

    kappa_pred_arr, cond_field = model(pore_sample)
    kappa_pred  = float(kappa_pred_arr[0])
    cond_np     = np.array(cond_field[0])   # [5, 5]
    pore_np     = np.array(test_pores[sample_idx])  # [5, 5]
    pore_np = pore_np.reshape(5, 5)             # force to (5, 5) for imshow

    pct_err = abs(kappa_pred - kappa_true) / abs(kappa_true) * 100

    # ── Layout ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # ── Panel (a): Pore geometry ──────────────────────────────────────────
    ax_pore = fig.add_subplot(gs[0, 0])
    im_p = ax_pore.imshow(pore_np, cmap="gray_r", interpolation="nearest",
                          vmin=0, vmax=1)
    ax_pore.set_title(f"(a) Pore Geometry (sample {sample_idx})", fontsize=12)
    ax_pore.set_xlabel("x");  ax_pore.set_ylabel("y")
    for i in range(pore_np.shape[0]):
        for j in range(pore_np.shape[1]):
            ax_pore.text(j, i, f"{int(pore_np[i,j])}", ha="center", va="center",
                         color="red", fontsize=9)
    fig.colorbar(im_p, ax=ax_pore, fraction=0.046, pad=0.04, label="Pore (1=solid)")

    # ── Panel (b): Generated conductivity field ───────────────────────────
    ax_cond = fig.add_subplot(gs[0, 1])
    im_c = ax_cond.imshow(cond_np, cmap="viridis", interpolation="nearest")
    ax_cond.set_title(
        f"(b) Generated Conductivity Field\n"
        f"κ_pred = {kappa_pred:.2f}  |  κ_true = {kappa_true:.2f}  |  err = {pct_err:.1f}%",
        fontsize=11,
    )
    ax_cond.set_xlabel("x");  ax_cond.set_ylabel("y")
    for i in range(cond_np.shape[0]):
        for j in range(cond_np.shape[1]):
            ax_cond.text(j, i, f"{cond_np[i,j]:.1f}", ha="center", va="center",
                         color="white", fontsize=8)
    fig.colorbar(im_c, ax=ax_cond, fraction=0.046, pad=0.04, label="Conductivity")

    epochs_range = np.arange(1, len(train_losses) + 1)

    # ── Panel (c): MSE loss curves ────────────────────────────────────────
    ax_loss = fig.add_subplot(gs[1, 0])
    ax_loss.plot(epochs_range, train_losses, "b-",  lw=1.5, label="Train MSE")
    ax_loss.plot(epochs_range, val_losses,   "r--", lw=1.5, label="Val MSE")
    ax_loss.set_xlabel("Epoch", fontsize=11)
    ax_loss.set_ylabel("Mean Squared Loss", fontsize=11)
    ax_loss.set_title("(c) Training & Validation MSE", fontsize=12)
    ax_loss.set_yscale("log")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    # ── Panel (d): Validation percentage error ────────────────────────────
    ax_pct = fig.add_subplot(gs[1, 1])
    ax_pct.plot(epochs_range, val_pct_errs, "g-", lw=1.5, label="Val % Error")
    ax_pct.axhline(5.0, color="gray", linestyle="--", lw=1, label="5% target")
    ax_pct.set_xlabel("Epoch", fontsize=11)
    ax_pct.set_ylabel("Mean |κ_pred - κ_true| / |κ_true| × 100%", fontsize=10)
    ax_pct.set_title("(d) Validation Percentage Error", fontsize=12)
    ax_pct.legend()
    ax_pct.grid(True, alpha=0.3)

    plt.suptitle("PEDS Educational Run — Single Device, No MPI", fontsize=14, y=1.01)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved to: {save_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # ── Hyperparameters (from config_experiment.py / config_model.py) ─────
    HP = dict(
        filepath      = "./data/highfidelity/high_fidelity_2_20000.npz",  # adjust path as needed
        train_size    = 5000,
        test_size     = 1000,
        batch_size    = 5000,
        epochs        = 1000,
        lr_max        = 5e-3,   # cosine schedule peak learning rate
        lr_min        = 5e-4,   # cosine schedule floor learning rate
        hidden_sizes  = [32, 32],  # matches model config m1
        resolution    = 5,
        seed          = 42,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    model, train_losses, val_losses, val_pct_errs = train(**HP)

    # ── Reload test set for final visualisation ───────────────────────────
    _, (test_pores, test_kappas) = load_data(
        HP["filepath"], HP["train_size"], HP["test_size"], HP["seed"]
    )

    # ── Visualise ─────────────────────────────────────────────────────────
 
    visualise_results(
        model        = model,
        test_pores   = test_pores,
        test_kappas  = test_kappas,
        train_losses = train_losses,
        val_losses   = val_losses,
        val_pct_errs = val_pct_errs,
        sample_idx   = 0,
        save_path    = "./experiments/coding/figures/MYCODE_results.png",
    )

    print("\nDone.")
    print(f"  Final train MSE     : {train_losses[-1]:.4f}")
    print(f"  Final val MSE       : {val_losses[-1]:.4f}")
    print(f"  Final val % error   : {val_pct_errs[-1]:.2f}%")
