"""
=============================================================================
PEDS Training Script
=============================================================================
 
Architecture overview:
  problem geometry (p vector) 
        │
        ▼
  GeneratorNN  ──►   XS tensor (num_regions x G x XS types)
        │
        ▼
  Diffusion solver  ──►  k-eff (scalar per sample)
        │
        ▼
  MSE loss vs k-eff from OpenMC
        │
        ▼
  Backprop through solver into NN weights

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

from matrix_JAX import diffusion_setup_jax
from config_def import GeometryConfig, MaterialSpec, BoundarySpec, BoundaryCondition, MatProperties
from config_run import GEO_CYL as GEO
from diffusion_solver import get_xs_basedon_geo, run_diffusion_solver


# ─────────────────────────────────────────────
# SECTION 1: Physics Solver
# ─────────────────────────────────────────────

xs_tensor = get_xs_basedon_geo(GEO)  # HERE I WILL ADD THE NN CONTRIB

def _run_NT_solver(xs_tensor):
    """Pure NumPy function — takes np array, returns np arrays.
    Neutron diffusion solver for the steady-state NT equation.
    GEO is captured from module scope (closure). 
    Safe because it is a fixed config, not a JAX-traced value,"""
    
    print("\n RUNNING the main NT solver!!!!!!!!!!!")
    R = GEO.boundaries[-1].radius
    I = int(R / GEO.mesh_size)

    k, phi_fwd, phi_adj = run_diffusion_solver(xs_tensor, GEO)
    print(f" \n ---------\n the k is {k}")

    # flatten phi: from [G, I] to [G*(I+1)] to match matrix size
    N_flat = GEO.G * (I + 1)
    phi_fwd_flat = np.zeros(N_flat, dtype=np.float32)
    phi_adj_flat = np.zeros(N_flat, dtype=np.float32)
    for g in range(GEO.G):
        phi_fwd_flat[g*(I+1) : g*(I+1)+I] = phi_fwd[g, :]
        phi_adj_flat[g*(I+1) : g*(I+1)+I] = phi_adj[g, :]

    return np.float32(k), phi_fwd_flat.astype(np.float32), \
           phi_adj_flat.astype(np.float32)


# this is the function that runs when gradients are being computed
# _NTdiff_fwd must also return the residuals, which _NTdiff_bwd will need later.
def _NTdiff_fwd(xs_tensor):
    """Forward pass: run the solver and save the residuals (input, output) for the backward.
    Args:
        XS_tensor: [batch, num_regions x M] neutron cross-sections (NN prediction), 
        to be assigned to specific cells in A,F matrixes 
    Returns:
        k_fwd: scalar, dominant eigenvalue from forward solve
        phi: [batch, N]  — steady-state neutron flux evolution over space
    """
    print("\n Inside the forward pass!!!!")
    R    = GEO.boundaries[-1].radius
    I    = int(R / GEO.mesh_size)

    N_flat = GEO.G * (I + 1)   # total size of flux vector

    keff, phi_fwd, phi_adj = jax.pure_callback(
        _run_NT_solver,
        (
            jax.ShapeDtypeStruct((),        jnp.float32),
            jax.ShapeDtypeStruct((N_flat,), jnp.float32),
            jax.ShapeDtypeStruct((N_flat,), jnp.float32),
        ),
        xs_tensor
    )
    residuals = (xs_tensor, keff, phi_fwd, phi_adj)        
    # MUST return (primal_output, residuals)
    # primal_output must match exactly what NTdiff_solver returns
    # The second return value is the "residuals" passed to the backward function.
    return keff, residuals


def _NTdiff_bwd(residuals, g):
    """
    Custom backward pass.
    Args:
        residuals: the residuals saved by _NTdiff_fwd
        g:   scalar, dL/dk arriving from the loss function
             (= k - k_ref  when loss is MSE)
    Returns:
        dL/d(xs_tensor), shape [batch, M]
    """
    print("\n Inside the backward pass!!!!")

    # --- unpack everything ---
    xs_tensor, k, phi_fwd, phi_adj = residuals
    dL_dk = g  

    # --- define the two matrix-vector functions ---
    # phi_fwd is captured from outer scope (treated as constant)    
    def A_phi(xs):
        A, _ = diffusion_setup_jax(xs)
        return A @ phi_fwd 
    def F_phi(xs):
        _, F = diffusion_setup_jax(xs)
        return F @ phi_fwd 

    value_A, vjp_fn_A = jax.vjp(A_phi, xs_tensor)
    dA_dp_phi = vjp_fn_A(phi_adj)[0]
    term1 = dA_dp_phi

    value_F, vjp_fn_F = jax.vjp(F_phi, xs_tensor)
    dF_dp_phi = vjp_fn_F(phi_adj)[0]
    term2 = (1/k) * dF_dp_phi

    numerator = term1 - term2
    phiadj_F_phi = phi_adj @ F_phi(xs_tensor)  # this is a scalar (dot product of two vectors)
    denominator =  (1/k**2) * phiadj_F_phi
    dk_dp = - numerator / denominator

    dL_dp = dL_dk * dk_dp
    dL_dxs_tensor = dL_dp
    return dL_dxs_tensor

# this function represents what the solver does when called 
# outside of differentiation context.
@jax.custom_vjp
def NTdiff_solver(xs_tensor):
    keff, _ = _NTdiff_fwd(xs_tensor)  # only care about the primal output
    return keff    


NTdiff_solver.defvjp(_NTdiff_fwd, _NTdiff_bwd)


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
    """MLP that maps a flat geometry vector → a 2D conductivity field.

    Architecture (matching PEDS config m1):
        Input  : [batch, 6]    — 6 config parameters
        Hidden : [32, 32]       — fully-connected ReLU layers  
        Output : [batch, 6]    → reshaped to [batch,6]
                                  with hardtanh_positive activation
                                  so all values ∈ (1e-16, 160)
    """

    def __init__(self, layer_sizes: list, n_regions: int, xs_per_region: int, rngs: nnx.Rngs):
        """
        Args:
            layer_sizes: e.g. [6, 32, 32, 6]  (input → hiddens → output)
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
        #  THE OUTPUT OF THE NN IS THE XS_TENSOR, WHICH HAS SIZE (N_regions, XS_per_region)
        self.n_regions    = n_regions     # 3  (b4c_rod, fuel_annulus, water)
        self.xs_per_region = xs_per_region  # 12  (4*G + G² for G=2)
        # self.resolution = int(layer_sizes[-1] ** 0.5)  # output side-length (5)

    def __call__(self, geoms: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """
        Args:
            geoms:    [batch, 6]  binary pore mask
            training: flag for layers like Dropout (unused here, kept for API parity)

        Returns:
            conductivity_field: [batch,6]  physical conductivity values
        """
        x = geoms
        # Apply ReLU on all but the final layer
        for layer in self.layers[:-1]:
            x = nnx.relu(layer(x))        # nnx.relu is just jax.nn.relu, re-exported
        x = self.layers[-1](x)  # Final layer: linear projection → clip to physical range
        x = _hardtanh_positive(x)  # ensures κ > 0 everywhere; could have used softplus(x)

        # Reshape flat vector [batch, 6] → spatial grid [batch,6]
        batch_size = geoms.shape[0]
        # print(f"The last layer is {x}, the size of the batch being {batch_size} with a resolution of {self.resolution}")
        return jnp.reshape(x, (batch_size, self.n_regions, self.xs_per_region))


# ─────────────────────────────────────────────
# SECTION 3: Combined PEDS Model
# ─────────────────────────────────────────────

class PEDSModel(nnx.Module):
    """Wraps GeneratorNN + GaussSolver into a single differentiable forward pass.

    Gradients from the scalar κ loss flow back through the solver (via the
    custom VJP) and then into the NN weights.
    """

    def __init__(self, hidden_sizes: list, n_regions: int, G:int, rngs: nnx.Rngs):
        super().__init__()
        from diffusion_solver import xs_per_region   # already built for other things
        xs_region = xs_per_region(G)            # = 12 for G=2
        output_size = n_regions * xs_region     # = 3 * 12 = 36        
        layer_sizes = [6] + hidden_sizes + [output_size]  # 6 geometry inputs → 36 XS outputs
    
        # rngs is stored by nnx and thread through all sub-modules that need it
        self.generator = GeneratorNN(layer_sizes=layer_sizes, n_regions=n_regions,
                                 xs_per_region=xs_region, rngs=rngs)
        self.n_regions = n_regions
        self.G = G
        
    def __call__(self, geoms: jnp.ndarray, training: bool = False):
        """
        Args:
            geoms: [batch,6] binary pore geometry (flattened inside)

        Returns:
            keff:               [batch]    effective thermal conductivity
            conductivity_field:  [batch, N, N]  intermediate field (for visualisation)
        """
        batch_size = geoms.shape[0]
        geoms_flat = jnp.reshape(geoms, (batch_size, 6))  # flatten spatial dims

        # 1. NN generates a plausible conductivity field
        xs_tensor = self.generator(geoms_flat, training)  # [batch,6]

        # 2. Physics solver maps conductivity field → κ
        #    Gradients propagate back through this call via the custom VJP
        keff = NTdiff_solver(xs_tensor[0])  # TODO why only the first elem????'      

        return keff, xs_tensor


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
    """Load geometries specs and keff labels from the .npz dataset created with MC. Returns NumPy arrays 
    """
    data = np.load(filepath, allow_pickle=True)
    print(f"Dataset keys: {list(data.keys())}")
    geoms  = np.array(data['params'],  dtype=np.float32)   # [N,6]
    keffs = np.array(data['keffs'], dtype=np.float32)   # [N]
    print(f"Loaded the data: example geom {geoms[0]} and keff {keffs[0]}")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(geoms)) # randomly shuffle the indices of the dataset to ensure random sampling of training and test sets
    train_idx = idx[:train_size] # index to save array from location 0 to location of training set dimension
    test_idx  = idx[train_size:train_size + test_size] # index for the array that takes the following chunk of data, of dimension test set

    return (geoms[train_idx], keffs[train_idx]), (geoms[test_idx], keffs[test_idx])


# ─────────────────────────────────────────────
# SECTION 5: Training Loop
# ─────────────────────────────────────────────

def train(filepath, train_size, test_size, batch_size, epochs, lr_max, lr_min,
    hidden_sizes, n_regions, G, seed):
    if hidden_sizes is None:
        hidden_sizes = [32, 32]   # matches config m1 in the original codebase
    # ── 5.1  Data ──────────────────────────────────────────────────────────
    print("Loading data …")
    (train_geoms, train_keffs), (test_geoms, test_keffs) = load_data(
        filepath, train_size, test_size, seed
    )   # train_geoms is a numpy array of shape (train_size, 6), while train_keffs is a numpy array of shape (train_size,) 
    print(f"  Train: {train_geoms.shape}  Test: {test_geoms.shape}")
    
    # ── 5.2  Model & Optimiser ─────────────────────────────────────────────
    # nnx.Rngs(seed) creates a named-key container to get a fresh, unique PRNGKey.
    rngs  = nnx.Rngs(seed)
    model = PEDSModel(hidden_sizes=hidden_sizes, n_regions=n_regions, G=G, rngs=rngs)
    # Cosine decay schedule: learning rate anneals smoothly from lr_max → lr_min.
    # `alpha` is the ratio min/max, so the schedule bottoms out at lr_min.
    lr_schedule = optax.cosine_decay_schedule(
        init_value=lr_max,
        decay_steps=epochs,
        alpha=lr_min / lr_max,   # final lr = lr_max * alpha = lr_min
    )
    optimizer = nnx.Optimizer(model, optax.adam(lr_schedule))     # nnx.Optimizer couples the model's parameters with an Optax update rule.

    # ── 5.3  Loss / gradient function ─────────────────────────────────────
    def loss_fn(model, geoms, keffs_true):
        # model is passed explicitly so nnx.value_and_grad knows which pytree to differentiate with respect to.
        keff_pred, _ = model(geoms, training=True) # keff predicted by PEDS model
        residuals = keff_pred - keffs_true # keff from the data batch
        return jnp.sum(residuals ** 2)   # Sum (not mean) MSE — consistent with original training.py
    # nnx.value_and_grad is the Flax-nnx analogue of jax.value_and_grad.
    # It differentiates `loss_fn` with respect to its FIRST argument (the model),
    grad_fn = nnx.value_and_grad(loss_fn)  # returning (loss_value, grad_pytree_of_model_params)

    # ── 5.4  Validation helper ─────────────────────────────────────────────
    def validation_step(geoms_np, keffs_np):
        """Compute mean squared loss and mean percentage error over the test set."""
        total_sq  = 0.0
        total_pct = 0.0
        # UNDERSTAND HOW IS THIS LOOP REPEATED
        for batch_idx, (batch_geoms, batch_keffs) in enumerate(data_loader(geoms_np, keffs_np, batch_size=batch_size)):
            # the batch index is always 0 because the test set is smaller than the batch size, so we only have one batch containing the whole test set.
            keff_pred, _ = model(jnp.array(batch_geoms))   # no training=True → no stochasticity
            sq_err  = jnp.sum((keff_pred - batch_keffs) ** 2) # squared error
            pct_err = jnp.sum(jnp.abs(keff_pred - batch_keffs) / jnp.abs(batch_keffs) * 100.0) # percent error
            total_sq  += float(sq_err)
            total_pct += float(pct_err)
        n = len(keffs_np)
        return total_sq / n, total_pct / n   # mean over all test samples

    # ── 5.5  Epoch loop ────────────────────────────────────────────────────
    train_losses   = []
    val_losses     = []
    val_pct_errors = []
    iterations = 0 
    print(f"\nTraining for {epochs} epochs …")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_geoms, batch_keffs in data_loader(train_geoms, train_keffs, batch_size=batch_size):
            bp = jnp.array(batch_geoms) # Convert NumPy → JAX arrays once per batch 
            bk = jnp.array(batch_keffs)
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
        avg_val_loss, avg_pct_err = validation_step(test_geoms, test_keffs)
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
    test_geoms:   np.ndarray,
    test_keffs:  np.ndarray,
    train_losses: list,
    val_losses:   list,
    val_pct_errs: list,
    sample_idx:   int = 0,
    save_path:    str = "./experiments/coding/figures/MYNT_results.png",
):
    """Four-panel figure:
       (a) pore geometry for one test sample
       (b) generated conductivity field for that sample
       (c) training vs validation MSE loss curve
       (d) validation percentage-error curve
    """
    # ── Run one sample through the model ──────────────────────────────────
    pore_sample   = jnp.array(test_geoms[sample_idx:sample_idx + 1])   # [1,6]
    kappa_true    = float(test_keffs[sample_idx])

    keff_pred_arr, cond_field = model(pore_sample)
    keff_pred  = float(keff_pred_arr[0])
    cond_np     = np.array(cond_field[0])   # [5, 5]
    pore_np     = np.array(test_geoms[sample_idx])  # [5, 5]
    pore_np = pore_np.reshape(5, 5)             # force to (5, 5) for imshow

    pct_err = abs(keff_pred - kappa_true) / abs(kappa_true) * 100

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
        f"κ_pred = {keff_pred:.2f}  |  κ_true = {kappa_true:.2f}  |  err = {pct_err:.1f}%",
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

    plt.suptitle("PEDS Run — Single Device, No MPI", fontsize=14, y=1.01)
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
        filepath      = "../data/highfidelity/NT_smallMCrun.npz",  # adjust path as needed
        train_size    = 80,
        test_size     = 20,
        batch_size    = 80,
        epochs        = 10,
        lr_max        = 5e-3,   # cosine schedule peak learning rate
        lr_min        = 5e-4,   # cosine schedule floor learning rate
        hidden_sizes  = [32, 32],  # matches model config m1
        n_regions    = 3,    # b4c_rod, fuel_annulus, water
        G            = 2,    # energy groups
        seed          = 42,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    model, train_losses, val_losses, val_pct_errs = train(**HP)

    # ── Reload test set for final visualisation ───────────────────────────
    _, (test_geoms, test_keffs) = load_data(
        HP["filepath"], HP["train_size"], HP["test_size"], HP["seed"]
    )

    # ── Visualise ─────────────────────────────────────────────────────────
 
    visualise_results(
        model        = model,
        test_geoms   = test_geoms,
        test_keffs  = test_keffs,
        train_losses = train_losses,
        val_losses   = val_losses,
        val_pct_errs = val_pct_errs,
        sample_idx   = 0,
        save_path    = "./experiments/coding/figures/NT/results.png",
    )

    print("\nDone.")
    print(f"  Final train MSE     : {train_losses[-1]:.4f}")
    print(f"  Final val MSE       : {val_losses[-1]:.4f}")
    print(f"  Final val % error   : {val_pct_errs[-1]:.2f}%")
