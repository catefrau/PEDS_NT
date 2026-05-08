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
import time
from contextlib import contextmanager
from collections import defaultdict
import csv

from matrix_JAX_optimized import diffusion_setup_jax, Aphi_Fphi_scan
from config_def import GeometryConfig, MaterialSpec, BoundarySpec, BoundaryCondition, MatProperties
from config_run import GEO_CYL as GEO
from diffusion_solver import get_xs_basedon_geo, run_diffusion_solver, predict_xs, precompute_geometry, xs_layout
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from plot_functions.xs_heatmap import plot_xs_heatmap, plot_xs_subplots

# ─────────────────────────────────────────────
# SECTION 0: variables initiation and global constants
# ─────────────────────────────────────────────
train_size    = 5
test_size     = 3
batch_size    = train_size
epochs        = 6
# predict the XS with the polynomial reg built in the other code
# acts as a first guess???
# TODO remove this and thefunction used 
XS_BASELINE = get_xs_basedon_geo(GEO)  # HERE I WILL ADD THE NN CONTRIB
XS_BASELINE = jnp.array(XS_BASELINE, dtype=jnp.float32)   # convert to JAX

GEO_DATA = precompute_geometry(GEO)
# Side channel to store the geometry information for each run — indexed by sample position in batch
_GEO_DATA_CACHE = {}
SLAY      = xs_layout(GEO.G) # Index slices for each XS type, given G groups

_XS_CACHE: dict = {}
HEATMAP_INTERVAL = 2
XS_HEATMAP_SNAPSHOTS = []
XS_HEATMAP_LABELS = []
_xs_baseline_ref = [None]        # list-box so inner assignment doesn't shadow
_snap_final      = None

# ─────────────────────────────────────────────
# SECTION 1: Physics Solver
# ─────────────────────────────────────────────
def _run_NT_solver(xs_tensor, params_raw_single, sample_id):
    """ NumPy function — Neutron diffusion solver for the steady-state NT equation.
    GEO is captured from the cached _GEO_DATA_CACHE for the sample."""
    
    geo_i    = update_geo(GEO, np.array(params_raw_single))
    geo_data = precompute_geometry(geo_i)       # per-sample geometry
    _GEO_DATA_CACHE[int(sample_id[0])] = geo_data

    R      = geo_i.boundaries[-1].radius
    I      = int(R / geo_i.mesh_size)

    with timer("  solver: eigenvalue solve (fwd)", verbose=False):
        k, phi_fwd, phi_adj = run_diffusion_solver(xs_tensor, geo_i)

    # flatten phi: from [G, I] to [G*(I+1)] to match matrix size
    N_flat = geo_i.G * (I + 1)
    phi_fwd_flat = np.zeros(N_flat, dtype=np.float32)
    phi_adj_flat = np.zeros(N_flat, dtype=np.float32)
    for g in range(geo_i.G):
        phi_fwd_flat[g*(I+1) : g*(I+1)+I] = phi_fwd[g, :]
        phi_adj_flat[g*(I+1) : g*(I+1)+I] = phi_adj[g, :]

    return np.float32(k), phi_fwd_flat.astype(np.float32), \
           phi_adj_flat.astype(np.float32)


# this is the function that runs when gradients are being computed
def _NTdiff_fwd(xs_tensor, params_raw_single, sample_id):
    """Forward pass: run the solver and save the residuals (input, output) for the backward.
    Args:
        XS_tensor: [batch, num_regions x M] neutron cross-sections (NN prediction), 
        to be assigned to specific cells in A,F matrixes 
    Returns:
        k_fwd: scalar, dominant eigenvalue from forward solve
        phi: [batch, N]  — steady-state neutron flux evolution over space
    """
    geo_i    = update_geo(GEO, np.array(params_raw_single))

    R      = geo_i.boundaries[-1].radius
    I      = int(R / geo_i.mesh_size)

    N_flat = geo_i.G * (I + 1)   # total size of flux vector

    keff, phi_fwd, phi_adj = jax.pure_callback(
        _run_NT_solver,
        (
            jax.ShapeDtypeStruct((),        jnp.float32),
            jax.ShapeDtypeStruct((N_flat,), jnp.float32),
            jax.ShapeDtypeStruct((N_flat,), jnp.float32),
        ),
        xs_tensor,  params_raw_single, sample_id
    )  
    geo_data = _GEO_DATA_CACHE[int(sample_id[0])]
    _, F = diffusion_setup_jax(xs_tensor, geo_data, SLAY)
    Fphi = F @ phi_fwd 

    residuals = (xs_tensor, keff, phi_fwd, phi_adj, Fphi, sample_id)        
    # primal must match exactly what NTdiff_solver returns
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
    xs_tensor, k, phi_fwd, phi_adj, Fphi, sample_id = residuals
    geo_data = _GEO_DATA_CACHE[int(sample_id[0])]
    dL_dk = g  

    # --- define the two matrix-vector functions ---
    # TODO phi_fwd is captured from outer scope (treated as constant)    
    def AF_phi(xs):
        with timer("  (Matrixes building part)", verbose=False):
            return Aphi_Fphi_scan(xs, geo_data, SLAY, phi_fwd)  # returns (Aphi, Fphi)
        
    with timer("  bwd: vjp A@phi, F@phi", verbose=False):
        _, vjp_fn = jax.vjp(AF_phi, xs_tensor)
        numerator, = vjp_fn((phi_adj, -(1.0 / k) * phi_adj))

    with timer("  bwd: denominator", verbose=False):
        phiadj_F_phi = phi_adj @ Fphi  # this is a scalar (dot product of two vectors)
        denominator =  (1/k**2) * phiadj_F_phi
    
    dk_dp = - numerator / denominator
    dL_dxs_tensor = dL_dk * dk_dp
    return (dL_dxs_tensor, None, None)  # only return gradients for xs_tensor; the other two inputs have no gradients

# this function represents what the solver does when called outside of differentiation context.
@jax.custom_vjp
def NTdiff_solver(xs_tensor, params_raw_single, sample_id):
    keff, _ = _NTdiff_fwd(xs_tensor, params_raw_single, sample_id) 
    return keff    

NTdiff_solver.defvjp(_NTdiff_fwd, _NTdiff_bwd)

# ─────────────────────────────────────────────
# SECTION 2: Utils: timings, geometry updates, sanity checks
# ─────────────────────────────────────────────

# Global registry — stores all measured times
_TIMINGS = defaultdict(list)

@contextmanager
def timer(label: str, verbose: bool = True):
    """
    Context manager that measures wall time and stores it.
    Usage:  with timer("forward pass"):
                result = model(x)
    """
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    _TIMINGS[label].append(elapsed)
    if verbose:
        print(f"  ⏱  {label:<35s} {elapsed*1000:.1f} ms")

def print_timing_report():
    """Print a summary of all measured times after training."""
    print("\n" + "="*60)
    print(f"{'TIMING REPORT':^60}")
    print("="*60)
    print(f"{'Step':<35s} {'calls':>6s} {'total(s)':>10s} {'mean(ms)':>10s} {'min(ms)':>10s}")
    print("-"*60)
    for label, times in sorted(_TIMINGS.items()):
        total   = sum(times)
        mean_ms = (total / len(times)) * 1000
        min_ms  = min(times) * 1000
        print(f"{label:<35s} {len(times):>6d} {total:>10.2f} {mean_ms:>10.1f} {min_ms:>10.1f}")
    print("="*60)


def compute_batch_baselines(params_raw: np.ndarray, geo: GeometryConfig) -> jnp.ndarray:
    """
    params_raw: [batch, 6] — un-normalized geometry parameters
    Returns: [batch, 3, 12] — regression-predicted XS for each sample
    """
    baselines = []
    for i in range(params_raw.shape[0]):
        # Build a modified GEO for this sample's raw parameters vector
        geo_i = update_geo(geo, params_raw[i]) #
        xs_i  = predict_xs(geo_i)        # regression function
        baselines.append(xs_i)

    return jnp.array(np.stack(baselines), dtype=jnp.float32)  # [batch, 3, 12]

def update_geo(geo: GeometryConfig, params_raw: np.ndarray) -> GeometryConfig:
    """
    Build a new GeometryConfig from a single raw parameter vector.
    params_raw: [6] — [b4c_outer_r, cr_fraction, fuel_outer_r, enrichment, f_mod, water_outer_r]
    """
    new_boundaries = (
        BoundarySpec(name='CR_outer',         radius=float(params_raw[0])),
        BoundarySpec(name='core_outer',       radius=float(params_raw[2])),
        BoundarySpec(name='moderator_outer',  radius=float(params_raw[5])),
    )
    new_mat = MatProperties(
        cr_fraction = float(params_raw[1]),
        enrichment  = float(params_raw[3]),
        f_mod       = float(params_raw[4]),
    )
    return GeometryConfig(
        G              = geo.G,
        regions        = geo.regions,
        boundaries     = new_boundaries,
        geometry       = geo.geometry,
        mat_properties = new_mat,
        bc             = geo.bc,
        mesh_size      = geo.mesh_size,
    )

def physics_sanity_check(xs_baseline):
    """
    Check gradient signs against physical intuition.
    Run these BEFORE finite differences — they're faster to interpret.
    """
    xs = jnp.array(xs_baseline, dtype=jnp.float32)
    _, vjp_fn = jax.vjp(NTdiff_solver, xs)
    grad = np.array(vjp_fn(jnp.ones(()))[0])   # shape (3, 12)

    lay = xs_layout(2)   # G=2
    print("=== Physics Sanity Check ===")

    for reg_idx, reg_name in enumerate(['b4c_rod', 'fuel_annulus', 'water']):
        d_keff_d_D      = grad[reg_idx, lay['D']]
        d_keff_d_Siga   = grad[reg_idx, lay['Sigma_a']]
        d_keff_d_nuSigf = grad[reg_idx, lay['nuSigma_f']]

        print(f"\nRegion: {reg_name}")
        print(f"  ∂keff/∂D         = {d_keff_d_D}")
        print(f"  ∂keff/∂Σ_a       = {d_keff_d_Siga}")
        print(f"  ∂keff/∂νΣ_f      = {d_keff_d_nuSigf}")
        
# ─────────────────────────────────────────────
# SECTION 2: Neural Network (Generator)
# ─────────────────────────────────────────────

def _hardtanh_correction(x):
    """Clip activations to (1e-16, 160.0) — ensures conductivity stays physical.
    This replaces the final activation layer so the network cannot output
    negative conductivities (which would break the physics solver).
    """
    return 0.5 * jnp.tanh(x)  # it only influences the output for 50%


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

        self.layers[-1] = nnx.Linear(
            in_features=layer_sizes[-2],
            out_features=layer_sizes[-1],
            kernel_init=nnx.initializers.zeros,   # ← zero init
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        )

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
        x = _hardtanh_correction(x)  # ensures κ > 0 everywhere; could have used softplus(x)

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
        
    def __call__(self, geoms: jnp.ndarray, params_raw: np.ndarray, training: bool = False):
        """
        Args:
            geoms: [batch,6] binary pore geometry (flattened inside)

        Returns:
            keff:               [batch]    effective thermal conductivity
            conductivity_field:  [batch, N, N]  intermediate field (for visualisation)
        """
        batch_size = geoms.shape[0]
        geoms_flat = jnp.reshape(geoms, (batch_size, 6))  # flatten spatial dims

        # Per-sample baselines from regression (NumPy, not traced by JAX)
        with timer("forward: regression baselines", verbose=False):        
            xs_baselines = compute_batch_baselines(params_raw, GEO)  # [batch, 3, 12]
        
        # NN learns corrections on top of each sample's own baseline
        with timer("forward: NN generated XS", verbose=False):
            xs_corrections = self.generator(geoms_flat, training)  # [batch, 3, 12]
        #print(f"example of this tensor {xs_tensor}")  
        xs_final = xs_baselines * (1.0 + xs_corrections) # final mix
        
        #print("RUNNING a gradients physics check ")
        #physics_sanity_check(xs_baselines[0])        
        #print(f" the baseline starting point XS tensor was {xs_baselines[0]}")
        #print(f"and the final corrected one is {xs_final[0]}")

        with timer("forward: solver loop (all samples)", verbose=False):
            keffs = [] # runs a calc for each of the 10 and then stores
            for i in range(batch_size):
                #print(f"the XS prediction for this geom is {xs_final[i]}")
                #print(f"where the baseline is {XS_BASELINE} \n and the NN gen is {xs_corrections[i]}")
                keff_i = NTdiff_solver(xs_final[i], 
                    jnp.array(params_raw[i], dtype=jnp.float32),       # (6,)   raw geometry
                    jnp.array([i], dtype=jnp.int32), )         # (1,)   sample ID)   # pass (3, 12) slice
                keffs.append(keff_i)
            keffs = jnp.stack(keffs)   # [batch]
        """ keffs = jnp.stack([
            NTdiff_solver_with_geo(xs_final[i], ALL_GEO_DATAS[sample_indices[i]])
            for i in range(batch_size)
        ]) """

        #print(f"are these all the keffs together? {keffs}, its size is {keffs.size}") 
        # same size as batch
        return keffs, xs_final
        #return keff, xs_tensor


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
    rawparams = np.array(data['params_raw'], dtype=np.float32)   # [N]
    print(f"Loaded the data: example geom {geoms[0]} and keff {keffs[0]}")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(geoms)) # randomly shuffle the indices of the dataset to ensure random sampling of training and test sets
    train_idx = idx[:train_size] # index to save array from location 0 to location of training set dimension
    test_idx  = idx[train_size:train_size + test_size] # index for the array that takes the following chunk of data, of dimension test set

    return (geoms[train_idx], keffs[train_idx], rawparams[train_idx]), (geoms[test_idx], keffs[test_idx], rawparams[test_idx])


# ─────────────────────────────────────────────
# SECTION 5: Training Loop
# ─────────────────────────────────────────────

def train(filepath, train_size, test_size, batch_size, epochs, lr_max, lr_min,
    hidden_sizes, n_regions, G, seed):
    if hidden_sizes is None:
        hidden_sizes = [32, 32]   # matches config m1 in the original codebase
    # ── 5.1  Data ──────────────────────────────────────────────────────────
    print("Loading data …")
    (train_geoms, train_keffs, train_rawparams), (test_geoms, test_keffs, test_rawparams) = load_data(
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
    
    # TODO move the call in a safer place
    print(f"an example of a raw param set is {train_rawparams[0]}")

    # ── 5.3  Loss / gradient function ─────────────────────────────────────
    def loss_fn(model, geoms, keffs_true, rawparams_batch):
        # model is passed explicitly so nnx.value_and_grad knows which pytree to differentiate with respect to.
        keff_pred, _ = model(geoms, rawparams_batch, training=True) # keff predicted by PEDS model
        print(f"the prediction gives a keff of {keff_pred.primal} (size {keff_pred.size}), compared to {keffs_true} (size {keffs_true.size})")
        residuals = keff_pred - keffs_true # keff from the data batch
        return jnp.sum(residuals ** 2)   # Sum (not mean) MSE — consistent with original training.py
    # nnx.value_and_grad is the Flax-nnx analogue of jax.value_and_grad.
    # It differentiates `loss_fn` with respect to its FIRST argument (the model),
    grad_fn = nnx.value_and_grad(loss_fn)  # returning (loss_value, grad_pytree_of_model_params)

    # ── 5.4  Validation helper ─────────────────────────────────────────────
    def validation_step(geoms_np, keffs_np, rawparams_np):
        """Compute mean squared loss and mean percentage error over the test set.
            Also returns per-sample keff_pred and keff_ref arrays for logging."""

        total_sq  = 0.0
        total_pct = 0.0
        all_keff_pred = []
        all_keff_ref  = []
        for batch_idx, (batch_geoms, batch_keffs, batch_rawparams) in enumerate(data_loader(geoms_np, keffs_np, rawparams_np, batch_size=batch_size)):
            # the batch index is always 0 because the test set is smaller than the batch size, so we only have one batch containing the whole test set.
            keff_pred, _ = model(jnp.array(batch_geoms), batch_rawparams, training=False)   # no training=True → no stochasticity
            sq_err  = jnp.sum((keff_pred - batch_keffs) ** 2) # squared error
            pct_err = jnp.sum(jnp.abs(keff_pred - batch_keffs) / jnp.abs(batch_keffs) * 100.0) # percent error
            total_sq  += float(sq_err)
            total_pct += float(pct_err)
            all_keff_pred.extend(np.array(keff_pred).tolist())   # ← collect predictions
            all_keff_ref.extend(np.array(batch_keffs).tolist())  # ← collect references            
        n = len(keffs_np)
        return total_sq / n, total_pct / n, all_keff_pred, all_keff_ref   
        # mean over all test samples

    # ── 5.5  Epoch loop ────────────────────────────────────────────────────
    train_losses   = []
    val_losses     = []
    val_pct_errors = []
    iterations = 0 
    print(f"\nTraining for {epochs} epochs …")

        # ── Open keff log file ────────────────────────────────────────────────
    log_path = "./LOGS/keff_epoch_log.csv"
    log_file  = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "sample_idx", "keff_openmc", "keff_peds", "delta_rho_pcm",  "epoch_train_loss", "epoch_val_loss"])

    for epoch in range(epochs):
        epoch_loss = 0.0
        # these are the parameters used for the training loop
        for batch_geoms, batch_keffs, batch_rawparams_tr in data_loader(train_geoms, train_keffs, train_rawparams, batch_size=batch_size):
            bp = jnp.array(batch_geoms) # Convert NumPy → JAX arrays once per batch 
            bk = jnp.array(batch_keffs)
            print("Running forward pass + AD \n ")
            # Part of training: forward pass + automatic differentiation in one call.
            # `loss` is a scalar; `grads` is a pytree mirroring model's parameter tree.
            with timer("train step: forward+backward+update"):
                with timer("train step: forward+loss+grad"):
                    loss, grads = grad_fn(model, bp, bk, batch_rawparams_tr) # model is the call to PEDS 
                if epoch % 2 == 0:
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
                with timer("train step: optimizer update"):
                    optimizer.update(grads)
            
            # still inside the epoch loop 
            # ── Collect heatmap snapshot every HEATMAP_INTERVAL epochs ────────
            if (epoch + 1) % HEATMAP_INTERVAL == 0 or epoch == 0:
                _snap_geo_i       = update_geo(GEO, train_rawparams[0])          # single GeometryConfig
                _snap_baseline    = np.array(predict_xs(_snap_geo_i))            # (3, 12) guaranteed

                _snap_geom_flat   = jnp.array(train_geoms[0:1])                  # (1, 6)
                _snap_corrections = np.array(
                    model.generator(_snap_geom_flat, training=False)[0]          # (3, 12)
                )
                _snap_final = _snap_baseline * (1.0 + _snap_corrections)         # (3, 12)

                # Freeze baseline reference on the very first snapshot
                if _xs_baseline_ref[0] is None:
                    _xs_baseline_ref[0] = _snap_baseline.copy()

                XS_HEATMAP_SNAPSHOTS.append(_snap_corrections.copy())
                XS_HEATMAP_LABELS.append(f"Epoch {epoch + 1}")

        # Normalise by dataset size (matches `avg_loss` in training.py)
        avg_train_loss = epoch_loss / train_size
        # Validation (no gradient tracking needed)
        with timer("validation step"):
            avg_val_loss, avg_pct_err, keff_preds, keff_refs = validation_step(test_geoms, test_keffs, test_rawparams)
        
        # ── Write per-sample keff values to the log file ─────────────────────
        for s_idx, (k_ref, k_pred) in enumerate(zip(keff_refs, keff_preds)):
            delta_rho_pcm = abs((k_pred - k_ref) / (k_pred * k_ref)) * 1e5   # ✅ correct
            log_writer.writerow([epoch + 1, s_idx, f"{k_ref:.6f}", f"{k_pred:.6f}", f"{delta_rho_pcm:.1f}", avg_train_loss, avg_val_loss])
        log_file.flush()   # write to disk immediately, so you can tail the file during a long run
                
        print(f"this is batch iteration {iterations + 1} with the average train loss {avg_train_loss:.4f}, the average validation loss {avg_val_loss:.4f} and the average percentage error {avg_pct_err:.4f} \n")
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
    log_file.close()
    print(f"\n keff log saved to: {log_path}")

    # ── Generate XS heatmap ───────────────────────────────────────────
    print(f"[xs_heatmap] baseline ref is None: {_xs_baseline_ref[0] is None}")
    print(f"[xs_heatmap] number of snapshots collected: {len(XS_HEATMAP_SNAPSHOTS)}")
    print(f"[xs_heatmap] snapshot labels: {XS_HEATMAP_LABELS}")

    if _xs_baseline_ref[0] is not None and len(XS_HEATMAP_SNAPSHOTS) > 0:
        print(f"\nGenerating XS heatmap with {len(XS_HEATMAP_SNAPSHOTS)} snapshots …")
        print(f"saving it to ")
        plot_xs_heatmap(
            baseline           = _xs_baseline_ref[0],
            snapshots          = XS_HEATMAP_SNAPSHOTS,
            epoch_labels       = XS_HEATMAP_LABELS,
            final_xs           = _snap_final,
            G                  = G,
            save_path          = "./LOGS/figures/xs_heatmap.png",
            plot_interval_note = f"snapshot every {HEATMAP_INTERVAL} epochs",
        )
        plot_xs_subplots(
            baseline           = _xs_baseline_ref[0],
            final_xs           = _snap_final,
            G                  = G,
            save_path          = "./LOGS/figures/xs_subplots.png",
        )
    else:
        print("[xs_heatmap] skipped — no snapshots collected (epochs < HEATMAP_INTERVAL?)")
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
        train_size    = train_size,
        test_size     = test_size,
        batch_size    = batch_size,
        epochs        = epochs,
        lr_max        = 5e-4,   # cosine schedule peak learning rate
        lr_min        = 5e-5,   # cosine schedule floor learning rate
        hidden_sizes  = [64, 32],  # matches model config m1
        n_regions    = 3,    # b4c_rod, fuel_annulus, water
        G            = 2,    # energy groups
        seed          = 42,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    model, train_losses, val_losses, val_pct_errs = train(**HP)
    print_timing_report()
    # ── Reload test set for final visualisation ───────────────────────────
    _, (test_geoms, test_keffs, test_rawparams) = load_data(
        HP["filepath"], HP["train_size"], HP["test_size"], HP["seed"]
    )

    # ── Visualise ─────────────────────────────────────────────────────────
 
    """ visualise_results(
        model        = model,
        test_geoms   = test_geoms,
        test_keffs  = test_keffs,
        train_losses = train_losses,
        val_losses   = val_losses,
        val_pct_errs = val_pct_errs,
        sample_idx   = 0,
        save_path    = "./experiments/coding/figures/NT/results.png",
    ) """

    print("\nDone.")
    print(f"  Final train MSE     : {train_losses[-1]:.4f}")
    print(f"  Final val MSE       : {val_losses[-1]:.4f}")
    print(f"  Final val % error   : {val_pct_errs[-1]:.2f}%")
