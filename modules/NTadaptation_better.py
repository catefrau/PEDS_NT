"""==========================================================================
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

=========================================================================="""
import multiprocessing
import os
multiprocessing.set_start_method("spawn", force=True)
os.environ["JAX_PLATFORMS"] = "cpu"   # force CPU
import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np
import pandas as pd
import optax
from flax import nnx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import csv
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
ctx = mp.get_context("spawn")   # explicit spawn context
import functools
import threading
_csv_lock = threading.Lock()

from matrix_JAX_optimized import diffusion_setup_jax, Aphi_Fphi_scan
from NTcode_config_data.config_def import GeometryConfig, MaterialSpec, BoundarySpec, BoundaryCondition, MatProperties
from NTcode_config_data.config_run import GEO_CYL as GEO
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from plot_functions.xs_heatmap import plot_xs_heatmap, plot_xs_subplots
from solvers.NTdiffusion.core.MG1D_eigenvalue_nregions import diffusion_setup, region_index
from solvers.NTdiffusion.diffusion_solver import (get_xs_basedon_geo, run_diffusion_solver, is_homogeneous,
    predict_xs, precompute_geometry, xs_layout, build_xs_callables, fn_xs_per_region, bc_to_coeffs, GEOMETRY_CODE)

from NTcode_destructure.timing_utils import timer, print_timing_report, _TIMINGS
from NTcode_destructure.diagnostics import full_check, solver_floor_check
from NTcode_destructure.visualize_notused import visualise_results

# ─────────────────────────────────────────────
# SECTION 0: variables initiation and global constants
# ─────────────────────────────────────────────
train_size    = 350
test_size     = 50
batch_size    = 25
epochs        = 100
exp_name = "4jun3pm_noweight" 

N_WORKERS = min(
    int(os.environ.get("SLURM_CPUS_PER_TASK", 16)),
    batch_size 
)
print(f"Using {N_WORKERS} parallel workers for solver")
executor = ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=ctx)
print(f"SLURM_CPUS_PER_TASK = {os.environ.get('SLURM_CPUS_PER_TASK', 'NOT SET')}", flush=True)

os.makedirs(f"./LOGS/{exp_name}", exist_ok=True)
val_log_path   = f"./LOGS/{exp_name}/keff_epoch_log_val.csv"
train_log_path = f"./LOGS/{exp_name}/keff_epoch_log_train.csv"
# Redirect all print output to a log file
loggy_file = open(f"train_log_{exp_name}.txt", "a", buffering=1)  # buffering=1 = write every line immediately
sys.stdout = loggy_file
sys.stderr = loggy_file
print("JAX devices:", jax.devices())
print("Backend:", jax.default_backend())

# predict the XS with the polynomial reg built in the other code
XS_BASELINE = get_xs_basedon_geo(GEO)  # HERE I WILL ADD THE NN CONTRIB
XS_BASELINE = jnp.array(XS_BASELINE, dtype=jnp.float32)   # convert to JAX
# ----------------------
GEO_DATA = precompute_geometry(GEO)
# Side channel to store the geometry information for each run — indexed by sample position in batch
_GEO_DATA_CACHE = {}
SLAY      = xs_layout(GEO.G) # Index slices for each XS type, given G groups

# Fixed reference scale for the NN output — computed ONCE from the default geometry.
# The NN learns log-ratio offsets relative to this scale; exp(0) * XS_SCALE = XS_SCALE at init.
XS_SCALE = jnp.array(get_xs_basedon_geo(GEO), dtype=jnp.float32)  # shape (3, 12), constant

_XS_CACHE: dict = {}
HEATMAP_INTERVAL = 25
XS_HEATMAP_SNAPSHOTS = []
XS_HEATMAP_LABELS = []
_xs_baseline_ref = [None]        # list-box so inner assignment doesn't shadow
_snap_final      = None

# Near the top of the module, after the imports and GEO definition,
# replace:  N_FLAT_MAX = None
# with this block that reads the raw params and computes the max immediately:


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

def _compute_n_flat_max_from_file(filepath: str, geo: GeometryConfig) -> int:
    data = np.load(filepath, allow_pickle=True)
    rawparams = np.array(data['params_raw'], dtype=np.float32)
    n_flat_max = 0
    for i in range(len(rawparams)):
        geo_i = update_geo(geo, rawparams[i])
        R = geo_i.boundaries[-1].radius
        I = int(R / geo_i.mesh_size)
        n_flat_max = max(n_flat_max, geo_i.G * (I + 1))
    return n_flat_max

_DATA_FILEPATH = "../data/highfidelity/MCruns_filtered.npz"
N_FLAT_MAX = _compute_n_flat_max_from_file(_DATA_FILEPATH, GEO)
print(f"N_FLAT_MAX computed at module load: {N_FLAT_MAX}")

# ─────────────────────────────────────────────
# SECTION 0: Solutions for better runspeed and memory efficiency
# ─────────────────────────────────────────────

def _compute_n_flat_max(rawparams: np.ndarray) -> int:
    n_flat_max = 0
    for i in range(len(rawparams)):
        geo_i = update_geo(GEO, rawparams[i])
        R = geo_i.boundaries[-1].radius
        I = int(R / geo_i.mesh_size)
        n_flat_max = max(n_flat_max, geo_i.G * (I + 1))
    return n_flat_max

MAX_CACHE_SIZE = 256   # tune based on your RAM
from collections import OrderedDict

class LRUCache(OrderedDict):
    def __init__(self, maxsize):
        super().__init__()
        self.maxsize = maxsize
    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if len(self) > self.maxsize:
            self.popitem(last=False)   # evict oldest entry

_GEO_DATA_CACHE = LRUCache(maxsize=MAX_CACHE_SIZE)

_PRESOLVE_CACHE: dict = {}   # stores (k, phi_fwd, phi_adj) from parallel pre-solve
def _solve_sample_worker(args):
    """
    Runs in a separate process — no JAX, pure NumPy/SciPy.
    args: (i, xs_np, params_np, sample_id_int)
    Returns: (i, k, phi_fwd, phi_adj, geo_data)
    """
    t0 = time.perf_counter()
    pid = os.getpid()

    i, xs_np, params_np, sample_id_int, epoch = args
    geo_i    = update_geo(GEO, params_np)
    geo_data = precompute_geometry(geo_i)
    
    k, phi_fwd, phi_adj = _run_NT_solver(
        xs_np,
        params_np,
        np.array([sample_id_int])
    )
    elapsed = time.perf_counter() - t0
    #print(f"  worker pid={pid} sample={i} took {elapsed:.2f}s", flush=True)
    return i, k, phi_fwd, phi_adj, geo_data, epoch


_EXECUTOR = ProcessPoolExecutor(max_workers=N_WORKERS)


# ─────────────────────────────────────────────
# SECTION 1: Physics Solver
# ─────────────────────────────────────────────
def _run_NT_solver(xs_tensor, params_raw_single, sample_id):
    """ NumPy function — Neutron diffusion solver for the steady-state NT equation.
    GEO is captured from the cached _GEO_DATA_CACHE for the sample."""
    
    geo_i    = update_geo(GEO, np.array(params_raw_single))
    geo_data = precompute_geometry(geo_i)       # per-sample geometry
    
    sid = int(sample_id[0])
    _GEO_DATA_CACHE[sid] = geo_data

    if sid in _PRESOLVE_CACHE:
        k, phi_fwd, phi_adj, geo_data_pre = _PRESOLVE_CACHE.pop(sid)
        _GEO_DATA_CACHE[sid] = geo_data_pre
        return k, phi_fwd, phi_adj

    R      = geo_i.boundaries[-1].radius
    I      = int(R / geo_i.mesh_size)
    G      = geo_i.G
    N_actual = G * (I + 1)

    with timer("  solver: eigenvalue solve (fwd)", verbose=False):
        k, phi_fwd, phi_adj= run_diffusion_solver(xs_tensor, geo_i)
    
    #print(f" DOUBLECHECKINGGGG [solver] k = {float(k):.6f}, ||phi_fwd||={float(jnp.linalg.norm(phi_fwd)):.6e}, " 
    #      f"||phi_adj||={float(jnp.linalg.norm(phi_adj)):.6e}, ")
    
    # flatten phi: from [G, I] to [G*(I+1)] to match matrix size
    N_flat = geo_i.G * (I + 1)
    phi_fwd_flat = np.zeros(N_flat, dtype=np.float64)
    phi_adj_flat = np.zeros(N_flat, dtype=np.float64)
    for g in range(geo_i.G):
        phi_fwd_flat[g*(I+1) : g*(I+1)+I] = phi_fwd[g, :]
        phi_adj_flat[g*(I+1) : g*(I+1)+I] = phi_adj[g, :]

    # ── Biorthonormalization ──────────────────────────────────────────
    # Enforce ⟨φ†, F·φ⟩ = k²  so the VJP denominator = 1.0 exactly.
    # This removes the arbitrary scaling ambiguity from inverse_power.

    r_divisions   = [b.radius for b in geo_i.boundaries[:-1]] \
                    if not is_homogeneous(geo_i) else []
    BC_coeffs     = bc_to_coeffs(geo_i.bc)               # [A, B, C]
    geometry_code = GEOMETRY_CODE[geo_i.geometry]         # 0/1/2

    D_fn, Sa_fn, nF_fn, Ss_fn, chi_fn = build_xs_callables(np.array(xs_tensor), geo_i)
    _, A_real, F_real = diffusion_setup(R, I, geo_i.G, r_divisions,
                                D_fn, Sa_fn, nF_fn, Ss_fn, chi_fn,
                                BC_coeffs, geometry_code)
    Fphi = F_real @ phi_fwd_flat
    biorth = phi_adj_flat @ Fphi   # scalar ⟨φ†, F·φ⟩

    # Step 3: rescale φ† so that ⟨φ†, F·φ⟩ = k²
    # This makes denominator = (1/k²) * k² = 1.0
    phi_adj_flat *= 1.0 / biorth 
    #print(f"bwd sanity: phiadj·F·phi = {float(phi_adj_flat @ Fphi):.6f}  (should be ~1.0)")

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
    _, Fphi = Aphi_Fphi_scan(xs_tensor, geo_data, SLAY, phi_fwd)

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


def log_keff_batch(writer, filehandle, epoch, keff_preds, keff_refs, avg_train_loss, avg_val_loss, sample_id_offset=0):
    """Log per-sample keff predictions vs OpenMC reference to a CSV."""
    with _csv_lock:                          # ← protect the whole loop
        for sidx, (kref, kpred) in enumerate(zip(keff_refs, keff_preds)):
            delta_rho_pcm = abs(kpred - kref) / (kpred * kref) * 1e5
            writer.writerow([epoch + 1, sidx + sample_id_offset, f"{kref:.6f}", 
                f"{kpred:.6f}", f"{delta_rho_pcm:.1f}", avg_train_loss, avg_val_loss])
        filehandle.flush()


val_logfile   = open(val_log_path, "w", newline="", buffering=1)
train_logfile = open(train_log_path, "w", newline="", buffering=1)
val_writer   = csv.writer(val_logfile)
train_writer = csv.writer(train_logfile)
header = ["epoch", "sample_idx", "keff_openmc", "keff_peds", "delta_rho_pcm", "train_loss", "val_loss"]
val_writer.writerow(header)
train_writer.writerow(header)

# ─────────────────────────────────────────────
# SECTION 2: Neural Network (Generator)
# ─────────────────────────────────────────────

def _hardtanh_correction(x):
    """Clip activations to (1e-16, 160.0) — ensures conductivity stays physical.
    This replaces the final activation layer so the network cannot output
    negative conductivities (which would break the physics solver).
    """
    return 0.5 * jnp.tanh(x)  # it only influences the output for 50%

def _xs_log_scale_activation(x):
    """Maps NN raw output to a log-ratio multiplier.
    exp(0) = 1.0  →  XS_SCALE * 1.0 at init (identity start).
    exp(+2) ≈ 7.4 →  up to ~7x the reference scale.
    exp(-2) ≈ 0.14 → down to ~14% of the reference scale.
    Always positive — no clipping needed; exp() is strictly > 0.
    """
    return x  # raw output; exp() is applied in PEDSModel.__ call__

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
        #bias_init = nnx.initializers.constant(0.0)
        zero_init = nnx.initializers.zeros

        # Shared trunk — ALL layers get He init, ALL get ReLU in __call__
        # layer_sizes is now [6, 128, 128, 64], no output dim here
        self.layers = nnx.List([
            nnx.Linear(
                in_features=layer_sizes[i],
                out_features=layer_sizes[i + 1],
                kernel_init=he_init,
                bias_init=nnx.initializers.constant(0.0),
                rngs=rngs,
            )
            for i in range(len(layer_sizes) - 1)
        ])
        self.skip_proj = nnx.Linear(layer_sizes[0], layer_sizes[-1], rngs=rngs)

        trunk_out = layer_sizes[-1]  # 64
        self.heads = nnx.List([
            nnx.Linear(
                in_features=trunk_out + xs_per_region,  # 64 from trunk + 12 log-baseline for this region
                out_features=xs_per_region,   # 12
                kernel_init=zero_init,
                bias_init=zero_init,
                rngs=rngs,
            )
            for _ in range(n_regions)         # 3 heads: b4c, fuel, water
        ])

        self.n_regions    = n_regions     # 3  (b4c_rod, fuel_annulus, water)
        self.xs_per_region = xs_per_region  # 12  (4*G + G² for G=2)
        # self.resolution = int(layer_sizes[-1] ** 0.5)  # output side-length (5)

    def __call__(self, geoms: jnp.ndarray, xs_baselines_log, training: bool = False) -> jnp.ndarray:
        """
        Args:
            geoms:    [batch, 6]  binary pore mask
            training: flag for layers like Dropout (unused here, kept for API parity)

        Returns:
            conductivity_field: [batch,6]  physical conductivity values
        """
        x = geoms
        skip = nnx.relu(self.skip_proj(geoms))   # project input to trunk-out dim

        # Apply ReLU on all but the final layer
        """ for layer in self.layers[:-1]:
            x = nnx.relu(layer(x))        """
        for layer in self.layers:          # ALL trunk layers get ReLU now
            x = nnx.relu(layer(x))         # x shape stays [batch, 64] after last layer
        x = x + skip                      # add skip connection to trunk output
        #x = self.layers[-1](x)  # Final layer: linear projection → raw log-ratio output

        region_outputs = []
        for r, head in enumerate(self.heads):
            # Concatenate trunk features WITH the log-baseline for this region
            region_feat = jnp.concatenate(
                [x, xs_baselines_log[:, r, :]], axis=-1   # [batch, trunk_out + 12]
            )
            region_outputs.append(head(region_feat))  # head now takes trunk_out+12 input

        #return jnp.reshape(x, (batch_size, self.n_regions, self.xs_per_region))
        return jnp.stack(region_outputs, axis=1)             # [batch, 3, 12]


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
        xs_region = fn_xs_per_region(G)            # = 12 for G=2
        output_size = n_regions * xs_region     # = 3 * 12 = 36        
        layer_sizes = [6] + hidden_sizes #+ [output_size]  # 6 geometry inputs → 36 XS outputs
    
        # rngs is stored by nnx and thread through all sub-modules that need it
        self.generator = GeneratorNN(layer_sizes=layer_sizes, n_regions=n_regions,
                                 xs_per_region=xs_region, rngs=rngs)
        self.n_regions = n_regions
        self.G = G

    def compute_xs(self, geoms, xs_baselines, epoch):
        """Pure NN forward — called OUTSIDE jax.grad, returns concrete numpy."""
        batch_size = geoms.shape[0]
        assert xs_baselines.ndim == 3, \
            f"xs_baselines must be [batch, n_regions, xs_per_region], got shape {xs_baselines.shape}"
        geoms_flat = jnp.reshape(geoms, (batch_size, 6))
        print(f"the geometry input to the NN is {geoms_flat[0]}!!!!!")
        xs_baselines_safe = jnp.where(xs_baselines > 1e-10, xs_baselines, jnp.ones_like(xs_baselines))
        xs_baselines_log = jnp.log(xs_baselines_safe)  # zeros become log(1)=0, not -inf
        xs_log_ratios = self.generator(geoms_flat, xs_baselines_log, training=False)
        
        # Same clipping as __call__
        mult = jnp.exp(xs_log_ratios)
        if epoch is not None and epoch < 3:
            mult = jnp.clip(mult, 0.8, 1.2)
        elif epoch is not None and epoch < 15:
            mult = jnp.clip(mult, 0.5, 2.0)
        else:
            mult = jnp.clip(mult, 0.3, 3.0)
        
        xs_final = mult * xs_baselines
        
        return np.array(xs_final)   # ← concrete numpy, exits JAX world

    def __call__(self, geoms: jnp.ndarray, params_raw, xs_baselines, epoch, training: bool = False, sample_id_offset=0):
        """
        Args:
            geoms: [batch,6] binary pore geometry (flattened inside)

        Returns:
            keff:               [batch]    effective thermal conductivity
            conductivity_field:  [batch, N, N]  intermediate field (for visualisation)
        """
        batch_size = geoms.shape[0]
        geoms_flat = jnp.reshape(geoms, (batch_size, 6))  # flatten spatial dims

        # NN outputs log-ratios relative to XS_SCALE (the fixed default-geometry reference).
        # exp(0) * XS_SCALE = XS_SCALE at init; the NN freely learns any multiplier during training.
        with timer("forward: NN generated XS", verbose=False):
            xs_baselines_safe = jnp.where(xs_baselines > 1e-10, xs_baselines, jnp.ones_like(xs_baselines))
            xs_baselines_log = jnp.log(xs_baselines_safe)  # zeros become log(1)=0, not -inf

            xs_log_ratios = self.generator(geoms_flat, xs_baselines_log, training)       # [batch, 3, 12], values near 0 at init
            warmup_scale = jnp.clip(epoch / 10.0, 0.0, 1.0) if epoch is not None else 1.0
            xs_log_ratios = xs_log_ratios * warmup_scale
        
        #xs_log_ratios_clamped = jnp.tanh(xs_log_ratios) * 3.0  # restricts to (-2, +2) → factors of (0.14x, 7.4x)
        #xs_final = jnp.exp(xs_log_ratios_clamped) * xs_baselines
        
        #xs_final = jnp.exp(xs_log_ratios) * xs_baselines              # [batch, 3, 12], always positive
        mult = jnp.exp(xs_log_ratios)           # [b,3,12]
        if epoch is not None:
            if epoch < 3:
                mult = jnp.clip(mult, 0.8, 1.2)   # very gentle at start
            elif epoch < 15:
                mult = jnp.clip(mult, 0.5, 2.0)   # as now
            else:
                mult = jnp.clip(mult, 0.3, 3.0)   # allow stronger moves late
        else:
            mult = jnp.clip(mult, 0.5, 2.0)
        xs_final = mult * xs_baselines
        #print("RUNNING a gradients physics check ")
        #physics_sanity_check(xs_baselines[0])        
        #print(f" the baseline starting point XS tensor was {xs_baselines[0]}")
        #print(f"and the final corrected one is {xs_final[0]}")

        with timer("forward: solver loop (all samples)", verbose=False):
            keffs = [] # runs a calc for each of the 10 and then stores
            for i in range(batch_size):
                """ if i == 2:  # print the XS for the first sample in the batch as a sanity check
                    print(f"\nthe LOG RATIOS computed by the NN for this geom are {xs_log_ratios[i]}")
                    print(f"\n AND THE XS PREDICTIONNNNNNNNN for this geom is {xs_final[i]}")
                    print(f"the baseline starting point XS tensor was {xs_baselines[i]}")
                    #print(f"for the id number {i}, the offset is {sample_id_offset}") """

                keff_i = NTdiff_solver(xs_final[i], 
                    jnp.array(params_raw[i], dtype=jnp.float32),       # (6,)   raw geometry
                    jnp.array([i + sample_id_offset], dtype=jnp.int32), )         # (1,)   sample ID)   # pass (3, 12) slice
                keffs.append(keff_i)
            keffs = jnp.stack(keffs)   # [batch]

        return keffs, xs_final, xs_log_ratios


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
    
    n_total = len(geoms)
    # Sort all samples by keff so we can bin them
    sorted_idx = np.argsort(keffs)
    # Divide into n_bins equal-sized bins by keff rank
    # Then sample test_per_bin samples from each bin
    n_bins = 10
    test_per_bin = max(1, test_size // n_bins)
    
    test_idx  = []
    train_idx = []
    
    bins = np.array_split(sorted_idx, n_bins)
    for bin_indices in bins:
        rng.shuffle(bin_indices)
        # Take test_per_bin from this bin for test, rest available for training
        n_test_this_bin = min(test_per_bin, len(bin_indices) - 1)  # keep at least 1 for train
        test_idx.extend(bin_indices[:n_test_this_bin].tolist())
        train_idx.extend(bin_indices[n_test_this_bin:].tolist())
    
    # Trim to requested sizes and shuffle
    test_idx  = np.array(test_idx[:test_size])
    train_idx = np.array(train_idx)
    rng.shuffle(train_idx)
    train_idx = train_idx[:train_size]
    
    # Sanity check: print keff ranges
    print(f"  Train keff range: {keffs[train_idx].min():.3f} - {keffs[train_idx].max():.3f}")
    print(f"  Test  keff range: {keffs[test_idx].min():.3f}  - {keffs[test_idx].max():.3f}")
    print(f"  Train keff mean:  {keffs[train_idx].mean():.3f}  std: {keffs[train_idx].std():.3f}")
    print(f"  Test  keff mean:  {keffs[test_idx].mean():.3f}   std: {keffs[test_idx].std():.3f}")
 
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
    print(f"Data loaded: {train_geoms.shape[0]} training samples, {test_geoms.shape[0]} test samples.")
    # Print dataset statistics for sanity check
    print("GEOMS STATSSSSSSSSSSS:")
    print(f"  mean: {train_geoms.mean(0)}")
    print(f"  std:  {train_geoms.std(0)}")
    print(f"  min:  {train_geoms.min(0)}")
    print(f"  max:  {train_geoms.max(0)}")

    print(f"  Train: {train_geoms.shape}  Test: {test_geoms.shape}")
    print(" THE NEW PRINTINGGGGGGGGG")
    print(f"Train keff range: {train_keffs.min():.3f} – {train_keffs.max():.3f}")
    print(f"Test  keff range: {test_keffs.min():.3f}  – {test_keffs.max():.3f}")

    global N_FLAT_MAX
    all_rawparams = np.concatenate([train_rawparams, test_rawparams], axis=0)
    N_FLAT_MAX = _compute_n_flat_max(all_rawparams)
    print(f"N_FLAT_MAX = {N_FLAT_MAX}  (XLA will compile one kernel variant)")
    
    # ── 5.2  Model & Optimiser ─────────────────────────────────────────────
    # nnx.Rngs(seed) creates a named-key container to get a fresh, unique PRNGKey.
    rngs  = nnx.Rngs(seed)
    model = PEDSModel(hidden_sizes=hidden_sizes, n_regions=n_regions, G=G, rngs=rngs)

    # Cosine decay schedule: learning rate anneals smoothly from lr_max → lr_min.
    # `alpha` is the ratio min/max, so the schedule bottoms out at lr_min.
    lr_schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(0.0, lr_max, transition_steps=15),
            optax.cosine_decay_schedule(init_value=lr_max, decay_steps=epochs * (train_size // batch_size)- 15, alpha=lr_min / lr_max),
        ],
        boundaries=[15]
    )

    #optimizer = nnx.Optimizer(model, optax.adam(lr_schedule), wrt=nnx.Param)     # nnx.Optimizer couples the model's parameters with an Optax update rule.
    optimizer = nnx.Optimizer(model,
        optax.chain(
            optax.clip_by_global_norm(0.1),   # ← clips any gradient explosion
            optax.adam(lr_schedule)
        ),
        wrt=nnx.Param
    )
    # ── 5.3  Loss / gradient function ─────────────────────────────────────
    def loss_fn(model, geoms, keffs_true, rawparams_batch, xs_baselines_batch, batch_weights, epoch):
        # model is passed explicitly so nnx.value_and_grad knows which pytree to differentiate with respect to.
        with timer("loss_fn forward pass", verbose=False):
            keff_pred, xs_final, xs_log_ratios = model(geoms, rawparams_batch, xs_baselines_batch, epoch, training=True) # keff predicted by PEDS model
            print(f"the prediction gives a keff of {keff_pred} (size {keff_pred.size}), \n compared to {keffs_true} (size {keffs_true.size})")
        
        sq_errors = (keff_pred - keffs_true) ** 2      # [batch]
        loss_core = jnp.mean(sq_errors)               # simple MSE in k

        # optional tiny regularization:
        lambda_reg = 1e-3
        #reg = lambda_reg * jnp.mean(xs_log_ratios ** 2)
        xs_log_ratios_clamped = jnp.tanh(xs_log_ratios) * 3.0  # matches your call() method
        reg = jnp.mean(xs_log_ratios_clamped ** 2) * lambda_reg
        loss_value = loss_core + reg
        return loss_value, keff_pred
    
    # nnx.value_and_grad is the Flax-nnx analogue of jax.value_and_grad.
    # It differentiates `loss_fn` with respect to its FIRST argument (the model),
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)  # returning (loss_value, grad_pytree_of_model_params)

    # ── 5.4  Validation helper ─────────────────────────────────────────────
    def validation_step(geoms_np, keffs_np, rawparams_np):
        """Compute mean squared loss and mean percentage error over the test set.
            Also returns per-sample keff_pred and keff_ref arrays for logging."""

        total_sq  = 0.0
        total_pct = 0.0
        all_keff_pred = []
        all_keff_ref  = []
        for batch_idx, (batch_geoms, batch_keffs, batch_rawparams, batch_xs_baselines) in enumerate(data_loader(geoms_np, keffs_np, rawparams_np, test_xs_baselines, batch_size=batch_size)):
            # the batch index is always 0 because the test set is smaller than the batch size, so we only have one batch containing the whole test set.
            global_start = batch_idx * batch_size
            batch_indices = jnp.arange(global_start, global_start + len(batch_geoms))
            batch_weights = sample_weights_jax[batch_indices]
            batch_geoms_jax = jnp.array(batch_geoms)

            # ── Pre-solve in parallel, same as training ────────────────────
            xs_np = model.compute_xs(batch_geoms_jax, jnp.array(batch_xs_baselines), epoch=epoch)
            args_list = [
                (i, xs_np[i], np.array(batch_rawparams[i]), i + global_start, -1)
                for i in range(len(batch_geoms))
            ]
            results = list(executor.map(_solve_sample_worker, args_list))

            _PRESOLVE_CACHE.clear()
            for i, k, phi_fwd, phi_adj, geo_data, _ in results:
                _PRESOLVE_CACHE[i + global_start] = (k, phi_fwd, phi_adj, geo_data)
            # ──────────────────────────────────────────────────────────────

            keff_pred, _, xs_log_ratios = model(batch_geoms_jax, batch_rawparams, 
            xs_baselines=batch_xs_baselines, epoch=epoch, training=False, sample_id_offset=global_start)  
            sq_err  = jnp.mean((keff_pred - batch_keffs) ** 2) # squared error
            pct_err = jnp.mean(jnp.abs(keff_pred - batch_keffs) / jnp.abs(batch_keffs) * 100.0) # percent error
            total_sq  += float(sq_err)
            total_pct += float(pct_err)
            all_keff_pred.extend(np.array(keff_pred).tolist())   # ← collect predictions
            all_keff_ref.extend(np.array(batch_keffs).tolist())  # ← collect references            
        n = len(keffs_np)
        return total_sq / n, total_pct / n, all_keff_pred, all_keff_ref   
        # mean over all test samples

    sample_weights = np.ones(train_size, dtype=np.float32)  # uniform initially
    sample_weights_jax = jnp.array(sample_weights)

    print("Precomputing XS baselines for all samples (done once)...")
    train_xs_baselines = compute_batch_baselines(train_rawparams, GEO)  # [train_size, 3, 12]
    test_xs_baselines  = compute_batch_baselines(test_rawparams, GEO)   # [test_size,  3, 12]
    print("Done.")

    # ── 5.5  Epoch loop ────────────────────────────────────────────────────
    # ── 5.5  EPOCH LOOOOOOOP ───────────────────────────────────────────────
    # ── 5.5  Epoch loop ────────────────────────────────────────────────────

    train_losses   = []
    val_losses     = []
    val_pct_errors = []
    iterations = 0 
    print(f"\nTraining for {epochs} epochs …")

        # ── Open keff log file ────────────────────────────────────────────────

    count = 0

    for epoch in range(epochs):
        count += 1
        print(f"entering the epoch loop, at epoch {epoch} which is")
        epoch_loss = 0.0
        
        shuffle_idx = np.random.permutation(train_size)
        train_geoms_shuffled    = train_geoms[shuffle_idx]
        train_keffs_shuffled    = train_keffs[shuffle_idx]
        train_rawparams_shuffled = train_rawparams[shuffle_idx]
        train_xs_baselines_shuffled = np.array(train_xs_baselines)[shuffle_idx]

        # these are the parameters used for the training loop
        for batch_idx, (batch_geoms, batch_keffs, batch_rawparams_tr, batch_baselines) in enumerate(data_loader(
            train_geoms_shuffled, train_keffs_shuffled, train_rawparams_shuffled, train_xs_baselines_shuffled, batch_size=batch_size)):
            
            bp = jnp.array(batch_geoms) # Convert NumPy → JAX arrays once per batch 
            bk = jnp.array(batch_keffs)
            bxs = jnp.array(batch_baselines)  
            # Step 1: get concrete xs from NN (outside trace)
            xs_np = model.compute_xs(bp, bxs, epoch=epoch)   # concrete numpy [batch, 3, 12]
                # Step 0: build batch_weights from global indices
            global_start   = batch_idx * batch_size
            batch_len      = len(batch_geoms)
            batch_indices  = jnp.arange(global_start, global_start + batch_len)
            batch_weights  = sample_weights_jax[batch_indices]   # shape [batch_len]

            # Step 2: parallel solver calls — populates _GEO_DATA_CACHE
            args_list = [
                (i, xs_np[i], np.array(batch_rawparams_tr[i]), i, epoch)
                for i in range(len(batch_geoms))
            ]
            results = list(executor.map(_solve_sample_worker, args_list))
            _PRESOLVE_CACHE.clear()   # ← clear stale entries from previous batch/epoch
            for i, k, phi_fwd, phi_adj, geo_data, result_epoch in results:
                _PRESOLVE_CACHE[i] = (k, phi_fwd, phi_adj, geo_data)

            # Step 3: grad_fn traces normally — pure_callback reads from cache
            with timer("train step: forward+backward+optiupd"):
                with timer("train step: forward+loss+grad"):    
                    (loss, keff_preds_batch), grads = grad_fn(model, bp, 
                                bk, batch_rawparams_tr, bxs, batch_weights, epoch)

            with timer("train step: optimizer update"):            
                optimizer.update(model, grads)
            
            log_keff_batch(
                train_writer, train_logfile, epoch,
                np.array(keff_preds_batch), np.array(bk),
                float(loss), 0.0,
                sample_id_offset=batch_idx * batch_size
            )
            epoch_loss += float(loss)

            print("Running forward pass + AD \n ")
            
            # still inside the epoch loop 
            # ── Collect heatmap snapshot every HEATMAP_INTERVAL epochs ────────
            if (epoch + 1) % HEATMAP_INTERVAL == 0 or epoch == 0:
                _snap_geo_i       = update_geo(GEO, train_rawparams[0])          # single GeometryConfig
                _snap_baseline    = np.array(predict_xs(_snap_geo_i))            # (3, 12) guaranteed
                _snap_baselines_log = np.log(np.where(_snap_baseline > 1e-10, _snap_baseline, np.ones_like(_snap_baseline)))  # log of baseline, with safe fallback
                _snap_baselines_log_batched = _snap_baselines_log[np.newaxis, :, :]  # (1, 3, 12) ← add batch dim

                _snap_geom_flat   = jnp.array(train_geoms[0:1])                  # (1, 6)
                _snap_log_ratios = np.array(
                    model.generator(_snap_geom_flat, _snap_baselines_log_batched, training=False)[0]  # (3, 12), log-ratios
                )
                _snap_final = np.exp(_snap_log_ratios) * np.array(XS_SCALE)  # (3, 12), absolute XS

                XS_HEATMAP_SNAPSHOTS.append(_snap_log_ratios.copy())  # store log-ratios for heatmap
                XS_HEATMAP_LABELS.append(f"Epoch {epoch + 1}")
                
                if _xs_baseline_ref[0] is None:
                    _xs_baseline_ref[0] = _snap_baseline

        # Normalise by dataset size (matches `avg_loss` in training.py)
        avg_train_loss = epoch_loss / train_size
        # Validation (no gradient tracking needed)
        with timer("validation step"):
            avg_val_loss, avg_pct_err, keff_preds, keff_refs = validation_step(test_geoms, test_keffs, test_rawparams)

        # ── Write per-sample keff values to the log file ─────────────────────
        log_keff_batch(val_writer, val_logfile, epoch, keff_preds, keff_refs, avg_train_loss, avg_val_loss)
   
        print(f"this is batch iteration {iterations + 1} with the average train loss {avg_train_loss:.4f}, the average validation loss {avg_val_loss:.4f} and the average percentage error {avg_pct_err:.4f} \n")
        iterations = iterations+1
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_pct_errors.append(avg_pct_err)

        """ # --- every 5 epochs: update sample_weights based on latest errors ---
        if (epoch + 1) % 5 == 0:
            # Here you would ideally run a training_eval_step over train_geoms/train_keffs
            # For illustration, we show how to use keff_preds / keff_refs if they correspond to training.
            keff_preds_np = np.array(keff_preds)
            keff_refs_np  = np.array(keff_refs)

            # delta-rho in pcm
            delta_rho = np.abs(keff_preds_np - keff_refs_np) / (keff_preds_np * keff_refs_np) * 1e5

            alpha = 0.5  # tune this
            new_weights = 1.0 + alpha * (delta_rho / 1000.0)
            new_weights = np.clip(new_weights, 1.0, 3.0)

            # Make sure new_weights has length train_size and corresponds to training ordering
            sample_weights = new_weights.astype(np.float32)
            sample_weights_jax = jnp.array(sample_weights)

            print(f"Updated sample weights at epoch {epoch+1}: "
                f"min={sample_weights.min():.2f}, max={sample_weights.max():.2f}") """

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
        if epoch % 50 == 0 and epoch > 0:
            jax.clear_caches()
        iterations = 0
    
    final_errors = []
    for i in range(len(keff_refs)):
        k_pred = float(keff_preds[i])   # keff from PEDS at final epoch
        k_ref  = float(keff_refs[i])    # keff from OpenMC
        delta_rho = abs(k_pred - k_ref) / (k_pred * k_ref) * 1e5
        final_errors.append(delta_rho)

    final_errors = np.array(final_errors)

    # final_errors: [50] array of pcm values at last epoch for test set
    param_names = ['b4c_r', 'cr_frac', 'fuel_r', 'enrichment', 'f_mod', 'water_r']
    for j, name in enumerate(param_names):
        corr = np.corrcoef(test_rawparams[:, j], final_errors)[0, 1]
        print(f"  {name:12s}: correlation with error = {corr:+.3f}")
        
    print(f"the total number of iterations is {iterations}.")
    print(f"\n keff log saved to: {val_log_path}")

    # ─────────────────────────────────────────────────────────────
    # check the geometry of the badly converged samples
    # ─────────────────────────────────────────────────────────────
    df = pd.read_csv(val_log_path, header=0)
    # This reads the header properly, columns named automatically
    # Then cast types explicitly:
    df['epoch'] = df['epoch'].astype(int)
    df['delta_rho_pcm'] = df['delta_rho_pcm'].astype(float)
    df['sample_idx'] = df['sample_idx'].astype(int)
    # Get the LAST epoch only (final converged state)
    last_epoch = df['epoch'].max()
    print(f"df shape: {df.shape}")
    print(f"df dtypes:\n{df.dtypes}")
    print(f"df head:\n{df.head()}")
    print(f"last epoch = {last_epoch}, type = {type(last_epoch)}")
    df_final = df[df['epoch'] == last_epoch]
    print(f"df final shape: {df_final.shape}")
    # Sort by worst discrepancy
    df_final_sorted = df_final.sort_values('delta_rho_pcm', ascending=False)
    print("=== TOP 10 WORST SAMPLES AT FINAL EPOCH ===")
    print(df_final_sorted[['sample_idx', 'keff_openmc', 'keff_peds', 'delta_rho_pcm']].head(10).to_string())
    # Extract the bad sample indices
    bad_indices = df_final_sorted['sample_idx'].head(10).values
    print(f"\nBad sample indices: {bad_indices}")

    print("\n=== GEOMETRY OF WORST 10 SAMPLES AT FINAL EPOCH ===")
    feat_names = ['b4c_r', 'cr_frac', 'fuel_r', 'enrich', 'f_mod', 'water_r']
    for idx in bad_indices:
        geom = test_geoms[idx]
        raw  = test_rawparams[idx]
        k_ref = test_keffs[idx]
        delta = float(df_final[df_final['sample_idx']==idx]['delta_rho_pcm'].values[0])
        print(f"\n  Sample {idx} | k_ref={k_ref:.4f} | delta_rho={delta:.1f} pcm")
        for i, name in enumerate(feat_names):
            print(f"    {name}: normalized={geom[i]:.3f}  raw={raw[i]:.4f}")

    # ── Generate XS heatmap ───────────────────────────────────────────
    print(f"[xs_heatmap] baseline ref is None: {_xs_baseline_ref[0] is None}")
    print(f"[xs_heatmap] number of snapshots collected: {len(XS_HEATMAP_SNAPSHOTS)}")
    print(f"[xs_heatmap] snapshot labels: {XS_HEATMAP_LABELS}")
    print(f"CHECKING IF THE PLOT IS RUNNIG WITH {_xs_baseline_ref[0]} as the baseline and {_snap_final} as the final xs to plot")
    if _xs_baseline_ref[0] is not None and len(XS_HEATMAP_SNAPSHOTS) > 0:
        print(f"\nGenerating XS heatmap with {len(XS_HEATMAP_SNAPSHOTS)} snapshots …")
        print(f"saving it to ")
        plot_xs_heatmap(
            baseline           = _xs_baseline_ref[0],
            snapshots          = XS_HEATMAP_SNAPSHOTS,
            epoch_labels       = XS_HEATMAP_LABELS,
            final_xs           = _snap_final,
            G                  = G,
            save_path          = "./LOGS/heatmaps/xs_heatmap.png",
            plot_interval_note = f"snapshot every {HEATMAP_INTERVAL} epochs",
        )
        plot_xs_subplots(
            baseline           = _xs_baseline_ref[0],
            final_xs           = _snap_final,
            G                  = G,
            save_path          = "./LOGS/heatmaps/xs_subplots.png",
        )
    else:
        print("[xs_heatmap] skipped — no snapshots collected (epochs < HEATMAP_INTERVAL?)")
    
    loggy_file.close()
    return model, train_losses, val_losses, val_pct_errors



# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # ── Hyperparameters (from config_experiment.py / config_model.py) ─────
    HP = dict(
        filepath      = "../data/highfidelity/MCruns_filtered.npz",  # adjust path as needed
        train_size    = train_size,
        test_size     = test_size,
        batch_size    = batch_size,
        epochs        = epochs,
        lr_max        = 5e-4,   # cosine schedule peak learning rate
        lr_min        = 5e-6,   # cosine schedule floor learning rate
        hidden_sizes  = [128, 128, 64],  # matches model config m1
        n_regions    = 3,    # b4c_rod, fuel_annulus, water
        G            = 2,    # energy groups
        seed          = 42,
    )

    (train_geoms, train_keffs, train_rawparams), _ = load_data(
        HP["filepath"], HP["train_size"], HP["test_size"], seed=42)
    
    # ── checks ─────────────────────────────────────────────────────────
    """ print("Performing a check ....")
    solver_floor_check(
        train_rawparams = train_rawparams,
        train_keffs     = train_keffs,
        GEO             = GEO,
        NTdiff_solver   = NTdiff_solver,
        update_geo      = update_geo,
        predict_xs      = predict_xs,
        n_samples       = 10,
    )

    full_check(
        train_rawparams = train_rawparams,
        XS_BASELINE     = XS_BASELINE,
        GEO             = GEO,
        NTdiff_solver   = NTdiff_solver,
        _NTdiff_fwd     = _NTdiff_fwd,
        _run_NT_solver  = _run_NT_solver,
        _GEO_DATA_CACHE = _GEO_DATA_CACHE,
        SLAY            = SLAY,
        update_geo      = update_geo,
    ) """
    # ── Train ─────────────────────────────────────────────────────────────
    
    print("starting the training process... \n")
    model, train_losses, val_losses, val_pct_errs = train(**HP)
    print_timing_report()
    # ── Reload test set for final visualisation ───────────────────────────
    _, (test_geoms, test_keffs, test_rawparams) = load_data(
        HP["filepath"], HP["train_size"], HP["test_size"], HP["seed"]
    )
    executor.shutdown(wait=True)


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
