"""==========================================================================
PEDS  —  VERSION 2: remove the separation into 3 heads
==========================================================================

=========================================================================="""

import multiprocessing
import os

# ── anchor all paths relative to THIS script file, not the cwd ──────────────
# This means you can run the script from any directory, including from inside
# the output folder, without breaking imports or data paths.
THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
os.environ["XLA_CPU_ENABLE_FAST_MATH"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"
multiprocessing.set_start_method("spawn", force=True)

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import optax
from flax import nnx
import matplotlib.pyplot as plt
import csv
import time
import threading
import resource
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from collections import OrderedDict
import sys

# ── add the project root (parent of THIS_DIR) to sys.path ───────────────────
# Previously this used os.path.dirname(__file__) which breaks when you cd
# into a subdirectory and run the script.  Using THIS_DIR + PARENT_DIR is
# robust regardless of working directory.
sys.path.insert(0, PARENT_DIR)

ctx = mp.get_context("spawn")
_csv_lock = threading.Lock()

soft, hard = resource.getrlimit(resource.RLIMIT_AS)
print(f"Memory limit: soft={soft/1e9:.1f}GB hard={hard/1e9:.1f}GB")

from matrix_JAX_optimized import diffusion_setup_jax, Aphi_Fphi_scan, Aphi_Fphi_vjp
from NTcode_config_data.config_def import (GeometryConfig, MaterialSpec,
    BoundarySpec, BoundaryCondition, MatProperties)
from NTcode_config_data.config_run import GEO_CYL as GEO
from solvers.NTdiffusion.core.MG1D_eigenvalue_nregions import diffusion_setup, region_index
from solvers.NTdiffusion.diffusion_solver import (
    get_xs_basedon_geo, run_diffusion_solver, is_homogeneous,
    predict_xs, precompute_geometry, xs_layout, _plot_fluxes,
    build_xs_callables, fn_xs_per_region, bc_to_coeffs, GEOMETRY_CODE)
from NTcode_destructure.timing_utils import timer, print_timing_report, _TIMINGS
from plot_functions.xs_heatmap import plot_xs_subplots


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0: Global constants
# ─────────────────────────────────────────────────────────────────────────────
EXP_NAME   = "v2_noheads"
TRAIN_SIZE = 400
TEST_SIZE  = 100
BATCH_SIZE = 25
EPOCHS     = 100

LR         = 5e-4        # constant learning rate — no schedule in v1
SEED       = 42

# Epochs at which XS heatmap snapshots are saved.
# Add/remove values here to control checkpointing granularity.
XS_HEATMAP_EPOCHS = {1, 5, 10, 20, 50, 100}

# Sample indices (into the TRAINING set) for per-sample xs_subplots figures.
# These are saved at the same epochs as XS_HEATMAP_EPOCHS.
SUBPLOT_SAMPLE_INDICES: list = [0, 1, 2, 3, 4]

# Human-readable names for the 6 raw geometry parameters (used in info box).
PARAM_NAMES: list = ['b4c_r', 'cr_frac', 'fuel_r', 'enrichment', 'f_mod', 'water_r']

# ── data path — anchored to PARENT_DIR so it works from any cwd ─────────────
_DATA_FILEPATH = os.path.join(PARENT_DIR, "data", "highfidelity", "MCruns_filtered.npz")

N_WORKERS = min(int(os.environ.get("SLURM_CPUS_PER_TASK", 16)), BATCH_SIZE)
print(f"Using {N_WORKERS} parallel workers")
executor = ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=ctx)

# ── output directory: everything for this run lives under LOG_DIR ─────────────
# Whether you run from THIS_DIR or from the parent, files always land here.
LOG_DIR = os.path.join(THIS_DIR, "LOGS", EXP_NAME)
os.makedirs(LOG_DIR, exist_ok=True)

val_log_path    = os.path.join(LOG_DIR, "keff_epoch_log_val.csv")
train_log_path  = os.path.join(LOG_DIR, "keff_epoch_log_train.csv")
# NEW: one row per epoch with all aggregate metrics
epoch_stats_path = os.path.join(LOG_DIR, "epoch_metrics.csv")

loggy_file = open(os.path.join(LOG_DIR, f"train_log_{EXP_NAME}.txt"), "w", buffering=1)
sys.stdout = loggy_file
sys.stderr = loggy_file
print("JAX devices:", jax.devices())

# ── precomputed geometry constants ───────────────────────────────────────────
GEO_DATA = precompute_geometry(GEO)
SLAY     = xs_layout(GEO.G)


# ── max flat flux vector size (needed for padded pure_callback) ──────────────
def _compute_n_flat_max_from_file(filepath: str, geo: GeometryConfig) -> int:
    data = np.load(filepath, allow_pickle=True)
    rawparams = np.array(data['params_raw'], dtype=np.float32)
    n_flat_max = 0
    for i in range(len(rawparams)):
        geo_i  = update_geo(geo, rawparams[i])
        R      = geo_i.boundaries[-1].radius
        I      = int(R / geo_i.mesh_size)
        n_flat_max = max(n_flat_max, geo_i.G * (I + 1))
    return n_flat_max


class LRUCache(OrderedDict):
    def __init__(self, maxsize):
        super().__init__()
        self.maxsize = maxsize
    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if len(self) > self.maxsize:
            self.popitem(last=False)

_GEO_DATA_CACHE  = LRUCache(maxsize=50)
_PRESOLVE_CACHE: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────
def update_geo(geo: GeometryConfig, params_raw: np.ndarray) -> GeometryConfig:
    """Rebuild a GeometryConfig from a [6] raw-parameter vector."""
    new_boundaries = (
        BoundarySpec(name='CR_outer',        radius=float(params_raw[0])),
        BoundarySpec(name='core_outer',      radius=float(params_raw[2])),
        BoundarySpec(name='moderator_outer', radius=float(params_raw[5])),
    )
    new_mat = MatProperties(
        cr_fraction=float(params_raw[1]),
        enrichment =float(params_raw[3]),
        f_mod      =float(params_raw[4]),
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


# Defer N_FLAT_MAX until after update_geo is defined
N_FLAT_MAX = _compute_n_flat_max_from_file(_DATA_FILEPATH, GEO)
print(f"N_FLAT_MAX = {N_FLAT_MAX}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Physics solver + custom VJP  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────
def _run_NT_solver(xs_tensor, params_raw_single, sample_id):
    """NumPy/SciPy forward solve. Returns (k, phi_fwd_padded, phi_adj_padded, Fphi_padded)."""
    geo_i    = update_geo(GEO, np.array(params_raw_single))
    geo_data = precompute_geometry(geo_i)
    sid      = int(sample_id[0])
    _GEO_DATA_CACHE[sid] = geo_data

    # ── use cached solve if available ────────────────────────────────────────
    if sid in _PRESOLVE_CACHE:
        k, phi_fwd_padded, phi_adj_padded, geo_data_pre = _PRESOLVE_CACHE.pop(sid)
        _GEO_DATA_CACHE[sid] = geo_data_pre
        R      = geo_i.boundaries[-1].radius
        I      = int(R / geo_i.mesh_size)
        N_flat = geo_i.G * (I + 1)
        r_div  = [b.radius for b in geo_i.boundaries[:-1]] if not is_homogeneous(geo_i) else []
        BC     = bc_to_coeffs(geo_i.bc)
        gcode  = GEOMETRY_CODE[geo_i.geometry]
        D_fn, Sa_fn, nF_fn, Ss_fn, chi_fn = build_xs_callables(np.array(xs_tensor), geo_i)
        _, _, F_real = diffusion_setup(R, I, geo_i.G, r_div, D_fn, Sa_fn, nF_fn, Ss_fn, chi_fn, BC, gcode)
        Fphi_full = F_real @ phi_fwd_padded[:N_flat].astype(np.float64)
        Fphi_padded = np.zeros(N_FLAT_MAX, dtype=np.float32)
        Fphi_padded[:N_flat] = Fphi_full
        return np.float32(k), phi_fwd_padded, phi_adj_padded, Fphi_padded

    # ── full eigenvalue solve ─────────────────────────────────────────────────
    R      = geo_i.boundaries[-1].radius
    I      = int(R / geo_i.mesh_size)
    N_flat = geo_i.G * (I + 1)

    with timer("  solver: eigenvalue solve (fwd)", verbose=False):
        k, phi_fwd, phi_adj = run_diffusion_solver(xs_tensor, geo_i)

    phi_fwd_flat = np.zeros(N_flat, dtype=np.float64)
    phi_adj_flat = np.zeros(N_flat, dtype=np.float64)
    for g in range(geo_i.G):
        phi_fwd_flat[g*(I+1): g*(I+1)+I] = phi_fwd[g, :]
        phi_adj_flat[g*(I+1): g*(I+1)+I] = phi_adj[g, :]

    r_div  = [b.radius for b in geo_i.boundaries[:-1]] if not is_homogeneous(geo_i) else []
    BC     = bc_to_coeffs(geo_i.bc)
    gcode  = GEOMETRY_CODE[geo_i.geometry]
    D_fn, Sa_fn, nF_fn, Ss_fn, chi_fn = build_xs_callables(np.array(xs_tensor), geo_i)
    _, A_real, F_real = diffusion_setup(R, I, geo_i.G, r_div, D_fn, Sa_fn, nF_fn, Ss_fn, chi_fn, BC, gcode)

    Fphi     = F_real @ phi_fwd_flat
    biorth   = phi_adj_flat @ Fphi
    phi_adj_flat *= 1.0 / biorth   # enforce ⟨φ†, Fφ⟩ = 1
    #print(f"bwd sanity: phiadj·F·phi = {float(phi_adj_flat @ Fphi):.6f}  (should be ~1.0)")


    phi_fwd_padded = np.zeros(N_FLAT_MAX, dtype=np.float32)
    phi_adj_padded = np.zeros(N_FLAT_MAX, dtype=np.float32)
    Fphi_padded    = np.zeros(N_FLAT_MAX, dtype=np.float32)
    phi_fwd_padded[:N_flat] = phi_fwd_flat[:N_flat]
    phi_adj_padded[:N_flat] = phi_adj_flat[:N_flat]
    Fphi_padded[:N_flat]    = (F_real @ phi_fwd_flat)[:N_flat]

    return np.float32(k), phi_fwd_padded, phi_adj_padded, Fphi_padded


def _NTdiff_fwd(xs_tensor, params_raw_single, sample_id):
    geo_i  = update_geo(GEO, np.array(params_raw_single))
    R      = geo_i.boundaries[-1].radius
    I      = int(R / geo_i.mesh_size)

    keff, phi_fwd_padded, phi_adj_padded, Fphi_padded = jax.pure_callback(
        _run_NT_solver,
        (
            jax.ShapeDtypeStruct((),             jnp.float32),
            jax.ShapeDtypeStruct((N_FLAT_MAX,),  jnp.float32),
            jax.ShapeDtypeStruct((N_FLAT_MAX,),  jnp.float32),
            jax.ShapeDtypeStruct((N_FLAT_MAX,),  jnp.float32),
        ),
        xs_tensor, params_raw_single, sample_id,
    )
    residuals = (xs_tensor, keff, phi_fwd_padded, phi_adj_padded, Fphi_padded,
                 params_raw_single, sample_id)
    return keff, residuals


def _NTdiff_bwd(residuals, g):
    xs_tensor, k, phi_fwd_padded, phi_adj_padded, Fphi_padded, params_raw_single, sample_id = residuals
    geo_i  = update_geo(GEO, np.array(params_raw_single))
    R      = geo_i.boundaries[-1].radius
    I      = int(R / geo_i.mesh_size)
    N_flat = int(geo_i.G * (I + 1))

    phi_fwd  = phi_fwd_padded[:N_flat]
    phi_adj  = phi_adj_padded[:N_flat]
    Fphi     = Fphi_padded[:N_flat]
    geo_data = _GEO_DATA_CACHE[int(sample_id[0])]

    v_A = phi_adj
    v_F = -(1.0 / k) * phi_adj
    with timer("  bwd: vjp A@phi, F@phi -> numerator", verbose=False):
        numerator   = Aphi_Fphi_vjp(xs_tensor, geo_data, SLAY, phi_fwd, v_A, v_F)
    denominator = (1.0 / k**2) * (phi_adj @ Fphi)
    dk_dp       = -numerator / denominator
    return (g * dk_dp, None, None)


@jax.custom_vjp
def NTdiff_solver(xs_tensor, params_raw_single, sample_id):
    keff, _ = _NTdiff_fwd(xs_tensor, params_raw_single, sample_id)
    return keff

NTdiff_solver.defvjp(_NTdiff_fwd, _NTdiff_bwd)


def _solve_sample_worker(args):
    """Runs in a subprocess. Returns (i, k, phi_fwd, phi_adj, geo_data, epoch)."""
    import os
    os.environ["JAX_PLATFORMS"] = "cpu"
    i, xs_np, params_np, sample_id_int, epoch = args
    geo_i    = update_geo(GEO, params_np)
    geo_data = precompute_geometry(geo_i)
    k, phi_fwd, phi_adj, Fphi = _run_NT_solver(xs_np, params_np, np.array([sample_id_int]))
    return i, k, phi_fwd, phi_adj, geo_data, epoch


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Neural Network
# ─────────────────────────────────────────────────────────────────────────────
class GeneratorNN(nnx.Module):
    """
    Trunk:  [batch, 7]  →  128 → 128 → 64  (ReLU + skip)
    Heads:  3 × [trunk_64 + log_baseline_12]  →  12  (zero-init)

    Input vector (dim 7):
        geoms[0:6]  — 6 normalised geometry parameters
        delta_k_norm[0]  — normalised (k_openmc - k_reg)

    Output: [batch, 3, 12]  log-ratio offsets δ_ℓ
            XS_final = exp(δ_ℓ) × XS_baseline
    """

    def __init__(self, layer_sizes: list, n_regions: int,
                 xs_per_region: int, rngs: nnx.Rngs):
        super().__init__()
        he_init   = nnx.initializers.kaiming_normal()
        zero_init = nnx.initializers.zeros

        # Trunk layers — He init, all ReLU in __call__
        self.layers = nnx.List([
            nnx.Linear(
                in_features  = layer_sizes[i],
                out_features = layer_sizes[i + 1],
                kernel_init  = he_init,
                bias_init    = nnx.initializers.constant(0.0),
                rngs         = rngs,
            )
            for i in range(len(layer_sizes) - 1)
        ])

        # Skip connection: input dim → trunk output dim
        self.skip_proj = nnx.Linear(layer_sizes[0], layer_sizes[-1], rngs=rngs)

        # Region heads — zero-init so δ_ℓ = 0 at epoch 0 (identity correction)
        trunk_out = layer_sizes[-1]
        total_xs       = n_regions * xs_per_region  # 3 * 12 = 36

        # Single head: trunk output + all log-baselines flattened → all XS corrections
        self.head = nnx.Linear(
            in_features  = trunk_out + total_xs,   # 64 + 36 = 100
            out_features = total_xs,                # 36
            kernel_init  = zero_init,
            bias_init    = zero_init,
            rngs         = rngs,
        )

        self.n_regions     = n_regions
        self.xs_per_region = xs_per_region

    def __call__(self, geoms: jnp.ndarray, xs_baselines_log: jnp.ndarray,
                 delta_k_norm: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """
        geoms:            [batch, 6]
        xs_baselines_log: [batch, n_regions, xs_per_region]
        delta_k_norm:     [batch, 1]
        Returns:          [batch, n_regions, xs_per_region]  log-ratio offsets
        """
        x    = jnp.concatenate([geoms, delta_k_norm], axis=-1)  # [batch, 7]
        skip = nnx.relu(self.skip_proj(x))
        for layer in self.layers:
            x = nnx.relu(layer(x))
        x = x + skip  # [batch, 64]

        log_base_flat = jnp.reshape(xs_baselines_log, (geoms.shape[0], -1))  # [batch, 36]
        feat = jnp.concatenate([x, log_base_flat], axis=-1)                  # [batch, 100]
        out  = self.head(feat)                                                # [batch, 36]
        return jnp.reshape(out, (geoms.shape[0], self.n_regions, self.xs_per_region))


class PEDSModel(nnx.Module):
    """GeneratorNN + physics solver, end-to-end differentiable."""

    def __init__(self, hidden_sizes: list, n_regions: int, G: int, rngs: nnx.Rngs):
        super().__init__()
        xs_region  = fn_xs_per_region(G)
        layer_sizes = [7] + hidden_sizes
        self.generator   = GeneratorNN(layer_sizes, n_regions, xs_region, rngs)
        self.n_regions   = n_regions
        self.G           = G

    def _log_baselines(self, xs_baselines: jnp.ndarray) -> jnp.ndarray:
        """Safe log: zeros → log(1) = 0 instead of -inf."""
        safe = jnp.where(xs_baselines > 1e-10, xs_baselines, jnp.ones_like(xs_baselines))
        return jnp.log(safe)

    def compute_xs(self, geoms, xs_baselines, delta_k_norm):
        """Pure NN forward (no grad). Used for pre-solving and validation."""
        batch_size = geoms.shape[0]
        dk  = jnp.reshape(jnp.array(delta_k_norm, dtype=jnp.float32), (batch_size, 1))
        with timer("log baselines from xs first guess", verbose=False): 
            log_base    = self._log_baselines(xs_baselines)
        with timer("NN generated XS log ratios", verbose=False):
            log_ratios  = self.generator(geoms, log_base, dk, training=False)
        # ── v1: NO clip, NO warmup ─────────────────────────────────────────
        xs_final = jnp.exp(log_ratios) * xs_baselines
        return np.array(xs_final)  # concrete numpy, exits JAX world

    def __call__(self, geoms, params_raw, xs_baselines, delta_k_norm,
                 training: bool = False, sample_id_offset: int = 0):
        batch_size = geoms.shape[0]
        dk  = jnp.reshape(delta_k_norm, (batch_size, 1))
        log_base   = self._log_baselines(xs_baselines)
        log_ratios = self.generator(geoms, log_base, dk, training)
        # ── v1: NO clip, NO warmup ─────────────────────────────────────────
        xs_final = jnp.exp(log_ratios) * xs_baselines  # [batch, 3, 12]

        with timer("forward: solver loop (all samples)", verbose=False):
            keffs = []
            for i in range(batch_size):
                keff_i = NTdiff_solver(
                    xs_final[i],
                    jnp.array(params_raw[i], dtype=jnp.float32),
                    jnp.array([i + sample_id_offset], dtype=jnp.int32),
                )
                keffs.append(keff_i)
        keffs = jnp.stack(keffs)  # [batch]
        return keffs, xs_final, log_ratios


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Data loading
# ─────────────────────────────────────────────────────────────────────────────
def data_loader(*arrays, batch_size: int):
    n = arrays[0].shape[0]
    for start in range(0, n, batch_size):
        yield tuple(arr[start:start + batch_size] for arr in arrays)


def compute_batch_baselines(params_raw: np.ndarray, geo: GeometryConfig) -> jnp.ndarray:
    """Polynomial-regression XS for every sample. Returns [N, 3, 12]."""
    return jnp.array(
        np.stack([predict_xs(update_geo(geo, params_raw[i])) for i in range(len(params_raw))]),
        dtype=jnp.float32,
    )


def compute_k_reg_batch(rawparams: np.ndarray) -> np.ndarray:
    """Run baseline solver (no NN) for every sample. Returns k_reg [N]."""
    k_regs = []
    for i, p in enumerate(rawparams):
        geo_i = update_geo(GEO, p)
        xs_i  = np.array(predict_xs(geo_i), dtype=np.float32)
        k, _, _, _ = _run_NT_solver(xs_i, p, np.array([i], dtype=np.int32))
        k_regs.append(float(k))
        if i % 20 == 0:
            print(f"  k_reg precompute {i}/{len(rawparams)}", flush=True)
    return np.array(k_regs, dtype=np.float32)


def load_data(filepath: str, train_size: int, test_size: int, seed: int = 42):
    """Stratified split by k-eff range."""
    data      = np.load(filepath, allow_pickle=True)
    geoms     = np.array(data['params'],     dtype=np.float32)
    keffs     = np.array(data['keffs'],      dtype=np.float32)
    rawparams = np.array(data['params_raw'], dtype=np.float32)

    rng        = np.random.default_rng(seed)
    sorted_idx = np.argsort(keffs)
    n_bins     = 10
    test_per_bin = max(1, test_size // n_bins)

    test_idx, train_idx = [], []
    for bin_indices in np.array_split(sorted_idx, n_bins):
        rng.shuffle(bin_indices)
        n_t = min(test_per_bin, len(bin_indices) - 1)
        test_idx.extend(bin_indices[:n_t].tolist())
        train_idx.extend(bin_indices[n_t:].tolist())

    test_idx  = np.array(test_idx[:test_size])
    train_idx = np.array(train_idx)
    rng.shuffle(train_idx)
    train_idx = train_idx[:train_size]

    print(f"  Train k range: {keffs[train_idx].min():.3f} – {keffs[train_idx].max():.3f}")
    print(f"  Test  k range: {keffs[test_idx].min():.3f}  – {keffs[test_idx].max():.3f}")
    print(f"  Train keff mean:  {keffs[train_idx].mean():.3f}  std: {keffs[train_idx].std():.3f}")
    print(f"  Test  keff mean:  {keffs[test_idx].mean():.3f}   std: {keffs[test_idx].std():.3f}")

    print("Precomputing k_reg (baseline solver, done once)…")
    k_reg_train = compute_k_reg_batch(rawparams[train_idx])
    k_reg_test  = compute_k_reg_batch(rawparams[test_idx])

    return (
        (geoms[train_idx], keffs[train_idx], rawparams[train_idx], k_reg_train),
        (geoms[test_idx],  keffs[test_idx],  rawparams[test_idx],  k_reg_test),
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: Metrics
# ─────────────────────────────────────────────────────────────────────────────
def compute_delta_rho_pcm(k_pred: np.ndarray, k_ref: np.ndarray) -> np.ndarray:
    """Reactivity error |Δρ| in pcm for each sample."""
    return np.abs(k_pred - k_ref) / (k_pred * k_ref) * 1e5


def compute_metrics(k_pred: np.ndarray, k_ref: np.ndarray) -> dict:
    """
    Returns a dict of scalar metrics for one epoch.

    Keys
    ----
    mse_k          : mean squared error in k units²
    MAE_k          : mean absolute error in k units
    mean_pcm       : mean |Δρ|  (pcm)  ← headline physics metric
    median_pcm     : median |Δρ| (pcm) ← robust to outliers
    p95_pcm        : 95th-percentile |Δρ| (pcm) ← worst-case tail
    std_pcm        : std dev of |Δρ|  (pcm) ← spread / consistency
    frac_below_650 : fraction of samples with |Δρ| < 650 pcm (β_eff threshold)
    frac_below_100 : fraction of samples with |Δρ| < 100 pcm (typical regulatory limit)
    """
    dr = compute_delta_rho_pcm(k_pred, k_ref)
    return dict(
        mse_k          = float(np.mean((k_pred - k_ref) ** 2)),
        MAE_k          = float(np.mean(np.abs(k_pred - k_ref))),
        mean_pcm       = float(np.mean(dr)),
        median_pcm     = float(np.median(dr)),
        p95_pcm        = float(np.percentile(dr, 95)),
        std_pcm        = float(np.std(dr)),
        frac_below_650 = float(np.mean(dr < 650.0)),
        frac_below_100 = float(np.mean(dr < 100.0)),
    )


def print_metrics(epoch: int, tag: str, m: dict):
    print(
        f"[Epoch {epoch:4d}] {tag:5s} | "
        f"MSE={m['mse_k']:.6f} | "
        f"MAE_k={m['MAE_k']:7.1f} | "
        f"mean={m['mean_pcm']:7.1f} pcm | "
        f"median={m['median_pcm']:7.1f} pcm | "
        f"p95={m['p95_pcm']:7.1f} pcm | "
        f"std={m['std_pcm']:6.1f} | "
        f"<650pcm={m['frac_below_650']*100:.1f}% | "
        f"<100pcm={m['frac_below_100']*100:.1f}%"
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: CSV logging
# ─────────────────────────────────────────────────────────────────────────────
val_logfile   = open(val_log_path,   "w", newline="", buffering=1)
train_logfile = open(train_log_path, "w", newline="", buffering=1)
val_writer    = csv.writer(val_logfile)
train_writer  = csv.writer(train_logfile)
_HEADER = ["epoch", "sample_idx", "keff_openmc", "keff_peds",
           "delta_rho_pcm", "train_loss", "val_loss"]
val_writer.writerow(_HEADER)
train_writer.writerow(_HEADER)

# ── NEW: epoch-level aggregate stats CSV ─────────────────────────────────────
# One row per epoch with every metric for both train and val splits.
# This lets you reconstruct the full training curve even if prints are lost.
epoch_stats_file = open(epoch_stats_path, "w", newline="", buffering=1)
epoch_stats_writer = csv.writer(epoch_stats_file)
_EPOCH_HEADER = [
    "epoch",
    "train_mse_k", "train_mae_k", "train_mean_pcm", "train_median_pcm",
    "train_p95_pcm", "train_std_pcm", "train_frac_below_650", "train_frac_below_100",
    "val_mse_k",   "val_mae_k",   "val_mean_pcm",   "val_median_pcm",
    "val_p95_pcm",   "val_std_pcm",   "val_frac_below_650",   "val_frac_below_100",
]
epoch_stats_writer.writerow(_EPOCH_HEADER)


def log_epoch_stats(epoch: int, train_m: dict, val_m: dict):
    """Write one row of aggregate stats per epoch to epoch_metrics.csv."""
    row = [epoch] + [
        train_m["mse_k"], train_m["MAE_k"], train_m["mean_pcm"], train_m["median_pcm"],
        train_m["p95_pcm"], train_m["std_pcm"], train_m["frac_below_650"], train_m["frac_below_100"],
        val_m["mse_k"],   val_m["MAE_k"],   val_m["mean_pcm"],   val_m["median_pcm"],
        val_m["p95_pcm"],   val_m["std_pcm"],   val_m["frac_below_650"],   val_m["frac_below_100"],
    ]
    with _csv_lock:
        epoch_stats_writer.writerow(row)
        epoch_stats_file.flush()


def log_keff_batch(writer, filehandle, epoch, k_pred, k_ref,
                   avg_train_loss, avg_val_loss, sample_id_offset=0):
    with _csv_lock:
        for sidx, (kr, kp) in enumerate(zip(k_ref, k_pred)):
            dr = abs(kp - kr) / (kp * kr) * 1e5
            writer.writerow([epoch, sidx + sample_id_offset,
                              f"{kr:.6f}", f"{kp:.6f}", f"{dr:.1f}",
                              avg_train_loss, avg_val_loss])
        filehandle.flush()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

# ── XS name labels — adjust to match your actual xs_layout ordering ──────────
# These are used as axis tick labels on the heatmap.
# G=2 → groups 1,2; typical order: D1 D2 Sa1 Sa2 nF1 nF2 Ss12 Ss21 chi1 chi2 ...
# (12 entries per region for G=2). Modify if your SLAY ordering differs.
_XS_LABELS = [
    "D₁","D₂",
    "Σa₁","Σa₂",
    "νΣf₁","νΣf₂",
    "Σs₁₂","Σs₂₁",
    "χ₁","χ₂",
    "XS₁₁","XS₁₂",   # placeholder names for remaining slots
]


def _save_xs_subplots_for_samples(
    model,
    sample_indices,
    geoms_all,
    keffs_all,
    rawparams_all,
    xs_baselines_all,
    delta_k_norm_all,
    epoch: int,
):
    """
    For each index in sample_indices, run a lightweight NN forward pass and
    save one plot_xs_subplots figure per sample.

    Files land in LOG_DIR/xs_subplots/epoch_{E:04d}_sample{IDX:03d}.png

    Arguments mirror _plot_xs_heatmap so they can share the same call-site data.
    delta_k_norm_all: 1-D float32 array [N] — normalised Δk for each sample.
    """
    subplots_dir = os.path.join(LOG_DIR, "xs_subplots")
    os.makedirs(subplots_dir, exist_ok=True)
    epoch_label = f"Epoch {epoch}"

    for idx in sample_indices:
        idx = int(idx)
        if idx >= len(geoms_all):
            print(f"  [subplot] sample_idx={idx} out of range, skipping")
            continue

        # ── per-sample baseline ───────────────────────────────────────────────
        geo_i    = update_geo(GEO, rawparams_all[idx])
        baseline = np.array(predict_xs(geo_i), dtype=np.float32)   # (3, 12)

        xs_b  = jnp.array(xs_baselines_all[idx:idx+1], dtype=jnp.float32)   # (1,3,12)
        geom  = jnp.array(geoms_all[idx:idx+1],        dtype=jnp.float32)   # (1,6)
        dk    = jnp.array([[float(delta_k_norm_all[idx])]], dtype=jnp.float32)  # (1,1)

        # ── NN forward (no gradient needed) ──────────────────────────────────
        log_base   = model._log_baselines(xs_b)                      # (1,3,12)
        log_ratios = np.array(
            model.generator(geom, log_base, dk, training=False)[0]  # (3,12)
        )
        final_xs = np.exp(log_ratios) * baseline                     # (3,12)

        # ── optional: run solver for keff_pred ───────────────────────────────
        keff_pred_val = None
        try:
            SENTINEL  = 7000 + idx
            keff_pred_val = float(NTdiff_solver(
                jnp.array(final_xs,              dtype=jnp.float32),
                jnp.array(rawparams_all[idx],    dtype=jnp.float32),
                jnp.array([SENTINEL],            dtype=jnp.int32),
            ))
        except Exception as e:
            print(f"  [subplot] solver failed for sample {idx} epoch {epoch}: {e}")

        save_path = os.path.join(
            subplots_dir, f"epoch_{epoch:04d}_sample{idx:03d}.png"
        )
        plot_xs_subplots(
            baseline    = baseline,
            final_xs    = final_xs,
            G           = GEO.G,
            save_path   = save_path,
            epoch_label = epoch_label,
            sample_idx  = idx,
            geo_params  = rawparams_all[idx],
            param_names = PARAM_NAMES,
            keff_ref    = float(keffs_all[idx]),
            keff_pred   = keff_pred_val,
        )
        print(f"  [subplot] epoch {epoch}  sample {idx}  →  {save_path}")


def _plot_xs_heatmap(model, geoms_batch, xs_baselines_batch, delta_k_norm_batch,
                     epoch: int, n_show: int = 8):
    """
    Save a heatmap of the NN-corrected XS for a small representative batch.

    Layout: rows = samples (up to n_show), columns = XS types.
            One sub-figure per region, side by side.
    Shows the ratio  xs_final / xs_baseline  so 1.0 = no correction.
    Saved to LOG_DIR/xs_heatmap/epoch_{epoch:04d}.png
    """
    heatmap_dir = os.path.join(LOG_DIR, "xs_heatmap")
    os.makedirs(heatmap_dir, exist_ok=True)

    # ── get corrected XS (numpy, no grad) ────────────────────────────────────
    n    = min(n_show, geoms_batch.shape[0])
    xs_b = jnp.array(xs_baselines_batch[:n], dtype=jnp.float32)
    xs_f = model.compute_xs(
        jnp.array(geoms_batch[:n], dtype=jnp.float32),
        xs_b,
        jnp.array(delta_k_norm_batch[:n], dtype=jnp.float32),
    )  # [n, n_regions, xs_per_region]

    # ratio relative to baseline; clip extreme values for display
    xs_base_np = np.array(xs_b)
    safe_base  = np.where(xs_base_np > 1e-10, xs_base_np, np.ones_like(xs_base_np))
    ratio      = xs_f / safe_base                   # [n, n_regions, xs_per_region]
    ratio      = np.clip(ratio, 0.5, 2.0)           # display range

    n_regions    = ratio.shape[1]
    xs_per_region = ratio.shape[2]
    labels = _XS_LABELS[:xs_per_region]

    region_names = ["CR", "Core", "Moderator"]

    fig, axes = plt.subplots(1, n_regions, figsize=(5 * n_regions, 0.5 * n + 1.5),
                             squeeze=False)
    fig.suptitle(f"XS correction ratio  (epoch {epoch})\n"
                 f"colour = xs_final / xs_baseline   [clipped 0.5–2.0]", fontsize=10)

    for r in range(n_regions):
        ax  = axes[0, r]
        mat = ratio[:, r, :]            # [n, xs_per_region]
        im  = ax.imshow(mat, aspect="auto", vmin=0.5, vmax=2.0, cmap="RdBu_r")
        ax.set_xticks(range(xs_per_region))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(n))
        ax.set_yticklabels([f"s{i}" for i in range(n)], fontsize=7)
        ax.set_title(region_names[r] if r < len(region_names) else f"Region {r}", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    path = os.path.join(heatmap_dir, f"epoch_{epoch:04d}.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"XS heatmap saved → {path}")


def _plot_history(history: dict, exp_name: str):
    """5-panel summary plot saved to LOG_DIR/training_history.png

    Panels:
      [0,0] MSE loss (log scale)
      [0,1] Val reactivity error in pcm
      [1,0] MAE in k-units (log scale)   ← NEW: correct label, log y-axis
      [1,1] % samples below 650 pcm
      [2,0] MAE in pcm (log scale)        ← NEW: physics-unit MAE
    """
    epochs_range = range(1, len(history["train_mse"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"PEDS training history — {exp_name}", fontsize=13)

    # ── [0,0] MSE ─────────────────────────────────────────────────────────────
    axes[0, 0].semilogy(epochs_range, history["train_mse"], label="train MSE")
    axes[0, 0].semilogy(epochs_range, history["val_mse"],   label="val MSE")
    axes[0, 0].set_title("MSE loss (k-units²)")
    axes[0, 0].set_ylabel("MSE  [k²]")
    axes[0, 0].legend()

    # ── [0,1] Reactivity error pcm ───────────────────────────────────────────
    axes[0, 1].plot(epochs_range, history["val_mean_pcm"],   label="mean |Δρ|")
    axes[0, 1].plot(epochs_range, history["val_median_pcm"], label="median |Δρ|")
    axes[0, 1].axhline(650, ls="--", color="grey", label="β_eff = 650 pcm")
    axes[0, 1].set_title("Val reactivity error (pcm)")
    axes[0, 1].set_ylabel("|Δρ|  [pcm]")
    axes[0, 1].legend()

    # ── [1,0] MAE in k-units, log scale ──────────────────────────────────────
    # MAE_k is in k-units (dimensionless k-eigenvalue differences).
    # Log scale is appropriate because it spans several orders of magnitude
    # during training and makes early improvement visible.
    axes[1, 0].semilogy(epochs_range, history["train_mae"], label="train MAE")
    axes[1, 0].semilogy(epochs_range, history["val_mae"],   label="val MAE")
    axes[1, 0].set_title("MAE  (k-units, log scale)")
    axes[1, 0].set_ylabel("MAE  [Δk]")
    axes[1, 0].legend()

    # ── [1,1] Fraction below 650 pcm ─────────────────────────────────────────
    axes[1, 1].plot(epochs_range, [x * 100 for x in history["val_frac_below_650"]])
    axes[1, 1].axhline(95, ls="--", color="grey", label="95% target")
    axes[1, 1].set_ylim(0, 105)
    axes[1, 1].set_title("% samples below 650 pcm")
    axes[1, 1].set_ylabel("fraction  [%]")
    axes[1, 1].legend()

    for ax in axes.flat:
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(LOG_DIR, "training_history.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"History plot saved → {path}")

    # ── Separate plot: val mean |Δρ| in pcm on log scale ─────────────────────
    # Useful when early epochs have very large errors that compress the linear scale.
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.semilogy(epochs_range, history["val_mean_pcm"],   label="mean |Δρ|")
    ax2.semilogy(epochs_range, history["val_median_pcm"], label="median |Δρ|")
    ax2.semilogy(epochs_range, history["val_p95_pcm"],    label="p95 |Δρ|",
                 ls="--", alpha=0.7)
    ax2.axhline(650, ls=":", color="grey", label="β_eff = 650 pcm")
    ax2.axhline(100, ls=":", color="navy", label="100 pcm target")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("|Δρ|  [pcm]")
    ax2.set_title(f"Val reactivity error — {exp_name}  (log scale)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    path2 = os.path.join(LOG_DIR, "val_reactivity_logscale.png")
    plt.savefig(path2, dpi=150)
    plt.close()
    print(f"Reactivity log-scale plot saved → {path2}")

def _save_flux_plots(
    sample_indices: list,
    train_geoms: np.ndarray,
    train_keffs: np.ndarray,
    train_rawparams: np.ndarray,
    train_xs_baselines: np.ndarray,
    train_delta_k_norm: np.ndarray,
    model,
    epoch: int,
):
    """
    For each sample index, run a fresh forward solve with the current NN
    XS corrections and plot the flux shape using _plot_fluxes.

    Files land in LOG_DIR/flux_plots/epoch_{E:04d}_sample{IDX:03d}.png

    This uses unshuffled arrays so sample identity is stable across epochs.
    """
    flux_dir = os.path.join(LOG_DIR, "flux_plots")
    os.makedirs(flux_dir, exist_ok=True)

    for idx in sample_indices:
        idx = int(idx)
        if idx >= len(train_geoms):
            print(f"  [flux_plot] sample_idx={idx} out of range, skipping")
            continue

        # ── Geometry for this sample ──────────────────────────────────────────
        geo_i    = update_geo(GEO, train_rawparams[idx])
        R        = geo_i.boundaries[-1].radius
        I        = int(R / geo_i.mesh_size)
        Delta_r  = geo_i.mesh_size

        # ── Get NN-corrected XS (no gradient needed) ──────────────────────────
        xs_b  = jnp.array(train_xs_baselines[idx:idx+1], dtype=jnp.float32)  # (1,3,12)
        geom  = jnp.array(train_geoms[idx:idx+1],        dtype=jnp.float32)  # (1,6)
        dk    = jnp.array([[float(train_delta_k_norm[idx])]], dtype=jnp.float32)  # (1,1)

        log_base   = model._log_baselines(xs_b)
        log_ratios = model.generator(geom, log_base, dk, training=False)  # (1,3,12)
        xs_corrected = np.array(jnp.exp(log_ratios[0]) * xs_b[0])        # (3,12)

        # ── Run the solver directly (NumPy, no JAX trace) ─────────────────────
        # We call run_diffusion_solver because it returns phi_fwd and phi_adj
        # in (G, I) shape, which is exactly what _plot_fluxes expects.
        try:
            k_pred, phi_fwd, phi_adj = run_diffusion_solver(xs_corrected, geo_i)
        except Exception as e:
            print(f"  [flux_plot] solver failed for sample {idx} epoch {epoch}: {e}")
            continue

        # phi_fwd and phi_adj already have shape (G, I) — no reshaping needed.
        # run_diffusion_solver volume-normalises them before returning.

        # ── Build the radial grid x (cell centres) ────────────────────────────
        # _plot_fluxes uses x as the horizontal axis. Cell centres are at
        # r = (i + 0.5) * Delta_r for i in 0..I-1.
        x = np.array([(i + 0.5) * Delta_r for i in range(I)])  # shape (I,)

        # ── Also run baseline (no NN) for comparison ──────────────────────────
        xs_base_np = np.array(train_xs_baselines[idx], dtype=np.float32)  # (3,12)
        try:
            k_base, phi_fwd_base, phi_adj_base = run_diffusion_solver(xs_base_np, geo_i)
        except Exception as e:
            print(f"  [flux_plot] baseline solver failed for sample {idx}: {e}")
            phi_fwd_base = None
            phi_adj_base = None
            k_base       = None

        # ── Plot ──────────────────────────────────────────────────────────────
        save_path = os.path.join(
            flux_dir, f"epoch_{epoch:04d}_sample{idx:03d}.png"
        )

        # _plot_fluxes from diffusion_solver.py takes:
        #   x              (I,)     radial grid
        #   geo            GeometryConfig
        #   phi_fwd_norm   (G, I)   forward flux
        #   phi_adj_norm   (G, I)   adjoint flux
        #   an_fwd_groups  optional — we use this slot for the baseline flux
        #   an_adj_groups  optional
        #   plot_output    str path
        #
        # We pass phi_fwd_base as an_fwd_groups so the plot shows both
        # the corrected flux (solid line) and the baseline flux (circles)
        # on the same axes. The legend labels them "Analytic Fwd" but
        # you can rename them inside _plot_fluxes if you prefer.

        _plot_fluxes(
            x            = x,
            geo          = geo_i,
            phi_fwd_norm = phi_fwd,
            phi_adj_norm = phi_adj,
            an_fwd_groups = phi_fwd_base,   # baseline for comparison
            an_adj_groups = None,
            plot_output  = save_path,
        )

        k_ref = float(train_keffs[idx])
        dr    = abs(k_pred - k_ref) / (k_pred * k_ref) * 1e5
        print(
            f"  [flux_plot] epoch {epoch}  sample {idx}  "
            f"k_ref={k_ref:.5f}  k_pred={k_pred:.5f}  |Δρ|={dr:.1f} pcm  → {save_path}"
        )

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: Training
# ─────────────────────────────────────────────────────────────────────────────
def train(filepath, train_size, test_size, batch_size, epochs, lr,
          hidden_sizes, n_regions, G, seed):

    # ── 8.1 Data ─────────────────────────────────────────────────────────────
    print("Loading data …")
    (train_geoms, train_keffs, train_rawparams, train_k_reg), \
    (test_geoms,  test_keffs,  test_rawparams,  test_k_reg) = load_data(
        filepath, train_size, test_size, seed,
    )

    # ── Δk normalisation — fit on train, apply everywhere ────────────────────
    train_dk_pcm  = (train_keffs - train_k_reg) * 1e5
    train_dk_mean = float(train_dk_pcm.mean())
    train_dk_std  = float(train_dk_pcm.std()) + 1e-8
    print(f"Δk: mean={train_dk_mean:.1f} pcm  std={train_dk_std:.1f} pcm")

    train_delta_k_norm = ((train_dk_pcm - train_dk_mean) / train_dk_std).astype(np.float32)
    test_delta_k_norm  = (((test_keffs  - test_k_reg) * 1e5 - train_dk_mean) / train_dk_std).astype(np.float32)

    # ── 8.2 Model — v1: plain Adam, constant LR, NO grad clip ────────────────
    rngs      = nnx.Rngs(seed)
    model     = PEDSModel(hidden_sizes=hidden_sizes, n_regions=n_regions, G=G, rngs=rngs)
    optimizer = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)

    # ── 8.3 Loss — pure MSE, NO regularization ───────────────────────────────
    def loss_fn(model, geoms, keffs_true, rawparams_batch,
                xs_baselines_batch, delta_k_norm_batch):
        with timer("loss_fn forward pass", verbose=False):
            keff_pred, _, _ = model(
                geoms, rawparams_batch, xs_baselines_batch, delta_k_norm_batch,
                training=True,
            )
        loss = jnp.mean((keff_pred - keffs_true) ** 2)
        return loss, keff_pred

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)

    # ── 8.4 Pre-compute XS baselines once ────────────────────────────────────
    print("Precomputing XS baselines …")
    train_xs_baselines = compute_batch_baselines(train_rawparams, GEO)
    test_xs_baselines  = compute_batch_baselines(test_rawparams,  GEO)
    print("Done.")

    # ── 8.5 History arrays ───────────────────────────────────────────────────
    history = {k: [] for k in
               ["train_mse","val_mse", "train_mae","val_mae","val_mean_pcm","val_median_pcm",
                "val_p95_pcm","val_std_pcm","val_frac_below_650"]}

    print(f"\nTraining for {epochs} epochs …")

    for epoch in range(1, epochs + 1):
        # ── shuffle ──────────────────────────────────────────────────────────
        idx = np.random.permutation(train_size)
        t_geoms   = train_geoms[idx]
        t_keffs   = train_keffs[idx]
        t_raw     = train_rawparams[idx]
        t_base    = np.array(train_xs_baselines)[idx]
        t_dk      = train_delta_k_norm[idx]

        epoch_loss = 0.0
        all_kp_train, all_kr_train = [], []

        for batch_idx, (bg, bk, br, bb, bdk) in enumerate(data_loader(
                t_geoms, t_keffs, t_raw, t_base, t_dk, batch_size=batch_size)):

            bp   = jnp.array(bg)
            bkj  = jnp.array(bk)
            bxs  = jnp.array(bb)
            bdkj = jnp.array(bdk)

            # Step 1: concrete XS from NN (outside JAX trace)
            xs_np = model.compute_xs(bp, bxs, bdkj)

            # Step 2: parallel pre-solve
            args_list = [(i, xs_np[i], np.array(br[i]), i, epoch)
                         for i in range(len(bg))]
            results = list(executor.map(_solve_sample_worker, args_list))
            _PRESOLVE_CACHE.clear()
            for i, k, phi_fwd, phi_adj, geo_data, _ in results:
                _PRESOLVE_CACHE[i] = (k, phi_fwd, phi_adj, geo_data)

            # Step 3: forward + backward + update
            with timer("train step: forward+backward+optiupd", verbose = False):
                (loss, keff_preds_batch), grads = grad_fn(
                    model, bp, bkj, br, bxs, bdkj,
                )
            optimizer.update(model, grads)

            epoch_loss    += float(loss)
            all_kp_train.extend(np.array(keff_preds_batch).tolist())
            all_kr_train.extend(bk.tolist())

            log_keff_batch(
                train_writer, train_logfile, epoch,
                np.array(keff_preds_batch), bk, float(loss), 0.0,
                sample_id_offset=batch_idx * batch_size,
            )

        # ── validation ───────────────────────────────────────────────────────
        _PRESOLVE_CACHE.clear()
        all_kp_val, all_kr_val = [], []
        with timer("validation step"):
            for batch_idx, (bg, bk, br, bb, bdk) in enumerate(data_loader(
                    test_geoms, test_keffs, test_rawparams,
                    np.array(test_xs_baselines), test_delta_k_norm,
                    batch_size=batch_size)):

                global_start = batch_idx * batch_size
                xs_np = model.compute_xs(
                    jnp.array(bg), jnp.array(bb), jnp.array(bdk),
                )
                VAL_OFFSET = train_size + 10000
                args_list = [(i, xs_np[i], np.array(br[i]),
                            VAL_OFFSET + i + global_start, -1)
                            for i in range(len(bg))]
                results = list(executor.map(_solve_sample_worker, args_list))
                for i, k, *_ in results:
                    all_kp_val.append(float(k))
                    all_kr_val.append(float(bk[i]))

        kp_val = np.array(all_kp_val)
        kr_val = np.array(all_kr_val)
        kp_tr  = np.array(all_kp_train)
        kr_tr  = np.array(all_kr_train)

        # ── metrics ─────────────────────────────────────────────────────────
        train_m = compute_metrics(kp_tr, kr_tr)
        val_m   = compute_metrics(kp_val, kr_val)
        print_metrics(epoch, "TRAIN", train_m)
        print_metrics(epoch, "VAL",   val_m)

        # ── per-sample CSV ───────────────────────────────────────────────────
        log_keff_batch(val_writer, val_logfile, epoch, kp_val, kr_val,
                       train_m["mse_k"], val_m["mse_k"])

        # ── epoch-level aggregate CSV (NEW) ──────────────────────────────────
        log_epoch_stats(epoch, train_m, val_m)

        # ── history for final plot ───────────────────────────────────────────
        history["train_mse"].append(train_m["mse_k"])
        history["val_mse"].append(val_m["mse_k"])
        history["train_mae"].append(train_m["MAE_k"])
        history["val_mae"].append(val_m["MAE_k"])
        history["val_mean_pcm"].append(val_m["mean_pcm"])
        history["val_median_pcm"].append(val_m["median_pcm"])
        history["val_p95_pcm"].append(val_m["p95_pcm"])
        history["val_std_pcm"].append(val_m["std_pcm"])
        history["val_frac_below_650"].append(val_m["frac_below_650"])

        # ── XS heatmap + subplot checkpoints ─────────────────────────────────
        if epoch in XS_HEATMAP_EPOCHS:
            _plot_xs_heatmap(
                model,
                train_geoms[:8],                        # ← unshuffled: always same 8 samples
                np.array(train_xs_baselines)[:8],       # ← unshuffled
                train_delta_k_norm[:8], 
                epoch=epoch,
                n_show=8,
            )
            _save_xs_subplots_for_samples(
                model            = model,
                sample_indices   = SUBPLOT_SAMPLE_INDICES,
                geoms_all        = train_geoms,              # ← unshuffled
                keffs_all        = train_keffs,              # ← unshuffled
                rawparams_all    = train_rawparams,          # ← unshuffled
                xs_baselines_all = np.array(train_xs_baselines),  # ← unshuffled
                delta_k_norm_all = train_delta_k_norm,       # ← unshuffled
                epoch            = epoch,
            )
            _save_flux_plots(                         
                sample_indices      = SUBPLOT_SAMPLE_INDICES,
                train_geoms         = train_geoms,
                train_keffs         = train_keffs,
                train_rawparams     = train_rawparams,
                train_xs_baselines  = np.array(train_xs_baselines),
                train_delta_k_norm  = train_delta_k_norm,
                model               = model,
                epoch               = epoch,
            )

        if epoch % 20 == 0:
            jax.clear_caches()

    # ── final report ─────────────────────────────────────────────────────────
    _plot_history(history, EXP_NAME)
    print_timing_report()
    val_logfile.close()
    train_logfile.close()
    epoch_stats_file.close()
    loggy_file.close()
    return model, history


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    HP = dict(
        filepath    = _DATA_FILEPATH,   # uses anchored path, not hard-coded string
        train_size  = TRAIN_SIZE,
        test_size   = TEST_SIZE,
        batch_size  = BATCH_SIZE,
        epochs      = EPOCHS,
        lr          = LR,
        hidden_sizes= [128, 128, 64],
        n_regions   = 3,
        G           = 2,
        seed        = SEED,
    )
    print("Starting PEDS v1 (clean baseline) …")
    model, history = train(**HP)
    print("\nDone.")
    print(f"  Final val mean |Δρ|   : {history['val_mean_pcm'][-1]:.1f} pcm")
    print(f"  Final val median |Δρ| : {history['val_median_pcm'][-1]:.1f} pcm")
    print(f"  Final val p95 |Δρ|    : {history['val_p95_pcm'][-1]:.1f} pcm")
    print(f"  Final % below 650 pcm : {history['val_frac_below_650'][-1]*100:.1f}%")
