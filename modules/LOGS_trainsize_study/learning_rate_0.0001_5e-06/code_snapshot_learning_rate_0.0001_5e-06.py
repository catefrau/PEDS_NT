"""==========================================================================
PEDS  —  VERSION 7 - normalization to log output
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
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={os.cpu_count()}"
os.environ["XLA_CPU_ENABLE_FAST_MATH"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"
multiprocessing.set_start_method("spawn", force=True)

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import optax
import pickle
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
import traceback
from datetime import datetime


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
from PEDS_core.timing_utils import timer, print_timing_report, _TIMINGS
from plot_functions.xs_heatmap import plot_xs_subplots


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0: Global constants
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_SIZE   = 500
VAL_SIZE     = 100    # used every epoch — was previously called val_SIZE
TEST_SIZE    = 100    # held out, evaluated only once at the very end
HOLDOUT_SEED = 0      # FIXED — keeps val/test identical across all runs
BATCH_SIZE = 32
EPOCHS     = 80
SEED       = 0
LR_max     = 1e-4   # cosine schedule peak learning rate
LR_min     = 5e-6

EXP_NAME   = f"learning_rate_{LR_max}_{LR_min}"
LOG_DIR = os.path.join(THIS_DIR, "LOGS_trainsize_study", EXP_NAME)


LOG_RATIO_CLIP_LO = -1.8
LOG_RATIO_CLIP_HI = 0.8

# Epochs at which XS heatmap snapshots are saved.
# Add/remove values here to control checkpointing granularity.
XS_HEATMAP_EPOCHS = {1, 5, 10, 20, 50, 100}

# Sample indices (into the TRAINING set) for per-sample xs_subplots figures.
# These are saved at the same epochs as XS_HEATMAP_EPOCHS.
SUBPLOT_SAMPLE_INDICES: list = [0, 1, 2, 3, 4]

TRACKED_SAMPLES: list = [0, 1, 2, 3, 4]

# Human-readable names for the 6 raw geometry parameters (used in info box).
PARAM_NAMES: list = ['b4c_r', 'cr_frac', 'fuel_r', 'enrichment', 'f_mod', 'water_r']

# ── data path — anchored to PARENT_DIR so it works from any cwd ─────────────
_DATA_FILEPATH = os.path.join(PARENT_DIR, "data", "highfidelity", "1082_0.8_1.2.npz")

N_WORKERS = min(int(os.environ.get("SLURM_CPUS_PER_TASK", 32)), BATCH_SIZE)
print(f"Using {N_WORKERS} parallel workers")

executor = None

val_logfile = None
train_logfile = None
val_writer = None
train_writer = None

epoch_stats_file = None
epoch_stats_writer = None

xs_proposal_stats_file = None
xs_proposal_stats_writer = None
xs_history_file = None
xs_history_writer = None
split_logfile = None
split_writer = None

# ── output directory: everything for this run lives under LOG_DIR ─────────────
XS_DIR = os.path.join(LOG_DIR, "XS")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(XS_DIR, exist_ok=True)

val_log_path    = os.path.join(LOG_DIR, "keff_epoch_log_val.csv")
train_log_path  = os.path.join(LOG_DIR, "keff_epoch_log_train.csv")
epoch_stats_path = os.path.join(LOG_DIR, "epoch_metrics.csv")
xs_proposal_stats_path = os.path.join(LOG_DIR, "xs_proposal_stats.csv")
xs_history_path = os.path.join(XS_DIR, "history_first5.csv")
logratio_stats_path = os.path.join(XS_DIR, "logratio_saturation.csv")
split_log_path = os.path.join(LOG_DIR, "split_log.csv")
# ── checkpointing ────────────────────────────────────────────────────────────
CHECKPOINT_DIR = os.path.join(LOG_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
BEST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pkl")
LAST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, "last_checkpoint.pkl")

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

_GEO_DATA_CACHE  = LRUCache(maxsize=BATCH_SIZE)
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

def build_xs_mask(filepath: str, geo: GeometryConfig) -> np.ndarray:
    """
    Returns shape [nregions, xsperregion] with 1.0 where XS can be nonzero,
    0.0 where physics requires exactly zero.
    Built from the polynomial-regression XS averaged over training data,
    or simply from a known reference sample.
    """
    data = np.load(filepath, allow_pickle=True)
    rawparams = np.array(data['params_raw'], dtype=np.float32)    
    ref_params = rawparams[0]  # or average over a few
    ref_xs = predict_xs(update_geo(geo, ref_params))  # shape [3, 12]
    mask = (np.abs(ref_xs) > 1e-10).astype(np.float32)
    return mask  # shape [3, 12]

XS_MASK = build_xs_mask(_DATA_FILEPATH, GEO)  # [3, 12], fixed for the whole run
XS_MASK_J = jnp.array(XS_MASK)  # JAX version

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

def clip_ste(x, lo, hi):
    clipped = jnp.clip(x, lo, hi)
    # forward: use clipped value. backward: gradient flows as if x passed through unchanged.
    return x + jax.lax.stop_gradient(clipped - x)

class GeneratorNN(nnx.Module):
    """
    Trunk:  [batch, 7]  →  128 → 128 → 64  (ReLU)
    Heads:  3 × [trunk_64 + log_baseline_12]  →  12  (zero-init)

    Input vector (dim 7):
        geoms[0:6]  — 6 normalised geometry parameters

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
                 phi_norm: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """
        geoms:            [batch, 6]
        xs_baselines_log: [batch, n_regions, xs_per_region]
        phi_norm:       [batch, n_phi_feats]
        Returns:          [batch, n_regions, xs_per_region]  log-ratio offsets
        """
        x    = jnp.concatenate([geoms, phi_norm], axis=-1)  # [batch, 7]
        for layer in self.layers:
            x = nnx.relu(layer(x))

        log_base_flat = jnp.reshape(xs_baselines_log, (geoms.shape[0], -1))  # [batch, 36]
        feat = jnp.concatenate([x, log_base_flat], axis=-1)                  # [batch, 100]
        out  = self.head(feat)                                                # [batch, 36]
        return jnp.reshape(out, (geoms.shape[0], self.n_regions, self.xs_per_region))


class PEDSModel(nnx.Module):
    """GeneratorNN + physics solver, end-to-end differentiable."""

    def __init__(self, hidden_sizes: list, n_regions: int, G: int, n_phi_feats: int, rngs: nnx.Rngs):
        super().__init__()
        xs_region  = fn_xs_per_region(G)
        input_dim   = 6 + n_phi_feats   # geom + phi = 13
        layer_sizes = [input_dim] + hidden_sizes
        self.generator   = GeneratorNN(layer_sizes, n_regions, xs_region, rngs)
        self.n_regions   = n_regions
        self.G           = G

    def _log_baselines(self, xs_baselines: jnp.ndarray,
                       log_xs_mean=None, log_xs_std=None) -> jnp.ndarray:
        """Safe log of baselines, optionally z-score normalized per XS slot.
        
        If log_xs_mean/std are provided (jnp arrays of shape [3,12]),
        the output is (log(xs) - mean) / std per slot.
        Pass them during training and validation; omit only in diagnostic calls
        that don't go through the solver (old plotting code).
        """
        if log_xs_mean is not None:
            fallback = jnp.exp(log_xs_mean)
        else:
            fallback = jnp.ones_like(xs_baselines)
        safe = jnp.where(xs_baselines > 1e-10, xs_baselines, fallback)
        log_xs = jnp.log(safe)
        if log_xs_mean is not None and log_xs_std is not None:
            log_xs = (log_xs - log_xs_mean) / log_xs_std
        return log_xs

    def compute_xs(self, geoms, xs_baselines, phi_norm,
                             log_xs_mean=None, log_xs_std=None):
        """Pure NN forward (no grad). Used for pre-solving and validation."""
        batch_size = geoms.shape[0]
        with timer("log baselines from xs first guess", verbose=False): 
            log_base    = self._log_baselines(xs_baselines,  log_xs_mean, log_xs_std)
        with timer("NN generated XS log ratios", verbose=False):
            log_ratios  = self.generator(geoms, log_base, phi_norm, training=False)
        # ── v1: NO clip, NO warmup ─────────────────────────────────────────
        log_ratios = clip_ste(log_ratios, LOG_RATIO_CLIP_LO, LOG_RATIO_CLIP_HI)  # now consistent
        xs_final = jnp.exp(log_ratios) * xs_baselines * XS_MASK_J
        return np.array(xs_final)  # concrete numpy, exits JAX world
        
    def compute_log_ratios(self, geoms, xs_baselines, phi_norm,
                log_xs_mean=None, log_xs_std=None):
            """Pure NN forward, returns CLIPPED log_ratios (no solver). For diagnostics."""
            batch_size = geoms.shape[0]
            log_base = self._log_baselines(xs_baselines, log_xs_mean, log_xs_std)
            log_ratios = self.generator(geoms, log_base, phi_norm, training=False)
            log_ratios = clip_ste(log_ratios, LOG_RATIO_CLIP_LO, LOG_RATIO_CLIP_HI)
            return np.array(log_ratios)
    
    def __call__(self, geoms, params_raw, xs_baselines, phi_norm,
                 training: bool = False, sample_id_offset: int = 0,
                 log_xs_mean=None, log_xs_std=None):
        batch_size = geoms.shape[0]
        log_base   = self._log_baselines(xs_baselines, log_xs_mean, log_xs_std)
        phi = jnp.reshape(phi_norm, (batch_size, -1))
        log_ratios = self.generator(geoms, log_base, phi, training)
        log_ratios = clip_ste(log_ratios, LOG_RATIO_CLIP_LO, LOG_RATIO_CLIP_HI)  # exp(±0.7) ≈ 0.5x to 2x

        xs_final = jnp.exp(log_ratios) * xs_baselines * XS_MASK_J # [batch, 3, 12]

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
    raw = np.stack([predict_xs(update_geo(geo, params_raw[i])) for i in range(len(params_raw))])
    floored = np.maximum(raw, 1e-6)
    masked = np.where(XS_MASK[None, :, :], floored, 0.0)
    return jnp.array(masked, dtype=jnp.float32)


def compute_phi_features(rawparams: np.ndarray) -> tuple:
    """
    Returns:
        phi_features: [N, G*3]  — volume-weighted mean flux per group per region
                                  order: [phi_g0_CR, phi_g0_Core, phi_g0_Mod,
                                          phi_g1_CR, phi_g1_Core, phi_g1_Mod]
    """
    phi_features = []

    for i, p in enumerate(rawparams):
        geo_i   = update_geo(GEO, p)
        xs_i    = np.array(predict_xs(geo_i), dtype=np.float32)
        k, phi_fwd_padded, _, _ = _run_NT_solver(xs_i, p, np.array([i], dtype=np.int32))

        # ── unpack geometry ───────────────────────────────────────────────────
        R       = geo_i.boundaries[-1].radius
        I       = int(R / geo_i.mesh_size)
        Delta_r = geo_i.mesh_size
        G       = geo_i.G

        # Region outer radii (CR_outer, core_outer, mod_outer)
        region_radii = [b.radius for b in geo_i.boundaries]

        # ── extract unpadded flux ─────────────────────────────────────────────
        # phi_fwd_padded layout: group g occupies indices [g*(I+1) : g*(I+1)+I]
        # (the +1 slot is the boundary point, left as zero)

        feats = []
        for g in range(G):
            phi_g = phi_fwd_padded[g * (I + 1) : g * (I + 1) + I]   # shape (I,)

            r_prev = 0.0
            for r_reg in region_radii:
                # Cell i has centre at (i + 0.5) * Delta_r
                centres = np.array([(ic + 0.5) * Delta_r for ic in range(I)])
                mask    = (centres >= r_prev) & (centres < r_reg)

                if mask.any():
                    weighted_mean = np.mean(phi_g[mask])
                    feats.append(float(weighted_mean))
                else:
                    # This region has no cells (e.g. CR radius < mesh_size)
                    feats.append(0.0)

                r_prev = r_reg

        phi_features.append(feats)  # length = G * 3

        if i % 20 == 0:
            print(f"  phi_reg precompute {i}/{len(rawparams)}", flush=True)

    return np.array(phi_features, dtype=np.float32)
    

def derive_split_indices(filepath, train_size, val_size, test_size,
                          train_seed=42, holdout_seed=0):
    """Pure index selection — no phi/XS computation, so it's cheap to call standalone."""
    data  = np.load(filepath, allow_pickle=True)
    keffs = np.array(data['keffs'], dtype=np.float32)
    sorted_idx = np.argsort(keffs)
    n_bins = 10

    holdout_rng  = np.random.default_rng(holdout_seed)
    val_per_bin  = max(1, val_size  // n_bins)
    test_per_bin = max(1, test_size // n_bins)

    val_idx, test_idx, pool_idx = [], [], []
    for bin_indices in np.array_split(sorted_idx, n_bins):
        bin_indices = bin_indices.copy()
        holdout_rng.shuffle(bin_indices)
        n_v = min(val_per_bin,  len(bin_indices) - 1)
        n_t = min(test_per_bin, len(bin_indices) - 1 - n_v)
        val_idx.extend(bin_indices[:n_v].tolist())
        test_idx.extend(bin_indices[n_v:n_v + n_t].tolist())
        pool_idx.extend(bin_indices[n_v + n_t:].tolist())

    val_idx  = np.array(val_idx[:val_size])
    test_idx = np.array(test_idx[:test_size])
    pool_idx = np.array(pool_idx)

    train_rng = np.random.default_rng(train_seed)
    train_rng.shuffle(pool_idx)
    train_idx = pool_idx[:train_size]
    if len(train_idx) < train_size or len(val_idx) < val_size or len(test_idx) < test_size:
        raise ValueError(
            f"Not enough samples! Dataset has ~{len(pool_idx) + len(val_idx) + len(test_idx)} total, "
            f"but requested train={train_size}, val={val_size}, test={test_size}. "
            f"Got: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
        )
    return train_idx, val_idx, test_idx

def _save_split_cache(cache_path, train_idx, val_idx, test_idx, metadata):
    """Save splits to disk for reuse across runs."""
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    payload = {
        "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx,
        "metadata": metadata,
    }
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f)


def load_or_create_split_cache(filepath, train_size, val_size, test_size,
                                train_seed=42, holdout_seed=0, cache_dir=None):
    """
    Load train/val/test splits from cache if available and compatible.
    If dataset grew: val/test stay the same, train extends with new samples.
    """
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(filepath) or ".", ".split_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # One cache file per (train_seed, holdout_seed, val/test config)
    cache_fname = f"split_t{train_size}_ts{train_seed}_hs{holdout_seed}_v{val_size}_t{test_size}.pkl"
    cache_path = os.path.join(cache_dir, cache_fname)
    
    data = np.load(filepath, allow_pickle=True)
    current_dataset_size = len(data['params'])
    
    # ── Try to load cache ────────────────────────────────────────────────────
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            meta = cache["metadata"]
            old_train_idx = cache["train_idx"]
            old_val_idx   = cache["val_idx"]
            old_test_idx  = cache["test_idx"]
            old_dataset_size = meta.get("dataset_size")
            
            # ── Seeds match: cache is for the same holdout config ────────────
            if (meta.get("train_seed") == train_seed and
                meta.get("holdout_seed") == holdout_seed and
                meta.get("val_size") == val_size and
                meta.get("test_size") == test_size):
                
                # Dataset unchanged: reuse exactly
                if current_dataset_size == old_dataset_size:
                    if meta.get("train_size") == train_size:
                        print(f"  [cache hit] reusing {len(old_train_idx)} train, "
                              f"{len(old_val_idx)} val, {len(old_test_idx)} test samples")
                        return old_train_idx, old_val_idx, old_test_idx
                
                # Dataset grew: extend training set
                if current_dataset_size > old_dataset_size:
                    print(f"  [cache hit, dataset grew] {old_dataset_size} → {current_dataset_size} samples")
                    print(f"    extending train from {len(old_train_idx)} to {train_size}…")
                    
                    # Identify newly available samples (not in old train/val/test)
                    old_all = np.union1d(old_train_idx, np.union1d(old_val_idx, old_test_idx))
                    new_pool = np.setdiff1d(np.arange(current_dataset_size), old_all)
                    
                    n_needed = train_size - len(old_train_idx)
                    if len(new_pool) < n_needed:
                        raise ValueError(
                            f"Not enough new samples ({len(new_pool)}) to reach "
                            f"train_size={train_size} (need {n_needed} more)"
                        )
                    
                    # Draw from new pool using the same seed (consistent sampling)
                    extend_rng = np.random.default_rng(train_seed)
                    extend_rng.shuffle(new_pool)
                    new_indices = new_pool[:n_needed]
                    
                    train_idx = np.concatenate([old_train_idx, new_indices])
                    val_idx   = old_val_idx
                    test_idx  = old_test_idx
                    
                    # Update cache
                    meta["dataset_size"] = current_dataset_size
                    meta["train_size"] = train_size
                    _save_split_cache(cache_path, train_idx, val_idx, test_idx, meta)
                    return train_idx, val_idx, test_idx
        
        except Exception as e:
            print(f"  [cache load failed] {e}, recomputing…")
    
    # ── Cache miss: compute from scratch ─────────────────────────────────────
    print(f"  [cache miss] computing new splits (train_seed={train_seed})…")
    train_idx, val_idx, test_idx = derive_split_indices(
        filepath, train_size, val_size, test_size, train_seed, holdout_seed
    )
    
    meta = {
        "train_size": train_size, "val_size": val_size, "test_size": test_size,
        "train_seed": train_seed, "holdout_seed": holdout_seed,
        "dataset_size": current_dataset_size,
    }
    _save_split_cache(cache_path, train_idx, val_idx, test_idx, meta)
    return train_idx, val_idx, test_idx


def load_data(filepath, train_size, val_size, test_size,
              train_seed=42, holdout_seed=0, split_cache_path=None, cache_dir=None):

    if split_cache_path is not None:
        cache_dir = os.path.dirname(split_cache_path)
    
    data      = np.load(filepath, allow_pickle=True)
    geoms     = np.array(data['params'],     dtype=np.float32)
    keffs     = np.array(data['keffs'],      dtype=np.float32)
    rawparams = np.array(data['params_raw'], dtype=np.float32)

    train_idx, val_idx, test_idx = load_or_create_split_cache(
        filepath, train_size, val_size, test_size, train_seed, holdout_seed, cache_dir=cache_dir)

    print(f"  Train k range: {keffs[train_idx].min():.3f} – {keffs[train_idx].max():.3f}")
    print(f"  Val   k range: {keffs[val_idx].min():.3f}  – {keffs[val_idx].max():.3f}")
    print(f"  Test  k range: {keffs[test_idx].min():.3f}  – {keffs[test_idx].max():.3f}")

    print("Precomputing φ_reg features (done once)…")
    train_phi_features = compute_phi_features(rawparams[train_idx])
    val_phi_features    = compute_phi_features(rawparams[val_idx])
    test_phi_features   = compute_phi_features(rawparams[test_idx])

    return (
        (train_idx, geoms[train_idx], keffs[train_idx], rawparams[train_idx], train_phi_features),
        (val_idx, geoms[val_idx],   keffs[val_idx],   rawparams[val_idx],   val_phi_features),
        (test_idx, geoms[test_idx],  keffs[test_idx],  rawparams[test_idx],  test_phi_features),
    )
    

def load_data_balanced_ranges(filepath, train_size, val_size, seed=2,
                               bin_edges=None, train_per_bin=50, val_per_bin=10):
    """
    Fixed-range balanced split by k_eff.
    Uses the same number of samples from every k_eff range.

    Returns the same outputs as your current loaddata().
    """

    data = np.load(filepath, allow_pickle=True)
    geoms = np.array(data["params"], dtype=np.float32)
    keffs = np.array(data["keffs"], dtype=np.float32)
    rawparams = np.array(data["params_raw"], dtype=np.float32)

    if bin_edges is None:
        bin_edges = np.array([0.80, 0.90, 0.95,
                              1.00, 1.05, 1.10, 1.20], dtype=np.float32)

    rng = np.random.default_rng(seed)

    nbins = len(bin_edges) - 1

    if train_size % nbins != 0 or val_size % nbins != 0:
        raise ValueError(
            f"train_size={train_size} and val_size={val_size} must both be "
            f"divisible by nbins={nbins}"
        )

    train_per_bin = train_size // nbins   # 500 // 10 = 50
    val_per_bin  = val_size  // nbins   # 100 // 10 = 10

    trainidx, testidx = [], []

    print("\n=== Balanced fixed-range split ===")
    print(f"train_per_bin = {train_per_bin}, val_per_bin = {val_per_bin}")

    for b in range(nbins):
        lo = bin_edges[b]
        hi = bin_edges[b + 1]

        if b < nbins - 1:
            idx = np.where((keffs >= lo) & (keffs < hi))[0]
            label = f"[{lo:.2f}, {hi:.2f})"
        else:
            idx = np.where((keffs >= lo) & (keffs <= hi))[0]
            label = f"[{lo:.2f}, {hi:.2f}]"

        rng.shuffle(idx)

        needed = train_per_bin + val_per_bin
        if len(idx) < needed:
            raise ValueError(
                f"Bin {label} has only {len(idx)} samples, but needs {needed}."
            )

        val_bin = idx[:val_per_bin]
        train_bin = idx[val_per_bin:val_per_bin + train_per_bin]

        testidx.extend(val_bin.tolist())
        trainidx.extend(train_bin.tolist())

        print(f"{label}: available={len(idx):3d}, train={len(train_bin):2d}, test={len(val_bin):2d}")

    trainidx = np.array(trainidx, dtype=int)
    testidx = np.array(testidx, dtype=int)

    rng.shuffle(trainidx)
    rng.shuffle(testidx)

    print(f"\nFinal train size = {len(trainidx)}")
    print(f"Final test size  = {len(testidx)}")
    print(f"Train k range {keffs[trainidx].min():.3f} to {keffs[trainidx].max():.3f}")
    print(f"Test  k range {keffs[testidx].min():.3f} to {keffs[testidx].max():.3f}")
    print(f"Train keff mean {keffs[trainidx].mean():.3f} std {keffs[trainidx].std():.3f}")
    print(f"Test  keff mean {keffs[testidx].mean():.3f} std {keffs[testidx].std():.3f}")

    print("Precomputing kreg and reg features done once...")
    trainkreg, trainphifeatures = compute_phi_features(rawparams[trainidx])
    testkreg, testphifeatures = compute_phi_features(rawparams[testidx])

    return (
        (geoms[trainidx], keffs[trainidx], rawparams[trainidx], trainkreg, trainphifeatures),
        (geoms[testidx],  keffs[testidx],  rawparams[testidx],  testkreg,  testphifeatures),
    )

def print_keff_bin_counts(name, keffs):
    edges = np.array([0.75, 0.80, 0.85, 0.90, 0.95,
                      1.00, 1.05, 1.10, 1.15, 1.20, 1.25], dtype=np.float32)
    print(f"\n{name} bin counts:")
    for b in range(len(edges) - 1):
        lo, hi = edges[b], edges[b + 1]
        if b < len(edges) - 2:
            n = np.sum((keffs >= lo) & (keffs < hi))
            label = f"[{lo:.2f}, {hi:.2f})"
        else:
            n = np.sum((keffs >= lo) & (keffs <= hi))
            label = f"[{lo:.2f}, {hi:.2f}]"
        print(f"{label}: {n}")

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
def init_csv_logs():
    global val_logfile, train_logfile, val_writer, train_writer
    global epoch_stats_file, epoch_stats_writer
    global logratio_stats_file, logratio_stats_writer
    global xs_history_file, xs_history_writer
    global split_logfile, split_writer

    os.makedirs(LOG_DIR, exist_ok=True)

    for p in [val_log_path, train_log_path, epoch_stats_path, xs_history_path, split_log_path,
              logratio_stats_path]:
        if os.path.exists(p):
            os.remove(p)

    val_logfile = open(val_log_path, "w", newline="", buffering=1)
    train_logfile = open(train_log_path, "w", newline="", buffering=1)

    val_writer = csv.writer(val_logfile)
    train_writer = csv.writer(train_logfile)

    header = ["epoch", "sample_idx", "keff_openmc", "keff_peds",
              "delta_rho_pcm", "train_loss", "val_loss"]
    val_writer.writerow(header)
    train_writer.writerow(header)
    val_logfile.flush()
    train_logfile.flush()

    epoch_stats_file = open(epoch_stats_path, "w", newline="", buffering=1)
    epoch_stats_writer = csv.writer(epoch_stats_file)
    epoch_header = [
        "epoch",
        "train_mse_k", "train_mae_k", "train_mean_pcm", "train_median_pcm",
        "train_p95_pcm", "train_std_pcm", "train_frac_below_650", "train_frac_below_100",
        "val_mse_k", "val_mae_k", "val_mean_pcm", "val_median_pcm",
        "val_p95_pcm", "val_std_pcm", "val_frac_below_650", "val_frac_below_100",
    ]
    epoch_stats_writer.writerow(epoch_header)
    epoch_stats_file.flush()

    logratio_stats_file = open(logratio_stats_path, "w", newline="", buffering=1)
    logratio_stats_writer = csv.writer(logratio_stats_file)
    lr_header = ["epoch", "region", "xs_idx",
                 "min", "max", "mean", "std",
                 "frac_at_lower_clip", "frac_at_upper_clip"]
    logratio_stats_writer.writerow(lr_header)
    logratio_stats_file.flush()

    xs_history_file = open(xs_history_path, "w", newline="", buffering=1)
    xs_history_writer = csv.writer(xs_history_file)

    region_names = ["CR", "Core", "Mod"]
    xs_labels = ["D1","D2","Sa1","Sa2","nSf1","nSf2","Ss11","Ss22","Ss12","Ss21","chi1","chi2"]
    xs_header = [f"{reg}_{xs}" for reg in region_names for xs in xs_labels]

    xs_history_writer.writerow(["epoch", "sample_idx", *PARAM_NAMES, *xs_header])
    xs_history_file.flush()

    split_logfile = open(split_log_path, "w", newline="", buffering=1)
    split_writer = csv.writer(split_logfile)
    split_writer.writerow(["sample_idx", "split", "keff"])
    split_logfile.flush()

def close_csv_logs():
    for fh in [val_logfile, train_logfile, epoch_stats_file, logratio_stats_file, split_logfile]:
        if fh is not None:
            fh.close()

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
                   avg_train_loss, avg_val_loss, sample_id_offset=0, sample_ids_override=None):
    with _csv_lock:
        for sidx, (kr, kp) in enumerate(zip(k_ref, k_pred)):
            dr = abs(kp - kr) / (kp * kr) * 1e5
            sample_idx = int(sample_ids_override[sidx]) if sample_ids_override is not None else sidx + sample_id_offset
            writer.writerow([epoch, sample_idx,
                              f"{kr:.6f}", f"{kp:.6f}", f"{dr:.1f}",
                              avg_train_loss, avg_val_loss])
        filehandle.flush()

_CLIP_EPS = 1e-4  # tolerance for "at the clip boundary"

region_names_lr = ["CR", "Core", "Moderator"]

def log_logratio_saturation(epoch: int, log_ratios_all: np.ndarray):
    """
    log_ratios_all: [N_samples, n_regions, xs_per_region]  (post-clip values)
    Writes one row per (region, xs_idx) summarizing saturation across all samples.
    """
    n_regions, xs_per_region = log_ratios_all.shape[1], log_ratios_all.shape[2]
    with _csv_lock:
        for r in range(n_regions):
            for x in range(xs_per_region):
                vals = log_ratios_all[:, r, x]
                frac_lo = float(np.mean(vals <= LOG_RATIO_CLIP_LO + _CLIP_EPS))
                frac_hi = float(np.mean(vals >= LOG_RATIO_CLIP_HI - _CLIP_EPS))
                logratio_stats_writer.writerow([
                    epoch, region_names_lr[r] if r < 3 else f"R{r}", x,
                    f"{vals.min():.5f}", f"{vals.max():.5f}",
                    f"{vals.mean():.5f}", f"{vals.std():.5f}",
                    f"{frac_lo:.3f}", f"{frac_hi:.3f}",
                ])
        logratio_stats_file.flush()

def log_xs_history_samples(model, epoch, sample_indices,
                           geom_sample_all, rawparams_sample_all, xs_baselines_sample_all, phi_norm_sample_all,
                           log_xs_mean_j, log_xs_std_j):
    global xs_history_file, xs_history_writer

    if len(sample_indices) == 0:
        return

    idx = np.array(sample_indices, dtype=int)

    xsf = model.compute_xs(
        jnp.array(geom_sample_all[idx], dtype=jnp.float32),
        jnp.array(xs_baselines_sample_all[idx], dtype=jnp.float32),
        jnp.array(phi_norm_sample_all[idx], dtype=jnp.float32),
        log_xs_mean_j, log_xs_std_j
    )  # shape (nsamples, 3, 12)

    with _csv_lock:
        for j, s in enumerate(idx):
            row = [epoch, int(s), *rawparams_sample_all[j].tolist(), *xsf[j].reshape(-1).tolist()]
            xs_history_writer.writerow(row)
        xs_history_file.flush()


def log_splits(train_idx, train_keffs, valid_idx, valid_keffs, test_idx, test_keffs,):
    global split_logfile, split_writer
    with _csv_lock:
        for idx, k in zip(train_idx, train_keffs):
            split_writer.writerow([int(idx), "train", float(k)])
        if valid_idx is not None and valid_keffs is not None:
            for idx, k in zip(valid_idx, valid_keffs):
                split_writer.writerow([int(idx), "val", float(k)])
        for idx, k in zip(test_idx, test_keffs):
            split_writer.writerow([int(idx), "test", float(k)])
        split_logfile.flush()

def save_final_xs_csv(model, geoms, raw_params, xs_baselines, phi_norm,
                      keffs_ref, log_xs_mean, log_xs_std, file_path, tag="train"):
    region_names = ["CR", "Core", "Mod"]
    xs_labels    = ["D1","D2","Sa1","Sa2","nSf1","nSf2","Ss11","Ss22","Ss12","Ss21","chi1","chi2"]
    
    geo_header = PARAM_NAMES
    xs_header  = [f"{reg}_{xs}" for reg in region_names for xs in xs_labels]
    # ↓ added keff_pred and delta_rho_pcm to the header
    header = ["sample_idx"] + geo_header + ["keff_ref", "keff_pred", "delta_rho_pcm"] + xs_header

    rows = []
    N = geoms.shape[0]
    batch_size = 25

    for start in range(0, N, batch_size):
        sl = slice(start, start + batch_size)
        xsf = model.compute_xs(
            jnp.array(geoms[sl],         dtype=jnp.float32),
            jnp.array(xs_baselines[sl],  dtype=jnp.float32),
            jnp.array(phi_norm[sl],      dtype=jnp.float32),
            log_xs_mean, log_xs_std,
        )  # (batch, 3, 12)

        for i, global_idx in enumerate(range(start, min(start + batch_size, N))):
            # ── run solver to get keff_pred ──────────────────────────────────
            keff_pred = None
            try:
                k, _, _, _ = _run_NT_solver(
                    xsf[i],                                     # (3,12) final XS
                    raw_params[global_idx],                     # (6,)   geometry
                    np.array([global_idx], dtype=np.int32),     # sample id
                )
                keff_pred = float(k)
            except Exception as e:
                print(f"  [save_final_xs_csv] solver failed for {tag} sample {global_idx}: {e}")

            # ── compute delta_rho_pcm ────────────────────────────────────────
            keff_ref_val = float(keffs_ref[global_idx])
            if keff_pred is not None:
                dr = abs(keff_pred - keff_ref_val) / (keff_pred * keff_ref_val) * 1e5
            else:
                dr = float("nan")   # solver failed → mark as NaN

            row = (
                [global_idx]
                + raw_params[global_idx].tolist()   # 6 geometry params
                + [keff_ref_val, keff_pred, round(dr, 2)]
                + xsf[i].flatten().tolist()          # 36 XS values
            )
            rows.append(row)

        if start % 100 == 0:
            print(f"  [{tag}] saving XS CSV: {start}/{N}", flush=True)

    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Saved final XS ({tag}): {file_path}  [{N} samples]")

import os
import shutil
import hashlib

def save_code_snapshot(logdir, expname, log_handle=None):
    src = os.path.abspath(__file__)
    snapshot_path = os.path.join(logdir, f"code_snapshot_{expname}.py")

    # 1) Save a real copy of the script
    shutil.copy2(src, snapshot_path)

    # 2) Read code text
    with open(src, "r", encoding="utf-8") as f:
        code_text = f.read()

    # 3) Optional hash, useful to identify exact version
    code_hash = hashlib.sha256(code_text.encode("utf-8")).hexdigest()

    # 4) Optional: also append the full code into the txt log
    if log_handle is not None:
        log_handle.write("\n" + "=" * 100 + "\n")
        log_handle.write("CODE SNAPSHOT\n")
        log_handle.write(f"Source file : {src}\n")
        log_handle.write(f"Saved copy  : {snapshot_path}\n")
        log_handle.write(f"SHA256      : {code_hash}\n")
        log_handle.write("=" * 100 + "\n")
        log_handle.write(code_text)
        log_handle.write("\n" + "=" * 100 + "\n")
        log_handle.flush()

    return snapshot_path, code_hash


def save_checkpoint(state_obj, filepath, metadata=None):
    """state_obj: an nnx.State, e.g. nnx.state(model) or nnx.state(optimizer)."""
    np_state = jax.tree_util.tree_map(np.asarray, state_obj)   # portable, no live JAX arrays
    with open(filepath, "wb") as f:
        pickle.dump({"state": np_state, "metadata": metadata or {}}, f)


def load_checkpoint(filepath):
    with open(filepath, "rb") as f:
        payload = pickle.load(f)
    jax_state = jax.tree_util.tree_map(jnp.asarray, payload["state"])
    return jax_state, payload["metadata"]

    
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
    phi_norm_all,
    log_xs_mean_j,
    log_xs_std_j,
    epoch: int,
):
    """
    For each index in sample_indices, run a lightweight NN forward pass and
    save one plot_xs_subplots figure per sample.

    Files land in LOG_DIR/xs_subplots/epoch_{E:04d}_sample{IDX:03d}.png

    Arguments mirror _plot_xs_heatmap so they can share the same call-site data.
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

        # ── NN forward (no gradient needed) ──────────────────────────────────
        log_base   = model._log_baselines(xs_b, log_xs_mean_j, log_xs_std_j)   
        phi = jnp.array(phi_norm_all[idx:idx+1], dtype=jnp.float32)   # (1, nphifeats)                   # (1,3,12)
        log_ratios = np.array(
            model.generator(geom, log_base, phi, training=False)[0]  # (3,12)
        )
        final_xs = np.exp(log_ratios) * baseline                     # (3,12)

        final_xs = model.compute_xs(
            jnp.array(geoms_all[idx:idx+1], dtype=jnp.float32),
            jnp.array(xs_baselines_all[idx:idx+1], dtype=jnp.float32),
            jnp.array(phi_norm_all[idx:idx+1], dtype=jnp.float32),
            log_xs_mean_j,
            log_xs_std_j,
        )[0]
        keff_pred_val = None
        try:
            keff_pred_val = float(NTdiff_solver(
                jnp.array(final_xs, dtype=jnp.float32),
                jnp.array(rawparams_all[idx], dtype=jnp.float32),
                jnp.array([idx], dtype=jnp.int32),
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


def _plot_xs_heatmap(model, geoms_batch, xs_baselines_batch,
                     phi_norm_batch, log_xs_mean_j, log_xs_std_j, epoch: int, n_show: int = 8):
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
        jnp.array(phi_norm_batch[:n], dtype=jnp.float32),
        log_xs_mean_j, log_xs_std_j,
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
    val_geoms: np.ndarray,
    val_keffs: np.ndarray,
    val_rawparams: np.ndarray,
    val_xs_baselines: np.ndarray,
    log_xs_mean_j: jnp.ndarray,
    log_xs_std_j: jnp.ndarray,
    val_phi_norm: np.ndarray,
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
        if idx >= len(val_geoms):
            print(f"  [flux_plot] sample_idx={idx} out of range, skipping")
            continue

        # ── Geometry for this sample ──────────────────────────────────────────
        geo_i    = update_geo(GEO, val_rawparams[idx])
        R        = geo_i.boundaries[-1].radius
        I        = int(R / geo_i.mesh_size)
        Delta_r  = geo_i.mesh_size

        # ── Get NN-corrected XS (no gradient needed) ──────────────────────────
        xs_b  = jnp.array(val_xs_baselines[idx:idx+1], dtype=jnp.float32)  # (1,3,12)
        geom  = jnp.array(val_geoms[idx:idx+1],        dtype=jnp.float32)  # (1,6)
        phi   = jnp.array((val_phi_norm[idx:idx+1]), dtype=jnp.float32) 

        log_base   = model._log_baselines(xs_b)
        log_ratios = model.generator(geom, log_base, phi, training=False)  # (1,3,12)
        #xs_corrected = np.array(jnp.exp(log_ratios[0]) * xs_b[0])        # (3,12)
        # TODO check
        xs_corrected = model.compute_xs(
                jnp.array(val_geoms[idx:idx+1], dtype=jnp.float32),
                jnp.array(val_xs_baselines[idx:idx+1], dtype=jnp.float32),
                jnp.array(val_phi_norm[idx:idx+1], dtype=jnp.float32),
                log_xs_mean_j,
                log_xs_std_j,
            )[0]
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
        xs_base_np = np.array(val_xs_baselines[idx], dtype=np.float32)  # (3,12)
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

        k_ref = float(val_keffs[idx])
        dr    = abs(k_pred - k_ref) / (k_pred * k_ref) * 1e5
        print(
            f"  [flux_plot] epoch {epoch}  sample {idx}  "
            f"k_ref={k_ref:.5f}  k_pred={k_pred:.5f}  |Δρ|={dr:.1f} pcm  → {save_path}"
        )

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: Training
# ─────────────────────────────────────────────────────────────────────────────
def train(filepath, train_size, val_size, test_size, batch_size, epochs, lr_max, lr_min,
          hidden_sizes, n_regions, G, seed):

    init_csv_logs()
        
    # ── 8.1 Data ─────────────────────────────────────────────────────────────
    USE_BALANCED_RANGES = False
    print("Loading data …")
    if USE_BALANCED_RANGES: 
        (train_geoms, train_keffs, train_rawparams, train_phi_features), \
        (val_geoms,  val_keffs,  val_rawparams, val_phi_features) = load_data_balanced_ranges(
            filepath, train_size, val_size, seed,
        )
    else:    
        split_cache_dir = os.path.join(PARENT_DIR, "data", "highfidelity", ".split_cache")
        (train_idx, train_geoms, train_keffs, train_rawparams, train_phi_features), \
        (val_idx, val_geoms,   val_keffs,   val_rawparams,   val_phi_features),  \
        (test_idx, test_geoms,  test_keffs,  test_rawparams,  test_phi_features) = load_data(
            filepath, train_size, val_size, test_size,
            train_seed=seed, holdout_seed=HOLDOUT_SEED,
            split_cache_path=None, cache_dir=split_cache_dir)
    train_size = len(train_geoms)
    val_size   = len(val_geoms)
    test_size  = len(test_geoms)
    log_splits(train_idx, train_keffs, val_idx, val_keffs, test_idx, test_keffs)

    print(f"Actual train size: {train_size}")
    print(f"Actual test size:  {val_size}")
    print_keff_bin_counts("TRAIN", train_keffs)
    print_keff_bin_counts("VAL", val_keffs)
    print_keff_bin_counts("TEST", test_keffs)

    # NEW: normalise phi features per-column (each group/region combination separately)
    phi_mean = train_phi_features.mean(axis=0)         # shape [G*3]
    phi_std  = train_phi_features.std(axis=0) + 1e-8   # shape [G*3]
    train_phi_norm = ((train_phi_features - phi_mean) / phi_std).astype(np.float32)
    val_phi_norm  = ((val_phi_features  - phi_mean) / phi_std).astype(np.float32)
    test_phi_norm = ((test_phi_features - phi_mean) / phi_std).astype(np.float32)   # add this

    n_phi_feats = GEO.G * 3
    # ── 8.2 Model — v1: plain Adam, constant LR, NO grad clip ────────────────
    rngs      = nnx.Rngs(seed)
    model     = PEDSModel(hidden_sizes=hidden_sizes, n_regions=n_regions, G=G, n_phi_feats=n_phi_feats, rngs=rngs)
    # cosine decay schedule + grads clipping
    lr_schedule = optax.cosine_decay_schedule(
        init_value=lr_max,
        decay_steps=epochs * (train_size // batch_size),
        alpha=lr_min / lr_max,   # final lr = lr_max * alpha = lr_min
    )
    optimizer = nnx.Optimizer(
        model,
        optax.chain(
            optax.clip_by_global_norm(0.5),   # clip before Adam sees the gradient
            optax.adam(lr_schedule),
        ),
        wrt=nnx.Param,
    )
    RESUME_FROM = None  # set to LAST_CKPT_PATH (or any saved path) to resume training
    if RESUME_FROM is not None:
        state, meta = load_checkpoint(RESUME_FROM)
        nnx.update(optimizer, state)
        print(f"Resumed from {RESUME_FROM}: epoch={meta.get('epoch')}, "
            f"val_mean_pcm={meta.get('val_mean_pcm')}")

    # ── 8.3 Loss — pure MSE, NO regularization ───────────────────────────────
    def loss_fn(model, geoms, keffs_true, rawparams_batch,
                xs_baselines_batch, phi_norm_batch,
                log_xs_mean_j, log_xs_std_j):
        with timer("loss_fn forward pass", verbose=False):
            keff_pred, _, log_ratios = model(
                geoms, rawparams_batch, xs_baselines_batch,
                phi_norm_batch, training=True,  log_xs_mean=log_xs_mean_j, log_xs_std=log_xs_std_j,
            )
        loss = jnp.mean((keff_pred - keffs_true) ** 2)
        return loss, keff_pred

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)

    # ── 8.4 Pre-compute XS baselines once ────────────────────────────────────
    print("Precomputing XS baselines …")
    train_xs_baselines = compute_batch_baselines(train_rawparams, GEO)
    val_xs_baselines  = compute_batch_baselines(val_rawparams,  GEO)
    print("Done.")
     # ── Compute log-baseline normalization stats (fit on train only) ──────────
    # We take log of the baselines the same way _log_baselines() does:
    # zeros stay at 0 (log(1)=0), everything else gets log applied.
    _train_xs_np = np.array(train_xs_baselines)  # [N, 3, 12]
    _safe = np.where(_train_xs_np > 1e-10, _train_xs_np, np.ones_like(_train_xs_np))
    _log_train = np.log(_safe)                   # [N, 3, 12]
    log_xs_mean = _log_train.mean(axis=0).astype(np.float32)   # [3, 12]
    log_xs_std  = (_log_train.std(axis=0) + 1e-8).astype(np.float32)  # [3, 12]
    
    # Zero-valued XS slots (like chi_g2, nuSf in CR) have zero variance —
    # their std is ~1e-8, so normalization leaves them at 0. That's correct.
    print(f"  log_xs_mean range: [{log_xs_mean.min():.2f}, {log_xs_mean.max():.2f}]")
    print(f"  log_xs_std  range: [{log_xs_std.min():.4f},  {log_xs_std.max():.2f}]")
    # Convert to JAX arrays once — reused every batch
    log_xs_mean_j = jnp.array(log_xs_mean)
    log_xs_std_j  = jnp.array(log_xs_std)

    # ── 8.5 History arrays ───────────────────────────────────────────────────
    history = {k: [] for k in
               ["train_mse","val_mse", "train_mae","val_mae","val_mean_pcm","val_median_pcm",
                "val_p95_pcm","val_std_pcm","val_frac_below_650"]}
    
    best_val_mean_pcm = float("inf")
    best_epoch = 0

    print(f"\nTraining for {epochs} epochs …")
    epoch_grad_norms = []

    # ── Epoch 0: true pre-training baseline ──────────────────
    print("Computing epoch 0 baseline for train set...")
    kp_train_baseline = []
    for i in range(len(train_rawparams)):
        xs_i = np.array(predict_xs(update_geo(GEO, train_rawparams[i])), dtype=np.float32)
        k, _, _, _ = _run_NT_solver(xs_i, train_rawparams[i], np.array([i], dtype=np.int32))
        kp_train_baseline.append(float(k))

    print("Computing epoch 0 baseline for val/test set...")
    kp_val_baseline = []
    for i in range(len(val_rawparams)):
        xs_i = np.array(predict_xs(update_geo(GEO, val_rawparams[i])), dtype=np.float32)
        k, _, _, _ = _run_NT_solver(xs_i, val_rawparams[i], np.array([i], dtype=np.int32))
        kp_val_baseline.append(float(k))    

    log_keff_batch(train_writer, train_logfile, epoch=0,
                k_pred=np.array(kp_train_baseline),
                k_ref=train_keffs,
                avg_train_loss=0.0, avg_val_loss=0.0)
    log_keff_batch(val_writer, val_logfile, epoch=0,
                k_pred=np.array(kp_val_baseline),
                k_ref=val_keffs,
                avg_train_loss=0.0, avg_val_loss=0.0)

    log_xs_history_samples(
        model=model,
        epoch=0,
        sample_indices=TRACKED_SAMPLES,
        geom_sample_all=train_geoms,          
        rawparams_sample_all=train_rawparams,
        xs_baselines_sample_all=train_xs_baselines,
        phi_norm_sample_all=train_phi_norm,
        log_xs_mean_j=log_xs_mean_j,
        log_xs_std_j=log_xs_std_j,
    )
    # EMA smoothing for checkpoint selection.
    # alpha=0.2 → time-constant ~5 epochs; small enough to damp transient spikes
    # (e.g. the ~+75% spike seen at epoch 17 in previous runs) while still being
    # responsive enough to track genuine improvement.
    # Initialization: run epoch 1 first, then seed the EMA with that real value so
    # the warm-up bias from an arbitrary starting point is entirely avoided.
    EMA_ALPHA     = 0.15
    MIN_SAVE_EPOCH = 20   # don't save before the model has passed early oscillations
    ema_val_mean_pcm = None   # will be set at end of epoch 1

    for epoch in range(1, epochs + 1):
        # ── shuffle ──────────────────────────────────────────────────────────
        idx = np.random.permutation(train_size)
        t_geoms   = train_geoms[idx]
        t_keffs   = train_keffs[idx]
        t_raw     = train_rawparams[idx]
        t_base    = np.array(train_xs_baselines)[idx]
        t_phi     = train_phi_norm[idx]
        t_sample_ids = np.arange(train_size)[idx]  # ← original sample IDs in shuffled order
        current_lr = float(lr_schedule(optimizer.step.value))
        print(f"Epoch {epoch:4d} | LR = {current_lr:.2e}")

        epoch_loss = 0.0
        all_kp_train, all_kr_train, all_sample_ids  = [], [], []

        for batch_idx, (bg, bk, br, bb, bphi, bsid) in enumerate(data_loader(
                t_geoms, t_keffs, t_raw, t_base, t_phi, t_sample_ids, batch_size=batch_size)):

            bp   = jnp.array(bg)
            bkj  = jnp.array(bk)
            bxs  = jnp.array(bb)
            b_phi_j = jnp.array(bphi)

            # Step 1: concrete XS from NN (outside JAX trace)
            xs_np = model.compute_xs(bp, bxs, b_phi_j, log_xs_mean_j, log_xs_std_j)

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
                    model, bp, bkj, br, bxs, b_phi_j,
                    log_xs_mean_j, log_xs_std_j,
                )
            grad_norm = float(optax.global_norm(grads))
            epoch_grad_norms.append(grad_norm)
            optimizer.update(model, grads)

            epoch_loss    += float(loss)
            all_kp_train.extend(np.array(keff_preds_batch).tolist())
            all_kr_train.extend(bk.tolist())
            all_sample_ids.extend(bsid.tolist())

        # After all batches — one row per sample, sorted by dataset index
        ids = np.asarray(all_sample_ids, dtype=int)
        order = np.argsort(ids)
        log_keff_batch(
            train_writer, train_logfile, epoch,
            np.array(all_kp_train)[order],
            np.array(all_kr_train)[order],
            epoch_loss / max(1, (train_size + batch_size - 1) // batch_size),
            0.0,
            sample_id_offset=0,
            sample_ids_override=ids[order],
        )
        
        log_xs_history_samples(
            model=model,
            epoch=epoch,
            sample_indices=TRACKED_SAMPLES,
            geom_sample_all=train_geoms,
            rawparams_sample_all=train_rawparams,
            xs_baselines_sample_all=train_xs_baselines,
            phi_norm_sample_all=train_phi_norm,
            log_xs_mean_j=log_xs_mean_j,
            log_xs_std_j=log_xs_std_j,
        )

        # ── validation ───────────────────────────────────────────────────────
        gn = np.array(epoch_grad_norms)
        print(f"  grad_norm: min={gn.min():.3f} mean={gn.mean():.3f} max={gn.max():.3f} "
        f"frac_clipped(>0.5)={np.mean(gn > 0.5):.2%}")
        
        _PRESOLVE_CACHE.clear()
        all_kp_val, all_kr_val = [], []
        with timer("validation step"):
            for batch_idx, (bg, bk, br, bb, bphi) in enumerate(data_loader(
                    val_geoms, val_keffs, val_rawparams,
                    np.array(val_xs_baselines), val_phi_norm,
                    batch_size=batch_size)):

                global_start = batch_idx * batch_size
                xs_np = model.compute_xs(
                    jnp.array(bg), jnp.array(bb),
                    jnp.array(bphi), log_xs_mean_j, log_xs_std_j,
                )
                VAL_OFFSET = train_size + 10000
                args_list = [(i, xs_np[i], np.array(br[i]),
                            VAL_OFFSET + i + global_start, -1)
                            for i in range(len(bg))]
                results = list(executor.map(_solve_sample_worker, args_list))
                for i, k, *_ in results:
                    all_kp_val.append(float(k))
                    all_kr_val.append(float(bk[i]))

        # ── log_ratios saturation diagnostic (every epoch, val set) ─────────
        val_log_ratios = model.compute_log_ratios(
            jnp.array(val_geoms), jnp.array(val_xs_baselines),
            jnp.array(val_phi_norm),
            log_xs_mean_j, log_xs_std_j,
        )
        log_logratio_saturation(epoch, val_log_ratios)

        kp_val = np.array(all_kp_val)
        kr_val = np.array(all_kr_val)
        kp_tr  = np.array(all_kp_train)
        kr_tr  = np.array(all_kr_train)

        # ── metrics ─────────────────────────────────────────────────────────
        train_m = compute_metrics(kp_tr, kr_tr)
        val_m   = compute_metrics(kp_val, kr_val)
        print_metrics(epoch, "TRAIN", train_m)
        print_metrics(epoch, "VAL",   val_m)
        if ema_val_mean_pcm is None:
            ema_val_mean_pcm = val_m["mean_pcm"]   # seed with first real observation
        else:
            ema_val_mean_pcm = EMA_ALPHA * val_m["mean_pcm"] + (1 - EMA_ALPHA) * ema_val_mean_pcm
        if epoch >= MIN_SAVE_EPOCH and ema_val_mean_pcm < best_val_mean_pcm:
            best_val_mean_pcm = ema_val_mean_pcm
            best_epoch = epoch
            save_checkpoint(nnx.state(model), BEST_CKPT_PATH,
                            metadata={"epoch": epoch, "val_mean_pcm": best_val_mean_pcm,
                                    "hidden_sizes": hidden_sizes, "n_regions": n_regions,
                                    "G": G, "n_phi_feats": n_phi_feats})
            print(f"  ✓ new best val_mean_pcm = {best_val_mean_pcm:.1f} pcm "
                f"(epoch {epoch}) — saved → {BEST_CKPT_PATH}")

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
                train_phi_norm[:8],
                log_xs_mean_j, log_xs_std_j,
                epoch=epoch,
                n_show=8,
            )
            _save_xs_subplots_for_samples(
                model            = model,
                sample_indices   = SUBPLOT_SAMPLE_INDICES,
                geoms_all        = val_geoms,              # ← unshuffled
                keffs_all        = val_keffs,              # ← unshuffled
                rawparams_all    = val_rawparams,          # ← unshuffled
                xs_baselines_all = np.array(val_xs_baselines),  # ← unshuffled
                phi_norm_all     = val_phi_norm,   
                log_xs_mean_j    = log_xs_mean_j,
                log_xs_std_j    = log_xs_std_j,
                epoch            = epoch,
            )
            _save_flux_plots(                         
                sample_indices      = SUBPLOT_SAMPLE_INDICES,
                val_geoms         = val_geoms,
                val_keffs         = val_keffs,
                val_rawparams     = val_rawparams,
                val_xs_baselines  = np.array(val_xs_baselines),
                log_xs_mean_j      = log_xs_mean_j,
                log_xs_std_j      = log_xs_std_j,
                val_phi_norm      = val_phi_norm,
                model               = model,
                epoch               = epoch,
            )

        if epoch % 20 == 0:
            jax.clear_caches()

    # end of the epoch loop

    save_checkpoint(nnx.state(optimizer), LAST_CKPT_PATH,
                    metadata={"epoch": epochs, "val_mean_pcm": val_m["mean_pcm"]})
    print(f"Saved last-epoch checkpoint (model+optimizer) → {LAST_CKPT_PATH}")
    print(f"Best epoch was {best_epoch}  (val_mean_pcm = {best_val_mean_pcm:.1f} pcm)  → {BEST_CKPT_PATH}")

    # --- Save final-epoch XS for all train and val samples ---
    save_final_xs_csv(
        model, train_geoms, train_rawparams, np.array(train_xs_baselines),
        train_phi_norm, train_keffs,
        log_xs_mean_j, log_xs_std_j,
        file_path=os.path.join(XS_DIR, "final_xs_train.csv"),
        tag="train"
    )
    save_final_xs_csv(
        model, val_geoms, val_rawparams, np.array(val_xs_baselines),
        val_phi_norm, val_keffs,
        log_xs_mean_j, log_xs_std_j,
        file_path=os.path.join(XS_DIR, "final_xs_val.csv"),
        tag="val"
    )

    print("\n=== Final evaluation on held-out TEST set (never seen during training) ===")
    test_xs_baselines = compute_batch_baselines(test_rawparams, GEO)
    all_kp_test, all_kr_test = [], []
    for bg, bk, br, bb, bphi in data_loader(
            test_geoms, test_keffs, test_rawparams,
            np.array(test_xs_baselines), test_phi_norm, batch_size=batch_size):
        xs_np = model.compute_xs(jnp.array(bg), jnp.array(bb), jnp.array(bphi),
                                log_xs_mean_j, log_xs_std_j)
        for i in range(len(bg)):
            k, _, _, _ = _run_NT_solver(xs_np[i], np.array(br[i]), np.array([i], dtype=np.int32))
            all_kp_test.append(float(k))
            all_kr_test.append(float(bk[i]))

    test_m = compute_metrics(np.array(all_kp_test), np.array(all_kr_test))
    print_metrics(epoch, "TEST", test_m)

    save_final_xs_csv(
        model, test_geoms, test_rawparams, np.array(test_xs_baselines),
        test_phi_norm, test_keffs, log_xs_mean_j, log_xs_std_j,
        file_path=os.path.join(XS_DIR, "test_final_xs.csv"), tag="test",
    )
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
    print(f"last epoch = {last_epoch}, type = {type(last_epoch)}")
    df_final = df[df['epoch'] == last_epoch]
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
        geom = val_geoms[idx]
        raw  = val_rawparams[idx]
        k_ref = val_keffs[idx]
        delta = float(df_final[df_final['sample_idx']==idx]['delta_rho_pcm'].values[0])
        print(f"\n  Sample {idx} | k_ref={k_ref:.4f} | delta_rho={delta:.1f} pcm")
        for i, name in enumerate(feat_names):
            print(f"    {name}: normalized={geom[i]:.3f}  raw={raw[i]:.4f}")

    # ── final report ─────────────────────────────────────────────────────────
    _plot_history(history, EXP_NAME)
    print_timing_report()

    close_csv_logs()
    return model, history    


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
class _Tee:
    """Write to multiple streams (e.g. sbatch stdout + train log file)."""

    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for fh in self.files:
            fh.write(data)
            fh.flush()

    def flush(self):
        for fh in self.files:
            fh.flush()




if __name__ == "__main__":
    _orig_stdout = sys.stdout
    _orig_stderr = sys.stderr
    loggy_file = open(os.path.join(LOG_DIR, f"train_log_{EXP_NAME}.txt"), "w", buffering=1)
    snapshot_path, code_hash = save_code_snapshot(LOG_DIR, EXP_NAME)
    sys.stdout = _Tee(_orig_stdout, loggy_file)
    sys.stderr = _Tee(_orig_stderr, loggy_file)
    start_dt = datetime.now()
    start_perf = time.time()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Started at   : {now_str}")
    print(f"Run log      : {loggy_file}")
    print(f"Code copy    : {snapshot_path}")
    print(f"Code SHA256  : {code_hash}")
    
    executor = ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=ctx)

    try:
        print(f"N_FLAT_MAX = {N_FLAT_MAX}")
        HP = dict(
            filepath    = _DATA_FILEPATH,   # uses anchored path, not hard-coded string
            train_size  = TRAIN_SIZE,
            val_size    = VAL_SIZE,
            test_size   = TEST_SIZE,
            batch_size  = BATCH_SIZE,
            epochs      = EPOCHS,
            lr_max        = LR_max,   # cosine schedule peak learning rate
            lr_min        = LR_min,        
            hidden_sizes= [128, 256, 128],
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

    except Exception:
        traceback.print_exc(file=loggy_file)
        traceback.print_exc(file=_orig_stderr)
        raise

    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        end_dt = datetime.now()
        elapsed_sec = time.time() - start_perf

        print(f"Finished at  : {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Elapsed time : {elapsed_sec/3600:.2f} hours")
        sys.stdout = _orig_stdout
        sys.stderr = _orig_stderr
        loggy_file.close()