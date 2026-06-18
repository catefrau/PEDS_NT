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
from contextlib import redirect_stdout, redirect_stderr

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
from PEDS_core.logs_plots import ( configure_training_helpers,
    init_csv_logs, close_csv_logs, log_epoch_stats, log_keff_batch,
    log_logratio_saturation, save_final_xs_csv, save_code_snapshot,
    _save_xs_subplots_for_samples, _plot_xs_heatmap, _plot_history, _save_flux_plots, )
from PEDS_core.fwd_bwd_vjp import (
    update_geo, _run_NT_solver, NTdiff_solver, _solve_sample_worker )

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0: Global constants
# ─────────────────────────────────────────────────────────────────────────────
EXP_NAME   = "v7_split"
TRAIN_SIZE = 800
TEST_SIZE  = 200
BATCH_SIZE = 50
EPOCHS     = 3

LR_max     = 5e-4   # cosine schedule peak learning rate
LR_min     = 5e-6
SEED       = 42

LOG_RATIO_CLIP_LO = -1.8
LOG_RATIO_CLIP_HI = 0.8

# Epochs at which XS heatmap snapshots are saved.
# Add/remove values here to control checkpointing granularity.
XS_HEATMAP_EPOCHS = {1, 5, 10, 20, 50, 100}

# Sample indices (into the TRAINING set) for per-sample xs_subplots figures.
# These are saved at the same epochs as XS_HEATMAP_EPOCHS.
SUBPLOT_SAMPLE_INDICES: list = [0, 1, 2, 3, 4]

# Human-readable names for the 6 raw geometry parameters (used in info box).
PARAM_NAMES: list = ['b4c_r', 'cr_frac', 'fuel_r', 'enrichment', 'f_mod', 'water_r']

# ── data path — anchored to PARENT_DIR so it works from any cwd ─────────────
_DATA_FILEPATH = os.path.join(PARENT_DIR, "data", "highfidelity", "1000_clean.npz")

N_WORKERS = min(int(os.environ.get("SLURM_CPUS_PER_TASK", 16)), BATCH_SIZE)
print(f"Using {N_WORKERS} parallel workers")

executor = None

# ── output directory: everything for this run lives under LOG_DIR ─────────────
LOG_DIR = os.path.join(THIS_DIR, "LOGS", EXP_NAME)
os.makedirs(LOG_DIR, exist_ok=True)
run_log_path = os.path.join(LOG_DIR, f"train_log_{EXP_NAME}.txt")

# ── precomputed geometry constants ───────────────────────────────────────────
GEO_DATA = precompute_geometry(GEO)
SLAY     = xs_layout(GEO.G)
_PRESOLVE_CACHE: dict = {}

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
        k_reg_norm[0]  — normalised (k_openmc - k_reg)

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
                 k_reg_norm: jnp.ndarray, phi_norm: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """
        geoms:            [batch, 6]
        xs_baselines_log: [batch, n_regions, xs_per_region]
        k_reg_norm:     [batch, 1]
        phi_norm:       [batch, n_phi_feats]
        Returns:          [batch, n_regions, xs_per_region]  log-ratio offsets
        """
        x    = jnp.concatenate([geoms, k_reg_norm, phi_norm], axis=-1)  # [batch, 7]
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
        input_dim   = 6 + 1 + n_phi_feats   # geom + k_reg + phi = 13
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
        safe = jnp.where(xs_baselines > 1e-10, xs_baselines, jnp.ones_like(xs_baselines))
        log_xs = jnp.log(safe)
        if log_xs_mean is not None and log_xs_std is not None:
            log_xs = (log_xs - log_xs_mean) / log_xs_std
        return log_xs

    def compute_xs(self, geoms, xs_baselines, k_reg_norm, phi_norm,
                             log_xs_mean=None, log_xs_std=None):
        """Pure NN forward (no grad). Used for pre-solving and validation."""
        batch_size = geoms.shape[0]
        k_reg  = jnp.reshape(jnp.array(k_reg_norm, dtype=jnp.float32), (batch_size, 1))
        with timer("log baselines from xs first guess", verbose=False): 
            log_base    = self._log_baselines(xs_baselines,  log_xs_mean, log_xs_std)
        with timer("NN generated XS log ratios", verbose=False):
            log_ratios  = self.generator(geoms, log_base, k_reg, phi_norm, training=False)
        # ── v1: NO clip, NO warmup ─────────────────────────────────────────
        log_ratios = clip_ste(log_ratios, LOG_RATIO_CLIP_LO, LOG_RATIO_CLIP_HI)  # now consistent
        xs_final = jnp.exp(log_ratios) * xs_baselines
        return np.array(xs_final)  # concrete numpy, exits JAX world
        
    def compute_log_ratios(self, geoms, xs_baselines, k_reg_norm, phi_norm,
                log_xs_mean=None, log_xs_std=None):
            """Pure NN forward, returns CLIPPED log_ratios (no solver). For diagnostics."""
            batch_size = geoms.shape[0]
            k_reg = jnp.reshape(jnp.array(k_reg_norm, dtype=jnp.float32), (batch_size, 1))
            log_base = self._log_baselines(xs_baselines, log_xs_mean, log_xs_std)
            log_ratios = self.generator(geoms, log_base, k_reg, phi_norm, training=False)
            log_ratios = clip_ste(log_ratios, LOG_RATIO_CLIP_LO, LOG_RATIO_CLIP_HI)
            return np.array(log_ratios)
    
    def __call__(self, geoms, params_raw, xs_baselines, k_reg_norm, phi_norm,
                 training: bool = False, sample_id_offset: int = 0,
                 log_xs_mean=None, log_xs_std=None):
        batch_size = geoms.shape[0]
        k_reg  = jnp.reshape(k_reg_norm, (batch_size, 1))
        log_base   = self._log_baselines(xs_baselines, log_xs_mean, log_xs_std)
        phi = jnp.reshape(phi_norm, (batch_size, -1))
        log_ratios = self.generator(geoms, log_base, k_reg, phi, training)
        log_ratios = clip_ste(log_ratios, LOG_RATIO_CLIP_LO, LOG_RATIO_CLIP_HI)  # exp(±0.7) ≈ 0.5x to 2x

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

def compute_k_reg_and_phi_features(rawparams: np.ndarray) -> tuple:
    """
    Returns:
        k_regs:       [N]       — baseline k-effective
        phi_features: [N, G*3]  — volume-weighted mean flux per group per region
                                  order: [phi_g0_CR, phi_g0_Core, phi_g0_Mod,
                                          phi_g1_CR, phi_g1_Core, phi_g1_Mod]
    """
    k_regs       = []
    phi_features = []

    for i, p in enumerate(rawparams):
        geo_i   = update_geo(GEO, p)
        xs_i    = np.array(predict_xs(geo_i), dtype=np.float32)
        k, phi_fwd_padded, _, _ = _run_NT_solver(xs_i, p, np.array([i], dtype=np.int32))
        k_regs.append(float(k))

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

    return np.array(k_regs, dtype=np.float32), np.array(phi_features, dtype=np.float32)
    

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

    print("Precomputing k_reg and φ_reg features (done once)…")
    train_k_reg, train_phi_features = compute_k_reg_and_phi_features(rawparams[train_idx])
    test_k_reg,  test_phi_features  = compute_k_reg_and_phi_features(rawparams[test_idx])

    return (
        (geoms[train_idx], keffs[train_idx], rawparams[train_idx],
        train_k_reg, train_phi_features),
        (geoms[test_idx],  keffs[test_idx],  rawparams[test_idx],
        test_k_reg,  test_phi_features),
    )

def load_data_balanced_ranges(filepath, train_size, test_size, seed=2,
                               bin_edges=None, train_per_bin=50, test_per_bin=10):
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

    if train_size % nbins != 0 or test_size % nbins != 0:
        raise ValueError(
            f"train_size={train_size} and test_size={test_size} must both be "
            f"divisible by nbins={nbins}"
        )

    train_per_bin = train_size // nbins   # 500 // 10 = 50
    test_per_bin  = test_size  // nbins   # 100 // 10 = 10

    trainidx, testidx = [], []

    print("\n=== Balanced fixed-range split ===")
    print(f"train_per_bin = {train_per_bin}, test_per_bin = {test_per_bin}")

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

        needed = train_per_bin + test_per_bin
        if len(idx) < needed:
            raise ValueError(
                f"Bin {label} has only {len(idx)} samples, but needs {needed}."
            )

        test_bin = idx[:test_per_bin]
        train_bin = idx[test_per_bin:test_per_bin + train_per_bin]

        testidx.extend(test_bin.tolist())
        trainidx.extend(train_bin.tolist())

        print(f"{label}: available={len(idx):3d}, train={len(train_bin):2d}, test={len(test_bin):2d}")

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
    trainkreg, trainphifeatures = compute_k_reg_and_phi_features(rawparams[trainidx])
    testkreg, testphifeatures = compute_k_reg_and_phi_features(rawparams[testidx])

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
# SECTION 8: Training
# ─────────────────────────────────────────────────────────────────────────────
def train(filepath, train_size, test_size, batch_size, epochs, lr_max, lr_min,
          hidden_sizes, n_regions, G, seed):
    
    configure_training_helpers(
        log_dir=LOG_DIR,
        param_names=PARAM_NAMES,
        log_ratio_clip_lo=LOG_RATIO_CLIP_LO,
        log_ratio_clip_hi=LOG_RATIO_CLIP_HI,
        geo=GEO,
    )

    init_csv_logs()
        
    # ── 8.1 Data ─────────────────────────────────────────────────────────────
    USE_BALANCED_RANGES = False
    print("Loading data …")
    if USE_BALANCED_RANGES: 
        (train_geoms, train_keffs, train_rawparams, train_k_reg, train_phi_features), \
        (test_geoms,  test_keffs,  test_rawparams,  test_k_reg, test_phi_features) = load_data_balanced_ranges(
            filepath, train_size, test_size, seed,
        )
    else:    
        (train_geoms, train_keffs, train_rawparams, train_k_reg, train_phi_features), \
        (test_geoms,  test_keffs,  test_rawparams,  test_k_reg, test_phi_features) = load_data(
            filepath, train_size, test_size, seed,
        )
    train_size = len(train_geoms)
    test_size = len(test_geoms)

    print(f"Actual train size: {train_size}")
    print(f"Actual test size:  {test_size}")
    print_keff_bin_counts("TRAIN", train_keffs)
    print_keff_bin_counts("TEST", test_keffs)

    # use k reg instead of delta k to be able to do inference
    k_reg_mean = float(train_k_reg.mean())
    k_reg_std  = float(train_k_reg.std()) + 1e-8
    train_k_reg_norm = ((train_k_reg - k_reg_mean) / k_reg_std).astype(np.float32)
    test_k_reg_norm  = ((test_k_reg  - k_reg_mean) / k_reg_std).astype(np.float32)

    # NEW: normalise phi features per-column (each group/region combination separately)
    phi_mean = train_phi_features.mean(axis=0)         # shape [G*3]
    phi_std  = train_phi_features.std(axis=0) + 1e-8   # shape [G*3]
    train_phi_norm = ((train_phi_features - phi_mean) / phi_std).astype(np.float32)
    test_phi_norm  = ((test_phi_features  - phi_mean) / phi_std).astype(np.float32)

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

    # ── 8.3 Loss — pure MSE, NO regularization ───────────────────────────────
    def loss_fn(model, geoms, keffs_true, rawparams_batch,
                xs_baselines_batch, k_reg_norm_batch, phi_norm_batch,
                log_xs_mean_j, log_xs_std_j):
        with timer("loss_fn forward pass", verbose=False):
            keff_pred, _, log_ratios = model(
                geoms, rawparams_batch, xs_baselines_batch, k_reg_norm_batch,
                phi_norm_batch, training=True,  log_xs_mean=log_xs_mean_j, log_xs_std=log_xs_std_j,
            )
        loss = jnp.mean((keff_pred - keffs_true) ** 2)
        return loss, keff_pred

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)

    # ── 8.4 Pre-compute XS baselines once ────────────────────────────────────
    print("Precomputing XS baselines …")
    train_xs_baselines = compute_batch_baselines(train_rawparams, GEO)
    test_xs_baselines  = compute_batch_baselines(test_rawparams,  GEO)
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
    for i in range(len(test_rawparams)):
        xs_i = np.array(predict_xs(update_geo(GEO, test_rawparams[i])), dtype=np.float32)
        k, _, _, _ = _run_NT_solver(xs_i, test_rawparams[i], np.array([i], dtype=np.int32))
        kp_val_baseline.append(float(k))    

    log_keff_batch(
        "train",
        epoch=0,
        k_pred=np.array(kp_train_baseline),
        k_ref=train_keffs,
        avg_train_loss=0.0,
        avg_val_loss=0.0,
    )
    log_keff_batch(
        "val",
        epoch=0,
        k_pred=np.array(kp_val_baseline),
        k_ref=test_keffs,
        avg_train_loss=0.0,
        avg_val_loss=0.0,
    )
                
    for epoch in range(1, epochs + 1):
        # ── shuffle ──────────────────────────────────────────────────────────
        idx = np.random.permutation(train_size)
        t_geoms   = train_geoms[idx]
        t_keffs   = train_keffs[idx]
        t_raw     = train_rawparams[idx]
        t_base    = np.array(train_xs_baselines)[idx]
        t_dk      = train_k_reg_norm[idx]
        t_phi     = train_phi_norm[idx]
        current_lr = float(lr_schedule(optimizer.step.value))
        print(f"Epoch {epoch:4d} | LR = {current_lr:.2e}")

        epoch_loss = 0.0
        all_kp_train, all_kr_train = [], []

        for batch_idx, (bg, bk, br, bb, bdk, bphi) in enumerate(data_loader(
                t_geoms, t_keffs, t_raw, t_base, t_dk, t_phi, batch_size=batch_size)):

            bp   = jnp.array(bg)
            bkj  = jnp.array(bk)
            bxs  = jnp.array(bb)
            bdkj = jnp.array(bdk)
            b_phi_j = jnp.array(bphi)

            # Step 1: concrete XS from NN (outside JAX trace)
            xs_np = model.compute_xs(bp, bxs, bdkj, b_phi_j, log_xs_mean_j, log_xs_std_j)

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
                    model, bp, bkj, br, bxs, bdkj, b_phi_j,
                    log_xs_mean_j, log_xs_std_j,
                )
            grad_norm = float(optax.global_norm(grads))
            epoch_grad_norms.append(grad_norm)
            optimizer.update(model, grads)

            epoch_loss    += float(loss)
            all_kp_train.extend(np.array(keff_preds_batch).tolist())
            all_kr_train.extend(bk.tolist())

            log_keff_batch("train", epoch,
                np.array(keff_preds_batch), bk, float(loss), 0.0,
                sample_id_offset=batch_idx * batch_size,
            )

        # ── validation ───────────────────────────────────────────────────────
        gn = np.array(epoch_grad_norms)
        print(f"  grad_norm: min={gn.min():.3f} mean={gn.mean():.3f} max={gn.max():.3f} "
        f"frac_clipped(>0.5)={np.mean(gn > 0.5):.2%}")
        
        _PRESOLVE_CACHE.clear()
        all_kp_val, all_kr_val = [], []
        with timer("validation step"):
            for batch_idx, (bg, bk, br, bb, bdk, bphi) in enumerate(data_loader(
                    test_geoms, test_keffs, test_rawparams,
                    np.array(test_xs_baselines), test_k_reg_norm, test_phi_norm,
                    batch_size=batch_size)):

                global_start = batch_idx * batch_size
                xs_np = model.compute_xs(
                    jnp.array(bg), jnp.array(bb), jnp.array(bdk), 
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
            jnp.array(test_geoms), jnp.array(test_xs_baselines),
            jnp.array(test_k_reg_norm), jnp.array(test_phi_norm),
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

        # ── per-sample CSV ───────────────────────────────────────────────────
        log_keff_batch("val", epoch, kp_val, kr_val,
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
                train_k_reg_norm[:8], 
                train_phi_norm[:8],
                log_xs_mean_j, log_xs_std_j,
                epoch=epoch,
                n_show=8,
            )
            _save_xs_subplots_for_samples(
                model            = model,
                sample_indices   = SUBPLOT_SAMPLE_INDICES,
                geoms_all        = test_geoms,              # ← unshuffled
                keffs_all        = test_keffs,              # ← unshuffled
                rawparams_all    = test_rawparams,          # ← unshuffled
                xs_baselines_all = np.array(test_xs_baselines),  # ← unshuffled
                k_reg_norm_all   = test_k_reg_norm,       # ← unshuffled
                phi_norm_all     = test_phi_norm,   
                log_xs_mean_j    = log_xs_mean_j,
                log_xs_std_j    = log_xs_std_j,
                epoch            = epoch,
            )
            _save_flux_plots(                         
                sample_indices      = SUBPLOT_SAMPLE_INDICES,
                test_geoms         = test_geoms,
                test_keffs         = test_keffs,
                test_rawparams     = test_rawparams,
                test_xs_baselines  = np.array(test_xs_baselines),
                test_k_reg_norm  = test_k_reg_norm,
                test_phi_norm      = test_phi_norm,
                model               = model,
                epoch               = epoch,
            )

        if epoch % 20 == 0:
            jax.clear_caches()

    # end of the epoch loop

    # --- Save final-epoch XS for all train and val samples ---
    save_final_xs_csv(
        model, train_geoms, train_rawparams, np.array(train_xs_baselines),
        train_k_reg_norm, train_phi_norm, train_keffs,
        log_xs_mean_j, log_xs_std_j,
        file_path=os.path.join(LOG_DIR, "final_xs_train.csv"),
        tag="train"
    )
    save_final_xs_csv(
        model, test_geoms, test_rawparams, np.array(test_xs_baselines),
        test_k_reg_norm, test_phi_norm, test_keffs,
        log_xs_mean_j, log_xs_std_j,
        file_path=os.path.join(LOG_DIR, "final_xs_val.csv"),
        tag="val"
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
        geom = test_geoms[idx]
        raw  = test_rawparams[idx]
        k_ref = test_keffs[idx]
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
if __name__ == "__main__":
    with open(run_log_path, "w", buffering=1) as run_log:
        with redirect_stdout(run_log), redirect_stderr(run_log):
            print(f"Run log      : {run_log_path}")
            try:
                snapshot_path, code_hash = save_code_snapshot(LOG_DIR, EXP_NAME, __file__)
                print(f"Code copy    : {snapshot_path}")
                print(f"Code SHA256  : {code_hash}")
    
                executor = ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=ctx)
                HP = dict(
                    filepath    = _DATA_FILEPATH,   # uses anchored path, not hard-coded string
                    train_size  = TRAIN_SIZE,
                    test_size   = TEST_SIZE,
                    batch_size  = BATCH_SIZE,
                    epochs      = EPOCHS,
                    lr_max        = LR_max,   # cosine schedule peak learning rate
                    lr_min        = LR_min,        
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
            except Exception:
                print("\n" + "=" * 100)
                print("UNCAUGHT EXCEPTION")
                traceback.print_exc()
                print("=" * 100)
                raise

            finally:
                executor.shutdown(wait=True, cancel_futures=True)
