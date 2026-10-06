import multiprocessing
import os

# ── anchor all paths relative to THIS script file, not the cwd ──────────────
# This means you can run the script from any directory, including from inside
# the output folder, without breaking imports or data paths.
THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)
CONFIG_RUN_DIR = os.path.join(PARENT_DIR, "config_and_run")

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
import glob

# ── import roots, so this runs from any cwd and from spawn workers ──────────
# - THIS_DIR       : model stack (PEDS_core, PEDS_subdivision, matrix_JAX_optimized)
# - CONFIG_RUN_DIR : config_peds, NTcode_config_data, plot_functions
# - PARENT_DIR     : repository root (solvers, data)
for _p in (THIS_DIR, CONFIG_RUN_DIR, PARENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

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
# SECTION 0: Run configuration
# ─────────────────────────────────────────────────────────────────────────────
# Every knob below is defined in config_and_run/config_peds.py. Edit that file
# (or set the PEDS_* environment variables it reads) rather than this one.
import config_peds as CFG

TRAIN_SIZE   = CFG.TRAIN_SIZE
VAL_SIZE     = CFG.VAL_SIZE
TEST_SIZE    = CFG.TEST_SIZE
SEED         = CFG.SEED
TRAIN_SEED   = CFG.TRAIN_SEED
VAL_SEED     = CFG.VAL_SEED
TEST_SEED    = CFG.TEST_SEED
HOLDOUT_SEED = CFG.HOLDOUT_SEED
BATCH_SIZE   = CFG.BATCH_SIZE
EPOCHS       = CFG.EPOCHS
LR_max       = CFG.LR_MAX
LR_min       = CFG.LR_MIN
DECAY_EPOCHS = CFG.DECAY_EPOCHS
EMA_ALPHA      = CFG.EMA_ALPHA
MIN_SAVE_EPOCH = CFG.MIN_SAVE_EPOCH
PATIENCE       = CFG.PATIENCE

HIDDEN_SIZES = CFG.HIDDEN_SIZES
N_REGIONS    = CFG.N_REGIONS
N_GROUPS     = CFG.N_GROUPS

EXP_NAME = CFG.EXP_NAME
LOG_DIR  = str(CFG.LOG_DIR)
_DATA_FILEPATH = str(CFG.DATA_FILEPATH)

USE_SPLIT_CACHE      = CFG.USE_SPLIT_CACHE
USE_BALANCED_RANGES  = CFG.USE_BALANCED_RANGES
USE_PARAM_STRATIFIED = CFG.USE_PARAM_STRATIFIED
PARAM_STRAT_BINS     = CFG.PARAM_STRAT_BINS
CACHE_DIR_NAME       = CFG.CACHE_DIR_NAME

RUN_GRAD_CHECK    = CFG.RUN_GRAD_CHECK
GRAD_CHECK_SAMPLE = CFG.GRAD_CHECK_SAMPLE

LOG_RATIO_CLIP_LO = CFG.LOG_RATIO_CLIP_LO
LOG_RATIO_CLIP_HI = CFG.LOG_RATIO_CLIP_HI

XS_HEATMAP_EPOCHS = CFG.XS_HEATMAP_EPOCHS
SUBPLOT_SAMPLE_INDICES: list = CFG.SUBPLOT_SAMPLE_INDICES
TRACKED_SAMPLES: list = CFG.TRACKED_SAMPLES
PARAM_NAMES: list = CFG.PARAM_NAMES

N_WORKERS = CFG.N_WORKERS
print(f"Using {N_WORKERS} parallel workers")


executor = None

# ── output directory: everything for this run lives under LOG_DIR ─────────────
XS_DIR = str(CFG.XS_DIR)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(XS_DIR, exist_ok=True)

# ── checkpointing ────────────────────────────────────────────────────────────
CHECKPOINT_DIR = str(CFG.CHECKPOINT_DIR)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
BEST_CKPT_PATH = str(CFG.BEST_CKPT_PATH)
LAST_CKPT_PATH = str(CFG.LAST_CKPT_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# Shared runtime context + moved implementation modules
# ─────────────────────────────────────────────────────────────────────────────
# The heavy/derived constants, caches and solver setup live in
# PEDS_subdivision.context; the implementation of the forward/backward calls,
# data loading, metrics, CSV logging and plotting live in the sibling modules.
# We fill the context here (at import scope, so spawn workers and snapshot
# reloads initialise it too) and re-import the moved symbols so the model,
# training loop, MAIN, and external importers keep referencing them directly.
from PEDS_subdivision import context
context.init(
    data_filepath     = _DATA_FILEPATH,
    log_dir           = LOG_DIR,
    xs_dir            = XS_DIR,
    param_names       = PARAM_NAMES,
    param_strat_bins  = PARAM_STRAT_BINS,
    log_ratio_clip_lo = LOG_RATIO_CLIP_LO,
    log_ratio_clip_hi = LOG_RATIO_CLIP_HI,
    batch_size        = BATCH_SIZE,
)

from PEDS_subdivision.context import update_geo
GEO_DATA        = context.GEO_DATA
SLAY            = context.SLAY
N_FLAT_MAX      = context.N_FLAT_MAX
XS_MASK         = context.XS_MASK
XS_MASK_J       = context.XS_MASK_J
_PRESOLVE_CACHE = context._PRESOLVE_CACHE     # same dict object; mutated in place
_GEO_DATA_CACHE = context._GEO_DATA_CACHE

from PEDS_subdivision.physics_solver import (
    _run_NT_solver, _NTdiff_fwd, _NTdiff_bwd, NTdiff_solver,
    _NT_batch_fwd, _NT_batch_bwd, NTdiff_solver_batch,
    _build_padded_geo_arrays, _solve_sample_worker,
)
from PEDS_subdivision.data_loading import (
    data_loader, compute_batch_baselines, compute_phi_features,
    derive_split_indices, derive_split_indices_by_params,
    _save_split_cache, load_or_create_split_cache,
    load_data, load_data_balanced_ranges, load_data_param_stratified,
    print_keff_bin_counts, print_param_bin_counts,
    check_keff_split_representation, check_keff_bin_geometry_dominance,
)
from PEDS_subdivision.metrics import (
    compute_delta_rho_pcm, compute_metrics, print_metrics,
)
from PEDS_subdivision import logging_csv
from PEDS_subdivision.logging_csv import (
    init_csv_logs, close_csv_logs, log_epoch_stats, log_keff_batch,
    log_logratio_saturation, log_xs_history_samples, log_splits,
    save_final_xs_csv, val_log_path,
)
from PEDS_subdivision.plotting import (
    _plot_xs_heatmap, _save_xs_subplots_for_samples, _plot_history, _save_flux_plots,
)
from PEDS_core.diagnostics import run_backward_grad_check


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Neural Network
# ─────────────────────────────────────────────────────────────────────────────

def clip_log_ratios(x, lo, hi):
    """Hard clip log-ratio corrections in forward *and* backward.

    Prefer this over a straight-through estimator: STE clips only the
    forward value, so Adam keeps receiving unbounded grads that push the
    network further past the bounds — updates then behave as if the clip
    were missing. Hard clip zeros the local gradient outside [lo, hi].
    """
    return jnp.clip(x, lo, hi)

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

    def _normalize_chi(self, xs_final: jnp.ndarray, xs_baselines: jnp.ndarray) -> jnp.ndarray:
        """
        Enforce chi normalization per region:
          - fissile regions: sum_g chi_g = 1
          - non-fissile regions: chi_g = 0 for all g

        Works for any number of energy groups G.
        """
        lay = xs_layout(self.G)
        chi_sl = lay['chi']
        nuf_sl = lay['nuSigma_f']
        eps = 1e-12

        chi = jnp.maximum(xs_final[:, :, chi_sl], 0.0)
        chi_sum = jnp.sum(chi, axis=-1, keepdims=True)
        chi_norm = chi / jnp.where(chi_sum > eps, chi_sum, 1.0)

        base_chi = jnp.maximum(xs_baselines[:, :, chi_sl], 0.0)
        base_sum = jnp.sum(base_chi, axis=-1, keepdims=True)
        base_norm = base_chi / jnp.where(base_sum > eps, base_sum, 1.0)
        uniform = jnp.ones_like(base_chi) / float(self.G)
        chi_fallback = jnp.where(base_sum > eps, base_norm, uniform)

        fissile = jnp.sum(jnp.maximum(xs_baselines[:, :, nuf_sl], 0.0), axis=-1, keepdims=True) > eps
        chi_final = jnp.where(fissile, jnp.where(chi_sum > eps, chi_norm, chi_fallback), jnp.zeros_like(chi))

        return xs_final.at[:, :, chi_sl].set(chi_final)

    def compute_xs(self, geoms, xs_baselines, phi_norm,
                             log_xs_mean=None, log_xs_std=None):
        """Pure NN forward (no grad). Used for pre-solving and validation."""
        batch_size = geoms.shape[0]
        with timer("log baselines from xs first guess", verbose=False): 
            log_base    = self._log_baselines(xs_baselines,  log_xs_mean, log_xs_std)
        with timer("NN generated XS log ratios", verbose=False):
            log_ratios  = self.generator(geoms, log_base, phi_norm, training=False)
        # Same hard clip as the training path (__call__) so pre-solve XS
        # matches the XS that the custom VJP / loss actually uses.
        log_ratios = clip_log_ratios(log_ratios, LOG_RATIO_CLIP_LO, LOG_RATIO_CLIP_HI)
        xs_final = jnp.exp(log_ratios) * xs_baselines * XS_MASK_J
        xs_final = self._normalize_chi(xs_final, xs_baselines)
        return np.array(xs_final)  # concrete numpy, exits JAX world
        
    def compute_log_ratios(self, geoms, xs_baselines, phi_norm,
                log_xs_mean=None, log_xs_std=None):
            """Pure NN forward, returns CLIPPED log_ratios (no solver). For diagnostics."""
            batch_size = geoms.shape[0]
            log_base = self._log_baselines(xs_baselines, log_xs_mean, log_xs_std)
            log_ratios = self.generator(geoms, log_base, phi_norm, training=False)
            log_ratios = clip_log_ratios(log_ratios, LOG_RATIO_CLIP_LO, LOG_RATIO_CLIP_HI)
            return np.array(log_ratios)
    
    def __call__(self, geoms, params_raw, xs_baselines, phi_norm,
                 training: bool = False, sample_id_offset: int = 0,
                 log_xs_mean=None, log_xs_std=None):
        batch_size = geoms.shape[0]
        log_base   = self._log_baselines(xs_baselines, log_xs_mean, log_xs_std)
        phi = jnp.reshape(phi_norm, (batch_size, -1))
        log_ratios = self.generator(geoms, log_base, phi, training)
        log_ratios = clip_log_ratios(log_ratios, LOG_RATIO_CLIP_LO, LOG_RATIO_CLIP_HI)

        xs_final = jnp.exp(log_ratios) * xs_baselines * XS_MASK_J # [batch, 3, 12]
        xs_final = self._normalize_chi(xs_final, xs_baselines)

        # Single batched forward+backward call — replaces the per-sample loop.
        # _NT_batch_fwd retrieves from _PRESOLVE_CACHE (no repeated eigen-solve);
        # _NT_batch_bwd runs one jit-vmapped Aphi_Fphi_vjp_padded over all samples.
        params_raw_j = jnp.array(params_raw, dtype=jnp.float32)          # [B, 6]
        sample_ids_j = (jnp.arange(batch_size, dtype=jnp.int32)
                        + sample_id_offset)                               # [B]

        with timer("forward: solver loop (all samples)", verbose=False):
            keffs = NTdiff_solver_batch(xs_final, params_raw_j, sample_ids_j)

        return keffs, xs_final, log_ratios


# ─────────────────────────────────────────────────────────────────────────────
import os
import shutil
import hashlib

def save_code_snapshot(logdir, expname, log_handle=None):
    src = os.path.abspath(__file__)
    snapshot_path = os.path.join(logdir, f"code_snapshot_{expname}.py")

    # 1) Save a real copy of the script
    shutil.copy2(src, snapshot_path)

    # 1b) The script now reads its knobs from config_peds, so the snapshot is
    # only reproducible if the resolved configuration travels with it.
    shutil.copy2(os.path.join(CONFIG_RUN_DIR, "config_peds.py"),
                 os.path.join(logdir, f"config_snapshot_{expname}.py"))
    with open(os.path.join(logdir, f"config_resolved_{expname}.txt"),
              "w", encoding="utf-8") as fh:
        fh.write(CFG.summary() + "\n")

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
# SECTION 8: Training
# ─────────────────────────────────────────────────────────────────────────────
def train(filepath, train_size, val_size, test_size, batch_size, epochs, lr_max, lr_min, decay_epochs,
          hidden_sizes, n_regions, G, seed):

    init_csv_logs()
        
    # ── 8.1 Data ─────────────────────────────────────────────────────────────
    print("Loading data …")
    print(f"  Split cache enabled: {USE_SPLIT_CACHE}")
    print(f"  Split seeds: train_seed={TRAIN_SEED}, val_seed={VAL_SEED}, "
          f"test_seed={TEST_SEED} (test fixed across runs)")
    if USE_PARAM_STRATIFIED:
        print(f"  Split mode: parameter-stratified fuel_r-primary (bins/param={PARAM_STRAT_BINS})")
        (train_idx, train_geoms, train_keffs, train_rawparams, train_phi_features), \
        (val_idx, val_geoms, val_keffs, val_rawparams, val_phi_features), \
        (test_idx, test_geoms, test_keffs, test_rawparams, test_phi_features) = load_data_param_stratified(
            filepath, train_size, val_size, test_size,
            train_seed=TRAIN_SEED, val_seed=VAL_SEED, test_seed=TEST_SEED,
            holdout_seed=HOLDOUT_SEED,
            n_bins_per_param=PARAM_STRAT_BINS,
            balanced=False,
        )
    elif USE_BALANCED_RANGES:
        print("  Split mode: keff balanced fixed-range")
        (train_idx, train_geoms, train_keffs, train_rawparams, train_phi_features), \
        (val_idx, val_geoms, val_keffs, val_rawparams, val_phi_features), \
        (test_idx, test_geoms, test_keffs, test_rawparams, test_phi_features) = load_data_balanced_ranges(
            filepath, train_size, val_size, test_size,
            train_seed=TRAIN_SEED, holdout_seed=HOLDOUT_SEED,
        )
    else:
        print("  Split mode: keff quantile-stratified (cached)")
        split_cache_dir = os.path.join(PARENT_DIR, "data", "highfidelity", CACHE_DIR_NAME)
        (train_idx, train_geoms, train_keffs, train_rawparams, train_phi_features), \
        (val_idx, val_geoms,   val_keffs,   val_rawparams,   val_phi_features),  \
        (test_idx, test_geoms,  test_keffs,  test_rawparams,  test_phi_features) = load_data(
            filepath, train_size, val_size, test_size,
            train_seed=TRAIN_SEED, holdout_seed=HOLDOUT_SEED,
            split_cache_path=None, cache_dir=split_cache_dir,
            use_split_cache=USE_SPLIT_CACHE)
    train_size = len(train_geoms)
    val_size   = len(val_geoms)
    test_size  = len(test_geoms)
    log_splits(train_idx, train_keffs, val_idx, val_keffs, test_idx, test_keffs)

    print(f"Actual train size: {train_size}")
    print(f"Actual val size:   {val_size}")
    print(f"Actual test size:  {test_size}")
    print_keff_bin_counts("TRAIN", train_keffs)
    print_keff_bin_counts("VAL", val_keffs)
    print_keff_bin_counts("TEST", test_keffs)
    # Re-print representation diagnostics even for non-param-stratified modes.
    if not USE_PARAM_STRATIFIED:
        check_keff_split_representation(train_keffs, val_keffs, test_keffs)
        check_keff_bin_geometry_dominance(
            ("TRAIN", train_keffs, train_rawparams),
            ("VAL",   val_keffs,   val_rawparams),
            ("TEST",  test_keffs,  test_rawparams),
        )

    # ── Optional: custom VJP vs finite-difference (before any training) ──────
    if RUN_GRAD_CHECK:
        gc_idx = int(GRAD_CHECK_SAMPLE)
        xs_gc = jnp.array(
            compute_batch_baselines(train_rawparams[gc_idx:gc_idx+1], GEO)[0],
            dtype=jnp.float32,
        )
        run_backward_grad_check(
            train_rawparams, xs_gc, GEO, update_geo,
            NTdiff_solver, NTdiff_solver_batch, _NTdiff_fwd,
            _GEO_DATA_CACHE, SLAY, xs_mask=XS_MASK,
            log_dir=LOG_DIR, sample_idx=gc_idx,
        )

    # NEW: normalise phi features per-column (each group/region combination separately)
    phi_mean = train_phi_features.mean(axis=0)         # shape [G*3]
    phi_std  = train_phi_features.std(axis=0) + 1e-8   # shape [G*3]
    train_phi_norm = ((train_phi_features - phi_mean) / phi_std).astype(np.float32)
    val_phi_norm  = ((val_phi_features  - phi_mean) / phi_std).astype(np.float32)
    test_phi_norm = ((test_phi_features - phi_mean) / phi_std).astype(np.float32)   # add this

    n_phi_feats = GEO.G * 3
    # ── 8.2 Model — Adam + cosine/warmup LR + global-norm grad clip ───────────
    rngs      = nnx.Rngs(seed)
    model     = PEDSModel(hidden_sizes=hidden_sizes, n_regions=n_regions, G=G, n_phi_feats=n_phi_feats, rngs=rngs)
    # Cosine-to-floor up to `decay_epochs`, then hold LR at lr_min.
    # This decouples LR horizon from total training epochs for sweep studies.
    if decay_epochs <= 0:
        raise ValueError(f"decay_epochs must be > 0, got {decay_epochs}")
    steps_per_epoch = max(int(np.ceil(train_size / batch_size)), 1)
    decay_steps = int(decay_epochs) * steps_per_epoch
    cosine_schedule = optax.cosine_decay_schedule(
        init_value=lr_max,
        decay_steps=decay_steps,
        alpha=lr_min / lr_max,   # final lr = lr_max * alpha = lr_min
    )
    hold_schedule = optax.constant_schedule(lr_min)
    warmup_steps = int(0.1 * decay_steps)  # ~10% of training
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=lr_min,       # start near zero, not zero, to avoid divide issues
        peak_value=lr_max,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=lr_min,
    )
    """ lr_schedule = optax.join_schedules(
        schedules=[cosine_schedule, hold_schedule],
        boundaries=[decay_steps],
    ) """
    print(
        "LR schedule: cosine decay for "
        f"{decay_epochs} epochs ({decay_steps} steps), then hold at {lr_min:.2e}"
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

    log_keff_batch(logging_csv.train_writer, logging_csv.train_logfile, epoch=0,
                k_pred=np.array(kp_train_baseline),
                k_ref=train_keffs,
                avg_train_loss=0.0, avg_val_loss=0.0)
    log_keff_batch(logging_csv.val_writer, logging_csv.val_logfile, epoch=0,
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
    # EMA smoothing for checkpoint selection (EMA_ALPHA, MIN_SAVE_EPOCH and
    # PATIENCE come from config_peds). A small alpha damps transient spikes
    # while staying responsive enough to track genuine improvement.
    epochs_since_improvement = 0
    ema_val_mean_pcm = None   # will be set at end of epoch 1
    train_shuffle_rng = np.random.default_rng(seed)

    for epoch in range(1, epochs + 1):
        # ── shuffle ──────────────────────────────────────────────────────────
        idx = train_shuffle_rng.permutation(train_size)
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
        epoch_grad_norms = []  # reset each epoch (pre-clip norms for diagnostics)

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
            _t_batch = time.perf_counter()
            results = list(executor.map(_solve_sample_worker, args_list))
            _TIMINGS["fwd pre-solve batch wall"].append(time.perf_counter() - _t_batch)
            _PRESOLVE_CACHE.clear()
            for i, k, phi_fwd, phi_adj, geo_data, _, w_elapsed in results:
                _PRESOLVE_CACHE[i] = (k, phi_fwd, phi_adj, geo_data)
                _TIMINGS["fwd pre-solve worker sample"].append(w_elapsed)

            # Step 3: forward + backward + update
            with timer("train step: forward+backward+optiupd", verbose = False):
                (loss, keff_preds_batch), grads = grad_fn(
                    model, bp, bkj, br, bxs, b_phi_j,
                    log_xs_mean_j, log_xs_std_j,
                )
            # Pre-clip norm (clip_by_global_norm runs inside optimizer.tx).
            grad_norm = float(optax.global_norm(grads))
            epoch_grad_norms.append(grad_norm)
            optimizer.update(model, grads)  # every batch: clip → Adam → apply

            epoch_loss    += float(loss)
            all_kp_train.extend(np.array(keff_preds_batch).tolist())
            all_kr_train.extend(bk.tolist())
            all_sample_ids.extend(bsid.tolist())

        # After all batches — one row per sample, sorted by dataset index
        ids = np.asarray(all_sample_ids, dtype=int)
        order = np.argsort(ids)
        log_keff_batch(
            logging_csv.train_writer, logging_csv.train_logfile, epoch,
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
                for i, k, _pf, _pa, _gd, _ep, w_elapsed in results:
                    all_kp_val.append(float(k))
                    all_kr_val.append(float(bk[i]))
                    _TIMINGS["fwd val worker sample"].append(w_elapsed)

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
            epochs_since_improvement = 0
        elif epoch >= MIN_SAVE_EPOCH:
            epochs_since_improvement += 1
        # ── per-sample CSV ───────────────────────────────────────────────────
        log_keff_batch(logging_csv.val_writer, logging_csv.val_logfile, epoch, kp_val, kr_val,
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
        if epoch >= MIN_SAVE_EPOCH and epochs_since_improvement >= PATIENCE:
            print(f"Early stopping at epoch {epoch}: no EMA improvement for {PATIENCE} epochs "
                f"(best={best_val_mean_pcm:.1f} pcm at epoch {best_epoch})")
            break
        if epoch % 20 == 0:
            jax.clear_caches()

    # END OF THE EPOCH LOOOOOP =================================================

    save_checkpoint(nnx.state(optimizer), LAST_CKPT_PATH,
                    metadata={"epoch": epochs, "val_mean_pcm": val_m["mean_pcm"]})
    print(f"Saved last-epoch checkpoint (model+optimizer) → {LAST_CKPT_PATH}")
    print(f"Best epoch was {best_epoch}  (val_mean_pcm = {best_val_mean_pcm:.1f} pcm)  → {BEST_CKPT_PATH}")
    
    best_state, best_meta = load_checkpoint(BEST_CKPT_PATH)
    nnx.update(model, best_state)
    print(f"Loaded best checkpoint from epoch {best_meta['epoch']} "
        f"(val_mean_pcm={best_meta['val_mean_pcm']:.1f}) for final evaluation.")

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
            decay_epochs  = DECAY_EPOCHS,
            hidden_sizes= HIDDEN_SIZES,
            n_regions   = N_REGIONS,
            G           = N_GROUPS,
            seed        = SEED,
        )
        print(
            f"Starting PEDS training with train size = {TRAIN_SIZE}, "
            f"seed = {SEED} (train/val), test_seed = {TEST_SEED}, "
            f"decay_epochs = {DECAY_EPOCHS}…"
        )
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