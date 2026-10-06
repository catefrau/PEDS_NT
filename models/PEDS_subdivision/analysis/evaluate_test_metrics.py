"""
evaluate_test_metrics.py
=========================================================================
Walks a LOGS root directory structured as:

    LOGS_ROOT/train_<N>/train_<N>_seed_<S>/checkpoints/best_model.pkl

For every run found, reconstructs the model with the architecture stored
in the checkpoint metadata, loads the best-validation weights, evaluates
it ONLY on the held-out TEST set (identical across every run, since the
val/test split is derived from a fixed HOLDOUT_SEED), and appends one row
of test-set metrics to a single aggregate CSV. The row identifier is the
"train_<N>/seed_<S>" folder path, exactly as requested.

Two things are cached to disk to avoid repeating expensive physics solves:
  1. The TEST set itself (geoms/keffs/rawparams/phi_features) — keyed by split
     signature and snapshot hash so different training code versions do not mix.
  2. The per-(train_size, seed, snapshot) normalization stats (phi_mean/std and
     log_xs_mean/std) — cached per run snapshot the first time they're needed.

Each run is evaluated with its own code_snapshot_*.py (same code used at train
time), so changes to models/PEDS.py do not affect aggregate metrics. Use
--use-current-peds to force the live PEDS.py instead.

USAGE
-----
    python evaluate_test_metrics.py /path/to/LOGS
    python evaluate_test_metrics.py /path/to/LOGS --out my_summary.csv
    python evaluate_test_metrics.py /path/to/LOGS --use-current-peds

Per-seed alternate tests (after generate_alt_test_splits.py):
    python evaluate_test_metrics.py /path/to/LOGS \\
        --split-log alt_split_log.csv \\
        --results-dirname testset_results_alt

    python evaluate_test_metrics.py /path/to/LOGS \
        --with-val-study \
        --split-log alt_split_log.csv \
        --results-dirname testset_results_alt
=========================================================================
"""
import os
import re
import sys
import glob
import pickle
import csv
import hashlib
import importlib.util
import types
import inspect
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

MODELS_DIR = Path(__file__).resolve().parents[2]      # models/ — model stack
PROJECT_ROOT = MODELS_DIR.parent
CONFIG_RUN_DIR = PROJECT_ROOT / "config_and_run"      # launchers, RUNS/, plot_functions

# Must come before the project imports below: this file sits three levels under
# models/, so neither the model stack nor the launcher tree is importable yet.
for _root in (str(MODELS_DIR), str(CONFIG_RUN_DIR), str(PROJECT_ROOT)):
    if _root not in sys.path:
        sys.path.insert(0, _root)

sys.path.insert(0, str(CONFIG_RUN_DIR / "plot_functions"))

# Fallback when a run folder has no code_snapshot_*.py (imported lazily in get_run_eval_context)
from PEDS import _DATA_FILEPATH as _DEFAULT_DATA_FILEPATH  # noqa: E402
from plots_all import plot_keff_scatter, plot_parallel_coords, plot_error_vs_keff  # noqa: E402

# rawparams column order → names expected by plots_all's plot_parallel_coords (PARAMS).
PARAM_COLS = [
    "r0_b4c_rod_outer_radius",
    "r0_b4c_rod_cr_fraction",
    "r1_fuel_annulus_outer_radius",
    "r1_fuel_annulus_enrichment",
    "r1_fuel_annulus_f_mod",
    "r2_water_outer_radius",
]

# ── CONFIG: edit these paths once, here ────────────────────────────────────
#"RUNS/bounds1000_70decayepochs"
OUTPUT_DIRNAME = "testset_results"
OUTPUT_FILENAME = "test_metrics_all_runs.csv"
SPLIT_LOG_NAME = "split_log.csv"  # override via --split-log (e.g. alt_split_log.csv)

# Dataset path used to load arrays for evaluation.
# Set to None to auto-detect from each run's code_snapshot_*.py.
# to force every run to use the same file regardless of what the snapshot says.
DATA_FILEPATH_OVERRIDE = None

# If True, ignore per-run code_snapshot_*.py and use current models/PEDS.py.
USE_CURRENT_PEDS = False
SCALING_PLOTS = False
# Optional extras (off by default) — enable via CLI flags --with-error-scatter / --with-val-study.
GENERATE_ERROR_VS_KEFF_PLOTS = False
GENERATE_VAL_STUDY = False
ERROR_VS_KEFF_RED_FRAC = 0.05      # worst 5% -> red
ERROR_VS_KEFF_ORANGE_FRAC = 0.10   # next slice up to worst 10% (cumulative) -> orange
VAL_STUDY_DIRNAME = "valset_results"
VAL_KEFF_MATCH_TOL = 1e-4          # tolerance (in keff units) for backtracing val-log samples to dataset params

# keff-distribution analysis across train / val / test (cheap; uses logged CSVs)
N_KEFF_BINS = 10
PCM_THRESHOLD = 650.0             # β_eff-style success threshold used in frac_below_* metrics
GENERATE_KEFF_DIST_ONLY = False   # set True via --keff-dist-only (skip checkpoint evaluation)

# If True, report Δk = (k_pred - k_ref) × 10^5 pcm instead of
# Δρ = (k_pred - k_ref) / (k_pred·k_ref) × 10^5 pcm.  Δk is linear in keff
# and carries no dependence on how far the system is from criticality.
USE_DELTA_K = False

RUN_PATTERN = re.compile(r"^train_(\d+)_seed_(\d+)$")
SNAPSHOT_DATA_RE = re.compile(r'_DATA_FILEPATH\s*=\s*os\.path\.join\([^)]*\)\s*$|_DATA_FILEPATH\s*=\s*["\']([^"\']+)["\']', re.MULTILINE)
EVAL_BATCH_SIZE = 32

# ── Error-metric helpers ────────────────────────────────────────────────────
def _signed_error_pcm(kp: float, kr: float) -> float:
    """Signed prediction error in pcm.  Δk or Δρ depending on USE_DELTA_K."""
    if USE_DELTA_K:
        return (kp - kr) * 1e5
    return (kp - kr) / (kp * kr) * 1e5


def _error_col_name() -> str:
    """Primary (absolute) error column name written to comparison CSVs."""
    return "delta_k_pcm" if USE_DELTA_K else "delta_rho_pcm"


def _signed_error_col_name() -> str:
    return "signed_delta_k_pcm" if USE_DELTA_K else "signed_delta_rho_pcm"


def _error_label() -> str:
    """Short Greek-letter label for plots."""
    return "Δk" if USE_DELTA_K else "Δρ"


def _error_label_mathtext() -> str:
    """Matplotlib-safe math label (single math fragment, no nested $)."""
    if USE_DELTA_K:
        return r"$\left|\Delta k\right|$"
    return r"$\left|\Delta\rho\right|$"


def _abs_error_pcm_array(k_pred: np.ndarray, k_ref: np.ndarray) -> np.ndarray:
    """Per-sample |error| in pcm (Δk or Δρ depending on USE_DELTA_K)."""
    k_pred = np.asarray(k_pred, dtype=float)
    k_ref = np.asarray(k_ref, dtype=float)
    if USE_DELTA_K:
        return np.abs(k_pred - k_ref) * 1e5
    return np.abs(k_pred - k_ref) / (k_pred * k_ref) * 1e5


def _compute_pcm_metrics(k_pred: np.ndarray, k_ref: np.ndarray) -> dict:
    """Headline pcm metrics using the active error definition."""
    dr = _abs_error_pcm_array(k_pred, k_ref)
    return dict(
        mse_k=float(np.mean((k_pred - k_ref) ** 2)),
        MAE_k=float(np.mean(np.abs(k_pred - k_ref))),
        mean_pcm=float(np.mean(dr)),
        median_pcm=float(np.median(dr)),
        p95_pcm=float(np.percentile(dr, 95)),
        std_pcm=float(np.std(dr)),
        frac_below_650=float(np.mean(dr < 650.0)),
        frac_below_100=float(np.mean(dr < 100.0)),
    )


def _detect_error_col(df) -> str:
    """Return the error column present in *df* (handles both modes)."""
    if "delta_k_pcm" in df.columns:
        return "delta_k_pcm"
    return "delta_rho_pcm"

# ───────────────────────────────────────────────────────────────────────────

_SNAPSHOT_MODULE_CACHE: dict[str, types.ModuleType] = {}


class RunEvalContext:
    """Symbols needed for test evaluation, loaded from a run's training snapshot."""

    def __init__(self, source: str, snapshot_path: Optional[str], snapshot_hash: str, mod):
        self.source = source
        self.snapshot_path = snapshot_path
        self.snapshot_hash = snapshot_hash
        self.GEO = mod.GEO
        self.PEDSModel = mod.PEDSModel
        self._run_NT_solver = mod._run_NT_solver
        self.compute_phi_features = mod.compute_phi_features
        # Older snapshots (pre feature-engineering) won't define this — fall
        # back to None so phi-only behavior is preserved for those runs.
        self.compute_interaction_features = getattr(mod, "compute_interaction_features", None)
        self.compute_batch_baselines = mod.compute_batch_baselines
        self.compute_metrics = mod.compute_metrics
        self.data_loader = mod.data_loader
        self.LOG_RATIO_CLIP_LO = getattr(mod, "LOG_RATIO_CLIP_LO", None)
        self.LOG_RATIO_CLIP_HI = getattr(mod, "LOG_RATIO_CLIP_HI", None)
        self.STUDY_CONFIGS = getattr(mod, "STUDY_CONFIGS", {})
        self.default_study_name = getattr(mod, "STUDY_NAME", None)


def _snapshot_hash(snapshot_path: str) -> str:
    h = hashlib.sha1()
    with open(snapshot_path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _find_snapshot_path(run_dir: str) -> Optional[str]:
    snapshots = sorted(glob.glob(os.path.join(run_dir, "code_snapshot_*.py")))
    return snapshots[0] if snapshots else None


def load_snapshot_module(snapshot_path: str) -> types.ModuleType:
    """
    Import the run's code_snapshot_*.py as an isolated module.

    Snapshots saved into run folders would otherwise set THIS_DIR to that folder
    and break PARENT_DIR / data paths, so THIS_DIR is patched back to the
    directory the script lived in at training time.

    That directory differs by snapshot vintage: current snapshots come from
    ``models/PEDS.py`` and derive their output paths from ``config_peds``, while
    legacy snapshots came from the old ``modules/PEDS.py`` and built ``LOG_DIR`` as
    ``THIS_DIR/RUNS/...`` -- which is now ``config_and_run``.
    """
    snapshot_path = os.path.abspath(snapshot_path)
    if snapshot_path in _SNAPSHOT_MODULE_CACHE:
        return _SNAPSHOT_MODULE_CACHE[snapshot_path]

    with open(snapshot_path, encoding="utf-8") as f:
        source = f.read()

    this_dir_for_snapshot = (
        MODELS_DIR if "import config_peds" in source else CONFIG_RUN_DIR
    )
    source = source.replace(
        "THIS_DIR   = os.path.dirname(os.path.abspath(__file__))",
        f"THIS_DIR   = {repr(str(this_dir_for_snapshot))}  # patched by evaluate_test_metrics",
        1,
    )

    module_name = "peds_eval_" + hashlib.sha1(snapshot_path.encode()).hexdigest()[:12]
    mod = types.ModuleType(module_name)
    mod.__file__ = snapshot_path
    mod.__dict__["__name__"] = module_name

    real_makedirs = os.makedirs
    os.makedirs = lambda *_args, **_kwargs: None
    try:
        code = compile(source, snapshot_path, "exec")
        exec(code, mod.__dict__)  # noqa: S102 — intentional snapshot load
    finally:
        os.makedirs = real_makedirs

    _SNAPSHOT_MODULE_CACHE[snapshot_path] = mod
    return mod


def get_run_eval_context(run_dir: str) -> RunEvalContext:
    """Load evaluation symbols from the run's snapshot, or fall back to PEDS.py."""
    if USE_CURRENT_PEDS:
        import PEDS as mod
        return RunEvalContext(
            source="current PEDS.py (--use-current-peds)",
            snapshot_path=None,
            snapshot_hash="current_peds",
            mod=mod,
        )

    snapshot_path = _find_snapshot_path(run_dir)
    if snapshot_path is None:
        print("  [warn] no code_snapshot_*.py — falling back to current PEDS.py")
        import PEDS as mod
        return RunEvalContext(
            source="current PEDS.py (no snapshot)",
            snapshot_path=None,
            snapshot_hash="current_peds",
            mod=mod,
        )

    snap_hash = _snapshot_hash(snapshot_path)
    mod = load_snapshot_module(snapshot_path)
    clip_info = ""
    if hasattr(mod, "LOG_RATIO_CLIP_HI"):
        clip_info = f", LOG_RATIO_CLIP_HI={mod.LOG_RATIO_CLIP_HI}"
    print(f"  Eval code: {os.path.basename(snapshot_path)} (hash {snap_hash[:12]}{clip_info})")
    return RunEvalContext(
        source=f"snapshot:{os.path.basename(snapshot_path)}",
        snapshot_path=snapshot_path,
        snapshot_hash=snap_hash,
        mod=mod,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Discovery
# ─────────────────────────────────────────────────────────────────────────────
def find_runs(LOGS_ROOT):
    """Yield (run_id, train_size, seed, ckpt_path, run_dir) for each run."""
    pattern = os.path.join(LOGS_ROOT, "**", "checkpoints", "best_model.pkl")
    for ckpt_path in sorted(glob.glob(pattern, recursive=True)):
        run_dir = os.path.dirname(os.path.dirname(ckpt_path))
        run_folder = os.path.basename(run_dir)

        m = RUN_PATTERN.match(run_folder)
        if not m:
            print(f"  [skip] couldn't parse train_size/seed from folder: {run_folder}")
            continue

        train_size, seed = int(m.group(1)), int(m.group(2))
        run_id = os.path.relpath(run_dir, LOGS_ROOT).replace(os.sep, "/")
        yield run_id, train_size, seed, ckpt_path, run_dir


def _indices_signature(indices):
    """Stable signature for an index set (order-independent)."""
    arr = np.sort(np.asarray(indices, dtype=np.int64))
    return hashlib.sha1(arr.tobytes()).hexdigest()


def _read_indices_from_split_log(run_dir, train_size, seed, split_log_name=None):
    """Load train/val/test sample indices from a split CSV in this run folder."""
    name = split_log_name if split_log_name is not None else SPLIT_LOG_NAME
    csv_path = os.path.join(run_dir, name)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{name} not found for run train={train_size}, seed={seed}: {csv_path}"
        )
    train_idx, val_idx, test_idx = [], [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = str(row.get("split", "")).strip().lower()
            idx = int(row["sample_idx"])
            if split == "train":
                train_idx.append(idx)
            elif split == "val":
                val_idx.append(idx)
            elif split == "test":
                test_idx.append(idx)
    if not train_idx or not test_idx:
        raise ValueError(
            f"{name} for run train={train_size}, seed={seed} is missing "
            f"required split rows (train={len(train_idx)}, test={len(test_idx)})"
        )
    if len(train_idx) != train_size:
        print(
            f"  [warn] {name} train count ({len(train_idx)}) != folder train_size ({train_size}) "
            f"for seed={seed}"
        )
    return (
        np.array(train_idx, dtype=np.int64),
        np.array(val_idx, dtype=np.int64),
        np.array(test_idx, dtype=np.int64),
        f"split_log:{csv_path}",
    )


def _detect_data_filepath(run_dir):
    """
    Determine which dataset a run was trained on.

    Current runs ship a config_resolved_*.txt whose "data :" line holds the
    absolute path, so prefer that. Older runs only have code_snapshot_*.py with
    a line like:
        _DATA_FILEPATH = os.path.join(PARENT_DIR, "data", "highfidelity", "some.npz")
    from which the last quoted token (the filename) is resolved against the
    project root. Falls back to the global PEDS default if neither can be read.
    """
    for resolved in sorted(glob.glob(os.path.join(run_dir, "config_resolved_*.txt"))):
        with open(resolved) as f:
            for line in f:
                if line.strip().startswith("data") and ":" in line:
                    candidate = line.split(":", 1)[1].strip()
                    if candidate and os.path.exists(candidate):
                        return candidate

    snapshots = glob.glob(os.path.join(run_dir, "code_snapshot_*.py"))
    if not snapshots:
        return None

    snapshot_path = snapshots[0]
    with open(snapshot_path) as f:
        content = f.read()

    for line in content.splitlines():
        if "_DATA_FILEPATH" in line and "=" in line:
            # Extract all quoted strings from the line
            quoted = re.findall(r'["\']([^"\']+)["\']', line)
            if quoted:
                # The last quoted token is the filename (e.g. "complete_LHS_2900.npz")
                filename = quoted[-1]
                candidate = os.path.join(PROJECT_ROOT, "data", "highfidelity", filename)
                if os.path.exists(candidate):
                    return candidate
                # Maybe the snapshot stored a full subpath like "data/highfidelity/x.npz"
                candidate2 = os.path.join(PROJECT_ROOT, filename)
                if os.path.exists(candidate2):
                    return candidate2

    return None


def load_dataset_arrays(run_dir=None):
    """
    Load full dataset arrays for evaluation.
    Uses DATA_FILEPATH_OVERRIDE if set, otherwise detects the correct .npz
    from the run's code_snapshot_*.py, falling back to the PEDS default.
    """
    if DATA_FILEPATH_OVERRIDE is not None:
        path = DATA_FILEPATH_OVERRIDE
        source = "override"
    elif run_dir is not None:
        detected = _detect_data_filepath(run_dir)
        if detected is not None:
            path = detected
            source = f"snapshot:{os.path.basename(path)}"
        else:
            path = _DEFAULT_DATA_FILEPATH
            source = "PEDS default (snapshot not parsed)"
    else:
        path = _DEFAULT_DATA_FILEPATH
        source = "PEDS default"

    print(f"  Dataset: {path}  [{source}]")
    data = np.load(path, allow_pickle=True)
    return (
        np.array(data["params"], dtype=np.float32),
        np.array(data["keffs"], dtype=np.float32),
        np.array(data["params_raw"], dtype=np.float32),
    )


def _phi_plus_interaction_features(ctx: RunEvalContext, rawparams, phi_features):
    """Append engineered interaction features (if this snapshot defines them)
    to the phi features, matching the concatenation done in PEDS.py train()."""
    if ctx.compute_interaction_features is None:
        return phi_features
    interaction_features = ctx.compute_interaction_features(rawparams)
    return np.concatenate([phi_features, interaction_features], axis=1)


def get_test_payload(geoms, keffs, rawparams, test_idx, split_source, cache, ctx: RunEvalContext):
    """Build test payload from explicit test indices from split_log.csv."""
    sig = _indices_signature(test_idx)
    cache_key = (sig, ctx.snapshot_hash)
    if cache_key in cache:
        return cache[cache_key]

    print(f"  Building TEST payload from {split_source} ({len(test_idx)} samples)…")
    test_geoms, test_keffs, test_rawparams = geoms[test_idx], keffs[test_idx], rawparams[test_idx]
    test_phi_features = ctx.compute_phi_features(test_rawparams)
    test_phi_features = _phi_plus_interaction_features(ctx, test_rawparams, test_phi_features)

    payload = dict(geoms=test_geoms, keffs=test_keffs, rawparams=test_rawparams,
                    phi_features=test_phi_features,
                    test_idx=test_idx,
                    split_source=split_source,
                    split_signature=sig)
    cache[cache_key] = payload
    return payload


# ─────────────────────────────────────────────────────────────────────────────
# Caching: per-(train_size, seed) normalization stats
# ─────────────────────────────────────────────────────────────────────────────
def get_or_build_norm_stats(LOGS_ROOT, train_size, seed, rawparams, train_idx, ctx: RunEvalContext):
    split_sig = _indices_signature(train_idx)
    cache_path = os.path.join(
        LOGS_ROOT, "_evaluation_cache",
        f"norm_stats_train{train_size}_seed{seed}_{ctx.snapshot_hash[:12]}_{split_sig[:12]}.pkl",
    )
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    print(
        f"  Recomputing normalization stats from split_log train indices "
        f"for train_size={train_size}, seed={seed} …"
    )
    train_rawparams = rawparams[train_idx]

    train_phi_features = ctx.compute_phi_features(train_rawparams)
    train_phi_features = _phi_plus_interaction_features(ctx, train_rawparams, train_phi_features)
    phi_mean = train_phi_features.mean(axis=0).astype(np.float32)
    phi_std  = (train_phi_features.std(axis=0) + 1e-8).astype(np.float32)

    train_xs_baselines = ctx.compute_batch_baselines(train_rawparams, ctx.GEO)
    xs_np = np.array(train_xs_baselines)
    safe  = np.where(xs_np > 1e-10, xs_np, np.ones_like(xs_np))
    log_train = np.log(safe)
    log_xs_mean = log_train.mean(axis=0).astype(np.float32)
    log_xs_std  = (log_train.std(axis=0) + 1e-8).astype(np.float32)

    payload = dict(phi_mean=phi_mean, phi_std=phi_std,
                    log_xs_mean=log_xs_mean, log_xs_std=log_xs_std,
                    split_signature=split_sig, snapshot_hash=ctx.snapshot_hash)
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f)
    return payload


# ─────────────────────────────────────────────────────────────────────────────
# Model reconstruction + checkpoint loading
# ─────────────────────────────────────────────────────────────────────────────
def load_checkpoint(ckpt_path):
    with open(ckpt_path, "rb") as f:
        payload = pickle.load(f)
    return payload["state"], payload.get("metadata", {})


def _arch_kwargs_from_study_cfg(cfg: dict) -> dict:
    extra = 0
    if cfg.get("extra_feat_mode") == "regime":
        extra = 4
    elif cfg.get("use_keff_input"):
        extra = 1
    return dict(
        activation=cfg.get("activation", "relu"),
        use_residual=bool(cfg.get("use_residual", False)),
        use_dropout=bool(cfg.get("use_dropout", False)),
        dropout_rate=float(cfg.get("dropout_rate", 0.0)),
        use_keff_input=bool(cfg.get("use_keff_input", False)),
        extra_feats_dim=int(cfg.get("extra_feats_dim", extra)),
    )


def _infer_arch_from_run_dir(ctx: RunEvalContext, run_dir: str) -> dict:
    """Existing agent checkpoints only stored hidden_sizes. Recover ELU / extras
    from the study folder name (…/r13_1k_elu6e4/train_1000_seed_1) via the
    snapshot's STUDY_CONFIGS. Without this, ELU-trained weights are evaluated
    as ReLU and the test correction collapses toward the diffusion baseline.
    """
    study = os.path.basename(os.path.dirname(os.path.abspath(run_dir)))
    cfgs = ctx.STUDY_CONFIGS or {}
    if study in cfgs:
        print(f"  architecture from study folder {study!r}: "
              f"activation={cfgs[study].get('activation', 'relu')}")
        return _arch_kwargs_from_study_cfg(cfgs[study])
    if ctx.default_study_name in cfgs:
        print(f"  architecture from snapshot STUDY_NAME={ctx.default_study_name!r}")
        return _arch_kwargs_from_study_cfg(cfgs[ctx.default_study_name])
    return {}


def build_model_from_metadata(ctx: RunEvalContext, metadata, seed_for_init=0,
                              run_dir=None):
    """Falls back to the v1 architecture defaults if metadata wasn't recorded."""
    hidden_sizes = metadata.get("hidden_sizes", [128, 256, 128])
    n_regions    = metadata.get("n_regions", 3)
    G            = metadata.get("G", ctx.GEO.G)
    n_phi_feats  = metadata.get("n_phi_feats", ctx.GEO.G * 3)
    rngs = nnx.Rngs(seed_for_init)
    extras = {}
    for key in ("activation", "use_residual", "use_dropout", "dropout_rate",
                "use_keff_input", "extra_feats_dim"):
        if key in metadata and metadata[key] is not None:
            extras[key] = metadata[key]
    if "activation" not in extras and run_dir:
        extras.update(_infer_arch_from_run_dir(ctx, run_dir))
    sig = inspect.signature(ctx.PEDSModel.__init__)
    extras = {k: v for k, v in extras.items() if k in sig.parameters}
    if extras:
        print(f"  PEDSModel extras: {extras}")
    return ctx.PEDSModel(hidden_sizes=hidden_sizes, n_regions=n_regions,
                         G=G, n_phi_feats=n_phi_feats, rngs=rngs, **extras)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_on_test(ctx: RunEvalContext, model, test_payload, norm_stats):
    geoms     = test_payload["geoms"]
    keffs     = test_payload["keffs"]
    rawparams = test_payload["rawparams"]

    phi_norm = ((test_payload["phi_features"] - norm_stats["phi_mean"])
                / norm_stats["phi_std"]).astype(np.float32)
    xs_baselines  = ctx.compute_batch_baselines(rawparams, ctx.GEO)
    log_xs_mean_j = jnp.array(norm_stats["log_xs_mean"])
    log_xs_std_j  = jnp.array(norm_stats["log_xs_std"])

    k_pred_all = []
    for bg, bk, br, bb, bphi in ctx.data_loader(
            geoms, keffs, rawparams, np.array(xs_baselines), phi_norm,
            batch_size=EVAL_BATCH_SIZE):
        xs_np = model.compute_xs(jnp.array(bg), jnp.array(bb), jnp.array(bphi),
                                  log_xs_mean_j, log_xs_std_j)
        for i in range(len(bg)):
            k, _, _, _ = ctx._run_NT_solver(xs_np[i], np.array(br[i]),
                                            np.array([i], dtype=np.int32))
            k_pred_all.append(float(k))

    k_pred_all = np.array(k_pred_all)
    if USE_DELTA_K:
        return _compute_pcm_metrics(k_pred_all, keffs), k_pred_all
    return ctx.compute_metrics(k_pred_all, keffs), k_pred_all


def compute_baseline_keffs(ctx: RunEvalContext, rawparams: np.ndarray) -> np.ndarray:
    """keff from running the physics solver directly on the un-corrected
    (baseline, polynomial-regression) XS — i.e. no NN correction at all.
    This is the "initial" prediction used as the pre-training reference point."""
    xs_baselines = np.array(ctx.compute_batch_baselines(rawparams, ctx.GEO))
    k_pred = []
    for i in range(len(rawparams)):
        k, _, _, _ = ctx._run_NT_solver(xs_baselines[i], rawparams[i], np.array([i], dtype=np.int32))
        k_pred.append(float(k))
    return np.array(k_pred)


# ─────────────────────────────────────────────────────────────────────────────
# Representative-seed initial-vs-final keff comparison
# (mirrors plots_all.py's plot_keff_scatter / plot_parallel_coords, epoch 0 vs
#  best epoch, but computed directly on the held-out TEST set for one seed).
# ─────────────────────────────────────────────────────────────────────────────
def select_representative_runs(df_rows: pd.DataFrame, metric: str = "test_mean_pcm") -> dict:
    """For each train_size group, pick the run whose `metric` is closest to
    that group's mean across seeds — a single seed representative of typical
    behaviour at that training-set size."""
    reps = {}
    for train_size, grp in df_rows.groupby("train_size"):
        group_mean = grp[metric].mean()
        idx = (grp[metric] - group_mean).abs().idxmin()
        reps[train_size] = grp.loc[idx, "run_id"]
    return reps


def build_keff_comparison_df(test_payload, k_initial, k_final, best_epoch):
    rawparams = test_payload["rawparams"]
    keffs = test_payload["keffs"]
    test_idx = test_payload["test_idx"]
    ecol = _error_col_name()
    scol = _signed_error_col_name()
    records = []
    for epoch_label, k_pred in ((0, k_initial), (best_epoch, k_final)):
        for i in range(len(keffs)):
            kp, kr = float(k_pred[i]), float(keffs[i])
            signed_err = _signed_error_pcm(kp, kr)
            rec = dict(epoch=epoch_label, sample_idx=int(test_idx[i]),
                       keff_openmc=kr, keff_peds=kp,
                       **{ecol: abs(signed_err), scol: signed_err})
            for j, col in enumerate(PARAM_COLS):
                rec[col] = float(rawparams[i, j])
            records.append(rec)
    return pd.DataFrame(records)


def _run_test_comparison_path(out_dir, train_size, seed):
    """Per-seed test before/after CSV (used for cross-run keff-bin std)."""
    return os.path.join(out_dir, f"run_train{train_size}_seed{seed}_keff_comparison.csv")


def _rep_test_comparison_path(out_dir, train_size, seed):
    return os.path.join(out_dir, f"rep_train{train_size}_seed{seed}_keff_comparison.csv")


def _test_payload_cache_key(test_payload) -> str:
    idx = np.asarray(test_payload["test_idx"]).astype(np.int64)
    return f"n{len(idx)}_h{hashlib.md5(idx.tobytes()).hexdigest()[:12]}"


def save_all_run_test_comparisons(run_eval_cache: dict, out_dir: str):
    """Write a keff comparison CSV for every evaluated seed.

    The fixed test set's baseline (epoch-0) keffs are computed once per unique
    test payload and reused; final keffs come from each run's checkpoint.
    Without these files, only the representative seed has Δρ and test std
    columns stay empty after cross-seed aggregation.
    """
    if not run_eval_cache:
        return
    os.makedirs(out_dir, exist_ok=True)
    baseline_cache = {}  # payload_key -> k_initial
    n_written = 0
    for run_id, info in sorted(run_eval_cache.items(), key=lambda kv: (kv[1]["train_size"], kv[1]["seed"])):
        train_size = int(info["train_size"])
        seed = int(info["seed"])
        best_epoch = info["best_epoch"]
        try:
            best_epoch = int(best_epoch)
        except (TypeError, ValueError):
            best_epoch = 1
        test_payload = info["test_payload"]
        key = _test_payload_cache_key(test_payload)
        if key not in baseline_cache:
            print(f"  Computing shared baseline keffs for test set ({key}) …")
            baseline_cache[key] = compute_baseline_keffs(info["ctx"], test_payload["rawparams"])
        k_initial = baseline_cache[key]
        comp_df = build_keff_comparison_df(
            test_payload, k_initial, info["k_pred_final"], best_epoch,
        )
        path = _run_test_comparison_path(out_dir, train_size, seed)
        comp_df.to_csv(path, index=False)
        n_written += 1
    print(f"  Wrote {n_written} per-run test keff comparison CSVs → {out_dir}")


def generate_representative_comparison_plots(df_rows: pd.DataFrame, run_eval_cache: dict, out_dir: str):
    if df_rows.empty:
        return
    reps = select_representative_runs(df_rows)
    for train_size, run_id in sorted(reps.items()):
        info = run_eval_cache.get(run_id)
        if info is None:
            print(f"  [warn] representative run {run_id} missing from eval cache — skipping")
            continue

        seed = info["seed"]
        best_epoch = info["best_epoch"]
        try:
            best_epoch = int(best_epoch)
        except (TypeError, ValueError):
            best_epoch = 1
        print(f"\n=== Representative case for train_size={train_size}: {run_id} "
              f"(closest to group-mean test_mean_pcm) ===")

        # Prefer the per-run CSV written by save_all_run_test_comparisons
        run_comp = _run_test_comparison_path(out_dir, train_size, seed)
        if os.path.exists(run_comp):
            comp_df = pd.read_csv(run_comp)
        else:
            ctx = info["ctx"]
            test_payload = info["test_payload"]
            print("  Running solver on baseline (uncorrected) XS for the test set …")
            k_initial = compute_baseline_keffs(ctx, test_payload["rawparams"])
            comp_df = build_keff_comparison_df(
                test_payload, k_initial, info["k_pred_final"], best_epoch,
            )

        prefix = f"rep_train{train_size}_seed{seed}"
        comp_df.to_csv(_rep_test_comparison_path(out_dir, train_size, seed), index=False)

        both = comp_df[comp_df["epoch"].isin([0, best_epoch])]
        ecol = _detect_error_col(comp_df)
        vmin, vmax = both[ecol].min(), both[ecol].max()
        # plot_keff_scatter / plot_parallel_coords expect the column named
        # 'delta_rho_pcm'; alias it so those helpers work in both modes.
        comp_df_plot = comp_df.copy()
        if ecol != "delta_rho_pcm":
            comp_df_plot["delta_rho_pcm"] = comp_df_plot[ecol]
        os.makedirs(os.path.join(out_dir,"representative_plots"), exist_ok=True)
        plot_keff_scatter(comp_df_plot, epoch=0, vmin=vmin, vmax=vmax,
                           save_path=os.path.join(out_dir,"representative_plots" , f"{prefix}_keff_scatter_initial.png"))
        plot_keff_scatter(comp_df_plot, epoch=best_epoch, vmin=vmin, vmax=vmax,
                           save_path=os.path.join(out_dir,"representative_plots" , f"{prefix}_keff_scatter_final.png"))

        plot_parallel_coords(comp_df_plot, epoch=0,
                              save_path=os.path.join(out_dir,"representative_plots" , f"{prefix}_parallel_coords_initial.png"))
        plot_parallel_coords(comp_df_plot, epoch=best_epoch,
                              save_path=os.path.join(out_dir,"representative_plots" , f"{prefix}_parallel_coords_final.png"))

        if GENERATE_ERROR_VS_KEFF_PLOTS:
            plot_error_vs_keff(comp_df_plot, epoch=0,
                                red_frac=ERROR_VS_KEFF_RED_FRAC, orange_frac=ERROR_VS_KEFF_ORANGE_FRAC,
                                save_path=os.path.join(out_dir, "representative_plots", f"{prefix}_error_vs_keff_initial.png"))
            plot_error_vs_keff(comp_df_plot, epoch=best_epoch,
                                red_frac=ERROR_VS_KEFF_RED_FRAC, orange_frac=ERROR_VS_KEFF_ORANGE_FRAC,
                                save_path=os.path.join(out_dir, "representative_plots", f"{prefix}_error_vs_keff_final.png"))


# ─────────────────────────────────────────────────────────────────────────────
# keff distribution across train / val / test (all seeds → mean ± std)
# Overall mean/median keff per split, plus 10 shared keff bins with sample %,
# mean keff, and mean |delta_rho| before (epoch 0) / after (best epoch).
# Aggregate CSVs average across runs in the folder (train/val vary by seed).
# ─────────────────────────────────────────────────────────────────────────────
def _pair_before_after_from_epoch_log(log_path, best_epoch=None):
    """Return DataFrame with keff_openmc, delta_rho_before, delta_rho_after.

    Rows are matched on the local per-split sample_idx within the epoch log.
    The 'delta_rho_before/after' columns contain Δk values when USE_DELTA_K=True.
    """
    if not os.path.exists(log_path):
        return None
    df = pd.read_csv(log_path)
    if df.empty or "epoch" not in df.columns:
        return None
    max_epoch = int(df["epoch"].max())
    final_epoch = max_epoch
    if best_epoch is not None:
        try:
            be = int(best_epoch)
            if (df["epoch"] == be).any():
                final_epoch = be
        except (TypeError, ValueError):
            pass

    # In delta_k mode, compute Δk from keff columns (always present in epoch logs).
    # In delta_rho mode, read the pre-computed delta_rho_pcm column.
    if USE_DELTA_K and "keff_peds" in df.columns:
        df = df.copy()
        df["_err_pcm"] = (df["keff_peds"] - df["keff_openmc"]).abs() * 1e5
        src_col = "_err_pcm"
    elif "delta_rho_pcm" in df.columns:
        src_col = "delta_rho_pcm"
    else:
        return None

    d0 = df[df["epoch"] == 0][["sample_idx", "keff_openmc", src_col]].copy()
    dF = df[df["epoch"] == final_epoch][["sample_idx", src_col]].copy()
    if d0.empty or dF.empty:
        return None
    d0 = d0.rename(columns={src_col: "delta_rho_before"})
    dF = dF.rename(columns={src_col: "delta_rho_after"})
    merged = d0.merge(dF, on="sample_idx", how="inner")
    return merged[["keff_openmc", "delta_rho_before", "delta_rho_after"]]


def _pair_before_after_from_comparison_csv(comp_path):
    """Pair epoch-0 / final rows from a rep_*_keff_comparison.csv.

    Detects whether the CSV was written in delta_rho or delta_k mode.
    The returned columns are always named delta_rho_before/after for
    compatibility with _summarize_keff_distribution (they contain whichever
    quantity the CSV stores).
    """
    if not os.path.exists(comp_path):
        return None
    df = pd.read_csv(comp_path)
    if df.empty:
        return None
    epochs = sorted(df["epoch"].unique())
    if len(epochs) < 2:
        return None
    e0, eF = int(epochs[0]), int(epochs[-1])
    ecol = _detect_error_col(df)
    d0 = df[df["epoch"] == e0][["sample_idx", "keff_openmc", ecol]].copy()
    dF = df[df["epoch"] == eF][["sample_idx", ecol]].copy()
    d0 = d0.rename(columns={ecol: "delta_rho_before"})
    dF = dF.rename(columns={ecol: "delta_rho_after"})
    merged = d0.merge(dF, on="sample_idx", how="inner")
    return merged[["keff_openmc", "delta_rho_before", "delta_rho_after"]]


def _frame_from_split_log(run_dir, split, split_log_name=None):
    """Keff-only frame from split_log.csv (delta_rho left as NaN)."""
    name = split_log_name if split_log_name is not None else SPLIT_LOG_NAME
    path = os.path.join(run_dir, name)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if df.empty or "split" not in df.columns:
        return None
    keff_col = "keff" if "keff" in df.columns else (
        "keff_openmc" if "keff_openmc" in df.columns else None
    )
    if keff_col is None:
        return None
    sub = df[df["split"].astype(str).str.lower() == split]
    if sub.empty:
        return None
    out = pd.DataFrame({
        "keff_openmc": sub[keff_col].astype(float).to_numpy(),
        "delta_rho_before": np.nan,
        "delta_rho_after": np.nan,
    })
    return out


def _keff_bin_edges(all_keffs, n_bins=N_KEFF_BINS):
    lo, hi = float(np.min(all_keffs)), float(np.max(all_keffs))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1e-6
    return np.linspace(lo, hi, n_bins + 1)


def _frac_below_pcm(values, threshold=PCM_THRESHOLD):
    """Fraction of finite |Δρ| values strictly below ``threshold`` pcm; NaN if none finite."""
    v = np.asarray(values, dtype=float)
    finite = np.isfinite(v)
    if not finite.any():
        return np.nan
    return float(np.mean(v[finite] < threshold))


def _summarize_keff_distribution(split_frames, n_bins=N_KEFF_BINS, edges=None):
    """
    split_frames: dict {split_name: DataFrame[keff_openmc, delta_rho_before, delta_rho_after]}

    Returns (overall_df, bins_df, edges).
    If ``edges`` is provided they are reused (needed for cross-run aggregation).

    Both overall and per-bin tables include ``frac_below_650_{before,after}``:
    the fraction of samples with |Δρ| < PCM_THRESHOLD in that split / keff bin.
    """
    overall_rows = []
    for split, frame in split_frames.items():
        k = frame["keff_openmc"].astype(float).to_numpy()
        before = frame["delta_rho_before"].astype(float).to_numpy()
        after = frame["delta_rho_after"].astype(float).to_numpy()
        overall_rows.append({
            "split": split,
            "n_samples": int(len(k)),
            "keff_mean": float(np.mean(k)),
            "keff_median": float(np.median(k)),
            "keff_std": float(np.std(k)),
            "keff_min": float(np.min(k)),
            "keff_max": float(np.max(k)),
            "delta_rho_before_mean": float(np.nanmean(before))
                if np.isfinite(before).any() else np.nan,
            "delta_rho_before_median": float(np.nanmedian(before))
                if np.isfinite(before).any() else np.nan,
            "delta_rho_after_mean": float(np.nanmean(after))
                if np.isfinite(after).any() else np.nan,
            "delta_rho_after_median": float(np.nanmedian(after))
                if np.isfinite(after).any() else np.nan,
            "frac_below_650_before": _frac_below_pcm(before),
            "frac_below_650_after": _frac_below_pcm(after),
        })
    overall_df = pd.DataFrame(overall_rows)

    if edges is None:
        all_keffs = np.concatenate([
            f["keff_openmc"].astype(float).to_numpy() for f in split_frames.values()
        ])
        edges = _keff_bin_edges(all_keffs, n_bins=n_bins)
    else:
        edges = np.asarray(edges, dtype=np.float64)
        n_bins = len(edges) - 1

    bin_rows = []
    for split, frame in split_frames.items():
        k = frame["keff_openmc"].astype(float).to_numpy()
        before = frame["delta_rho_before"].astype(float).to_numpy()
        after = frame["delta_rho_after"].astype(float).to_numpy()
        # rightmost edge inclusive
        bin_idx = np.digitize(k, edges[1:-1], right=False)
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        n_tot = max(len(k), 1)
        for b in range(n_bins):
            mask = bin_idx == b
            n = int(mask.sum())
            bin_rows.append({
                "split": split,
                "bin": b,
                "keff_lo": float(edges[b]),
                "keff_hi": float(edges[b + 1]),
                "n_samples": n,
                "pct_samples": 100.0 * n / n_tot,
                "avg_keff": float(np.mean(k[mask])) if n else np.nan,
                "avg_delta_rho_before": float(np.nanmean(before[mask]))
                    if n and np.isfinite(before[mask]).any() else np.nan,
                "avg_delta_rho_after": float(np.nanmean(after[mask]))
                    if n and np.isfinite(after[mask]).any() else np.nan,
                "median_delta_rho_before": float(np.nanmedian(before[mask]))
                    if n and np.isfinite(before[mask]).any() else np.nan,
                "median_delta_rho_after": float(np.nanmedian(after[mask]))
                    if n and np.isfinite(after[mask]).any() else np.nan,
                "frac_below_650_before": _frac_below_pcm(before[mask]) if n else np.nan,
                "frac_below_650_after": _frac_below_pcm(after[mask]) if n else np.nan,
            })
    bins_df = pd.DataFrame(bin_rows)
    return overall_df, bins_df, edges


def _mean_std_cols(df, group_keys, value_cols):
    """Return DataFrame with {col}_mean / {col}_std for each value column."""
    grouped = df.groupby(group_keys, sort=True)
    out = grouped.size().rename("n_runs").reset_index()
    for col in value_cols:
        if col not in df.columns:
            continue
        stats = grouped[col].agg(["mean", "std"]).reset_index()
        stats.columns = list(group_keys) + [f"{col}_mean", f"{col}_std"]
        out = out.merge(stats, on=list(group_keys), how="left")
    return out


def _aggregate_keff_dist_across_runs(per_run_overall, per_run_bins):
    """
    Average per-run tables across seeds.

    overall: group by (train_size, split)
    bins:    group by (train_size, split, bin) — keff_lo/hi taken as mean (shared edges → identical)
    """
    if not per_run_overall:
        return pd.DataFrame(), pd.DataFrame()

    cat_o = pd.concat(per_run_overall, ignore_index=True)
    overall_value_cols = [
        "n_samples", "keff_mean", "keff_median", "keff_std", "keff_min", "keff_max",
        "delta_rho_before_mean", "delta_rho_before_median",
        "delta_rho_after_mean", "delta_rho_after_median",
        "frac_below_650_before", "frac_below_650_after",
    ]
    overall_agg = _mean_std_cols(cat_o, ["train_size", "split"], overall_value_cols)

    cat_b = pd.concat(per_run_bins, ignore_index=True)
    bins_value_cols = [
        "n_samples", "pct_samples", "avg_keff",
        "avg_delta_rho_before", "avg_delta_rho_after",
        "median_delta_rho_before", "median_delta_rho_after",
        "frac_below_650_before", "frac_below_650_after",
        "keff_lo", "keff_hi",
    ]
    bins_agg = _mean_std_cols(cat_b, ["train_size", "split", "bin"], bins_value_cols)
    # Prefer clearer names for the fraction the user cares about
    if "pct_samples_mean" in bins_agg.columns:
        bins_agg = bins_agg.rename(columns={
            "pct_samples_mean": "frac_pct_mean",
            "pct_samples_std": "frac_pct_std",
        })
        # also keep fraction in [0,1] for convenience
        bins_agg["frac_mean"] = bins_agg["frac_pct_mean"] / 100.0
        bins_agg["frac_std"] = bins_agg["frac_pct_std"] / 100.0
    return overall_agg, bins_agg


def _plot_keff_distribution(bins_df, overall_df, out_path, title,
                            pct_col="pct_samples", pct_err_col=None,
                            before_col="avg_delta_rho_before", after_col="avg_delta_rho_after",
                            frac650_before_col="frac_below_650_before",
                            frac650_after_col="frac_below_650_after",
                            frac650_before_err_col=None,
                            frac650_after_err_col=None,
                            lo_col="keff_lo", hi_col="keff_hi"):
    """Three-panel figure: sample % by keff bin × split, mean |Δρ| before/after,
    and fraction of samples with |Δρ| < 650 pcm before/after.

    Middle panel: both ``before`` and ``after`` on a shared log y-axis so the
    full dynamic range (baseline ~thousands of pcm → corrected ~tens–hundreds of
    pcm) is readable in a single panel.
    """
    splits = [s for s in ("train", "val", "test") if s in set(bins_df["split"])]
    colors = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"}
    bins = sorted(bins_df["bin"].unique())
    x = np.arange(len(bins))
    width = 0.8 / max(len(splits), 1)

    has_frac650 = (
        frac650_after_col in bins_df.columns
        or frac650_before_col in bins_df.columns
    )
    n_panels = 3 if has_frac650 else 2
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 4.0 * n_panels), sharex=True)
    if n_panels == 2:
        ax0, ax1 = axes
        ax2 = None
    else:
        ax0, ax1, ax2 = axes

    for i, split in enumerate(splits):
        sub = bins_df[bins_df["split"] == split].set_index("bin").reindex(bins)
        heights = sub[pct_col].fillna(0).values
        xpos = x + (i - (len(splits) - 1) / 2) * width
        ax0.bar(xpos, heights, width=width, color=colors.get(split, f"C{i}"),
                label=split, edgecolor="white")
        if pct_err_col is not None and pct_err_col in sub.columns:
            yerr = sub[pct_err_col].fillna(0).values
            ax0.errorbar(xpos, heights, yerr=yerr, fmt="none", ecolor="black",
                         elinewidth=1, capsize=2, alpha=0.7)
    ax0.set_ylabel("% of samples in split", fontsize=12)
    ax0.legend(fontsize=11)
    ax0.grid(True, axis="y", alpha=0.3)
    ax0.set_title(title)

    # both before and after on a single log y-axis
    handles, labels_leg = [], []
    all_positive_vals = []
    for i, split in enumerate(splits):
        sub = bins_df[bins_df["split"] == split].set_index("bin").reindex(bins)
        c = colors.get(split, f"C{i}")
        if before_col in sub.columns:
            yb = np.asarray(sub[before_col].values, dtype=float)
            h_b, = ax1.plot(x, yb, ls="--", marker="o", color=c,
                            alpha=0.75, label=f"{split} before")
            handles.append(h_b)
            labels_leg.append(f"{split} before")
            all_positive_vals.extend(yb[np.isfinite(yb) & (yb > 0)].tolist())
        if after_col in sub.columns:
            ya = np.asarray(sub[after_col].values, dtype=float)
            h_a, = ax1.plot(x, ya, ls="-", marker="s", color=c,
                            label=f"{split} after")
            handles.append(h_a)
            labels_leg.append(f"{split} after")
            all_positive_vals.extend(ya[np.isfinite(ya) & (ya > 0)].tolist())

    h_beta = ax1.axhline(PCM_THRESHOLD, ls=":", color="grey", alpha=0.7,
                         label=f"β_eff = {PCM_THRESHOLD:.0f} pcm")
    handles.append(h_beta)
    labels_leg.append(f"β_eff = {PCM_THRESHOLD:.0f} pcm")

    ax1.set_yscale("log")
    if all_positive_vals:
        ymin = max(float(np.min(all_positive_vals)) * 0.7, 1.0)
        ymax = float(np.max(all_positive_vals)) * 1.5
        ax1.set_ylim(ymin, ymax)
    ax1.set_ylabel(f"mean |{_error_label()}| (pcm, log scale)", fontsize=12)
    ax1.grid(True, alpha=0.3, which="both")
    ax1.legend(handles, labels_leg, fontsize=10, ncol=2, loc="upper right")

    # fraction below 650 pcm by bin
    if ax2 is not None:
        handles2, labels2 = [], []
        for i, split in enumerate(splits):
            sub = bins_df[bins_df["split"] == split].set_index("bin").reindex(bins)
            c = colors.get(split, f"C{i}")
            if frac650_before_col in sub.columns:
                yb = np.asarray(sub[frac650_before_col].values, dtype=float)
                h_b, = ax2.plot(x, yb, ls="--", marker="o", color=c,
                                alpha=0.75, label=f"{split} before")
                handles2.append(h_b)
                labels2.append(f"{split} before")
                if (frac650_before_err_col is not None
                        and frac650_before_err_col in sub.columns):
                    yerr = np.asarray(sub[frac650_before_err_col].values, dtype=float)
                    ax2.errorbar(x, yb, yerr=yerr, fmt="none", ecolor=c,
                                 elinewidth=1, capsize=2, alpha=0.55)
            if frac650_after_col in sub.columns:
                ya = np.asarray(sub[frac650_after_col].values, dtype=float)
                h_a, = ax2.plot(x, ya, ls="-", marker="s", color=c,
                                label=f"{split} after")
                handles2.append(h_a)
                labels2.append(f"{split} after")
                if (frac650_after_err_col is not None
                        and frac650_after_err_col in sub.columns):
                    yerr = np.asarray(sub[frac650_after_err_col].values, dtype=float)
                    ax2.errorbar(x, ya, yerr=yerr, fmt="none", ecolor=c,
                                 elinewidth=1, capsize=2, alpha=0.7)
        ax2.set_ylim(-0.02, 1.05)
        ax2.set_ylabel(f"frac |{_error_label()}| < {PCM_THRESHOLD:.0f} pcm", fontsize=12)
        ax2.grid(True, alpha=0.3)
        if handles2:
            ax2.legend(handles2, labels2, fontsize=10, ncol=2, loc="lower right")

    # tick labels from first split's edges
    edge_src = bins_df[bins_df["split"] == splits[0]].set_index("bin").reindex(bins)
    labels = [f"[{lo:.2f},{hi:.2f})" for lo, hi in zip(edge_src[lo_col], edge_src[hi_col])]
    if labels:
        # close last interval visually
        labels[-1] = labels[-1][:-1] + "]"
    ax_bottom = ax2 if ax2 is not None else ax1
    ax_bottom.set_xlabel("keff bin", fontsize=12)
    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels(labels, rotation=35, ha="right", fontsize=11)

    # small annotation of overall mean/median (+ overall frac_below_650 after when present)
    note_parts = []
    for _, r in overall_df.iterrows():
        if "keff_mean_mean" in overall_df.columns:
            part = (
                f"{r['split']}: μ={r['keff_mean_mean']:.3f}±{r.get('keff_mean_std', 0):.3f} "
                f"(n_runs={int(r.get('n_runs', 1))})"
            )
            if "frac_below_650_after_mean" in overall_df.columns and pd.notna(
                    r.get("frac_below_650_after_mean")
            ):
                part += (
                    f", |{_error_label()}|<{PCM_THRESHOLD:.0f}={r['frac_below_650_after_mean']:.2f}"
                    f"±{r.get('frac_below_650_after_std', 0):.2f}"
                )
            note_parts.append(part)
        else:
            part = (
                f"{r['split']}: μ={r['keff_mean']:.3f}, med={r['keff_median']:.3f} "
                f"(n={r['n_samples']})"
            )
            if "frac_below_650_after" in overall_df.columns and pd.notna(
                    r.get("frac_below_650_after")
            ):
                part += f", |{_error_label()}|<{PCM_THRESHOLD:.0f}={r['frac_below_650_after']:.2f}"
            note_parts.append(part)
    fig.text(0.5, 0.01, "  |  ".join(note_parts), ha="center", va="bottom",
             fontsize=10, color="dimgray")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Plot saved → {out_path}")


def _resolve_run_dir(logs_root, run_id, train_size, seed):
    run_dir = os.path.join(logs_root, run_id)
    if os.path.isdir(run_dir):
        return run_dir
    alt = os.path.join(logs_root, f"train_{train_size}_seed_{seed}")
    return alt if os.path.isdir(alt) else run_dir


def _load_split_frames_for_run(run_dir, out_dir, train_size, seed, best_epoch=None):
    """Build train/val/test frames for one run (epoch logs + comparison/split_log)."""
    split_frames = {}
    for split in ("train", "val"):
        paired = _pair_before_after_from_epoch_log(
            os.path.join(run_dir, f"keff_epoch_log_{split}.csv"),
            best_epoch=best_epoch,
        )
        if paired is None or paired.empty:
            paired = _frame_from_split_log(run_dir, split)
        if paired is not None and not paired.empty:
            split_frames[split] = paired

    # Test: per-run comparison (all seeds) → representative CSV → keff-only split_log.
    # Per-run files are required for meaningful avg_delta_rho_*_std on the fixed test set.
    test_paired = None
    for cand in (
        _run_test_comparison_path(out_dir, train_size, seed),
        _rep_test_comparison_path(out_dir, train_size, seed),
    ):
        test_paired = _pair_before_after_from_comparison_csv(cand)
        if test_paired is not None and not test_paired.empty:
            break
    if test_paired is None or test_paired.empty:
        test_paired = _frame_from_split_log(run_dir, "test")
    if test_paired is not None and not test_paired.empty:
        split_frames["test"] = test_paired
    return split_frames


def generate_keff_distribution_analysis(logs_root, out_dir, df_rows=None, run_eval_cache=None):
    """
    For every run under each train_size, summarize keff across train/val/test,
    then write folder-level aggregates as mean ± std across seeds:

      - keff_dist_overall_by_split.csv  (includes frac_below_650 before/after)
      - keff_dist_bins_by_split.csv     (sample fraction + frac_below_650 per keff bin)

    Also keeps a representative-seed detail CSV/plot for each train_size.
    """
    if df_rows is None:
        metrics_csv = os.path.join(out_dir, OUTPUT_FILENAME)
        if not os.path.exists(metrics_csv):
            print(f"[keff-dist] no metrics CSV at {metrics_csv} — skip")
            return
        df_rows = pd.read_csv(metrics_csv)
    if df_rows is None or df_rows.empty:
        print("[keff-dist] empty run metrics — skip")
        return

    run_eval_cache = run_eval_cache or {}
    reps = select_representative_runs(df_rows)
    os.makedirs(out_dir, exist_ok=True)

    # ── Pass 1: load all run split frames ─────────────────────────────────────
    run_payloads = []  # list of dicts
    all_keff_values = []
    for _, row in df_rows.sort_values(["train_size", "seed"]).iterrows():
        train_size = int(row["train_size"])
        seed = int(row["seed"])
        run_id = row["run_id"]
        best_epoch = row.get("best_epoch", None)
        info = run_eval_cache.get(run_id, {})
        run_dir = info.get("run_dir") or _resolve_run_dir(logs_root, run_id, train_size, seed)
        if not os.path.isdir(run_dir):
            print(f"  [keff-dist] missing run dir for {run_id} — skip")
            continue

        split_frames = _load_split_frames_for_run(
            run_dir, out_dir, train_size, seed, best_epoch=best_epoch
        )
        if not split_frames:
            print(f"  [keff-dist] nothing usable in {run_id} — skip")
            continue
        for frame in split_frames.values():
            all_keff_values.append(frame["keff_openmc"].astype(float).to_numpy())
        run_payloads.append(dict(
            train_size=train_size, seed=seed, run_id=run_id,
            best_epoch=best_epoch, run_dir=run_dir, split_frames=split_frames,
        ))

    if not run_payloads:
        print("[keff-dist] no runs loaded — skip")
        return

    shared_edges = _keff_bin_edges(np.concatenate(all_keff_values), n_bins=N_KEFF_BINS)
    print(f"\n[keff-dist] shared bin edges over {len(run_payloads)} runs: "
          f"[{shared_edges[0]:.4f}, {shared_edges[-1]:.4f}] × {N_KEFF_BINS} bins")

    # ── Pass 2: per-run summarize with shared edges ──────────────────────────
    per_run_overall, per_run_bins = [], []
    test_rho_coverage = 0
    for payload in run_payloads:
        overall_df, bins_df, _ = _summarize_keff_distribution(
            payload["split_frames"], n_bins=N_KEFF_BINS, edges=shared_edges
        )
        overall_df.insert(0, "train_size", payload["train_size"])
        overall_df.insert(1, "seed", payload["seed"])
        overall_df.insert(2, "run_id", payload["run_id"])
        bins_df.insert(0, "train_size", payload["train_size"])
        bins_df.insert(1, "seed", payload["seed"])
        bins_df.insert(2, "run_id", payload["run_id"])
        per_run_overall.append(overall_df)
        per_run_bins.append(bins_df)

        test_frame = payload["split_frames"].get("test")
        if test_frame is not None and np.isfinite(test_frame["delta_rho_after"]).any():
            test_rho_coverage += 1

        # Detail files + plot only for the representative seed of each train_size
        if reps.get(payload["train_size"]) == payload["run_id"]:
            prefix = f"rep_train{payload['train_size']}_seed{payload['seed']}"
            overall_path = os.path.join(out_dir, f"{prefix}_keff_dist_overall.csv")
            bins_path = os.path.join(out_dir, f"{prefix}_keff_dist_bins.csv")
            overall_df.to_csv(overall_path, index=False)
            bins_df.to_csv(bins_path, index=False)
            print(f"\n=== keff distribution (representative) train_size={payload['train_size']}, "
                  f"seed={payload['seed']} ===")
            print(f"  Overall → {overall_path}")
            print(overall_df[["split", "n_samples", "keff_mean", "keff_median",
                              "delta_rho_before_mean", "delta_rho_after_mean",
                              "frac_below_650_before", "frac_below_650_after"]].to_string(index=False))
            print(f"  Bins → {bins_path}")
            _plot_keff_distribution(
                bins_df, overall_df,
                out_path=os.path.join(out_dir, f"{prefix}_keff_dist_by_split.png"),
                title=(f"keff distribution by split "
                       f"(train_size={payload['train_size']}, seed={payload['seed']})"),
            )

    if test_rho_coverage < len(run_payloads):
        print(
            f"  [keff-dist] WARNING: only {test_rho_coverage}/{len(run_payloads)} runs have "
            f"test Δρ (need run_train*_seed*_keff_comparison.csv). "
            f"Re-run without --keff-dist-only to fill test *_std columns."
        )

    # ── Pass 3: mean ± std across seeds ──────────────────────────────────────
    overall_agg, bins_agg = _aggregate_keff_dist_across_runs(per_run_overall, per_run_bins)
    overall_path = os.path.join(out_dir, "keff_dist_overall_by_split.csv")
    bins_path = os.path.join(out_dir, "keff_dist_bins_by_split.csv")
    overall_agg.to_csv(overall_path, index=False)
    bins_agg.to_csv(bins_path, index=False)
    print(f"\n[keff-dist] cross-seed mean±std → {overall_path}")
    print(f"[keff-dist] cross-seed bin fractions → {bins_path}")
    if not overall_agg.empty:
        show = [c for c in ("train_size", "split", "n_runs",
                            "keff_mean_mean", "keff_mean_std",
                            "keff_median_mean", "keff_median_std",
                            "n_samples_mean",
                            "frac_below_650_after_mean", "frac_below_650_after_std")
                if c in overall_agg.columns]
        print(overall_agg[show].to_string(index=False))
    if not bins_agg.empty:
        show_b = [c for c in ("train_size", "split", "bin", "n_runs",
                              "frac_pct_mean", "frac_pct_std",
                              "n_samples_mean", "n_samples_std",
                              "frac_below_650_after_mean", "frac_below_650_after_std")
                  if c in bins_agg.columns]
        print(bins_agg[show_b].to_string(index=False))

    # Aggregate plot per train_size (mean fraction ± std error bars)
    for train_size, sub in bins_agg.groupby("train_size"):
        osub = overall_agg[overall_agg["train_size"] == train_size]
        # Map aggregate column names into the plot helper
        plot_bins = sub.copy()
        plot_bins = plot_bins.rename(columns={
            "frac_pct_mean": "pct_samples",
            "frac_pct_std": "pct_samples_std",
            "keff_lo_mean": "keff_lo",
            "keff_hi_mean": "keff_hi",
            # These columns hold delta_rho or delta_k values depending on mode;
            # the plot helper reads them via its before_col/after_col args.
            "avg_delta_rho_before_mean": "avg_delta_rho_before",
            "avg_delta_rho_after_mean": "avg_delta_rho_after",
            "frac_below_650_before_mean": "frac_below_650_before",
            "frac_below_650_after_mean": "frac_below_650_after",
            "frac_below_650_before_std": "frac_below_650_before_std",
            "frac_below_650_after_std": "frac_below_650_after_std",
        })
        _plot_keff_distribution(
            plot_bins, osub,
            out_path=os.path.join(out_dir, f"keff_dist_by_split_train{train_size}_mean.png"),
            title=f"keff distribution by split (mean±std over seeds, train_size={train_size})",
            pct_col="pct_samples", pct_err_col="pct_samples_std",
            before_col="avg_delta_rho_before", after_col="avg_delta_rho_after",
            frac650_before_col="frac_below_650_before",
            frac650_after_col="frac_below_650_after",
            frac650_before_err_col="frac_below_650_before_std",
            frac650_after_err_col="frac_below_650_after_std",
            lo_col="keff_lo", hi_col="keff_hi",
        )

# ─────────────────────────────────────────────────────────────────────────────
# Validation-set study (optional)
# Mirrors the representative test-set comparison, but reads keff_epoch_log_val.csv
# directly (epoch 0 and best epoch) instead of re-running the solver — those
# keff_peds values were already logged at training time. Since that log's
# "sample_idx" is a LOCAL index into the validation subset (not the global
# dataset index), the 6 parameters are recovered by matching keff_openmc
# against the full dataset's keffs array (values are unique per geometry).
# ─────────────────────────────────────────────────────────────────────────────
def _match_keff_to_dataset(target_keff, keffs_full, tol=VAL_KEFF_MATCH_TOL):
    """Find the dataset row whose keff is closest to target_keff.
    Returns (idx, diff) or (None, diff) if the closest match exceeds tol."""
    diffs = np.abs(keffs_full - target_keff)
    idx = int(np.argmin(diffs))
    best_diff = float(diffs[idx])
    n_close = int(np.sum(diffs <= tol))
    if best_diff > tol:
        return None, best_diff, n_close
    return idx, best_diff, n_close


def build_val_keff_comparison_df(run_dir, best_epoch, keffs_full, rawparams_full,
                                  tol=VAL_KEFF_MATCH_TOL):
    """Build the initial-vs-final comparison dataframe for the validation set,
    sourced from keff_epoch_log_val.csv (epoch 0 and best_epoch rows only),
    backtracing parameters via keff matching against the full dataset."""
    log_path = os.path.join(run_dir, "keff_epoch_log_val.csv")
    if not os.path.exists(log_path):
        print(f"  [warn] no keff_epoch_log_val.csv found in {run_dir} — skipping val study for this run")
        return None

    val_df = pd.read_csv(log_path)
    val_df = val_df[val_df["epoch"].isin([0, best_epoch])].copy()
    if val_df.empty:
        print(f"  [warn] keff_epoch_log_val.csv has no rows for epoch 0 or {best_epoch} — skipping")
        return None

    n_dropped, n_ambiguous = 0, 0
    records = []
    for _, row in val_df.iterrows():
        idx, diff, n_close = _match_keff_to_dataset(float(row["keff_openmc"]), keffs_full, tol=tol)
        if idx is None:
            n_dropped += 1
            continue
        if n_close > 1:
            n_ambiguous += 1
        kp, kr = float(row["keff_peds"]), float(row["keff_openmc"])
        signed_err = _signed_error_pcm(kp, kr)
        ecol, scol = _error_col_name(), _signed_error_col_name()
        rec = dict(epoch=int(row["epoch"]), sample_idx=idx,
                   keff_openmc=kr, keff_peds=kp,
                   **{ecol: abs(signed_err), scol: signed_err})
        for j, col in enumerate(PARAM_COLS):
            rec[col] = float(rawparams_full[idx, j])
        records.append(rec)

    if n_dropped:
        print(f"  [warn] {n_dropped} val-log sample(s) had no keff match within tol={tol} — dropped")
    if n_ambiguous:
        print(f"  [warn] {n_ambiguous} val-log sample(s) matched multiple dataset rows within tol={tol} "
              f"— used the closest match")

    return pd.DataFrame(records)


def generate_val_representative_outputs(df_rows: pd.DataFrame, run_eval_cache: dict, val_out_dir: str):
    """Validation-set counterpart of generate_representative_comparison_plots.
    Reuses the SAME representative run already chosen from test-set metrics,
    reads its keff_epoch_log_val.csv instead of re-running the solver, and
    produces the same representative-only outputs (comparison CSV, keff
    scatter, parallel coords, and optionally the error-vs-keff plot)."""
    if df_rows.empty:
        return
    os.makedirs(val_out_dir, exist_ok=True)
    reps = select_representative_runs(df_rows)
    for train_size, run_id in sorted(reps.items()):
        info = run_eval_cache.get(run_id)
        if info is None:
            print(f"  [warn] representative run {run_id} missing from eval cache — skipping val study")
            continue

        seed = info["seed"]
        best_epoch = info["best_epoch"]
        try:
            best_epoch = int(best_epoch)
        except (TypeError, ValueError):
            best_epoch = 1
        print(f"\n=== Validation-set study for train_size={train_size}: {run_id} "
              f"(same representative run as test set) ===")

        comp_df = build_val_keff_comparison_df(
            run_dir=info["run_dir"], best_epoch=best_epoch,
            keffs_full=info["keffs_full"], rawparams_full=info["rawparams_full"],
        )
        if comp_df is None or comp_df.empty:
            continue

        prefix = f"rep_train{train_size}_seed{seed}"
        comp_df.to_csv(os.path.join(val_out_dir, f"{prefix}_keff_comparison.csv"), index=False)

        both = comp_df[comp_df["epoch"].isin([0, best_epoch])]
        ecol = _detect_error_col(comp_df)
        vmin, vmax = both[ecol].min(), both[ecol].max()
        comp_df_plot = comp_df.copy()
        if ecol != "delta_rho_pcm":
            comp_df_plot["delta_rho_pcm"] = comp_df_plot[ecol]
        os.makedirs(os.path.join(val_out_dir, "representative_plots"), exist_ok=True)
        plot_keff_scatter(comp_df_plot, epoch=0, vmin=vmin, vmax=vmax,
                           save_path=os.path.join(val_out_dir, "representative_plots", f"{prefix}_keff_scatter_initial.png"))
        plot_keff_scatter(comp_df_plot, epoch=best_epoch, vmin=vmin, vmax=vmax,
                           save_path=os.path.join(val_out_dir, "representative_plots", f"{prefix}_keff_scatter_final.png"))

        plot_parallel_coords(comp_df_plot, epoch=0,
                              save_path=os.path.join(val_out_dir, "representative_plots", f"{prefix}_parallel_coords_initial.png"))
        plot_parallel_coords(comp_df_plot, epoch=best_epoch,
                              save_path=os.path.join(val_out_dir, "representative_plots", f"{prefix}_parallel_coords_final.png"))

        if GENERATE_ERROR_VS_KEFF_PLOTS:
            plot_error_vs_keff(comp_df_plot, epoch=0,
                                red_frac=ERROR_VS_KEFF_RED_FRAC, orange_frac=ERROR_VS_KEFF_ORANGE_FRAC,
                                save_path=os.path.join(val_out_dir, "representative_plots", f"{prefix}_error_vs_keff_initial.png"))
            plot_error_vs_keff(comp_df_plot, epoch=best_epoch,
                                red_frac=ERROR_VS_KEFF_RED_FRAC, orange_frac=ERROR_VS_KEFF_ORANGE_FRAC,
                                save_path=os.path.join(val_out_dir, "representative_plots", f"{prefix}_error_vs_keff_final.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Plotting functions
# ─────────────────────────────────────────────────────────────────────────────

METRIC_COLS = [
    "test_mse_k", "test_mae_k", "test_mean_pcm", "test_median_pcm",
    "test_p95_pcm", "test_std_pcm", "test_rmse_pcm",
    "test_frac_below_650", "test_frac_below_100",
]


def summarize(csv_path, out_csv=None):
    df = pd.read_csv(csv_path)

    n_seeds = df.groupby("train_size")["seed"].nunique().rename("n_seeds")

    agg = df.groupby("train_size")[METRIC_COLS].agg(["mean", "std"])
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.join(n_seeds).reset_index().sort_values("train_size")
    agg["nn_correction"] = "yes"

    priority_metrics = ["test_mean_pcm", "test_median_pcm", "test_rmse_pcm",
                         "test_frac_below_650", "test_frac_below_100"]
    other_metrics = ["test_mse_k", "test_mae_k"]
    remaining_metrics = ["test_p95_pcm", "test_std_pcm"]

    ordered_cols = ["train_size", "n_seeds", "nn_correction"]
    for m in priority_metrics:
        ordered_cols += [f"{m}_mean", f"{m}_std"]
    for m in other_metrics:
        ordered_cols += [f"{m}_mean", f"{m}_std"]
    for m in remaining_metrics:
        ordered_cols += [f"{m}_mean", f"{m}_std"]
    # Keep only columns that were actually computed (handles CSVs from older runs).
    ordered_cols = [c for c in ordered_cols if c in agg.columns]

    agg = agg[ordered_cols]

    rep_pattern = os.path.join(
        os.path.dirname(csv_path) or ".",
        "rep_train*_seed*_keff_comparison.csv",
    )
    # A folder can accumulate stale rep_train<N>_seed<S>_*.csv files from earlier
    # runs where a different seed was picked as the representative for that
    # train_size. Keep only the most-recently-written file per train_size.
    latest_rep_by_train_size = {}
    for rep_path in glob.glob(rep_pattern):
        m = re.search(r"rep_train(\d+)_seed(\d+)_keff_comparison\.csv$", os.path.basename(rep_path))
        if m is None:
            continue
        train_size = int(m.group(1))
        mtime = os.path.getmtime(rep_path)
        prev = latest_rep_by_train_size.get(train_size)
        if prev is None or mtime > prev[1]:
            latest_rep_by_train_size[train_size] = (rep_path, mtime)

    baseline_rows = []
    for train_size, (rep_path, _mtime) in sorted(latest_rep_by_train_size.items()):
        rep_df = pd.read_csv(rep_path)
        rep_df = rep_df[rep_df["epoch"] == 0].copy()
        if rep_df.empty:
            continue

        k_ref = rep_df["keff_openmc"].astype(float).to_numpy()
        k_pred = rep_df["keff_peds"].astype(float).to_numpy()
        ecol = _detect_error_col(rep_df)
        delta_pcm = rep_df[ecol].astype(float).to_numpy()
        # Signed error for RMSE: use the stored signed column if available,
        # otherwise recompute from keff columns.
        scol = _signed_error_col_name()
        if scol in rep_df.columns:
            signed_pcm = rep_df[scol].astype(float).to_numpy()
        elif "keff_peds" in rep_df.columns:
            signed_pcm = np.array([
                _signed_error_pcm(float(kp), float(kr))
                for kp, kr in zip(rep_df["keff_peds"], rep_df["keff_openmc"])
            ])
        else:
            signed_pcm = delta_pcm  # fallback: use absolute values
        baseline_rows.append({
            "train_size": train_size,
            "n_seeds": 1,
            "nn_correction": "no",
            "test_mean_pcm_mean": float(np.mean(delta_pcm)),
            "test_mean_pcm_std": 0.0,
            "test_median_pcm_mean": float(np.median(delta_pcm)),
            "test_median_pcm_std": 0.0,
            "test_frac_below_650_mean": float(np.mean(delta_pcm < 650.0)),
            "test_frac_below_650_std": 0.0,
            "test_frac_below_100_mean": float(np.mean(delta_pcm < 100.0)),
            "test_frac_below_100_std": 0.0,
            "test_mse_k_mean": float(np.mean((k_pred - k_ref) ** 2)),
            "test_mse_k_std": 0.0,
            "test_mae_k_mean": float(np.mean(np.abs(k_pred - k_ref))),
            "test_mae_k_std": 0.0,
            "test_p95_pcm_mean": float(np.percentile(delta_pcm, 95.0)),
            "test_p95_pcm_std": 0.0,
            "test_std_pcm_mean": float(np.std(delta_pcm)),
            "test_std_pcm_std": 0.0,
            "test_rmse_pcm_mean": float(np.sqrt(np.mean(signed_pcm ** 2))),
            "test_rmse_pcm_std": 0.0,
        })

    if baseline_rows:
        agg = pd.concat([agg, pd.DataFrame(baseline_rows)], ignore_index=True)
        agg = agg.sort_values(["train_size", "nn_correction"]).reset_index(drop=True)
        print(f"Added epoch-0 baseline rows for {len(baseline_rows)} train sizes.")

    if out_csv is None:
        out_csv = os.path.join(os.path.dirname(csv_path) or ".",
                                "new_test_metrics_summary_by_train_size.csv")
    agg.to_csv(out_csv, index=False)

    print(f"Summary saved → {out_csv}\n")
    cols_to_show = [c for c in ["train_size", "nn_correction", "n_seeds",
                                "test_mean_pcm_mean", "test_mean_pcm_std",
                                "test_rmse_pcm_mean",
                                "test_median_pcm_mean", "test_frac_below_650_mean"]
                    if c in agg.columns]
    print(agg[cols_to_show].to_string(index=False))
    return agg

plt.rc('font', size=16)
plt.rc('axes', labelsize=18)
plt.rc('xtick', labelsize=16)
plt.rc('ytick', labelsize=16)
plt.rc('legend', fontsize=16)
plt.rc('lines', markersize=8, linewidth=2)

def plot_scaling(agg, out_path, metric="test_mean_pcm", band="std", logx=True):
    """
    band: 'std'    -> mean +/- 1 std across seeds
          'minmax' -> mean with min/max whiskers (more honest when n_seeds is small)
    """
    x = agg["train_size"].values
    mean = agg[f"{metric}_mean"].values

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, mean, "-o", color="C0", label="mean across seeds")

    if band == "std":
        std = agg[f"{metric}_std"].fillna(0).values
        ax.fill_between(x, mean - std, mean + std, alpha=0.25, color="C0", label="± 1 std")
    else:
        lo, hi = agg[f"{metric}_min"].values, agg[f"{metric}_max"].values
        ax.fill_between(x, lo, hi, alpha=0.2, color="C0", label="min–max across seeds")

    if "pcm" in metric:
        ax.axhline(650, ls="--", color="grey", alpha=0.7, label="β_eff = 650 pcm")

    if logx:
        #ax.set_xscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([str(int(v)) for v in x])
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        ax.minorticks_off()  # optional: removes extra unlabeled log ticks
    ax.set_xlabel("Training set size")
    if metric == "test_frac_below_650":
        ax.set_ylabel("fraction below 650 pcm")
    elif metric == "test_mean_pcm":
        ax.set_ylabel(f"Mean |{_error_label()}| (pcm)")
    else:
        ax.set_ylabel(metric.replace("test_", "").replace("_", " "))
    #ax.set_title(f"Test-set {metric.replace('test_', '').replace('_', ' ')} vs. training size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Plot saved → {out_path}")


def plot_n_seeds_annotated(agg, out_path, metric="test_mean_pcm"):
    """Same as plot_scaling but annotates each point with how many seeds back it —
    useful since std is unreliable with very few seeds."""
    x = agg["train_size"].values
    mean = agg[f"{metric}_mean"].values
    std  = agg[f"{metric}_std"].fillna(0).values
    n    = agg["n_seeds"].values

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(x, mean, yerr=std, fmt="-o", color="C1", capsize=4)
    for xi, yi, ni in zip(x, mean, n):
        ax.annotate(f"n={ni}", (xi, yi), textcoords="offset points",
                     xytext=(6, 6), fontsize=8, color="grey")
    ax.set_xscale("log")
    ax.set_xlabel("Training set size")
    if metric == "test_frac_below_650":
        ax.set_ylabel("fraction below 650 pcm")
    else:
        ax.set_ylabel(metric.replace("test_", "").replace("_", " "))
    ax.set_title(f"{metric.replace('test_', '').replace('_', ' ')} vs. training size (seed count annotated)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Plot saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    out_dir = os.path.join(LOGS_ROOT, OUTPUT_DIRNAME)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, OUTPUT_FILENAME)

    if GENERATE_KEFF_DIST_ONLY:
        print(f"[keff-dist-only] skipping checkpoint eval; analyzing existing outputs under {out_dir}")
        metrics_csv = os.path.join(out_dir, OUTPUT_FILENAME)
        df_rows = pd.read_csv(metrics_csv) if os.path.exists(metrics_csv) else None
        generate_keff_distribution_analysis(LOGS_ROOT, out_dir, df_rows=df_rows)
        return

    runs = list(find_runs(LOGS_ROOT))
    if not runs:
        print(f"No runs found under {LOGS_ROOT}")
        return

    print(f"Using split log: {SPLIT_LOG_NAME}")
    print(f"Results dirname: {OUTPUT_DIRNAME}")

    # Cache loaded dataset arrays by resolved path (avoid reloading the same file).
    dataset_cache = {}
    test_payload_cache = {}
    rows = []
    run_eval_cache = {}

    for run_id, train_size, seed, ckpt_path, run_dir in runs:
        print(f"\n=== Evaluating {run_id} ===")
        try:
            ctx = get_run_eval_context(run_dir)
            train_idx, _, test_idx, split_source = _read_indices_from_split_log(
                run_dir, train_size, seed, split_log_name=SPLIT_LOG_NAME,
            )

            # Resolve and cache the dataset for this run.
            if DATA_FILEPATH_OVERRIDE is not None:
                dataset_key = DATA_FILEPATH_OVERRIDE
            else:
                detected = _detect_data_filepath(run_dir)
                dataset_key = detected if detected is not None else _DEFAULT_DATA_FILEPATH
            if dataset_key not in dataset_cache:
                dataset_cache[dataset_key] = load_dataset_arrays(run_dir)
            geoms, keffs, rawparams = dataset_cache[dataset_key]
            print(f"  Using dataset: {os.path.basename(dataset_key)}")

            test_payload = get_test_payload(
                geoms=geoms, keffs=keffs, rawparams=rawparams,
                test_idx=test_idx, split_source=split_source,
                cache=test_payload_cache, ctx=ctx,
            )
            norm_stats = get_or_build_norm_stats(
                LOGS_ROOT=LOGS_ROOT, train_size=train_size, seed=seed,
                rawparams=rawparams, train_idx=train_idx, ctx=ctx,
            )
            state, meta = load_checkpoint(ckpt_path)
            model = build_model_from_metadata(ctx, meta, seed_for_init=seed,
                                              run_dir=run_dir)
            nnx.update(model, jax.tree_util.tree_map(jnp.asarray, state))

            m, k_pred_final = evaluate_on_test(ctx, model, test_payload, norm_stats)
            # Compute RMSE in pcm from signed per-sample errors.
            _keffs_test = test_payload["keffs"]
            _signed_errs = np.array([
                _signed_error_pcm(float(kp), float(kr))
                for kp, kr in zip(k_pred_final, _keffs_test)
            ])
            rmse_pcm = float(np.sqrt(np.mean(_signed_errs ** 2)))
        except Exception as e:
            print(f"  [FAILED] {run_id}: {e}")
            continue

        rows.append(dict(
            run_id=run_id, train_size=train_size, seed=seed,
            best_epoch=meta.get("epoch", ""),
            train_time_val_mean_pcm=meta.get("val_mean_pcm", ""),
            test_mse_k=m["mse_k"], test_mae_k=m["MAE_k"],
            test_mean_pcm=m["mean_pcm"], test_median_pcm=m["median_pcm"],
            test_p95_pcm=m["p95_pcm"], test_std_pcm=m["std_pcm"],
            test_rmse_pcm=rmse_pcm,
            test_frac_below_650=m["frac_below_650"], test_frac_below_100=m["frac_below_100"],
        ))
        run_eval_cache[run_id] = dict(
            ctx=ctx, test_payload=test_payload, k_pred_final=k_pred_final,
            train_size=train_size, seed=seed, best_epoch=meta.get("epoch", ""),
            run_dir=run_dir, keffs_full=keffs, rawparams_full=rawparams,
        )
        print(f"  best_epoch={meta.get('epoch','?')}  "
              f"test_mean_pcm={m['mean_pcm']:.1f}  test_median_pcm={m['median_pcm']:.1f}")

    fieldnames = ["run_id", "train_size", "seed", "best_epoch", "train_time_val_mean_pcm",
                  "test_mse_k", "test_mae_k", "test_mean_pcm", "test_median_pcm",
                  "test_p95_pcm", "test_std_pcm", "test_rmse_pcm",
                  "test_frac_below_650", "test_frac_below_100"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved aggregate test metrics for {len(rows)} runs → {out_csv}")

    print("\nWriting per-run test keff comparisons (needed for test-bin std) …")
    save_all_run_test_comparisons(run_eval_cache, out_dir)

    print("\nGenerating representative-seed initial-vs-final keff comparison plots …")
    generate_representative_comparison_plots(pd.DataFrame(rows), run_eval_cache, out_dir)

    print("\nGenerating keff distribution analysis across train / val / test …")
    generate_keff_distribution_analysis(
        LOGS_ROOT, out_dir, df_rows=pd.DataFrame(rows), run_eval_cache=run_eval_cache,
    )

    if GENERATE_VAL_STUDY:
        val_out_dir = os.path.join(LOGS_ROOT, VAL_STUDY_DIRNAME)
        print(f"\nGenerating validation-set study (representative runs) → {val_out_dir} …")
        generate_val_representative_outputs(pd.DataFrame(rows), run_eval_cache, val_out_dir)


def evaluate_study(
    logs_root: str | os.PathLike | None,
    *,
    output_filename: str = OUTPUT_FILENAME,
    split_log_name: str = SPLIT_LOG_NAME,
    results_dirname: str = OUTPUT_DIRNAME,
    use_current_peds: bool = USE_CURRENT_PEDS,
    with_error_scatter: bool = GENERATE_ERROR_VS_KEFF_PLOTS,
    with_val_study: bool = GENERATE_VAL_STUDY,
    keff_dist_only: bool = GENERATE_KEFF_DIST_ONLY,
    scaling_plots: bool = SCALING_PLOTS,
    use_delta_k: bool = USE_DELTA_K,
) -> str:
    """Run full test-set evaluation for a study folder (LOGS_ROOT).

    Returns the path to the results directory (…/testset_results by default).
    When *use_delta_k* is True the output dirname gets a ``_dk`` suffix so
    both sets of results can coexist for comparison.
    """
    global LOGS_ROOT, OUTPUT_FILENAME, SPLIT_LOG_NAME, OUTPUT_DIRNAME
    global USE_CURRENT_PEDS, GENERATE_ERROR_VS_KEFF_PLOTS, GENERATE_VAL_STUDY
    global GENERATE_KEFF_DIST_ONLY, SCALING_PLOTS, USE_DELTA_K

    if logs_root is None:
        raise ValueError("logs_root is required")

    USE_DELTA_K = use_delta_k
    # Automatically suffix the output directory so delta_rho and delta_k
    # results sit in distinct folders and can be compared side-by-side.
    effective_dirname = results_dirname
    if use_delta_k and not results_dirname.endswith("_dk"):
        effective_dirname = results_dirname + "_dk"

    LOGS_ROOT = os.path.abspath(str(logs_root))
    OUTPUT_FILENAME = output_filename
    SPLIT_LOG_NAME = split_log_name
    OUTPUT_DIRNAME = effective_dirname
    USE_CURRENT_PEDS = use_current_peds
    GENERATE_ERROR_VS_KEFF_PLOTS = with_error_scatter
    GENERATE_VAL_STUDY = with_val_study
    GENERATE_KEFF_DIST_ONLY = keff_dist_only
    SCALING_PLOTS = scaling_plots

    main()

    if keff_dist_only:
        return os.path.join(LOGS_ROOT, OUTPUT_DIRNAME)

    csv_path = os.path.join(LOGS_ROOT, OUTPUT_DIRNAME, OUTPUT_FILENAME)
    band = "std"
    agg = summarize(csv_path)
    out_dir = os.path.join(LOGS_ROOT, OUTPUT_DIRNAME)
    agg_plot = agg[agg["nn_correction"] == "yes"].copy() if "nn_correction" in agg.columns else agg

    if not agg_plot.empty and SCALING_PLOTS:
        plot_scaling(agg_plot, os.path.join(out_dir, "scaling_mean_pcm.png"),
                     metric="test_mean_pcm", band=band)
        plot_scaling(agg_plot, os.path.join(out_dir, "scaling_median_pcm.png"),
                     metric="test_median_pcm", band=band)
        plot_scaling(agg_plot, os.path.join(out_dir, "scaling_frac_below_650.png"),
                     metric="test_frac_below_650", band=band, logx=True)
    else:
        print("plots not wanted or not possible")

    return out_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate best checkpoints on the held-out test set using each run's code snapshot.",
    )
    parser.add_argument("logs_root", nargs="?",
                        help="Path to LOGS_ROOT directory")
    parser.add_argument("--out", default=OUTPUT_FILENAME)
    parser.add_argument(
        "--split-log",
        default=SPLIT_LOG_NAME,
        help="Split CSV filename inside each run folder "
             f"(default {SPLIT_LOG_NAME}; use alt_split_log.csv for per-seed alt tests)",
    )
    parser.add_argument(
        "--results-dirname",
        default=OUTPUT_DIRNAME,
        help=f"Subdirectory under LOGS_ROOT for outputs (default {OUTPUT_DIRNAME})",
    )
    parser.add_argument(
        "--use-current-peds",
        action="store_true",
        help="Ignore per-run code_snapshot_*.py and evaluate with current models/PEDS.py",
    )
    parser.add_argument(
        "--with-error-scatter",
        action="store_true",
        help="Also generate the (prediction - target) vs. target-keff scatter plot, "
             "with the worst 5%% in red and the next slice up to 10%% in orange, "
             "for each representative run at epoch 0 and best epoch.",
    )
    parser.add_argument(
        "--with-val-study",
        action="store_true",
        help="Also run the validation-set study: for each representative run (same "
             "one selected from test-set metrics), reads keff_epoch_log_val.csv at "
             "epoch 0 and best epoch, backtraces parameters via keff matching, and "
             f"saves comparison CSV/plots to LOGS_ROOT/{VAL_STUDY_DIRNAME}/.",
    )
    parser.add_argument(
        "--keff-dist-only",
        action="store_true",
        help="Skip checkpoint evaluation; only rebuild the train/val/test keff "
             "distribution tables/plots from existing epoch logs and "
             "rep_*_keff_comparison.csv under --results-dirname.",
    )
    args = parser.parse_args()
    evaluate_study(
        args.logs_root,
        output_filename=args.out,
        split_log_name=args.split_log,
        results_dirname=args.results_dirname,
        use_current_peds=args.use_current_peds,
        with_error_scatter=args.with_error_scatter,
        with_val_study=args.with_val_study,
        keff_dist_only=args.keff_dist_only,
    )
