"""Experiment configuration for a PEDS training run.

This is the only file to edit when setting up a run. ``models/PEDS.py`` imports
every knob from here, so the model code itself holds no experiment settings.

Launch a configured run with::

    sbatch run_jobby.sh        # which calls: python run_peds.py

Every knob that ``run_jobby.sh`` sweeps is overridable through an environment
variable, so one Slurm array can vary train size, seed and decay horizon
without editing this file.
"""

from __future__ import annotations

import os
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_list_int(name: str, default: list[int]) -> list[int]:
    raw = os.environ.get(name)
    if raw is None:
        return list(default)
    return [int(v) for v in raw.replace(",", " ").split()]


# ─────────────────────────────────────────────────────────────────────────────
# Project layout
# ─────────────────────────────────────────────────────────────────────────────
CONFIG_RUN_DIR = Path(__file__).resolve().parent        # config_and_run/
PROJECT_ROOT = CONFIG_RUN_DIR.parent                    # repository root
MODELS_DIR = PROJECT_ROOT / "models"                    # model implementations
MODEL_SCRIPT = MODELS_DIR / "PEDS.py"                   # training entry module

# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
DATA_FILENAME = os.environ.get("PEDS_DATA_FILENAME", "17jul_0.8_1.2.npz")
DATA_FILEPATH = PROJECT_ROOT / "data" / "highfidelity" / DATA_FILENAME

TRAIN_SIZE = _env_int("PEDS_TRAIN_SIZE", 1000)
VAL_SIZE = _env_int("PEDS_VAL_SIZE", 500)      # evaluated every epoch
TEST_SIZE = _env_int("PEDS_TEST_SIZE", 300)    # held out until the very end

# ─────────────────────────────────────────────────────────────────────────────
# Seeds — train/val vary across a study, test stays fixed so runs stay comparable
# ─────────────────────────────────────────────────────────────────────────────
SEED = _env_int("PEDS_SEED", 0)
TRAIN_SEED = SEED
VAL_SEED = SEED
TEST_SEED = _env_int("PEDS_TEST_SEED", 0)
HOLDOUT_SEED = TEST_SEED  # alias used by the non-param-stratified loaders

# ─────────────────────────────────────────────────────────────────────────────
# Train/val/test splitting
# ─────────────────────────────────────────────────────────────────────────────
# USE_PARAM_STRATIFIED wins if both stratification flags are on.
USE_PARAM_STRATIFIED = _env_bool("PEDS_USE_PARAM_STRATIFIED", True)
USE_BALANCED_RANGES = _env_bool("PEDS_USE_BALANCED_RANGES", False)
USE_SPLIT_CACHE = _env_bool("PEDS_USE_SPLIT_CACHE", False)
PARAM_STRAT_BINS = _env_int("PEDS_PARAM_STRAT_BINS", 5)
# Subdirectory under data/highfidelity holding cached train/val/test splits.
CACHE_DIR_NAME = "split_cache/.split_cache_"

# ─────────────────────────────────────────────────────────────────────────────
# Optimisation
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE = _env_int("PEDS_BATCH_SIZE", 32)
EPOCHS = _env_int("PEDS_EPOCHS", 70)
# Cosine horizon, decoupled from EPOCHS so sweeps can vary it independently.
DECAY_EPOCHS = _env_int("PEDS_DECAY_EPOCHS", 70)
LR_MAX = _env_float("PEDS_LR_MAX", 2e-4)   # cosine schedule peak
LR_MIN = _env_float("PEDS_LR_MIN", 5e-6)   # cosine schedule floor

# Checkpoint selection on an EMA of val mean |Δρ|, plus early stopping.
EMA_ALPHA = _env_float("PEDS_EMA_ALPHA", 0.1)
MIN_SAVE_EPOCH = _env_int("PEDS_MIN_SAVE_EPOCH", 30)  # skip early oscillations
PATIENCE = _env_int("PEDS_PATIENCE", 15)              # epochs without EMA gain

# ─────────────────────────────────────────────────────────────────────────────
# Architecture
# ─────────────────────────────────────────────────────────────────────────────
HIDDEN_SIZES = _env_list_int("PEDS_HIDDEN_SIZES", [128, 256, 128])
N_REGIONS = _env_int("PEDS_N_REGIONS", 3)
N_GROUPS = _env_int("PEDS_N_GROUPS", 2)  # energy groups (G)

# Hard clip on the NN log-ratio XS correction, applied in forward *and* backward.
LOG_RATIO_CLIP_LO = _env_float("PEDS_LOG_RATIO_CLIP_LO", -1.8)
LOG_RATIO_CLIP_HI = _env_float("PEDS_LOG_RATIO_CLIP_HI", 0.5)

# ─────────────────────────────────────────────────────────────────────────────
# Diagnostics
# ─────────────────────────────────────────────────────────────────────────────
# Custom-VJP vs finite-difference check before training.
# Writes grad_check_per_sample.csv and grad_check_summary.csv under LOG_DIR/gradient_FDcheck.
RUN_GRAD_CHECK = _env_bool("PEDS_RUN_GRAD_CHECK", True)
GRAD_CHECK_SAMPLE = _env_int("PEDS_GRAD_CHECK_SAMPLE", 0)

# ─────────────────────────────────────────────────────────────────────────────
# Logging and figures
# ─────────────────────────────────────────────────────────────────────────────
# Epochs at which per-sample XS subplot and flux snapshots are written.
XS_HEATMAP_EPOCHS = _env_list_int(
    "PEDS_XS_HEATMAP_EPOCHS", [1, max(int(EPOCHS / 2), 1), EPOCHS]
)
# Indices into the VAL set for per-sample xs_subplots and flux figures.
SUBPLOT_SAMPLE_INDICES = _env_list_int("PEDS_SUBPLOT_SAMPLE_INDICES", [22, 44, 55, 68, 82])
PARAM_NAMES = ["b4c_r", "cr_frac", "fuel_r", "enrichment", "f_mod", "water_r"]

# ─────────────────────────────────────────────────────────────────────────────
# Output location + parallelism
# ─────────────────────────────────────────────────────────────────────────────
STUDY_NAME = os.environ.get("PEDS_STUDY_NAME", "precise_param_strat")
RUNS_ROOT = Path(os.environ.get("PEDS_RUNS_ROOT", CONFIG_RUN_DIR / "RUNS"))

EXP_NAME = os.environ.get("PEDS_EXP_NAME", f"train_{TRAIN_SIZE}_seed_{SEED}")
LOG_DIR = RUNS_ROOT / STUDY_NAME / EXP_NAME
XS_DIR = LOG_DIR / "XS"
CHECKPOINT_DIR = LOG_DIR / "checkpoints"
BEST_CKPT_PATH = CHECKPOINT_DIR / "best_model.pkl"
LAST_CKPT_PATH = CHECKPOINT_DIR / "last_checkpoint.pkl"

# One solver worker per CPU, capped at the batch size (never more useful).
N_WORKERS = min(_env_int("SLURM_CPUS_PER_TASK", 32), BATCH_SIZE)


def summary() -> str:
    """Human-readable dump of the resolved configuration, for run logs."""
    lines = [
        "PEDS run configuration",
        f"  data          : {DATA_FILEPATH}",
        f"  sizes         : train={TRAIN_SIZE} val={VAL_SIZE} test={TEST_SIZE}",
        f"  seeds         : train/val={SEED} test={TEST_SEED} (fixed)",
        f"  split mode    : param_stratified={USE_PARAM_STRATIFIED} "
        f"balanced_ranges={USE_BALANCED_RANGES} cache={USE_SPLIT_CACHE}",
        f"  optimisation  : epochs={EPOCHS} decay_epochs={DECAY_EPOCHS} "
        f"batch={BATCH_SIZE} lr={LR_MAX:.1e}->{LR_MIN:.1e}",
        f"  checkpointing : min_save_epoch={MIN_SAVE_EPOCH} patience={PATIENCE} "
        f"ema_alpha={EMA_ALPHA}",
        f"  architecture  : hidden={HIDDEN_SIZES} n_regions={N_REGIONS} G={N_GROUPS}",
        f"  grad check    : {RUN_GRAD_CHECK} (sample {GRAD_CHECK_SAMPLE})",
        f"  workers       : {N_WORKERS}",
        f"  output        : {LOG_DIR}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
