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
  1. The TEST set itself (geoms/keffs/rawparams/phi_features) — built once,
     reused for every run, since it never changes.
  2. The per-(train_size, seed) normalization stats (phi_mean/std and
     log_xs_mean/std) — these depend on which samples ended up in THAT
     run's training set, so they're cached per (train_size, seed) pair
     the first time they're needed and reused on any later script run.

USAGE
-----
    python evaluate_test_metrics.py /path/to/LOGS
    python evaluate_test_metrics.py /path/to/LOGS --out my_summary.csv

Before running, edit the TODO below:
 sys.path.insert(...) — point this at the folder containing your
     training script.
=========================================================================
"""
import os
import re
import sys
import glob
import pickle
import csv
import hashlib
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from PEDS import (
    GEO,
    PEDSModel,
    _run_NT_solver,
    compute_phi_features,
    compute_batch_baselines,
    compute_metrics,
    data_loader,
    _DATA_FILEPATH,
)

# ── CONFIG: edit these paths once, here ────────────────────────────────────
LOGS_ROOT = "RUNS/study_LHS_0.8_bounds"
OUTPUT_DIRNAME = "testset_results"
OUTPUT_FILENAME = "test_metrics_all_runs.csv"

# Dataset path used to load arrays for evaluation.
# Set to None to auto-detect from each run's code_snapshot_*.py.
# Set to an explicit path (e.g. "../data/highfidelity/merged_lhs_0.8_1.2.npz")
# to force every run to use the same file regardless of what the snapshot says.
DATA_FILEPATH_OVERRIDE = None

RUN_PATTERN = re.compile(r"^train_(\d+)_seed_(\d+)$")
SNAPSHOT_DATA_RE = re.compile(r'_DATA_FILEPATH\s*=\s*os\.path\.join\([^)]*\)\s*$|_DATA_FILEPATH\s*=\s*["\']([^"\']+)["\']', re.MULTILINE)
EVAL_BATCH_SIZE = 32


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


def _read_indices_from_split_log(run_dir, train_size, seed):
    """Load train/val/test sample indices from split_log.csv in this run folder."""
    csv_path = os.path.join(run_dir, "split_log.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"split_log.csv not found for run train={train_size}, seed={seed}: {csv_path}")
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
            f"split_log.csv for run train={train_size}, seed={seed} is missing "
            f"required split rows (train={len(train_idx)}, test={len(test_idx)})"
        )
    if len(train_idx) != train_size:
        print(
            f"  [warn] split_log train count ({len(train_idx)}) != folder train_size ({train_size}) "
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
    Read _DATA_FILEPATH from the run's code_snapshot_*.py.
    The snapshot stores a line like:
        _DATA_FILEPATH = os.path.join(PARENT_DIR, "data", "highfidelity", "some.npz")
    We extract the last quoted string on that line (the filename) and resolve it
    relative to the project root (two levels above the modules/ folder).
    Falls back to the global PEDS default if the snapshot cannot be parsed.
    """
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
                # Resolve: project root is two levels above THIS_DIR (modules/../..)
                THIS_DIR = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(THIS_DIR)
                candidate = os.path.join(project_root, "data", "highfidelity", filename)
                if os.path.exists(candidate):
                    return candidate
                # Maybe the snapshot stored a full subpath like "data/highfidelity/x.npz"
                candidate2 = os.path.join(project_root, filename)
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
            path = _DATA_FILEPATH
            source = "PEDS default (snapshot not parsed)"
    else:
        path = _DATA_FILEPATH
        source = "PEDS default"

    print(f"  Dataset: {path}  [{source}]")
    data = np.load(path, allow_pickle=True)
    return (
        np.array(data["params"], dtype=np.float32),
        np.array(data["keffs"], dtype=np.float32),
        np.array(data["params_raw"], dtype=np.float32),
    )


def get_test_payload(geoms, keffs, rawparams, test_idx, split_source, cache):
    """Build test payload from explicit test indices from split_log.csv."""
    sig = _indices_signature(test_idx)
    if sig in cache:
        return cache[sig]

    print(f"  Building TEST payload from {split_source} ({len(test_idx)} samples)…")
    test_geoms, test_keffs, test_rawparams = geoms[test_idx], keffs[test_idx], rawparams[test_idx]
    test_phi_features = compute_phi_features(test_rawparams)

    payload = dict(geoms=test_geoms, keffs=test_keffs, rawparams=test_rawparams,
                    phi_features=test_phi_features,
                    test_idx=test_idx,
                    split_source=split_source,
                    split_signature=sig)
    cache[sig] = payload
    return payload


# ─────────────────────────────────────────────────────────────────────────────
# Caching: per-(train_size, seed) normalization stats
# ─────────────────────────────────────────────────────────────────────────────
def get_or_build_norm_stats(LOGS_ROOT, train_size, seed, rawparams, train_idx):
    split_sig = _indices_signature(train_idx)
    cache_path = os.path.join(
        LOGS_ROOT, "_evaluation_cache", f"norm_stats_train{train_size}_seed{seed}_{split_sig[:12]}.pkl"
    )
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)  # ← this line is missing

    print(
        f"  Recomputing normalization stats from split_log train indices "
        f"for train_size={train_size}, seed={seed} …"
    )
    train_rawparams = rawparams[train_idx]

    train_phi_features = compute_phi_features(train_rawparams)
    phi_mean = train_phi_features.mean(axis=0).astype(np.float32)
    phi_std  = (train_phi_features.std(axis=0) + 1e-8).astype(np.float32)

    train_xs_baselines = compute_batch_baselines(train_rawparams, GEO)
    xs_np = np.array(train_xs_baselines)
    safe  = np.where(xs_np > 1e-10, xs_np, np.ones_like(xs_np))
    log_train = np.log(safe)
    log_xs_mean = log_train.mean(axis=0).astype(np.float32)
    log_xs_std  = (log_train.std(axis=0) + 1e-8).astype(np.float32)

    payload = dict(phi_mean=phi_mean, phi_std=phi_std,
                    log_xs_mean=log_xs_mean, log_xs_std=log_xs_std,
                    split_signature=split_sig)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
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


def build_model_from_metadata(metadata, seed_for_init=0):
    """Falls back to the v1 architecture defaults if metadata wasn't recorded."""
    hidden_sizes = metadata.get("hidden_sizes", [128, 256, 128])
    n_regions    = metadata.get("n_regions", 3)
    G            = metadata.get("G", GEO.G)
    n_phi_feats  = metadata.get("n_phi_feats", GEO.G * 3)
    rngs = nnx.Rngs(seed_for_init)
    return PEDSModel(hidden_sizes=hidden_sizes, n_regions=n_regions,
                      G=G, n_phi_feats=n_phi_feats, rngs=rngs)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_on_test(model, test_payload, norm_stats):
    geoms     = test_payload["geoms"]
    keffs     = test_payload["keffs"]
    rawparams = test_payload["rawparams"]

    phi_norm = ((test_payload["phi_features"] - norm_stats["phi_mean"])
                / norm_stats["phi_std"]).astype(np.float32)
    xs_baselines  = compute_batch_baselines(rawparams, GEO)
    log_xs_mean_j = jnp.array(norm_stats["log_xs_mean"])
    log_xs_std_j  = jnp.array(norm_stats["log_xs_std"])

    k_pred_all = []
    for bg, bk, br, bb, bphi in data_loader(
            geoms, keffs, rawparams, np.array(xs_baselines), phi_norm,
            batch_size=EVAL_BATCH_SIZE):
        xs_np = model.compute_xs(jnp.array(bg), jnp.array(bb), jnp.array(bphi),
                                  log_xs_mean_j, log_xs_std_j)
        for i in range(len(bg)):
            k, _, _, _ = _run_NT_solver(xs_np[i], np.array(br[i]),
                                         np.array([i], dtype=np.int32))
            k_pred_all.append(float(k))

    return compute_metrics(np.array(k_pred_all), keffs)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting functions
# ─────────────────────────────────────────────────────────────────────────────

METRIC_COLS = [
    "test_mse_k", "test_mae_k", "test_mean_pcm", "test_median_pcm",
    "test_p95_pcm", "test_std_pcm", "test_frac_below_650", "test_frac_below_100",
]


def summarize(csv_path, out_csv=None):
    df = pd.read_csv(csv_path)

    n_seeds = df.groupby("train_size")["seed"].nunique().rename("n_seeds")

    agg = df.groupby("train_size")[METRIC_COLS].agg(["mean", "std"])
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.join(n_seeds).reset_index().sort_values("train_size")

    priority_metrics = ["test_mean_pcm", "test_median_pcm",
                         "test_frac_below_650", "test_frac_below_100"]
    other_metrics = ["test_mse_k", "test_mae_k"]
    remaining_metrics = ["test_p95_pcm", "test_std_pcm"]

    ordered_cols = ["train_size", "n_seeds"]
    for m in priority_metrics:
        ordered_cols += [f"{m}_mean", f"{m}_std"]
    for m in other_metrics:
        ordered_cols += [f"{m}_mean", f"{m}_std"]
    for m in remaining_metrics:
        ordered_cols += [f"{m}_mean", f"{m}_std"]

    agg = agg[ordered_cols]

    if out_csv is None:
        out_csv = os.path.join(os.path.dirname(csv_path) or ".",
                                "new_test_metrics_summary_by_train_size.csv")
    agg.to_csv(out_csv, index=False)

    print(f"Summary saved → {out_csv}\n")
    cols_to_show = ["train_size", "n_seeds", "test_mean_pcm_mean", "test_mean_pcm_std",
                     "test_median_pcm_mean", "test_frac_below_650_mean"]
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
        ax.set_ylabel("Mean reactivity difference (pcm)")
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

    runs = list(find_runs(LOGS_ROOT))
    if not runs:
        print(f"No runs found under {LOGS_ROOT}")
        return

    # Cache loaded dataset arrays by resolved path (avoid reloading the same file).
    dataset_cache = {}
    test_payload_cache = {}
    rows = []

    for run_id, train_size, seed, ckpt_path, run_dir in runs:
        print(f"\n=== Evaluating {run_id} ===")
        try:
            train_idx, _, test_idx, split_source = _read_indices_from_split_log(run_dir, train_size, seed)

            # Resolve and cache the dataset for this run.
            if DATA_FILEPATH_OVERRIDE is not None:
                dataset_key = DATA_FILEPATH_OVERRIDE
            else:
                detected = _detect_data_filepath(run_dir)
                dataset_key = detected if detected is not None else _DATA_FILEPATH
            if dataset_key not in dataset_cache:
                dataset_cache[dataset_key] = load_dataset_arrays(run_dir)
            geoms, keffs, rawparams = dataset_cache[dataset_key]
            print(f"  Using dataset: {os.path.basename(dataset_key)}")

            test_payload = get_test_payload(
                geoms=geoms, keffs=keffs, rawparams=rawparams,
                test_idx=test_idx, split_source=split_source,
                cache=test_payload_cache,
            )
            norm_stats = get_or_build_norm_stats(
                LOGS_ROOT=LOGS_ROOT, train_size=train_size, seed=seed,
                rawparams=rawparams, train_idx=train_idx,
            )
            state, meta = load_checkpoint(ckpt_path)
            model = build_model_from_metadata(meta, seed_for_init=seed)
            nnx.update(model, jax.tree_util.tree_map(jnp.asarray, state))

            m = evaluate_on_test(model, test_payload, norm_stats)
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
            test_frac_below_650=m["frac_below_650"], test_frac_below_100=m["frac_below_100"],
        ))
        print(f"  best_epoch={meta.get('epoch','?')}  "
              f"test_mean_pcm={m['mean_pcm']:.1f}  test_median_pcm={m['median_pcm']:.1f}")

    fieldnames = ["run_id", "train_size", "seed", "best_epoch", "train_time_val_mean_pcm",
                  "test_mse_k", "test_mae_k", "test_mean_pcm", "test_median_pcm",
                  "test_p95_pcm", "test_std_pcm", "test_frac_below_650", "test_frac_below_100"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved aggregate test metrics for {len(rows)} runs → {out_csv}")


if __name__ == "__main__":
    main()
    csv_path = os.path.join(LOGS_ROOT, OUTPUT_DIRNAME, OUTPUT_FILENAME)
    band = "std"

    agg = summarize(csv_path)
    out_dir = os.path.join(LOGS_ROOT, OUTPUT_DIRNAME)

    if agg.empty:
        print("No successful runs — skipping plots.")
    else:
        plot_scaling(agg, os.path.join(out_dir, "scaling_mean_pcm.png"),
                     metric="test_mean_pcm", band=band)
        plot_scaling(agg, os.path.join(out_dir, "scaling_median_pcm.png"),
                     metric="test_median_pcm", band=band)
        plot_scaling(agg, os.path.join(out_dir, "scaling_frac_below_650.png"),
                     metric="test_frac_below_650", band=band, logx=True)
