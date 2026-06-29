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
import argparse

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

# ── TODO 1: point this at the folder containing your training script ───────
#sys.path.insert(0, "../")

# ── TODO 2: rename "PEDS_v7" to your training script's module name ─────────
from PEDS import (
    GEO,
    PEDSModel,
    _run_NT_solver,
    derive_split_indices,
    compute_phi_features,
    compute_batch_baselines,
    compute_metrics,
    data_loader,
    _DATA_FILEPATH,
    HOLDOUT_SEED,
    VAL_SIZE,
    TEST_SIZE,
)

RUN_PATTERN = re.compile(r"^train_(\d+)_seed_(\d+)$")
EVAL_BATCH_SIZE = 32


# ─────────────────────────────────────────────────────────────────────────────
# Discovery
# ─────────────────────────────────────────────────────────────────────────────
def find_runs(logs_root):
    """Yield (run_id, train_size, seed, ckpt_path) for every best_model.pkl found."""
    pattern = os.path.join(logs_root, "**", "checkpoints", "best_model.pkl")
    for ckpt_path in sorted(glob.glob(pattern, recursive=True)):
        run_dir = os.path.dirname(os.path.dirname(ckpt_path))
        run_folder = os.path.basename(run_dir)

        m = RUN_PATTERN.match(run_folder)
        if not m:
            print(f"  [skip] couldn't parse train_size/seed from folder: {run_folder}")
            continue

        train_size, seed = int(m.group(1)), int(m.group(2))
        run_id = os.path.relpath(run_dir, logs_root).replace(os.sep, "/")
        yield run_id, train_size, seed, ckpt_path

# ─────────────────────────────────────────────────────────────────────────────
# Caching: fixed TEST set (built once, shared by every run)
# ─────────────────────────────────────────────────────────────────────────────
def get_or_build_test_set(logs_root):
    cache_path = os.path.join(logs_root, "_eval_cache", "test_set.pkl")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.exists(cache_path):
        print(f"Loading cached TEST set ← {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print("Building TEST set (first time only — will be cached for future runs)…")
    data = np.load(_DATA_FILEPATH, allow_pickle=True)
    geoms     = np.array(data["params"],     dtype=np.float32)
    keffs     = np.array(data["keffs"],      dtype=np.float32)
    rawparams = np.array(data["params_raw"], dtype=np.float32)

    # train_size/train_seed don't affect test_idx (test is carved out first,
    # using only holdout_seed) — the value passed here is just a placeholder.
    _, _, test_idx = derive_split_indices(
        _DATA_FILEPATH, train_size=1, val_size=VAL_SIZE, test_size=TEST_SIZE,
        train_seed=0, holdout_seed=HOLDOUT_SEED,
    )

    test_geoms, test_keffs, test_rawparams = geoms[test_idx], keffs[test_idx], rawparams[test_idx]
    test_phi_features = compute_phi_features(test_rawparams)

    payload = dict(geoms=test_geoms, keffs=test_keffs, rawparams=test_rawparams,
                    phi_features=test_phi_features)
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"  cached → {cache_path}  ({len(test_idx)} samples)")
    return payload


# ─────────────────────────────────────────────────────────────────────────────
# Caching: per-(train_size, seed) normalization stats
# ─────────────────────────────────────────────────────────────────────────────
def get_or_build_norm_stats(logs_root, train_size, seed):
    cache_path = os.path.join(
        logs_root, "_eval_cache", f"norm_stats_train{train_size}_seed{seed}.pkl"
    )
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"  Recomputing normalization stats for train_size={train_size}, seed={seed} …")
    data = np.load(_DATA_FILEPATH, allow_pickle=True)
    rawparams = np.array(data["params_raw"], dtype=np.float32)

    train_idx, _, _ = derive_split_indices(
        _DATA_FILEPATH, train_size, VAL_SIZE, TEST_SIZE,
        train_seed=seed, holdout_seed=HOLDOUT_SEED,
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
                    log_xs_mean=log_xs_mean, log_xs_std=log_xs_std)
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
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main(logs_root, out_csv):
    test_payload = get_or_build_test_set(logs_root)
    rows = []

    for run_id, train_size, seed, ckpt_path in find_runs(logs_root):
        print(f"\n=== Evaluating {run_id} ===")
        try:
            norm_stats = get_or_build_norm_stats(logs_root, train_size, seed)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("logs_root", help="e.g. ../LOGS")
    parser.add_argument("--out", default=None,
                         help="output CSV path (default: <logs_root>/test_set_metrics_all_runs.csv)")
    args = parser.parse_args()
    out_csv = args.out or os.path.join(args.logs_root, "test_set_metrics_all_runs.csv")
    main(args.logs_root, out_csv)
