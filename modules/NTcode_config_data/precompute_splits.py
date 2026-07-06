"""
precompute_splits.py
=========================================================================
Pre-compute and cache train/val/test splits for ALL seeds (defined below)
in a single run, WITHOUT running training.

Splits are cached per (train_seed, holdout_seed, val_size, test_size).
When the dataset grows, training sets extend automatically while val/test
stay identical.

USAGE
-----
Default (uses SEED_LIST and defaults below):
    python precompute_splits.py

Override defaults from command line:
    python precompute_splits.py --data /new/data.npz --train-size 1500
    python precompute_splits.py --data /new/data.npz --train-size 1500 --cache-dir /other/cache

Specify different seeds (overrides SEED_LIST):
    python precompute_splits.py --seeds 0 1 2 3 4 5

=========================================================================
"""
import os
import sys
import argparse
import pickle
import csv

import numpy as np

sys.path.insert(0, "..")
from PEDS import derive_split_indices, _save_split_cache

# ── CONFIGURATION: defaults (can be overridden via command line) ──────────────
SEED_LIST = [0, 1, 2, 3, 4]                              # seeds to precompute
DEFAULT_DATA_PATH = ("../../data/highfidelity/1082_0.8_1.2.npz")
DEFAULT_TRAIN_SIZE = 200
DEFAULT_VAL_SIZE = 100
DEFAULT_TEST_SIZE = 100
DEFAULT_HOLDOUT_SEED = 0
DEFAULT_CACHE_DIR = None  # if None, will use (data parent)/.split_cache

def save_splits_to_csv(train_idx, val_idx, test_idx, train_seed, holdout_seed, 
                        train_size, val_size, test_size, cache_dir):
    """Append splits to a single CSV file grouped by train_size."""
    # One CSV per train_size, containing all seeds
    csv_fname = f"splits_train{train_size}_val{val_size}_test{test_size}.csv"
    csv_path = os.path.join(cache_dir, csv_fname)
    
    # Convert indices to space-separated strings
    train_str = " ".join(str(i) for i in sorted(train_idx))
    val_str   = " ".join(str(i) for i in sorted(val_idx))
    test_str  = " ".join(str(i) for i in sorted(test_idx))
    
    # Write/append to CSV
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["seed", "train_indices", "val_indices", "test_indices"])
        writer.writerow([train_seed, train_str, val_str, test_str])
        
    print(f"  ✓ Saved → {csv_path}")

def find_largest_smaller_cache(cache_dir, train_seed, holdout_seed, val_size, test_size):
    """Find the largest cached train_size < current request in this directory."""
    import glob
    pattern = os.path.join(cache_dir, f"split_t*_ts{train_seed}_hs{holdout_seed}_v{val_size}_t{test_size}.pkl")
    cached_files = glob.glob(pattern)
    
    candidates = []
    for fpath in cached_files:
        fname = os.path.basename(fpath)
        # Extract train_size from filename: split_t<N>_ts<seed>_hs<holdout>_v<val>_t<test>.pkl
        try:
            parts = fname.split("_")
            t_val = int(parts[0][1:])  # skip "split_t"
            candidates.append((t_val, fpath))
        except:
            continue
    
    if not candidates:
        return None, None
    
    # Return the largest train_size found
    candidates.sort(reverse=True)
    return candidates[0][0], candidates[0][1]


def precompute_split_for_seed(data_path, train_size, val_size, test_size,
                               train_seed, holdout_seed, cache_dir):
    """Compute and save splits for a single seed without running training."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_fname = f"split_t{train_size}_ts{train_seed}_hs{holdout_seed}_v{val_size}_t{test_size}.pkl"
    cache_path = os.path.join(cache_dir, cache_fname)

    data = np.load(data_path, allow_pickle=True)
    dataset_size = len(data['params'])

    print(f"\n{'='*70}")
    print(f"Dataset: {data_path}")
    print(f"  total samples: {dataset_size}")
    print(f"Splits (seed={train_seed}):")
    print(f"  train_size={train_size}, val_size={val_size}, test_size={test_size}")
    print(f"  train_seed={train_seed}, holdout_seed={holdout_seed}")

    # ── Check if exact cache exists ──────────────────────────────────────────
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        meta = cache["metadata"]
        old_ds = meta.get("dataset_size")

        # Dataset unchanged: nothing to do
        if old_ds == dataset_size and meta.get("train_size") == train_size:
            print(f"  ✓ Cache already exists and is up-to-date → {cache_path}")
            return

        # Dataset grew: extend training set
        if old_ds < dataset_size:
            print(f"  Dataset grew: {old_ds} → {dataset_size}")
            old_train = cache["train_idx"]
            old_val   = cache["val_idx"]
            old_test  = cache["test_idx"]

            # Find new samples
            old_all = np.union1d(old_train, np.union1d(old_val, old_test))
            new_pool = np.setdiff1d(np.arange(dataset_size), old_all)
            n_needed = train_size - len(old_train)

            if len(new_pool) < n_needed:
                raise ValueError(
                    f"Not enough new samples ({len(new_pool)}) to extend train "
                    f"from {len(old_train)} to {train_size} (need {n_needed})"
                )

            extend_rng = np.random.default_rng(train_seed)
            extend_rng.shuffle(new_pool)
            new_indices = new_pool[:n_needed]
            train_idx = np.concatenate([old_train, new_indices])
            val_idx   = old_val
            test_idx  = old_test

            meta["dataset_size"] = dataset_size
            meta["train_size"]   = train_size
            _save_split_cache(cache_path, train_idx, val_idx, test_idx, meta)
            save_splits_to_csv(train_idx, val_idx, test_idx, train_seed, holdout_seed, train_size, val_size, test_size, cache_dir) 
            print(f"  ✓ Extended train: {len(old_train)} → {len(train_idx)} samples")
            print(f"    Saved → {cache_path}")
            return

    # ── Fallback: look for largest smaller cached train_size ─────────────────
    prev_train_size, prev_cache_path = find_largest_smaller_cache(
        cache_dir, train_seed, holdout_seed, val_size, test_size
    )
    
    if prev_cache_path is not None:
        print(f"  ✓ Found previous cache for train_size={prev_train_size}")
        print(f"    Extending from {prev_cache_path}")
        
        with open(prev_cache_path, "rb") as f:
            cache = pickle.load(f)
        meta = cache["metadata"]
        old_train = cache["train_idx"]
        old_val   = cache["val_idx"]
        old_test  = cache["test_idx"]
        old_ds    = meta.get("dataset_size")

        # Find new samples
        old_all = np.union1d(old_train, np.union1d(old_val, old_test))
        new_pool = np.setdiff1d(np.arange(dataset_size), old_all)
        n_needed = train_size - len(old_train)

        if len(new_pool) < n_needed:
            raise ValueError(
                f"Not enough new samples ({len(new_pool)}) to extend train "
                f"from {len(old_train)} to {train_size} (need {n_needed})"
            )

        extend_rng = np.random.default_rng(train_seed)
        extend_rng.shuffle(new_pool)
        new_indices = new_pool[:n_needed]
        train_idx = np.concatenate([old_train, new_indices])
        val_idx   = old_val
        test_idx  = old_test

        meta["dataset_size"] = dataset_size
        meta["train_size"]   = train_size
        _save_split_cache(cache_path, train_idx, val_idx, test_idx, meta)
        save_splits_to_csv(train_idx, val_idx, test_idx, train_seed, holdout_seed, train_size, val_size, test_size, cache_dir) 
        print(f"  ✓ Extended train: {len(old_train)} → {len(train_idx)} samples")
        print(f"    Saved → {cache_path}")
        return

    # ── Compute from scratch ─────────────────────────────────────────────────
    print(f"  Computing new splits from scratch…")
    train_idx, val_idx, test_idx = derive_split_indices(
        data_path, train_size, val_size, test_size, train_seed, holdout_seed
    )

    meta = {
        "train_size": train_size, "val_size": val_size, "test_size": test_size,
        "train_seed": train_seed, "holdout_seed": holdout_seed,
        "dataset_size": dataset_size,
    }
    _save_split_cache(cache_path, train_idx, val_idx, test_idx, meta)
    print(f"  ✓ Computed and saved → {cache_path}")
    print(f"    train: {len(train_idx)} samples")
    print(f"    val:   {len(val_idx)} samples")
    print(f"    test:  {len(test_idx)} samples")
    save_splits_to_csv(train_idx, val_idx, test_idx, train_seed, holdout_seed, train_size, val_size, test_size, cache_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Precompute and cache data splits for multiple seeds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Edit SEED_LIST and defaults at the top of the script to change behavior."
    )
    parser.add_argument("--data", default=DEFAULT_DATA_PATH,
                         help=f"path to .npz data file (default: {DEFAULT_DATA_PATH})")
    parser.add_argument("--train-size", type=int, default=DEFAULT_TRAIN_SIZE,
                         help=f"training set size (default: {DEFAULT_TRAIN_SIZE})")
    parser.add_argument("--val-size", type=int, default=DEFAULT_VAL_SIZE,
                         help=f"validation set size (default: {DEFAULT_VAL_SIZE})")
    parser.add_argument("--test-size", type=int, default=DEFAULT_TEST_SIZE,
                         help=f"test set size (default: {DEFAULT_TEST_SIZE})")
    parser.add_argument("--seeds", nargs="*", type=int, default=None,
                         help="space-separated list of train seeds (default: SEED_LIST from script)")
    parser.add_argument("--seed-range", nargs=2, type=int, metavar=("START", "STOP"),
                         help="alternative: generate seeds from range(START, STOP)")
    parser.add_argument("--holdout-seed", type=int, default=DEFAULT_HOLDOUT_SEED,
                         help=f"holdout seed for val/test split (default: {DEFAULT_HOLDOUT_SEED})")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR,
                         help="cache directory (default: data parent / .split_cache)")
    args = parser.parse_args()

    # Determine which seeds to process
    if args.seed_range:
        seeds = list(range(args.seed_range[0], args.seed_range[1]))
    elif args.seeds is not None:  # CLI explicitly provided (even if empty list via --seeds alone)
        seeds = args.seeds
    else:
        seeds = SEED_LIST  # default: use SEED_LIST defined at top

    # Compute cache dir if not provided
    if args.cache_dir is None:
        args.cache_dir = os.path.join(os.path.dirname(args.data), ".split_cache")

    print(f"\n{'='*70}")
    print(f"Precomputing splits for {len(seeds)} seed(s): {seeds}")
    print(f"Data: {args.data}")
    print(f"Config: train_size={args.train_size}, val_size={args.val_size}, test_size={args.test_size}")
    print(f"{'='*70}")

    for train_seed in seeds:
        try:
            precompute_split_for_seed(
                args.data, args.train_size, args.val_size, args.test_size,
                train_seed, args.holdout_seed, args.cache_dir
            )
        except Exception as e:
            print(f"  ✗ FAILED for seed {train_seed}: {e}")
            continue

    print(f"\n{'='*70}")
    print(f"✓ Finished processing {len(seeds)} seed(s)")
    print(f"Cache location: {args.cache_dir}")
    print(f"{'='*70}\n")
