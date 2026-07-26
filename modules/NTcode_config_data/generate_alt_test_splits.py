"""
generate_alt_test_splits.py
=========================================================================
For every train_<N>_seed_<S> run under a LOGS root, write an alternate
test split that:

  * keeps the original train and val indices (so norm stats / leakage
    rules stay identical to training),
  * draws a new TEST set from geometries that were unused by that run
    (not in train ∪ val ∪ original test),
  * balances the new test across the same 10 keff-quantile bins used by
    PEDS.derive_split_indices.

Output per run folder (same schema as split_log.csv):

    alt_split_log.csv   columns: sample_idx, split, keff

Then evaluate with:

    python evaluate_test_metrics.py /path/to/LOGS \\
        --split-log alt_split_log.csv \\
        --results-dirname testset_results_alt

USAGE
-----
    python generate_alt_test_splits.py /path/to/LOGS
    python generate_alt_test_splits.py /path/to/LOGS --holdout-seed 1
    python generate_alt_test_splits.py /path/to/LOGS --force
=========================================================================
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
from typing import Optional

import numpy as np

MODULES_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(MODULES_DIR)

RUN_PATTERN = re.compile(r"^train_(\d+)_seed_(\d+)$")
DEFAULT_SPLIT_LOG = "split_log.csv"
DEFAULT_ALT_SPLIT_LOG = "alt_split_log.csv"
DEFAULT_HOLDOUT_SEED = 1  # original training uses HOLDOUT_SEED = 0
N_BINS = 10


def _detect_data_filepath(run_dir: str) -> Optional[str]:
    """Resolve the .npz path recorded in the run's code_snapshot_*.py."""
    snapshots = glob.glob(os.path.join(run_dir, "code_snapshot_*.py"))
    if not snapshots:
        return None
    with open(snapshots[0], encoding="utf-8") as f:
        content = f.read()
    for line in content.splitlines():
        if "_DATA_FILEPATH" in line and "=" in line:
            quoted = re.findall(r'["\']([^"\']+)["\']', line)
            if not quoted:
                continue
            filename = quoted[-1]
            candidate = os.path.join(PROJECT_ROOT, "data", "highfidelity", filename)
            if os.path.exists(candidate):
                return candidate
            candidate2 = os.path.join(PROJECT_ROOT, filename)
            if os.path.exists(candidate2):
                return candidate2
    return None


def _read_split_log(csv_path: str) -> dict[str, list[int]]:
    splits: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = str(row.get("split", "")).strip().lower()
            if split not in splits:
                continue
            splits[split].append(int(row["sample_idx"]))
    if not splits["train"] or not splits["test"]:
        raise ValueError(
            f"{csv_path} missing train/test rows "
            f"(train={len(splits['train'])}, test={len(splits['test'])})"
        )
    return splits


def find_run_dirs(logs_root: str) -> list[tuple[str, int, int, str]]:
    """Return (run_id, train_size, seed, run_dir) for folders with a split_log."""
    runs = []
    for split_path in sorted(
        glob.glob(os.path.join(logs_root, "**", DEFAULT_SPLIT_LOG), recursive=True)
    ):
        run_dir = os.path.dirname(split_path)
        folder = os.path.basename(run_dir)
        m = RUN_PATTERN.match(folder)
        if not m:
            continue
        train_size, seed = int(m.group(1)), int(m.group(2))
        run_id = os.path.relpath(run_dir, logs_root).replace(os.sep, "/")
        runs.append((run_id, train_size, seed, run_dir))
    return runs


def derive_alt_test_from_unused(
    keffs: np.ndarray,
    used_idx: np.ndarray,
    test_size: int,
    holdout_seed: int,
    n_bins: int = N_BINS,
) -> np.ndarray:
    """
    Draw `test_size` indices from the unused pool, balanced across the same
    keff-quantile bins as PEDS.derive_split_indices (n_bins equal-count bins
    on the full dataset, then sample from unused candidates inside each bin).
    """
    n = len(keffs)
    used = set(int(i) for i in np.asarray(used_idx, dtype=np.int64).tolist())
    unused = np.array(sorted(set(range(n)) - used), dtype=np.int64)
    if len(unused) < test_size:
        raise ValueError(
            f"Unused pool has only {len(unused)} samples, need test_size={test_size}"
        )

    sorted_idx = np.argsort(keffs)
    holdout_rng = np.random.default_rng(holdout_seed)
    test_per_bin = max(1, test_size // n_bins)
    test_idx: list[int] = []

    for bin_indices in np.array_split(sorted_idx, n_bins):
        cand = [int(i) for i in bin_indices.tolist() if int(i) not in used]
        if not cand:
            continue
        holdout_rng.shuffle(cand)
        take = min(test_per_bin, len(cand))
        test_idx.extend(cand[:take])

    if len(test_idx) < test_size:
        # Fill any shortfall from remaining unused (still shuffled).
        leftover = [int(i) for i in unused.tolist() if int(i) not in set(test_idx)]
        holdout_rng.shuffle(leftover)
        need = test_size - len(test_idx)
        test_idx.extend(leftover[:need])

    test_idx = np.array(test_idx[:test_size], dtype=np.int64)
    if len(test_idx) != test_size:
        raise ValueError(
            f"Could not build alt test of size {test_size} (got {len(test_idx)})"
        )
    if len(set(test_idx.tolist()) & used) != 0:
        raise RuntimeError("Alt test leaked into used train/val/original-test indices")
    return test_idx


def write_alt_split_log(
    out_path: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    keffs: np.ndarray,
) -> None:
    """Write split_log-compatible CSV (sorted by sample_idx within each split)."""
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_idx", "split", "keff"])
        for label, idx in (
            ("train", train_idx),
            ("val", val_idx),
            ("test", test_idx),
        ):
            order = np.argsort(np.asarray(idx, dtype=np.int64))
            arr = np.asarray(idx, dtype=np.int64)[order]
            for i in arr:
                writer.writerow([int(i), label, float(keffs[int(i)])])


def _bin_counts(keffs: np.ndarray, idx: np.ndarray, n_bins: int = N_BINS) -> list[int]:
    sorted_idx = np.argsort(keffs)
    idx_set = set(int(i) for i in np.asarray(idx).tolist())
    counts = []
    for bin_indices in np.array_split(sorted_idx, n_bins):
        counts.append(len(idx_set & set(int(i) for i in bin_indices.tolist())))
    return counts


def process_run(
    run_dir: str,
    run_id: str,
    keffs: np.ndarray,
    holdout_seed: int,
    alt_name: str,
    force: bool,
) -> dict:
    split_path = os.path.join(run_dir, DEFAULT_SPLIT_LOG)
    out_path = os.path.join(run_dir, alt_name)
    if os.path.exists(out_path) and not force:
        return {"run_id": run_id, "status": "skipped_exists", "path": out_path}

    splits = _read_split_log(split_path)
    train_idx = np.array(splits["train"], dtype=np.int64)
    val_idx = np.array(splits["val"], dtype=np.int64)
    orig_test = np.array(splits["test"], dtype=np.int64)
    used = np.concatenate([train_idx, val_idx, orig_test])

    alt_test = derive_alt_test_from_unused(
        keffs=keffs,
        used_idx=used,
        test_size=len(orig_test),
        holdout_seed=holdout_seed,
    )
    write_alt_split_log(out_path, train_idx, val_idx, alt_test, keffs)

    overlap_orig = len(set(alt_test.tolist()) & set(orig_test.tolist()))
    return {
        "run_id": run_id,
        "status": "written",
        "path": out_path,
        "test_size": int(len(alt_test)),
        "unused_pool": int(len(keffs) - len(set(used.tolist()))),
        "overlap_orig_test": overlap_orig,
        "keff_min": float(keffs[alt_test].min()),
        "keff_max": float(keffs[alt_test].max()),
        "bin_counts": _bin_counts(keffs, alt_test),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Write per-seed alternate test splits (unused geometries, quantile-balanced).",
    )
    parser.add_argument("logs_root", help="LOGS root containing train_<N>_seed_<S> folders")
    parser.add_argument(
        "--holdout-seed",
        type=int,
        default=DEFAULT_HOLDOUT_SEED,
        help=f"RNG seed for alt test draw (default {DEFAULT_HOLDOUT_SEED}; training uses 0)",
    )
    parser.add_argument(
        "--out-name",
        default=DEFAULT_ALT_SPLIT_LOG,
        help=f"Filename written inside each run folder (default {DEFAULT_ALT_SPLIT_LOG})",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Optional dataset .npz override (else detected from each run's code_snapshot)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing alt split logs",
    )
    args = parser.parse_args()

    logs_root = os.path.abspath(args.logs_root)
    if not os.path.isdir(logs_root):
        print(f"ERROR: logs_root not found: {logs_root}", file=sys.stderr)
        sys.exit(1)

    runs = find_run_dirs(logs_root)
    if not runs:
        print(f"No runs with {DEFAULT_SPLIT_LOG} found under {logs_root}")
        sys.exit(1)

    print(f"LOGS_ROOT     = {logs_root}")
    print(f"holdout_seed  = {args.holdout_seed}")
    print(f"out_name      = {args.out_name}")
    print(f"runs found    = {len(runs)}")

    dataset_cache: dict[str, np.ndarray] = {}
    summaries = []

    for run_id, _train_size, _seed, run_dir in runs:
        if args.data is not None:
            data_path = args.data
        else:
            detected = _detect_data_filepath(run_dir)
            if detected is None:
                print(f"  [skip] {run_id}: could not detect dataset from code_snapshot")
                continue
            data_path = detected

        if data_path not in dataset_cache:
            print(f"  Loading dataset: {data_path}")
            dataset_cache[data_path] = np.array(
                np.load(data_path, allow_pickle=True)["keffs"], dtype=np.float32
            )
        keffs = dataset_cache[data_path]

        info = process_run(
            run_dir=run_dir,
            run_id=run_id,
            keffs=keffs,
            holdout_seed=args.holdout_seed,
            alt_name=args.out_name,
            force=args.force,
        )
        summaries.append(info)
        if info["status"] == "skipped_exists":
            print(f"  [skip] {run_id}: {args.out_name} exists (use --force)")
        else:
            print(
                f"  [ok]   {run_id}: test={info['test_size']} "
                f"unused_pool={info['unused_pool']} "
                f"overlap_orig={info['overlap_orig_test']} "
                f"keff=[{info['keff_min']:.3f},{info['keff_max']:.3f}] "
                f"bins={info['bin_counts']}"
            )

    n_written = sum(1 for s in summaries if s["status"] == "written")
    print(f"\nDone. Wrote {n_written}/{len(summaries)} alt split logs.")
    print(
        "Evaluate with:\n"
        f"  python evaluate_test_metrics.py {logs_root} "
        f"--split-log {args.out_name} --results-dirname testset_results_alt"
    )


if __name__ == "__main__":
    main()
