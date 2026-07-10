#!/usr/bin/env python3
"""
check_split_consistency.py
===========================
Cross-experiment split-consistency checker for PEDS runs.

Point it at a "family" folder such as:

    RUNS/test_1000_oldDF_chi/
        train_500_seed_0/split_log.csv
        train_500_seed_1/split_log.csv
        train_1000_seed_0/split_log.csv
        ...

It will:

1. Load every split_log.csv it can find under the family folder
   (one per experiment, written by log_splits() in the training script).

2. Check that any sample index appearing in MORE THAN ONE experiment is
   assigned the SAME split (train/val/test) everywhere. Because
   HOLDOUT_SEED is fixed, val and test are supposed to be identical
   across all experiments regardless of train_size/seed — if a sample
   is "val" in one run and "train" in another, that's a real leakage
   bug, not just a benign difference in train_size.

3. Optionally cross-checks each experiment's split_log.csv against the
   .pkl split-cache file that produced it (in
   data/highfidelity/.split_cache*/), so you can catch cases where the
   cache-fallback/extension logic in load_or_create_split_cache()
   silently picked a different split than what you expect.

Usage
-----
Edit the CONFIG block below with the paths for the family of runs you
want to check, then just run:

    python check_split_consistency.py

Command-line flags are still accepted and, if given, override the
values in CONFIG (handy for one-off checks without editing the file):

    python check_split_consistency.py \
        --family-dir /path/to/RUNS/test_1000_newDF_chi \
        --cache-dir  /path/to/data/highfidelity/.split_cache_bounded \
        [--holdout-seed 0] \
        [--out-csv mismatches.csv]
"""

import argparse
import glob
import os
import pickle
import re
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────
# CONFIG — edit these to point at the run family you want to check.
# ─────────────────────────────────────────────────────────────────────────
# Folder containing multiple experiment subfolders, e.g.
#   .../RUNS/test_1000_oldDF_chi/train_500_seed_0/split_log.csv
#   .../RUNS/test_1000_oldDF_chi/train_1000_seed_0/split_log.csv
FAMILY_DIR = "RUNS/LHS_trainsize"

# Folder holding the split_cache .pkl files (CACHE_DIR_NAME in the
# training script, normally PARENT_DIR/data/highfidelity/.split_cache_bounded).
# Set to None to skip the cache cross-check entirely.
CACHE_DIR = "../data/highfidelity/.split_cache_fullLHS"

# Must match HOLDOUT_SEED in the training script.
HOLDOUT_SEED = 0

# Where to write the detailed conflict/mismatch CSVs. Set to None to
# only print to stdout.
OUT_CSV = "split_consistency_mismatches.csv"

EXP_NAME_RE = re.compile(r"train_(\d+)_seed_(\d+)")


# ─────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────
def find_experiment_csvs(family_dir: str) -> dict:
    """
    Returns {experiment_folder_name: path_to_split_log.csv}
    Searches one level deep AND recursively, so it works whether
    LOG_DIR is family_dir/EXP_NAME or nested deeper.
    """
    pattern = os.path.join(family_dir, "**", "split_log.csv")
    paths = glob.glob(pattern, recursive=True)
    if not paths:
        print(f"[warn] no split_log.csv found under {family_dir}")
    result = {}
    for p in paths:
        exp_name = os.path.basename(os.path.dirname(p))
        result[exp_name] = p
    return result


def load_split_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["sample_idx"] = df["sample_idx"].astype(int)
    df["split"] = df["split"].astype(str)
    df["keff"] = df["keff"].astype(float)
    return df


def parse_train_size_seed(exp_name: str):
    m = EXP_NAME_RE.search(exp_name)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


# ─────────────────────────────────────────────────────────────────────────
# Cross-experiment consistency
# ─────────────────────────────────────────────────────────────────────────
def cross_experiment_check(dfs: dict) -> pd.DataFrame:
    """
    dfs: {exp_name: dataframe(sample_idx, split, keff)}

    Returns a "conflicts" dataframe: one row per sample_idx that is NOT
    assigned the same split in every experiment that contains it.
    Columns: sample_idx, n_experiments, splits_seen (dict-like str),
             experiments (str)
    """
    idx_to_splits = defaultdict(dict)  # sample_idx -> {exp_name: split}
    for exp_name, df in dfs.items():
        for sidx, split in zip(df["sample_idx"], df["split"]):
            idx_to_splits[sidx][exp_name] = split

    conflict_rows = []
    for sidx, exp_split_map in idx_to_splits.items():
        unique_splits = set(exp_split_map.values())
        if len(unique_splits) > 1:
            conflict_rows.append({
                "sample_idx": sidx,
                "n_experiments_with_this_idx": len(exp_split_map),
                "splits_seen": ",".join(sorted(unique_splits)),
                "assignment": "; ".join(f"{e}={s}" for e, s in sorted(exp_split_map.items())),
            })

    conflicts = pd.DataFrame(conflict_rows)
    if not conflicts.empty:
        conflicts = conflicts.sort_values("sample_idx").reset_index(drop=True)
    return conflicts


def summarize_conflicts(conflicts: pd.DataFrame, dfs: dict):
    n_total_indices = len(set().union(*[set(df["sample_idx"]) for df in dfs.values()])) if dfs else 0
    print("\n=== CROSS-EXPERIMENT SPLIT CONSISTENCY ===")
    print(f"Experiments checked : {len(dfs)}")
    for exp_name, df in sorted(dfs.items()):
        counts = df["split"].value_counts().to_dict()
        print(f"  {exp_name:35s} train={counts.get('train',0):5d}  "
              f"val={counts.get('val',0):4d}  test={counts.get('test',0):4d}")
    print(f"Distinct sample indices seen across all experiments: {n_total_indices}")

    if conflicts.empty:
        print("\nNo conflicts found: every sample index that appears in more than "
              "one experiment has the same split (train/val/test) in all of them.")
        return

    print(f"\n*** {len(conflicts)} sample indices have INCONSISTENT split assignment ***")
    # Break down by which pair of splits is being confused (e.g. val<->train)
    pair_counts = defaultdict(int)
    for splits_seen in conflicts["splits_seen"]:
        pair_counts[splits_seen] += 1
    print("Breakdown by conflicting split combination:")
    for pair, n in sorted(pair_counts.items(), key=lambda x: -x[1]):
        print(f"  {pair:20s}: {n} sample(s)")

    # This is the dangerous case: val or test leaking into train (or vice versa)
    leakage = conflicts[conflicts["splits_seen"].str.contains("train") &
                         (conflicts["splits_seen"].str.contains("val") |
                          conflicts["splits_seen"].str.contains("test"))]
    if not leakage.empty:
        print(f"\n  !! {len(leakage)} of these involve train vs. val/test leakage "
              f"(a sample used for training in one run and for evaluation in another) !!")

    print("\nFirst 10 conflicting rows:")
    print(conflicts.head(10).to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────
# Cache cross-check
# ─────────────────────────────────────────────────────────────────────────
def load_all_cache_pkls(cache_dir: str) -> list:
    """Returns list of (path, metadata_dict, train_idx, val_idx, test_idx)."""
    entries = []
    for p in glob.glob(os.path.join(cache_dir, "*.pkl")):
        try:
            with open(p, "rb") as f:
                payload = pickle.load(f)
            meta = payload.get("metadata", {})
            entries.append((p, meta,
                             np.asarray(payload["train_idx"]).reshape(-1),
                             np.asarray(payload["val_idx"]).reshape(-1),
                             np.asarray(payload["test_idx"]).reshape(-1)))
        except Exception as e:
            print(f"  [warn] could not read cache file {p}: {e}")
    return entries


def find_matching_cache(entries, train_size, train_seed, holdout_seed, val_size, test_size):
    """
    Find the cache entry whose metadata matches this experiment.
    Matches on train_seed / holdout_seed / val_size / test_size always;
    train_size is matched exactly if possible, otherwise we return the
    entry with the closest train_size <= requested (mirrors the
    "extend from smaller cache" fallback logic in the training script).
    """
    candidates = []
    for path, meta, tr, va, te in entries:
        if (meta.get("train_seed") == train_seed and
                meta.get("holdout_seed") == holdout_seed and
                meta.get("val_size") == val_size and
                meta.get("test_size") == test_size):
            candidates.append((path, meta, tr, va, te))

    if not candidates:
        return None

    exact = [c for c in candidates if c[1].get("train_size") == train_size]
    if exact:
        return exact[0]

    smaller = [c for c in candidates if c[1].get("train_size", -1) <= train_size]
    if smaller:
        smaller.sort(key=lambda c: c[1]["train_size"], reverse=True)
        return smaller[0]

    return None


def compare_csv_to_cache(exp_name, df, cache_entry):
    path, meta, cache_train, cache_val, cache_test = cache_entry
    csv_train = set(df.loc[df["split"] == "train", "sample_idx"])
    csv_val = set(df.loc[df["split"] == "val", "sample_idx"])
    csv_test = set(df.loc[df["split"] == "test", "sample_idx"])

    problems = []
    for name, csv_set, cache_set in [("train", csv_train, set(cache_train.tolist())),
                                      ("val", csv_val, set(cache_val.tolist())),
                                      ("test", csv_test, set(cache_test.tolist()))]:
        only_csv = csv_set - cache_set
        only_cache = cache_set - csv_set
        if only_csv or only_cache:
            problems.append((name, only_csv, only_cache))

    print(f"\n--- {exp_name}  vs.  {os.path.basename(path)}  (cache dataset_size={meta.get('dataset_size')}) ---")
    if not problems:
        print("  OK: train/val/test index sets match the cache exactly.")
        return None

    row = {"experiment": exp_name, "cache_file": os.path.basename(path)}
    for name, only_csv, only_cache in problems:
        print(f"  MISMATCH in '{name}': "
              f"{len(only_csv)} idx in CSV but not cache, "
              f"{len(only_cache)} idx in cache but not CSV")
        if only_csv:
            sample = sorted(only_csv)[:10]
            print(f"      e.g. CSV-only: {sample}{' ...' if len(only_csv) > 10 else ''}")
        if only_cache:
            sample = sorted(only_cache)[:10]
            print(f"      e.g. cache-only: {sample}{' ...' if len(only_cache) > 10 else ''}")
        row[f"{name}_only_in_csv_count"] = len(only_csv)
        row[f"{name}_only_in_cache_count"] = len(only_cache)
    return row


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family-dir", default=FAMILY_DIR,
                    help="Folder containing multiple experiment subfolders, "
                         "e.g. RUNS/test_1000_oldDF_chi "
                         f"(default: {FAMILY_DIR})")
    ap.add_argument("--cache-dir", default=CACHE_DIR,
                    help="Folder containing the split_cache .pkl files "
                         "(e.g. data/highfidelity/.split_cache_bounded). "
                         "Pass an empty string to skip the cache cross-check. "
                         f"(default: {CACHE_DIR})")
    ap.add_argument("--holdout-seed", type=int, default=HOLDOUT_SEED,
                    help=f"HOLDOUT_SEED used in the training script (default {HOLDOUT_SEED}).")
    ap.add_argument("--out-csv", default=OUT_CSV,
                    help="Path to write the conflicts table as CSV. Pass an "
                         "empty string to only print to stdout. "
                         f"(default: {OUT_CSV})")
    args = ap.parse_args()

    if not args.family_dir:
        print("[error] FAMILY_DIR is not set — edit the CONFIG block at the "
              "top of this script or pass --family-dir.")
        sys.exit(1)
    # allow "" to mean "disabled" for the optional paths
    args.cache_dir = args.cache_dir or None
    args.out_csv = args.out_csv or None

    csv_paths = find_experiment_csvs(args.family_dir)
    if not csv_paths:
        sys.exit(1)

    dfs = {exp: load_split_csv(p) for exp, p in csv_paths.items()}

    # ── 1. cross-experiment consistency ────────────────────────────────────
    conflicts = cross_experiment_check(dfs)
    summarize_conflicts(conflicts, dfs)

    if args.out_csv and not conflicts.empty:
        conflicts.to_csv(args.out_csv, index=False)
        print(f"\nFull conflict list written to {args.out_csv}")

    # ── 2. cache cross-check ────────────────────────────────────────────────
    if args.cache_dir:
        print("\n=== CACHE CROSS-CHECK ===")
        cache_entries = load_all_cache_pkls(args.cache_dir)
        if not cache_entries:
            print(f"[warn] no .pkl files found in {args.cache_dir}")
        cache_mismatch_rows = []
        for exp_name, df in sorted(dfs.items()):
            train_size, train_seed = parse_train_size_seed(exp_name)
            if train_size is None:
                print(f"\n--- {exp_name}: could not parse train_size/seed from folder "
                      f"name, skipping cache check ---")
                continue
            val_size = int((df["split"] == "val").sum())
            test_size = int((df["split"] == "test").sum())

            entry = find_matching_cache(cache_entries, train_size, train_seed,
                                         args.holdout_seed, val_size, test_size)
            if entry is None:
                print(f"\n--- {exp_name}: no matching cache file found "
                      f"(train_size={train_size}, train_seed={train_seed}, "
                      f"holdout_seed={args.holdout_seed}, val_size={val_size}, "
                      f"test_size={test_size}) ---")
                continue

            row = compare_csv_to_cache(exp_name, df, entry)
            if row:
                cache_mismatch_rows.append(row)

        if cache_mismatch_rows:
            cache_df = pd.DataFrame(cache_mismatch_rows)
            print(f"\n{len(cache_mismatch_rows)} experiment(s) disagree with their cache file.")
            if args.out_csv:
                cache_out = os.path.splitext(args.out_csv)[0] + "_cache.csv"
                cache_df.to_csv(cache_out, index=False)
                print(f"Cache mismatch details written to {cache_out}")
        else:
            print("\nAll checked experiments match their cache file exactly.")


if __name__ == "__main__":
    main()
