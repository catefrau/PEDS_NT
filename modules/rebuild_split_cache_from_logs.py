"""
==============================================================================
Rebuild split_cache .pkl files from previously-saved split_log.csv files.
==============================================================================

CONTEXT / WHY THIS EXISTS
--------------------------
Your training script (PEDS driver) normally derives train/val/test index
splits via `derive_split_indices()` and caches them as a .pkl so future runs
with the same (train_size, train_seed, holdout_seed, val_size, test_size)
hit the cache instead of recomputing.

For an older batch of runs (a train-size sweep), you only saved the human-
readable `split_log.csv` (columns: sample_idx, split, keff) inside each run's
log folder, not the .pkl cache. This script walks all the run subfolders,
reads each split_log.csv, and reconstructs the equivalent .pkl cache file so
future training runs can pick them up.

IMPORTANT ASSUMPTION — PLEASE VERIFY BEFORE TRUSTING THE OUTPUT
------------------------------------------------------------------
`sample_idx` in split_log.csv is a row index into the .npz dataset AS IT
EXISTED at the time that run was launched. Reusing that index as a valid
index into today's .npz is only correct if the dataset has grown
APPEND-ONLY since then (i.e. row i is still the same physical sample, and
any new samples were appended after the old ones, never inserted/reordered/
deleted). If the dataset was ever rebuilt, reordered, or had rows removed,
these reconstructed splits will silently point to the WRONG samples. This
script only checks that indices are in-bounds for the CURRENT dataset size;
it cannot detect reordering.

WHAT THIS SCRIPT DOES
----------------------
1. Scans STUDY_ROOT for subfolders named exactly `train_{N}_seed_{S}`
   (matching your training script's EXP_NAME convention).
2. Inside each matching subfolder, looks for `split_log.csv`.
3. Parses train_size (N) and train_seed (S) from the folder name.
4. Extracts train_idx / val_idx / test_idx from the CSV's 'split' column.
5. Cross-checks: len(train_idx) should equal N; warns (does not silently
   "fix") if it doesn't, since that indicates the folder name and the CSV
   disagree about something.
6. Loads the current dataset .npz once, to get dataset_size and to sanity-
   check all indices are in-bounds.
7. Writes one .pkl per run into OUTPUT_CACHE_DIR, using the SAME filename
   convention your training script's cache loader expects:
       split_t{train_size}_ts{train_seed}_hs{holdout_seed}_v{val_size}_t{test_size}.pkl
   (Per-run distinct filenames are required — a single fixed filename can't
   hold results for multiple train sizes/seeds at once. If you point your
   training script's cache_dir at OUTPUT_CACHE_DIR, cache-hit lookup will
   work automatically, since it searches by that same naming pattern.)

EDIT THE CONFIG BLOCK BELOW BEFORE RUNNING. No command-line arguments.
==============================================================================
"""

import os
import re
import glob
import pickle
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — edit these before running
# ─────────────────────────────────────────────────────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# Root folder containing one subfolder per run, each named train_{N}_seed_{S}
STUDY_ROOT = "RUNS/copy_of_1000case"
STUDY_ROOT = os.path.join(THIS_DIR, STUDY_ROOT)
# Where to write the reconstructed .pkl cache files (custom location).
# Point your training script's `cache_dir` argument here if you want those
# runs to auto cache-hit.
OUTPUT_CACHE_DIR = "../data/highfidelity/.split_cache_rebuilt_1000case"

# Path to the .npz dataset file used at training time (needed to get the
# true current dataset_size, and to bounds-check the reconstructed indices).
DATA_FILEPATH = "../data/highfidelity/MC_1315.npz"

# Fixed across the whole study, per your confirmation.
HOLDOUT_SEED = 0

# Folder-name pattern matching EXP_NAME = f"train_{TRAIN_SIZE}_seed_{SEED}"
FOLDER_PATTERN = re.compile(r"^trainnew_(\d+)_seed_(\d+)$")

# Name of the CSV inside each run folder (as written by log_splits()).
SPLIT_CSV_NAME = "split_log.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Validation helpers (mirrors _validate_split_indices from the training code,
# reimplemented standalone so this script has no dependency on your JAX /
# OpenMC environment)
# ─────────────────────────────────────────────────────────────────────────────

def _as_1d_int_indices(indices, name):
    arr = np.asarray(indices, dtype=np.int64).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape {arr.shape}")
    return arr


def _validate_split_indices(train_idx, val_idx, test_idx, dataset_size, context):
    train_idx = _as_1d_int_indices(train_idx, "train_idx")
    val_idx   = _as_1d_int_indices(val_idx, "val_idx")
    test_idx  = _as_1d_int_indices(test_idx, "test_idx")

    for name, arr in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        if arr.size == 0:
            raise ValueError(f"[{context}] {name} split is empty")
        if arr.min() < 0 or arr.max() >= dataset_size:
            raise ValueError(
                f"[{context}] {name} indices out of bounds for dataset_size={dataset_size} "
                f"(min={arr.min()}, max={arr.max()})"
            )
        if np.unique(arr).size != arr.size:
            raise ValueError(f"[{context}] duplicate indices inside {name} split")

    if np.intersect1d(train_idx, val_idx).size:
        raise ValueError(f"[{context}] train/val overlap detected")
    if np.intersect1d(train_idx, test_idx).size:
        raise ValueError(f"[{context}] train/test overlap detected")
    if np.intersect1d(val_idx, test_idx).size:
        raise ValueError(f"[{context}] val/test overlap detected")

    return train_idx, val_idx, test_idx


def _save_split_cache(cache_path, train_idx, val_idx, test_idx, metadata):
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    payload = {
        "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx,
        "metadata": metadata,
    }
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f)


# ─────────────────────────────────────────────────────────────────────────────
# Main reconstruction logic
# ─────────────────────────────────────────────────────────────────────────────

def find_run_folders(study_root):
    """Yield (folder_path, train_size, train_seed) for every direct subfolder
    matching FOLDER_PATTERN. Only immediate children of study_root are
    considered, since EXP_NAME folders are one level deep in your layout."""
    if not os.path.isdir(study_root):
        raise FileNotFoundError(f"STUDY_ROOT does not exist: {study_root}")

    for name in sorted(os.listdir(study_root)):
        full = os.path.join(study_root, name)
        if not os.path.isdir(full):
            continue
        m = FOLDER_PATTERN.match(name)
        if m is None:
            continue
        train_size_from_name = int(m.group(1))
        train_seed_from_name = int(m.group(2))
        yield full, train_size_from_name, train_seed_from_name


def find_split_csv(run_folder):
    """Locate split_log.csv. Checks the run folder directly first (expected
    location per log_splits()), then falls back to a recursive search in
    case your layout nests it deeper."""
    direct = os.path.join(run_folder, SPLIT_CSV_NAME)
    if os.path.isfile(direct):
        return direct
    matches = glob.glob(os.path.join(run_folder, "**", SPLIT_CSV_NAME), recursive=True)
    if matches:
        return matches[0]
    return None


def reconstruct_one_run(csv_path, train_size_from_name, train_seed_from_name,
                         holdout_seed, dataset_size, context):
    df = pd.read_csv(csv_path)

    required_cols = {"sample_idx", "split", "keff"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"[{context}] {csv_path} missing expected columns {required_cols}, "
            f"found {set(df.columns)}"
        )

    train_idx = df.loc[df["split"] == "train", "sample_idx"].to_numpy(dtype=np.int64)
    val_idx   = df.loc[df["split"] == "val",   "sample_idx"].to_numpy(dtype=np.int64)
    test_idx  = df.loc[df["split"] == "test",  "sample_idx"].to_numpy(dtype=np.int64)

    if len(train_idx) != train_size_from_name:
        print(
            f"  [WARNING][{context}] folder name says train_size={train_size_from_name}, "
            f"but CSV has {len(train_idx)} train rows. Using the CSV's actual count "
            f"({len(train_idx)}) for the cache filename/metadata — double check this run."
        )

    train_size = len(train_idx)
    val_size   = len(val_idx)
    test_size  = len(test_idx)

    train_idx, val_idx, test_idx = _validate_split_indices(
        train_idx, val_idx, test_idx, dataset_size, context=context
    )

    metadata = {
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "train_seed": train_seed_from_name,
        "holdout_seed": holdout_seed,
        "dataset_size": dataset_size,
    }

    return train_idx, val_idx, test_idx, metadata


def main():
    print(f"STUDY_ROOT       = {STUDY_ROOT}")
    print(f"OUTPUT_CACHE_DIR = {OUTPUT_CACHE_DIR}")
    print(f"DATA_FILEPATH    = {DATA_FILEPATH}")
    print(f"HOLDOUT_SEED     = {HOLDOUT_SEED}")

    print("\nLoading dataset to determine current dataset_size ...")
    data = np.load(DATA_FILEPATH, allow_pickle=True)
    # 'params' is the array load_data() uses to define dataset length.
    dataset_size = len(data["params"])
    print(f"  dataset_size = {dataset_size}")

    os.makedirs(OUTPUT_CACHE_DIR, exist_ok=True)

    run_folders = list(find_run_folders(STUDY_ROOT))
    print(f"\nFound {len(run_folders)} run folder(s) matching train_{{N}}_seed_{{S}}:")
    for folder, n, s in run_folders:
        print(f"  {os.path.basename(folder)}  (train_size={n}, seed={s})")

    n_ok, n_skipped = 0, 0

    for folder, train_size_from_name, train_seed_from_name in run_folders:
        context = os.path.basename(folder)
        csv_path = find_split_csv(folder)
        if csv_path is None:
            print(f"\n[SKIP] {context}: no {SPLIT_CSV_NAME} found under {folder}")
            n_skipped += 1
            continue

        try:
            train_idx, val_idx, test_idx, metadata = reconstruct_one_run(
                csv_path, train_size_from_name, train_seed_from_name,
                HOLDOUT_SEED, dataset_size, context,
            )
        except Exception as e:
            print(f"\n[SKIP] {context}: failed to reconstruct — {e}")
            n_skipped += 1
            continue

        cache_fname = (
            f"split_t{metadata['train_size']}_ts{metadata['train_seed']}_"
            f"hs{metadata['holdout_seed']}_v{metadata['val_size']}_t{metadata['test_size']}.pkl"
        )
        cache_path = os.path.join(OUTPUT_CACHE_DIR, cache_fname)

        if os.path.exists(cache_path):
            print(f"\n[SKIP] {context}: {cache_fname} already exists in OUTPUT_CACHE_DIR — "
                  f"not overwriting. Delete it manually first if you want to regenerate.")
            n_skipped += 1
            continue

        _save_split_cache(cache_path, train_idx, val_idx, test_idx, metadata)
        print(f"\n[OK] {context} -> {cache_fname}")
        print(f"     train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}  "
              f"train_seed={metadata['train_seed']}  holdout_seed={metadata['holdout_seed']}")
        n_ok += 1

    print(f"\nDone. Reconstructed {n_ok} cache file(s), skipped {n_skipped}.")


if __name__ == "__main__":
    main()
