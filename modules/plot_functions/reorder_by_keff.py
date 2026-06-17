"""
reorder_by_keff.py
------------------
Reassign stable sample_idx values based on unique keff_openmc identities.

The training CSV shuffles sample order each epoch, so sample_idx 0 at epoch 1
is not the same physical case as sample_idx 0 at epoch 2. Since each unique
keff_openmc value identifies a specific MC case, we can build a lookup once
and remap all rows to a consistent stable_idx.

Usage
-----
    python reorder_by_keff.py \
        --input  ../LOGS/v5_cleanDS/keff_epoch_log_train.csv \
        --output ../LOGS/v5_cleanDS/keff_epoch_log_train_stable.csv

Or import and call reorder_csv() directly.
"""

import argparse
import pandas as pd
import numpy as np


# ── tolerance for matching keff_openmc values across epochs ──────────────────
# MC keff values are stored as 6 decimal places (e.g. 1.059244).
# Two rows are considered the same physical case if their keff_openmc values
# agree to within this tolerance.
KEFF_MATCH_TOL = 1e-5


def build_stable_index(
    keff_values: np.ndarray,
    tol: float = KEFF_MATCH_TOL,
) -> np.ndarray:
    """
    Given a flat array of keff_openmc values (all epochs concatenated),
    return an integer stable_idx for each entry.

    Strategy
    --------
    1. Collect the unique keff_openmc values seen in epoch 1
       (or the earliest epoch present) — these define the canonical set.
    2. Sort them ascending so stable_idx 0 = lowest keff case.
    3. For every row in the full DataFrame, find the nearest canonical keff
       and assign that canonical rank as stable_idx.

    Raises
    ------
    ValueError if any row's keff cannot be matched within `tol`.
    """
    unique_sorted = np.unique(np.round(keff_values, decimals=6))
    # build a stable_idx: rank by ascending keff_openmc
    # (so stable_idx 0 is always the most sub-critical case)
    stable_map = {k: i for i, k in enumerate(unique_sorted)}

    stable_indices = np.empty(len(keff_values), dtype=np.int64)
    for row_i, kv in enumerate(keff_values):
        # find nearest canonical value
        nearest = unique_sorted[np.argmin(np.abs(unique_sorted - kv))]
        if abs(nearest - kv) > tol:
            raise ValueError(
                f"Row {row_i}: keff_openmc={kv:.8f} has no match within "
                f"tol={tol:.1e}. Nearest canonical value is {nearest:.8f}."
            )
        stable_indices[row_i] = stable_map[nearest]

    return stable_indices


def reorder_csv(
    input_path: str,
    output_path: str,
    tol: float = KEFF_MATCH_TOL,
    sort_output: bool = True,
) -> pd.DataFrame:
    """
    Read a per-epoch keff CSV, add a `stable_idx` column, and save.

    Parameters
    ----------
    input_path  : path to the original shuffled CSV
    output_path : where to write the reindexed CSV
    tol         : tolerance for keff_openmc matching
    sort_output : if True, sort by (epoch, stable_idx) in the output

    Returns
    -------
    The reindexed DataFrame (also written to output_path).
    """
    print(f"Reading {input_path} …")
    df = pd.read_csv(input_path)

    required = {"epoch", "sample_idx", "keff_openmc", "keff_peds", "delta_rho_pcm"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {missing}")

    df["keff_openmc"] = df["keff_openmc"].astype(float)

    # ── build stable index ────────────────────────────────────────────────────
    print("Building stable index from keff_openmc values …")
    stable_indices = build_stable_index(df["keff_openmc"].values, tol=tol)
    df["stable_idx"] = stable_indices

    # ── sanity check: every epoch should have the same set of stable_idx ─────
    epochs = df["epoch"].unique()
    first_set = set(df[df["epoch"] == epochs[0]]["stable_idx"].values)
    mismatches = []
    for ep in epochs[1:]:
        ep_set = set(df[df["epoch"] == ep]["stable_idx"].values)
        if ep_set != first_set:
            mismatches.append(ep)
    if mismatches:
        print(
            f"WARNING: {len(mismatches)} epoch(s) have a different stable_idx set "
            f"than epoch {epochs[0]}. First few: {mismatches[:5]}"
        )
    else:
        print(f"Sanity check passed: all {len(epochs)} epochs share the same {len(first_set)} stable indices.")

    # ── reorder columns: put stable_idx right after sample_idx ──────────────
    cols = list(df.columns)
    cols.remove("stable_idx")
    insert_pos = cols.index("sample_idx") + 1
    cols.insert(insert_pos, "stable_idx")
    df = df[cols]

    if sort_output:
        df = df.sort_values(["epoch", "stable_idx"]).reset_index(drop=True)

    print(f"Writing {output_path} …")
    df.to_csv(output_path, index=False)
    print(f"Done. {len(df)} rows, {df['stable_idx'].nunique()} unique cases, {len(epochs)} epochs.")

    # ── quick summary ─────────────────────────────────────────────────────────
    epoch1 = df[df["epoch"] == epochs[0]].sort_values("stable_idx")
    print(f"\nStable index mapping (epoch {epochs[0]}, first 5 rows):")
    print(epoch1[["stable_idx", "keff_openmc", "sample_idx"]].head(5).to_string(index=False))

    return df


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input",  required=True, help="Path to the original train CSV")
    parser.add_argument("--output", required=True, help="Path to write the reindexed CSV")
    parser.add_argument("--tol",    type=float, default=KEFF_MATCH_TOL,
                        help=f"keff matching tolerance (default {KEFF_MATCH_TOL})")
    args = parser.parse_args()

    reorder_csv(args.input, args.output, tol=args.tol)
