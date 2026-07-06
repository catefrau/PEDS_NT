"""
Merge all parameter-sweep CSVs in a folder into a single clean dataset,
check for duplicated samples, report parameter-space coverage, and produce
a second "good LHS" dataset where over-sampled regions of the 6D design
space have been thinned out.

Outputs (in ./lhs_analysis_outputs/):
  merged_raw.csv          -> all rows, relevant columns only, tagged with source file
  merged_deduped.csv      -> merged_raw.csv with exact-duplicate design points removed
  merged_good_lhs.csv     -> deduped data with over-populated regions of the 6D
                             parameter space thinned out (closer to uniform LHS coverage)
  histograms_before.png   -> per-parameter distributions before thinning
  histograms_after.png    -> per-parameter distributions after thinning
  summary_report.txt      -> human-readable summary of everything above

Usage:
    python merge_and_clean_lhs.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
FOLDER = Path("../FILES")
OUTDIR = Path("./lhs_analysis_outputs")
OUTDIR.mkdir(exist_ok=True)

PARAMS = [
    "r0_b4c_rod_outer_radius",
    "r0_b4c_rod_cr_fraction",
    "r1_fuel_annulus_outer_radius",
    "r1_fuel_annulus_enrichment",
    "r1_fuel_annulus_f_mod",
    "r2_water_outer_radius",
]
TARGET = "keff"
COLS_OF_INTEREST = PARAMS + [TARGET]

N_BINS_1D = 10          # bins per parameter, for reporting/plotting distributions
N_BINS_PER_DIM = 5      # bins per parameter, for the 6D thinning grid (5**6 = 15625 cells)
CELL_CAP_PERCENTILE = 75  # cap each 6D cell's count at this percentile of nonzero cell counts
RANDOM_SEED = 42


# ----------------------------------------------------------------------
# Step 1: load & merge
# ----------------------------------------------------------------------
def load_and_merge(folder: Path) -> pd.DataFrame:
    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {folder.resolve()}")

    frames = []
    for f in csv_files:
        header = pd.read_csv(f, nrows=0).columns.tolist()
        cols = [c for c in COLS_OF_INTEREST if c in header]
        missing = set(COLS_OF_INTEREST) - set(cols)
        if missing:
            print(f"  [warn] {f.name}: missing columns {missing}, skipping those")
        df = pd.read_csv(f, usecols=cols)
        df["source_file"] = f.name
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    # keep a clean, fixed column order
    ordered_cols = [c for c in COLS_OF_INTEREST if c in merged.columns] + ["source_file"]
    return merged[ordered_cols]


# ----------------------------------------------------------------------
# Step 2: duplicate check
# ----------------------------------------------------------------------
def check_and_drop_duplicates(df: pd.DataFrame):
    """Duplicates are judged on the 6 design parameters + keff (i.e. a repeated
    design point with a repeated result). source_file is ignored so the same
    point appearing in two different files is still caught."""
    dup_mask = df.duplicated(subset=COLS_OF_INTEREST, keep="first")
    n_dupes = int(dup_mask.sum())
    deduped = df.loc[~dup_mask].reset_index(drop=True)
    return deduped, n_dupes


# ----------------------------------------------------------------------
# Step 3: distribution reporting
# ----------------------------------------------------------------------
def plot_distributions(df: pd.DataFrame, title: str, out_png: Path):
    cols_present = [c for c in COLS_OF_INTEREST if c in df.columns]
    n = len(cols_present)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, col in zip(axes, cols_present):
        data = df[col].dropna()
        ax.hist(data, bins=N_BINS_1D, color="steelblue", edgecolor="black", alpha=0.8)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("value")
        ax.set_ylabel("count")

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def bounds_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in COLS_OF_INTEREST:
        if col not in df.columns:
            continue
        rows.append({
            "parameter": col,
            "min": df[col].min(),
            "max": df[col].max(),
            "mean": df[col].mean(),
            "std": df[col].std(),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Step 4: thin out over-sampled regions of the 6D design space
# ----------------------------------------------------------------------
def thin_to_good_lhs(df: pd.DataFrame, rng: np.random.Generator):
    """
    Bin each of the 6 design parameters into N_BINS_PER_DIM equal-width bins
    (based on the global min/max of the deduped data), forming a 6D grid.
    Any grid cell containing more samples than a cap (set from the
    distribution of cell occupancy) is randomly downsampled to that cap.
    This directly targets clusters/over-represented regions while leaving
    sparsely covered regions untouched, pushing the joint design closer to
    a uniform / well-spread LHS-like coverage.
    """
    bin_idx = pd.DataFrame(index=df.index)
    edges = {}
    for p in PARAMS:
        lo, hi = df[p].min(), df[p].max()
        e = np.linspace(lo, hi, N_BINS_PER_DIM + 1)
        edges[p] = e
        # np.digitize with right-open bins, clip last bin's right edge into range
        idx = np.digitize(df[p].values, e[1:-1], right=False)
        bin_idx[p] = idx

    cell_id = bin_idx.apply(lambda row: tuple(row.values), axis=1)
    cell_counts = cell_id.value_counts()
    nonzero_counts = cell_counts.values

    cap = int(np.percentile(nonzero_counts, CELL_CAP_PERCENTILE))
    cap = max(cap, 1)

    keep_indices = []
    for cid, group_idx in df.groupby(cell_id).groups.items():
        group_idx = list(group_idx)
        if len(group_idx) > cap:
            chosen = rng.choice(group_idx, size=cap, replace=False)
            keep_indices.extend(chosen.tolist())
        else:
            keep_indices.extend(group_idx)

    good = df.loc[sorted(keep_indices)].reset_index(drop=True)
    info = {
        "n_cells_total_possible": N_BINS_PER_DIM ** len(PARAMS),
        "n_cells_occupied": len(cell_counts),
        "cell_cap": cap,
        "max_cell_count_before": int(nonzero_counts.max()),
        "median_cell_count_before": float(np.median(nonzero_counts)),
    }
    return good, info


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    rng = np.random.default_rng(RANDOM_SEED)
    report_lines = []

    def log(msg=""):
        print(msg)
        report_lines.append(msg)

    log("Loading and merging CSVs...")
    merged = load_and_merge(FOLDER)
    merged.to_csv(OUTDIR / "merged_raw.csv", index=False)
    log(f"  merged_raw.csv: {len(merged)} rows from files in {FOLDER.resolve()}")

    log("\nChecking for duplicated design points...")
    deduped, n_dupes = check_and_drop_duplicates(merged)
    deduped.to_csv(OUTDIR / "merged_deduped.csv", index=False)
    log(f"  Found {n_dupes} duplicate rows (exact match on {COLS_OF_INTEREST})")
    log(f"  merged_deduped.csv: {len(deduped)} rows remaining")

    log("\nParameter space bounds (deduped data):")
    bounds_before = bounds_table(deduped)
    log(bounds_before.to_string(index=False))

    plot_distributions(deduped, "Distributions — before thinning", OUTDIR / "histograms_before.png")

    log("\nThinning over-sampled regions of the 6D design space...")
    good, thin_info = thin_to_good_lhs(deduped, rng)
    good.to_csv(OUTDIR / "merged_good_lhs.csv", index=False)
    for k, v in thin_info.items():
        log(f"  {k}: {v}")
    log(f"  merged_good_lhs.csv: {len(good)} rows kept out of {len(deduped)} "
        f"({len(good)/len(deduped)*100:.1f}%)")

    plot_distributions(good, "Distributions — after thinning (good LHS)", OUTDIR / "histograms_after.png")

    log("\nParameter space bounds (after thinning):")
    bounds_after = bounds_table(good)
    log(bounds_after.to_string(index=False))

    log("\nPer-source-file sample counts:")
    log("  before dedup/thinning:")
    log(merged["source_file"].value_counts().to_string())
    log("  after dedup + thinning:")
    log(good["source_file"].value_counts().to_string())

    log("\n=== SUMMARY ===")
    log(f"Total raw samples merged:      {len(merged)}")
    log(f"After removing duplicates:     {len(deduped)}  (-{n_dupes})")
    log(f"After thinning to good LHS:    {len(good)}  (-{len(deduped) - len(good)})")
    log(f"Design parameter space (6D) bounds:")
    for _, r in bounds_after.iterrows():
        if r["parameter"] in PARAMS:
            log(f"  {r['parameter']:35s}: [{r['min']:.4g}, {r['max']:.4g}]")
    if "keff" in bounds_after["parameter"].values:
        r = bounds_after[bounds_after["parameter"] == "keff"].iloc[0]
        log(f"  {'keff (output)':35s}: [{r['min']:.4g}, {r['max']:.4g}]  mean={r['mean']:.4g}, std={r['std']:.4g}")

    with open(OUTDIR / "summary_report.txt", "w") as fh:
        fh.write("\n".join(report_lines))

    print(f"\nAll outputs saved in {OUTDIR.resolve()}/")


if __name__ == "__main__":
    main()
