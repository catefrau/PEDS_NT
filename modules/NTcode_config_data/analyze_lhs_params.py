"""
Analyze parameter ranges, distributions, and LHS uniformity across a folder
of CSV files containing neutron-diffusion parameter sweeps.

For each CSV file:
  - min / max / mean / std of the 6 LHS parameters + keff
  - a histogram grid (saved as PNG) showing the empirical distribution
  - a chi-square goodness-of-fit test against a Uniform(min, max) distribution
    for each of the 6 parameters, to flag where LHS coverage looks broken
    (e.g. clumping, gaps, non-uniform density).

Also produces:
  - summary_stats.csv        -> one row per file with min/max/mean/std for each column
  - lhs_uniformity_stats.csv -> one row per (file, parameter) with chi2 stat & p-value

Usage:
    python analyze_lhs_params.py
    (edit FOLDER below if your CSVs aren't in ../FILES)
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chisquare

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
FOLDER = Path("../FILES/older_datasets")
OUTDIR = Path("./lhs_analysis_outputs/olderDS")
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

N_BINS = 10          # bins for both histograms and the chi-square uniformity test
ALPHA = 0.05         # significance threshold for "non-uniform" flag


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def load_relevant_columns(csv_path: Path) -> pd.DataFrame:
    """Read only the columns we care about, if present."""
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    cols = [c for c in COLS_OF_INTEREST if c in header]
    missing = set(COLS_OF_INTEREST) - set(cols)
    if missing:
        warnings.warn(f"{csv_path.name}: missing columns {missing}")
    return pd.read_csv(csv_path, usecols=cols)


def uniformity_chi2(series: pd.Series, n_bins: int = N_BINS):
    """
    Chi-square goodness-of-fit test of `series` against a Uniform(min, max).
    Returns (chi2_stat, p_value, observed_counts, bin_edges).
    Low p-value (< ALPHA) => distribution deviates from uniform => LHS coverage
    looks compromised for that parameter in that file.
    """
    data = series.dropna().values
    lo, hi = data.min(), data.max()
    if hi == lo:
        return np.nan, np.nan, None, None
    counts, edges = np.histogram(data, bins=n_bins, range=(lo, hi))
    expected = np.full(n_bins, data.size / n_bins)
    chi2, p = chisquare(f_obs=counts, f_exp=expected)
    return chi2, p, counts, edges


def summarize_file(csv_path: Path) -> dict:
    df = load_relevant_columns(csv_path)
    row = {"file": csv_path.name, "n_samples": len(df)}
    for col in COLS_OF_INTEREST:
        if col not in df.columns:
            continue
        row[f"{col}_min"] = df[col].min()
        row[f"{col}_max"] = df[col].max()
        row[f"{col}_mean"] = df[col].mean()
        row[f"{col}_std"] = df[col].std()
    return row


def plot_histograms(df: pd.DataFrame, csv_path: Path):
    cols_present = [c for c in COLS_OF_INTEREST if c in df.columns]
    n = len(cols_present)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, col in zip(axes, cols_present):
        data = df[col].dropna()
        ax.hist(data, bins=N_BINS, color="steelblue", edgecolor="black", alpha=0.8)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("value")
        ax.set_ylabel("count")

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(f"Distributions — {csv_path.name}", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = OUTDIR / f"{csv_path.stem}_histograms.png"
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    return out_png


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    csv_files = sorted(FOLDER.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {FOLDER.resolve()}")
        return

    summary_rows = []
    uniformity_rows = []

    for f in csv_files:
        print(f"Processing {f.name} ...")
        df = load_relevant_columns(f)

        # basic stats
        summary_rows.append(summarize_file(f))

        # histograms
        plot_histograms(df, f)

        # LHS uniformity check (only makes sense for the 6 design parameters,
        # not for keff which is an *output*, not a sampled input)
        for p in PARAMS:
            if p not in df.columns:
                continue
            chi2, pval, counts, edges = uniformity_chi2(df[p])
            uniformity_rows.append({
                "file": f.name,
                "parameter": p,
                "n_samples": df[p].notna().sum(),
                "chi2_stat": chi2,
                "p_value": pval,
                "likely_non_uniform (p<{:.2f})".format(ALPHA): (
                    pval < ALPHA if pd.notna(pval) else np.nan
                ),
            })

    summary = pd.DataFrame(summary_rows)
    uniformity = pd.DataFrame(uniformity_rows)

    summary_path = OUTDIR / "summary_stats.csv"
    uniformity_path = OUTDIR / "lhs_uniformity_stats.csv"
    summary.to_csv(summary_path, index=False)
    uniformity.to_csv(uniformity_path, index=False)

    print("\n=== Per-file min/max/mean/std summary ===")
    print(summary.to_string(index=False))

    print("\n=== LHS uniformity check (chi-square vs Uniform) ===")
    print(uniformity.to_string(index=False))

    flagged = uniformity[uniformity[f"likely_non_uniform (p<{ALPHA:.2f})"] == True]
    if not flagged.empty:
        print("\n⚠ Parameters/files where LHS coverage looks NOT uniform:")
        print(flagged[["file", "parameter", "p_value"]].to_string(index=False))
    else:
        print("\nAll checked parameters look consistent with uniform LHS sampling.")

    print(f"\nSaved:\n  {summary_path}\n  {uniformity_path}\n  histogram PNGs in {OUTDIR}/")


if __name__ == "__main__":
    main()
