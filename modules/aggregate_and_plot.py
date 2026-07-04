"""
aggregate_and_plot.py
=========================================================================
Reads the aggregate CSV produced by evaluate_test_metrics.py (one row per
run, columns include train_size, seed, test_mean_pcm, test_median_pcm, ...),
groups by train_size, computes mean/std/min/max across seeds for every
metric, saves that summary as its own CSV, and plots how the headline
metrics evolve with training-set size with a variability band.

USAGE
-----
    python aggregate_and_plot.py /path/to/LOGS/test_set_metrics_all_runs.csv
    python aggregate_and_plot.py ... --band minmax   # use min/max instead of ±std
=========================================================================
"""
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


if __name__ == "__main__":
    csv_path = "LHS_trainsize/testset_results/LHS_metrics_all_runs.csv"
    band = "std"

    agg = summarize(csv_path)
    out_dir = os.path.dirname(csv_path) or "."

    plot_scaling(agg, os.path.join(out_dir, "new_scaling_mean_pcm.png"),
                 metric="test_mean_pcm", band=band)
    plot_scaling(agg, os.path.join(out_dir, "new_scaling_median_pcm.png"),
                 metric="test_median_pcm", band=band)
    plot_scaling(agg, os.path.join(out_dir, "new_scaling_frac_below_650.png"),
                 metric="test_frac_below_650", band=band, logx=True)
    """ plot_n_seeds_annotated(agg, os.path.join(out_dir, "new_scaling_mean_pcm_nseeds.png"),
                            metric="test_mean_pcm") """