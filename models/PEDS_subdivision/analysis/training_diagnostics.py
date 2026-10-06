#!/usr/bin/env python3
"""Generate per-run and cross-run training diagnostics from epoch_metrics.csv."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parents[2]
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from PEDS_subdivision.analysis.config import (
    STUDY_FOLDER,
    STUDY_PARENT_FOLDER,
    all_train_run_glob,
    study_dir,
    training_diagnostics_outdir,
)


KEY_METRICS = [
    "train_mse_k",
    "val_mse_k",
    "train_mae_k",
    "val_mae_k",
    "train_mean_pcm",
    "val_mean_pcm",
    "train_median_pcm",
    "val_median_pcm",
    "train_p95_pcm",
    "val_p95_pcm",
    "train_frac_below_650",
    "val_frac_below_650",
    "train_frac_below_100",
    "val_frac_below_100",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-folder", default=STUDY_FOLDER)
    parser.add_argument("--study-parent-folder", default=STUDY_PARENT_FOLDER)
    parser.add_argument("--run-glob", default=None, help="Optional explicit run dir glob")
    parser.add_argument("--outdir", type=Path, default=None)
    return parser.parse_args()


def find_run_metric_files(study_folder: str, study_parent_folder: str, run_glob: str | None) -> list[tuple[str, Path]]:
    pattern = run_glob or all_train_run_glob(study_folder, study_parent_folder)
    run_dirs = [Path(p) for p in sorted(glob.glob(pattern)) if Path(p).is_dir()]
    pairs = []
    for run_dir in run_dirs:
        metric_path = run_dir / "epoch_metrics.csv"
        if metric_path.exists():
            pairs.append((run_dir.name, metric_path))
    return pairs


def _plot_full_history(df: pd.DataFrame, run_name: str, outpath: Path) -> None:
    epochs = df["epoch"].to_numpy()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"Training history: {run_name}")

    axes[0, 0].semilogy(epochs, df["train_mse_k"], label="train MSE")
    axes[0, 0].semilogy(epochs, df["val_mse_k"], label="val MSE")
    axes[0, 0].set_title("MSE in k units")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].semilogy(epochs, df["train_mae_k"], label="train MAE")
    axes[0, 1].semilogy(epochs, df["val_mae_k"], label="val MAE")
    axes[0, 1].set_title("MAE in k units")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, df["val_mean_pcm"], label="mean |Δρ|")
    axes[1, 0].plot(epochs, df["val_median_pcm"], label="median |Δρ|")
    axes[1, 0].plot(epochs, df["val_p95_pcm"], label="p95 |Δρ|", linestyle="--")
    axes[1, 0].axhline(650, linestyle=":", color="gray", label="650 pcm")
    axes[1, 0].axhline(100, linestyle=":", color="navy", label="100 pcm")
    axes[1, 0].set_title("Validation reactivity error (pcm)")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].plot(epochs, 100.0 * df["val_frac_below_650"], label="<650 pcm")
    axes[1, 1].plot(epochs, 100.0 * df["val_frac_below_100"], label="<100 pcm")
    axes[1, 1].set_ylim(0, 105)
    axes[1, 1].set_title("Validation fractions")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Fraction (%)")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    fig.tight_layout()
    fig.savefig(outpath, dpi=170)
    plt.close(fig)


def _plot_reactivity_fraction_only(df: pd.DataFrame, run_name: str, outpath: Path) -> None:
    epochs = df["epoch"].to_numpy()
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"Validation reactivity/fraction trends: {run_name}")

    axes[0].plot(epochs, df["val_mean_pcm"], label="mean |Δρ|")
    axes[0].plot(epochs, df["val_median_pcm"], label="median |Δρ|")
    axes[0].plot(epochs, df["val_p95_pcm"], label="p95 |Δρ|", linestyle="--")
    axes[0].axhline(650, linestyle=":", color="gray", label="650 pcm")
    axes[0].axhline(100, linestyle=":", color="navy", label="100 pcm")
    axes[0].set_ylabel("pcm")
    axes[0].set_title("Validation reactivity metrics")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, 100.0 * df["val_frac_below_650"], label="<650 pcm")
    axes[1].plot(epochs, 100.0 * df["val_frac_below_100"], label="<100 pcm")
    axes[1].set_ylim(0, 105)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Fraction (%)")
    axes[1].set_title("Validation fraction metrics")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(outpath, dpi=170)
    plt.close(fig)


def _plot_cross_run_summary(run_frames: dict[str, pd.DataFrame], outpath: Path) -> None:
    max_epoch = max(int(df["epoch"].max()) for df in run_frames.values())
    seed_order = sorted(run_frames.keys())
    metrics_by_seed = {}
    for seed in seed_order:
        df = run_frames[seed].set_index("epoch").sort_index()
        df = df.reindex(range(0, max_epoch + 1))
        df = df.interpolate(limit_direction="both")
        metrics_by_seed[seed] = df

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    plot_defs = [
        ("val_mse_k", axes[0, 0], "Val MSE (k units)", True),
        ("val_mae_k", axes[0, 1], "Val MAE (k units)", True),
        ("val_mean_pcm", axes[1, 0], "Val mean |Δρ| (pcm)", False),
        ("val_frac_below_650", axes[1, 1], "Val frac < 650 pcm", False),
    ]

    epochs = np.arange(0, max_epoch + 1)
    for metric, ax, title, semilogy in plot_defs:
        stack = np.vstack([metrics_by_seed[s][metric].to_numpy(dtype=float) for s in seed_order])
        mean = np.nanmean(stack, axis=0)
        std = np.nanstd(stack, axis=0)
        if semilogy:
            mean = np.clip(mean, 1e-12, None)
            lo = np.clip(mean - std, 1e-12, None)
            hi = np.clip(mean + std, 1e-12, None)
            ax.semilogy(epochs, mean, color="#1f77b4", label="mean across seeds")
            ax.fill_between(epochs, lo, hi, color="#1f77b4", alpha=0.2, label="±1 std")
        else:
            y = 100.0 * mean if "frac" in metric else mean
            ystd = 100.0 * std if "frac" in metric else std
            ax.plot(epochs, y, color="#1f77b4", label="mean across seeds")
            ax.fill_between(epochs, y - ystd, y + ystd, color="#1f77b4", alpha=0.2, label="±1 std")
            if metric == "val_frac_below_650":
                ax.set_ylim(0, 105)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("Cross-run summary — validation (mean ± std across seeds)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=170)
    plt.close(fig)


def _plot_cross_run_train_val_summary(run_frames: dict[str, pd.DataFrame], outpath: Path) -> None:
    """Cross-run mean±std with train and val on the same panels."""
    max_epoch = max(int(df["epoch"].max()) for df in run_frames.values())
    seed_order = sorted(run_frames.keys())
    metrics_by_seed = {}
    for seed in seed_order:
        df = run_frames[seed].set_index("epoch").sort_index()
        df = df.reindex(range(0, max_epoch + 1))
        df = df.interpolate(limit_direction="both")
        metrics_by_seed[seed] = df

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    panel_defs = [
        (axes[0, 0], "MSE (k units)", True, "train_mse_k", "val_mse_k"),
        (axes[0, 1], "MAE (k units)", True, "train_mae_k", "val_mae_k"),
        (axes[1, 0], "mean |Δρ| (pcm)", False, "train_mean_pcm", "val_mean_pcm"),
        (axes[1, 1], "frac < 650 pcm", False, "train_frac_below_650", "val_frac_below_650"),
    ]
    split_colors = {"train": "#4C72B0", "val": "#DD8452"}
    epochs = np.arange(0, max_epoch + 1)

    for ax, title, semilogy, train_col, val_col in panel_defs:
        for split_col, split_name in ((train_col, "train"), (val_col, "val")):
            stack = np.vstack([
                metrics_by_seed[s][split_col].to_numpy(dtype=float) for s in seed_order
            ])
            mean = np.nanmean(stack, axis=0)
            std = np.nanstd(stack, axis=0)
            color = split_colors[split_name]
            if semilogy:
                mean = np.clip(mean, 1e-12, None)
                lo = np.clip(mean - std, 1e-12, None)
                hi = np.clip(mean + std, 1e-12, None)
                ax.semilogy(epochs, mean, color=color, label=f"{split_name} mean")
                ax.fill_between(epochs, lo, hi, color=color, alpha=0.2)
            else:
                y = 100.0 * mean if "frac" in split_col else mean
                ystd = 100.0 * std if "frac" in split_col else std
                ax.plot(epochs, y, color=color, label=f"{split_name} mean")
                ax.fill_between(epochs, y - ystd, y + ystd, color=color, alpha=0.2)
                if "frac" in split_col:
                    ax.set_ylim(0, 105)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("Cross-run summary — train vs val (mean ± std across seeds)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=170)
    plt.close(fig)


def generate_training_diagnostics(study_folder: str, study_parent_folder: str, outdir: Path, run_glob: str | None = None) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    run_metric_files = find_run_metric_files(study_folder, study_parent_folder, run_glob)
    if not run_metric_files:
        raise FileNotFoundError(
            f"No epoch_metrics.csv found under {study_dir(study_folder, study_parent_folder)}"
        )

    run_frames = {}
    rows = []
    for run_name, metrics_path in run_metric_files:
        df = pd.read_csv(metrics_path)
        missing = [c for c in ["epoch"] + KEY_METRICS if c not in df.columns]
        if missing:
            print(f"Skipping {run_name}: missing columns {missing}")
            continue
        run_frames[run_name] = df

        run_out = outdir / run_name
        run_out.mkdir(parents=True, exist_ok=True)
        _plot_full_history(df, run_name, run_out / "training_history_full.png")
        _plot_reactivity_fraction_only(df, run_name, run_out / "reactivity_fraction_history.png")

        last = df.sort_values("epoch").iloc[-1]
        rows.append(
            {
                "run_name": run_name,
                "final_epoch": int(last["epoch"]),
                "final_val_mse_k": float(last["val_mse_k"]),
                "final_val_mae_k": float(last["val_mae_k"]),
                "final_val_mean_pcm": float(last["val_mean_pcm"]),
                "final_val_median_pcm": float(last["val_median_pcm"]),
                "final_val_p95_pcm": float(last["val_p95_pcm"]),
                "final_val_frac_below_650": float(last["val_frac_below_650"]),
                "final_val_frac_below_100": float(last["val_frac_below_100"]),
            }
        )

    if not run_frames:
        raise ValueError("No run metrics were usable after column checks.")

    pd.DataFrame(rows).sort_values("run_name").to_csv(outdir / "final_epoch_metrics_by_run.csv", index=False)
    _plot_cross_run_summary(run_frames, outdir / "cross_run_mean_std_summary.png")
    _plot_cross_run_train_val_summary(run_frames, outdir / "cross_run_train_val_mean_std_summary.png")
    print(f"Wrote training diagnostics to {outdir}")


def main() -> None:
    args = parse_args()
    outdir = args.outdir or training_diagnostics_outdir(args.study_folder, args.study_parent_folder)
    generate_training_diagnostics(
        study_folder=args.study_folder,
        study_parent_folder=args.study_parent_folder,
        outdir=outdir,
        run_glob=args.run_glob,
    )


if __name__ == "__main__":
    main()

