"""
Two-panel k_eff distributions from a training keff epoch log:

  (left)  OpenMC (HF) + diffusion only (LF, epoch 0)
  (right) OpenMC (HF) + diffusion corrected (PEDS, final epoch)

No solver re-run required.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_LOG = (
    THIS_DIR.parent
    / "RUNS"
    / "precise_param_strat"
    / "train_1000_seed_0"
    / "keff_epoch_log_train.csv"
)
DEFAULT_OUT = THIS_DIR / "keff_histogram_openmc_diff_peds.png"

FULL_LHS_LOG = (
    THIS_DIR.parent
    / "RUNS"
    / "older"
    / "study_fullLHS_trainsize"
    / "train_1000_seed_2"
    / "keff_epoch_log_train.csv"
)
FULL_LHS_OUT = THIS_DIR / "keff_histogram_openmc_diff_peds_fullLHS_seed2.png"

KEFF_LABEL = r"$k_{\mathrm{eff}}$"
COLOR_HF = "#4C78A8"
COLOR_LF = "#F58518"
COLOR_PEDS = "#54A24B"


def plot_keff_inclusion_histogram(
    log_path: Path | str = DEFAULT_LOG,
    plot_output: Path | str = DEFAULT_OUT,
    epoch_lf: int = 0,
    epoch_corrected: int | None = None,
    bins: int = 40,
    dpi: int = 400,
) -> Path:
    log_path = Path(log_path)
    plot_output = Path(plot_output)

    df = pd.read_csv(log_path)
    if epoch_corrected is None:
        epoch_corrected = int(df["epoch"].max())

    missing = {epoch_lf, epoch_corrected} - set(df["epoch"].unique())
    if missing:
        raise ValueError(f"Missing requested epoch(s) in {log_path}: {sorted(missing)}")

    df_lf = df[df["epoch"] == epoch_lf].sort_values("sample_idx")
    df_corr = df[df["epoch"] == epoch_corrected].sort_values("sample_idx")

    keff_openmc = df_lf["keff_openmc"].to_numpy()
    keff_diff = df_lf["keff_peds"].to_numpy()
    keff_corr = df_corr["keff_peds"].to_numpy()

    # Shared bin edges so both panels are comparable
    all_vals = np.concatenate([keff_openmc, keff_diff, keff_corr])
    bin_edges = np.linspace(all_vals.min(), all_vals.max(), bins + 1)

    label_fs = 22
    tick_fs = 18
    legend_fs = 16
    title_fs = 20

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), sharey=True)

    hist_kw = dict(bins=bin_edges, alpha=0.6, edgecolor="white", linewidth=0.4)

    axes[0].hist(keff_openmc, label="OpenMC (HF)", color=COLOR_HF, **hist_kw)
    axes[0].hist(keff_diff, label="Diffusion (LF)", color=COLOR_LF, **hist_kw)
    axes[0].set_title("OpenMC vs diffusion", fontsize=title_fs, pad=10)

    axes[1].hist(keff_openmc, label="OpenMC (HF)", color=COLOR_HF, **hist_kw)
    axes[1].hist(
        keff_corr,
        label="Diffusion corrected (PEDS)",
        color=COLOR_PEDS,
        **hist_kw,
    )
    axes[1].set_title("OpenMC vs diffusion corrected", fontsize=title_fs, pad=10)

    for ax in axes:
        ax.set_xlabel(KEFF_LABEL, fontsize=label_fs)
        ax.tick_params(axis="both", labelsize=tick_fs)
        ax.legend(fontsize=legend_fs, framealpha=0.9)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Count", fontsize=label_fs)
    fig.tight_layout()

    plot_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"OpenMC (HF)            : n={len(keff_openmc)}, "
          f"[{keff_openmc.min():.4f}, {keff_openmc.max():.4f}]")
    print(f"Diffusion (LF) epoch {epoch_lf}: n={len(keff_diff)}, "
          f"[{keff_diff.min():.4f}, {keff_diff.max():.4f}]")
    print(f"PEDS corrected epoch {epoch_corrected}: n={len(keff_corr)}, "
          f"[{keff_corr.min():.4f}, {keff_corr.max():.4f}]")
    print(f"Saved: {plot_output}")
    return plot_output


if __name__ == "__main__":
    plot_keff_inclusion_histogram(
        log_path=FULL_LHS_LOG,
        plot_output=FULL_LHS_OUT,
        epoch_lf=0,
        epoch_corrected=None,  # final epoch in this log (51)
    )
