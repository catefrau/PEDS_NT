"""
Publication figures for the held-out test set (precise_param_strat).

Reads the representative comparison CSV that underlies
``new_test_metrics_summary_by_train_size.csv``:

  epoch 0  -> baseline (uncorrected NT / no PEDS XS correction)
  last epoch -> PEDS

Figures are sized for on-screen / single-column viewing: large canvas
with publication-scale fonts, PNG only.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

THIS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = THIS_DIR.parent / "RUNS" / "precise_param_strat" / "testset_results"
DEFAULT_CSV = RESULTS_DIR / "rep_train1000_seed2_keff_comparison.csv"
DEFAULT_OUT = RESULTS_DIR / "pub_plots"

FIGSIZE_SCATTER = (6.6, 6.2)
FIGSIZE_HIST = (8.6, 6.4)
DPI = 200

FS_TITLE = 18
FS_LABEL = 17
FS_TICK = 15
FS_CBAR = 16
FS_LEGEND = 14

# Wider hist canvas makes the same pt sizes look smaller on screen;
# scale so hist text matches parity-plot readability.
_FS_SCALE_HIST = FIGSIZE_HIST[0] / FIGSIZE_SCATTER[0]
FS_LABEL_H = round(FS_LABEL * _FS_SCALE_HIST)
FS_TICK_H = round(FS_TICK * _FS_SCALE_HIST)
FS_CBAR_H = round(FS_CBAR * _FS_SCALE_HIST)
FS_LEGEND_H = round(FS_LEGEND * _FS_SCALE_HIST)

BETA_EFF_PCM = 650
TARGET_PCM = 100
N_TICKS = 5
N_BINS = 28

LABEL_BASELINE = "Baseline"
LABEL_PEDS = "PEDS"


def _apply_pub_rc():
    plt.rcParams.update({
        "font.size": FS_TICK,
        "axes.titlesize": FS_TITLE,
        "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEGEND,
        "axes.linewidth": 0.9,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    })


def _style_axes(ax, n_ticks=N_TICKS, tick_fs=FS_TICK):
    ax.tick_params(axis="both", labelsize=tick_fs, length=3.2, width=0.8, pad=2)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=n_ticks, min_n_ticks=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=n_ticks, min_n_ticks=3))
    ax.grid(True, alpha=0.22, linestyle="--", linewidth=0.6)
    ax.set_axisbelow(True)


def _style_cbar(cbar, label, n_ticks=N_TICKS, label_fs=FS_CBAR, tick_fs=FS_TICK):
    cbar.set_label(label, fontsize=label_fs)
    cbar.ax.tick_params(labelsize=tick_fs, length=3, width=0.8, pad=1.5)
    cbar.locator = MaxNLocator(nbins=n_ticks)
    cbar.update_ticks()


def load_comparison(csv_path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    needed = {"epoch", "keff_openmc", "keff_peds", "delta_rho_pcm"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")
    df = df.copy()
    df["abs_delta_rho"] = df["delta_rho_pcm"].abs()
    return df


def split_baseline_peds(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    epoch0 = int(df["epoch"].min())
    epoch_peds = int(df["epoch"].max())
    return df[df["epoch"] == epoch0].copy(), df[df["epoch"] == epoch_peds].copy()


def mean_pcm(frame: pd.DataFrame) -> float:
    return float(frame["abs_delta_rho"].mean())


def plot_keff_parity(
    frame: pd.DataFrame,
    save_path: Path | str,
    title_prefix: str,
    ylabel: str,
    vmin: float | None = None,
    vmax: float | None = None,
    lims: tuple[float, float] | None = None,
):
    """keff_openmc vs keff_predicted, coloured by |Δρ| (pcm)."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    x = frame["keff_openmc"].to_numpy()
    y = frame["keff_peds"].to_numpy()
    c = frame["abs_delta_rho"].to_numpy()
    avg = mean_pcm(frame)

    if vmin is None:
        vmin = float(np.nanmin(c))
    if vmax is None:
        vmax = float(np.nanmax(c))
    if lims is None:
        lo = min(np.nanmin(x), np.nanmin(y))
        hi = max(np.nanmax(x), np.nanmax(y))
        pad = 0.02 * (hi - lo)
        lims = (lo - pad, hi + pad)

    fig, ax = plt.subplots(figsize=FIGSIZE_SCATTER)
    sc = ax.scatter(
        x, y,
        c=c, cmap="RdYlGn_r",
        vmin=vmin, vmax=vmax,
        s=42, edgecolors="0.15", linewidths=0.3, zorder=3,
    )
    ax.plot(lims, lims, "k--", lw=1.3, zorder=2, label=r"$y=x$")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(r"$k_{\mathrm{eff}}$ (OpenMC)", fontsize=FS_LABEL, labelpad=3)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL, labelpad=3)
    ax.set_title(
        rf"{title_prefix}: mean $|\Delta\rho|$ = {avg:.0f} pcm",
        fontsize=FS_TITLE, pad=6, fontweight="bold",
    )
    _style_axes(ax)
    ax.legend(
        loc="upper left", framealpha=0.92, handlelength=1.6, borderpad=0.3,
        fontsize=FS_LEGEND,
    )

    cbar = fig.colorbar(sc, ax=ax, pad=0.03, fraction=0.046)
    _style_cbar(cbar, r"$|\Delta\rho|$ (pcm)")

    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved {save_path}")


def _draw_sample_tiles(ax, frame, bin_edges, cmap, norm):
    """One coloured tile per test case, stacked in Δρ bins (sorted by keff).

    A faint count envelope sits behind the tiles so the distribution shape
    stays readable when a bin is densely packed.
    """
    df = frame.copy()
    df["bin"] = pd.cut(df["abs_delta_rho"], bins=bin_edges, include_lowest=True)
    width = (bin_edges[1] - bin_edges[0]) * 0.84
    tile_h = 0.88

    counts, _ = np.histogram(df["abs_delta_rho"].to_numpy(), bins=bin_edges)
    ax.stairs(
        counts, bin_edges, fill=True, color="0.90",
        edgecolor="0.75", linewidth=0.6, zorder=1,
    )

    patches, facecolors = [], []
    max_h = 0
    for interval, group in df.groupby("bin", observed=True):
        group = group.sort_values("keff_openmc")
        x0 = (interval.left + interval.right) / 2.0 - width / 2.0
        n = len(group)
        max_h = max(max_h, n)
        for stack_pos, (_, row) in enumerate(group.iterrows()):
            patches.append(Rectangle((x0, stack_pos + (1.0 - tile_h) / 2.0), width, tile_h))
            facecolors.append(cmap(norm(row["keff_openmc"])))

    coll = PatchCollection(
        patches,
        facecolors=facecolors,
        edgecolors="white",
        linewidths=0.45,
        zorder=3,
    )
    ax.add_collection(coll)
    return max_h


def plot_rho_histogram(
    frame: pd.DataFrame,
    save_path: Path | str,
    keff_vmin: float | None = None,
    keff_vmax: float | None = None,
    x_max: float | None = None,
    n_bins: int = N_BINS,
):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    keff_vals = frame["keff_openmc"].to_numpy()
    rho = frame["abs_delta_rho"].to_numpy()
    if keff_vmin is None:
        keff_vmin = float(keff_vals.min())
    if keff_vmax is None:
        keff_vmax = float(keff_vals.max())
    if x_max is None:
        x_max = float(np.nanmax(rho))

    bin_edges = np.linspace(0.0, x_max, n_bins + 1)
    cmap = plt.colormaps["coolwarm"]
    norm = plt.Normalize(vmin=keff_vmin, vmax=keff_vmax)

    fig, ax = plt.subplots(figsize=FIGSIZE_HIST)
    max_h = _draw_sample_tiles(ax, frame, bin_edges, cmap, norm)

    ax.axvline(
        BETA_EFF_PCM, color="#00008B", linewidth=1.6,
        linestyle=(0, (4, 2.5)), label=rf"$\beta_{{\mathrm{{eff}}}}$",
        zorder=4,
    )
    ax.axvline(
        TARGET_PCM, color="0.15", linewidth=1.5,
        linestyle=":", label="100 pcm", zorder=4,
    )

    ax.set_xlabel("Reactivity difference (pcm)", fontsize=FS_LABEL_H, labelpad=4)
    ax.set_ylabel("Cases", fontsize=FS_LABEL_H, labelpad=4)

    ax.set_xlim(0.0, x_max * 1.03)
    ax.set_ylim(0.0, max(max_h, 1) * 1.08)
    _style_axes(ax, tick_fs=FS_TICK_H)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=N_TICKS, integer=True, min_n_ticks=3))
    ax.legend(
        loc="upper right", framealpha=0.92, handlelength=1.8, borderpad=0.3,
        fontsize=FS_LEGEND_H,
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.046)
    _style_cbar(
        cbar, r"$k_{\mathrm{eff}}$ (OpenMC)",
        label_fs=FS_CBAR_H, tick_fs=FS_TICK_H,
    )

    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved {save_path}")


def generate_all(csv_path: Path | str = DEFAULT_CSV, out_dir: Path | str = DEFAULT_OUT):
    _apply_pub_rc()
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_comparison(csv_path)
    baseline, peds = split_baseline_peds(df)

    rho_vmin = float(df["abs_delta_rho"].min())
    rho_vmax = float(df["abs_delta_rho"].max())
    k_all = np.concatenate([
        baseline["keff_openmc"].to_numpy(),
        baseline["keff_peds"].to_numpy(),
        peds["keff_peds"].to_numpy(),
    ])
    lo, hi = float(k_all.min()), float(k_all.max())
    pad = 0.02 * (hi - lo)
    keff_lims = (lo - pad, hi + pad)

    keff_cmin = float(df["keff_openmc"].min())
    keff_cmax = float(df["keff_openmc"].max())

    print(f"Loaded {csv_path.name}: N={len(baseline)}  "
          f"baseline mean |Δρ|={mean_pcm(baseline):.1f} pcm  "
          f"PEDS mean |Δρ|={mean_pcm(peds):.1f} pcm")

    plot_keff_parity(
        baseline,
        out_dir / "keff_parity_baseline.png",
        title_prefix=LABEL_BASELINE,
        ylabel=r"$k_{\mathrm{eff}}$ (baseline)",
        vmin=rho_vmin, vmax=rho_vmax, lims=keff_lims,
    )
    plot_keff_parity(
        peds,
        out_dir / "keff_parity_peds.png",
        title_prefix=LABEL_PEDS,
        ylabel=r"$k_{\mathrm{eff}}$ (PEDS)",
        vmin=rho_vmin, vmax=rho_vmax, lims=keff_lims,
    )

    plot_rho_histogram(
        baseline,
        out_dir / "rho_hist_baseline.png",
        keff_vmin=keff_cmin, keff_vmax=keff_cmax,
        x_max=float(baseline["abs_delta_rho"].max()),
    )
    plot_rho_histogram(
        peds,
        out_dir / "rho_hist_peds.png",
        keff_vmin=keff_cmin, keff_vmax=keff_cmax,
        x_max=float(peds["abs_delta_rho"].max()),
    )


if __name__ == "__main__":
    generate_all()
