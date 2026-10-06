"""Publication plots for held-out test-set keff parity and error histograms."""

from __future__ import annotations

from pathlib import Path
import argparse
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

MODELS_DIR = Path(__file__).resolve().parents[2]
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from PEDS_subdivision.analysis.config import (
    STUDY_FOLDER,
    STUDY_PARENT_FOLDER,
    representative_test_csv,
    pub_keff_outdir,
)

FIGSIZE_SCATTER = (6.6, 6.2)
FIGSIZE_HIST = (8.6, 6.4)
DPI = 200

FS_TITLE = 18
FS_LABEL = 17
FS_TICK = 15
FS_CBAR = 16
FS_LEGEND = 14

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-folder", default=STUDY_FOLDER)
    parser.add_argument("--study-parent-folder", default=STUDY_PARENT_FOLDER)
    parser.add_argument("--csv-path", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def _apply_pub_rc():
    plt.rcParams.update(
        {
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
        }
    )


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


def _mathtext_error_label(error_label: str) -> str:
    if error_label == "Δk":
        return r"$\left|\Delta k\right|$"
    return r"$\left|\Delta\rho\right|$"


def load_comparison(csv_path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Accept either delta_rho_pcm (reactivity) or delta_k_pcm (Δk) as the
    # primary error column; normalise to 'abs_delta_rho' for downstream helpers.
    if "delta_k_pcm" in df.columns:
        error_col = "delta_k_pcm"
        error_label = "Δk"
    elif "delta_rho_pcm" in df.columns:
        error_col = "delta_rho_pcm"
        error_label = "Δρ"
    else:
        needed = {"epoch", "keff_openmc", "keff_peds"}
        missing = needed - set(df.columns)
        if missing:
            raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")
        raise ValueError(f"{csv_path} has neither 'delta_rho_pcm' nor 'delta_k_pcm'")
    needed = {"epoch", "keff_openmc", "keff_peds"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")
    df = df.copy()
    df["abs_delta_rho"] = df[error_col].abs()
    df.attrs["error_label"] = error_label  # carry label metadata for plot functions
    return df


def split_baseline_peds(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    epoch0 = int(df["epoch"].min())
    epoch_peds = int(df["epoch"].max())
    return df[df["epoch"] == epoch0].copy(), df[df["epoch"] == epoch_peds].copy()


def mean_pcm(frame: pd.DataFrame) -> float:
    return float(frame["abs_delta_rho"].mean())


def plot_keff_parity(frame, save_path, title_prefix, ylabel, vmin=None, vmax=None, lims=None,
                     error_label: str | None = None):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Derive label from DataFrame metadata if not explicitly provided.
    if error_label is None:
        error_label = frame.attrs.get("error_label", "Δρ")

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
        x,
        y,
        c=c,
        cmap="RdYlGn_r",
        vmin=vmin,
        vmax=vmax,
        s=42,
        edgecolors="0.15",
        linewidths=0.3,
        zorder=3,
    )
    ax.plot(lims, lims, "k--", lw=1.3, zorder=2, label=r"$y=x$")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$k_{\mathrm{eff}}$ (OpenMC)", fontsize=FS_LABEL, labelpad=3)
    ax.set_ylabel(ylabel, fontsize=FS_LABEL, labelpad=3)
    ax.set_title(
        rf"{title_prefix}: mean {_mathtext_error_label(error_label)} = {avg:.0f} pcm",
        fontsize=FS_TITLE,
        pad=6,
        fontweight="bold",
    )
    _style_axes(ax)
    ax.legend(loc="upper left", framealpha=0.92, handlelength=1.6, borderpad=0.3, fontsize=FS_LEGEND)
    cbar = fig.colorbar(sc, ax=ax, pad=0.03, fraction=0.046)
    _style_cbar(cbar, f"{_mathtext_error_label(error_label)} (pcm)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI)
    plt.close(fig)
    print(f"Saved {save_path}")


def _draw_sample_tiles(ax, frame, bin_edges, cmap, norm):
    df = frame.copy()
    df["bin"] = pd.cut(df["abs_delta_rho"], bins=bin_edges, include_lowest=True)
    width = (bin_edges[1] - bin_edges[0]) * 0.84
    tile_h = 0.88
    counts, _ = np.histogram(df["abs_delta_rho"].to_numpy(), bins=bin_edges)
    ax.stairs(counts, bin_edges, fill=True, color="0.90", edgecolor="0.75", linewidth=0.6, zorder=1)

    patches, facecolors = [], []
    max_h = 0
    for interval, group in df.groupby("bin", observed=True):
        group = group.sort_values("keff_openmc")
        x0 = (interval.left + interval.right) / 2.0 - width / 2.0
        max_h = max(max_h, len(group))
        for stack_pos, (_, row) in enumerate(group.iterrows()):
            patches.append(Rectangle((x0, stack_pos + (1.0 - tile_h) / 2.0), width, tile_h))
            facecolors.append(cmap(norm(row["keff_openmc"])))

    coll = PatchCollection(patches, facecolors=facecolors, edgecolors="white", linewidths=0.45, zorder=3)
    ax.add_collection(coll)
    return max_h


def plot_rho_histogram(frame, save_path, keff_vmin=None, keff_vmax=None, x_max=None, n_bins=N_BINS,
                       error_label: str | None = None):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if error_label is None:
        error_label = frame.attrs.get("error_label", "Δρ")
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
    ax.axvline(BETA_EFF_PCM, color="#00008B", linewidth=1.6, linestyle=(0, (4, 2.5)), label=rf"$\beta_{{\mathrm{{eff}}}}$")
    ax.axvline(TARGET_PCM, color="0.15", linewidth=1.5, linestyle=":", label="100 pcm")
    # Label clearly distinguishes Δk from Δρ so plots cannot be confused.
    x_label = f"{_mathtext_error_label(error_label)} (pcm)" if error_label != "Δρ" else "Reactivity difference (pcm)"
    ax.set_xlabel(x_label, fontsize=FS_LABEL_H, labelpad=4)
    ax.set_ylabel("Cases", fontsize=FS_LABEL_H, labelpad=4)
    ax.set_xlim(0.0, x_max * 1.03)
    ax.set_ylim(0.0, max(max_h, 1) * 1.08)
    _style_axes(ax, tick_fs=FS_TICK_H)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=N_TICKS, integer=True, min_n_ticks=3))
    ax.legend(loc="upper right", framealpha=0.92, handlelength=1.8, borderpad=0.3, fontsize=FS_LEGEND_H)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.046)
    _style_cbar(cbar, r"$k_{\mathrm{eff}}$ (OpenMC)", label_fs=FS_CBAR_H, tick_fs=FS_TICK_H)
    fig.tight_layout()
    fig.savefig(save_path, dpi=DPI)
    plt.close(fig)
    print(f"Saved {save_path}")


def generate_all(
    csv_path: Path | str,
    out_dir: Path | str,
):
    _apply_pub_rc()
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_comparison(csv_path)
    error_label = df.attrs.get("error_label", "Δρ")
    baseline, peds = split_baseline_peds(df)

    rho_vmin = float(df["abs_delta_rho"].min())
    rho_vmax = float(df["abs_delta_rho"].max())
    k_all = np.concatenate([baseline["keff_openmc"].to_numpy(), baseline["keff_peds"].to_numpy(), peds["keff_peds"].to_numpy()])
    lo, hi = float(k_all.min()), float(k_all.max())
    pad = 0.02 * (hi - lo)
    keff_lims = (lo - pad, hi + pad)
    keff_cmin = float(df["keff_openmc"].min())
    keff_cmax = float(df["keff_openmc"].max())

    print(
        f"Loaded {csv_path.name} [metric: |{error_label}|]: "
        f"N={len(baseline)} baseline mean={mean_pcm(baseline):.1f} pcm "
        f"PEDS mean={mean_pcm(peds):.1f} pcm"
    )
    plot_keff_parity(
        baseline,
        out_dir / "keff_parity_baseline.png",
        title_prefix=LABEL_BASELINE,
        ylabel=r"$k_{\mathrm{eff}}$ (baseline)",
        vmin=rho_vmin,
        vmax=rho_vmax,
        lims=keff_lims,
        error_label=error_label,
    )
    plot_keff_parity(
        peds,
        out_dir / "keff_parity_peds.png",
        title_prefix=LABEL_PEDS,
        ylabel=r"$k_{\mathrm{eff}}$ (PEDS)",
        vmin=rho_vmin,
        vmax=rho_vmax,
        lims=keff_lims,
        error_label=error_label,
    )
    plot_rho_histogram(baseline, out_dir / "rho_hist_baseline.png", keff_vmin=keff_cmin, keff_vmax=keff_cmax,
                       x_max=float(baseline["abs_delta_rho"].max()), error_label=error_label)
    plot_rho_histogram(peds, out_dir / "rho_hist_peds.png", keff_vmin=keff_cmin, keff_vmax=keff_cmax,
                       x_max=float(peds["abs_delta_rho"].max()), error_label=error_label)


def main() -> None:
    args = parse_args()
    csv_path = args.csv_path or representative_test_csv(args.study_folder, args.study_parent_folder)
    out_dir = args.out_dir or pub_keff_outdir(args.study_folder, args.study_parent_folder)
    generate_all(csv_path=csv_path, out_dir=out_dir)


if __name__ == "__main__":
    main()

