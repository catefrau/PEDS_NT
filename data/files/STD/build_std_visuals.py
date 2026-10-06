#!/usr/bin/env python
"""
Build STD-threshold datasets and publication-style baseline histograms.

Outputs:
  - std_all_cases_with_solver.csv
  - std_below40(.csv/_with_solver.csv), std_above40(.csv/_with_solver.csv)
  - std_below50(.csv/_with_solver.csv), std_above50(.csv/_with_solver.csv)
  - keff histograms + 6-parameter histograms for each subset
  - baseline-style rho histograms (colored by keff) for each subset
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODULES_DIR = PROJECT_ROOT / "config_and_run"
if str(MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(MODULES_DIR))

from PEDS_subdivision.analysis.pub_keff_plots import _apply_pub_rc, plot_rho_histogram

STD_DIR = Path(__file__).resolve().parent

PARAM_COLS = [
    "r0_b4c_rod_outer_radius",
    "r0_b4c_rod_cr_fraction",
    "r1_fuel_annulus_outer_radius",
    "r1_fuel_annulus_enrichment",
    "r1_fuel_annulus_f_mod",
    "r2_water_outer_radius",
]


def _make_keff_hist(df: pd.DataFrame, out_png: Path, title_prefix: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["keff"].dropna(), bins=40, edgecolor="black", alpha=0.82)
    ax.set_xlabel("keff")
    ax.set_ylabel("Count")
    ax.set_title(f"{title_prefix} (N={len(df)})")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _make_param_hists(df: pd.DataFrame, out_png: Path, title_prefix: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for i, col in enumerate(PARAM_COLS):
        axes[i].hist(df[col].dropna(), bins=35, edgecolor="black", alpha=0.82)
        axes[i].set_title(col)
        axes[i].set_ylabel("Count")
    fig.suptitle(f"{title_prefix} (N={len(df)})")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _to_pub_frame(df: pd.DataFrame) -> pd.DataFrame:
    pub = pd.DataFrame(
        {
            "keff_openmc": df["keff"].to_numpy(),
            "abs_delta_rho": df["delta_pcm"].abs().to_numpy(),
        }
    )
    pub.attrs["error_label"] = "Δρ"
    return pub


def _save_subset_plots(base_df: pd.DataFrame, tag: str) -> None:
    _make_keff_hist(base_df, STD_DIR / f"keff_hist_{tag}.png", f"keff distribution ({tag})")
    _make_param_hists(base_df, STD_DIR / f"param_hist_{tag}.png", f"Parameter distributions ({tag})")


def _save_rho_style_plot(with_solver_df: pd.DataFrame, tag: str, keff_vmin: float, keff_vmax: float) -> None:
    ok_df = with_solver_df[with_solver_df["status"] == "ok"].copy()
    if ok_df.empty:
        return
    pub = _to_pub_frame(ok_df)
    plot_rho_histogram(
        pub,
        STD_DIR / f"rho_hist_baseline_style_{tag}.png",
        keff_vmin=keff_vmin,
        keff_vmax=keff_vmax,
        x_max=float(pub["abs_delta_rho"].max()),
        error_label="Δρ",
    )


def _split_and_save(all_df: pd.DataFrame, all_solver_df: pd.DataFrame, thr: float, suffix: str) -> None:
    below = all_df[all_df["keff_std"] < thr].copy()
    above = all_df[all_df["keff_std"] > thr].copy()

    below.to_csv(STD_DIR / f"std_below{suffix}.csv", index=False)
    above.to_csv(STD_DIR / f"std_above{suffix}.csv", index=False)

    keys = PARAM_COLS + ["keff", "keff_std", "file_name"]
    below_solver = all_solver_df.merge(below[keys], on=keys, how="inner")
    above_solver = all_solver_df.merge(above[keys], on=keys, how="inner")

    below_solver.to_csv(STD_DIR / f"std_below{suffix}_with_solver.csv", index=False)
    above_solver.to_csv(STD_DIR / f"std_above{suffix}_with_solver.csv", index=False)

    _save_subset_plots(below, f"below{suffix}")
    _save_subset_plots(above, f"above{suffix}")

    keff_vmin = float(all_solver_df["keff"].min())
    keff_vmax = float(all_solver_df["keff"].max())
    _save_rho_style_plot(below_solver, f"below{suffix}", keff_vmin=keff_vmin, keff_vmax=keff_vmax)
    _save_rho_style_plot(above_solver, f"above{suffix}", keff_vmin=keff_vmin, keff_vmax=keff_vmax)

    print(f"std threshold {thr:.5f}: below={len(below)} above={len(above)}")


def main() -> None:
    _apply_pub_rc()

    all_df = pd.read_csv(STD_DIR / "std_all_cases.csv")
    below40_solver = pd.read_csv(STD_DIR / "std_below40_with_solver.csv")
    above40_solver = pd.read_csv(STD_DIR / "std_above40_with_solver.csv")
    all_solver_df = pd.concat([below40_solver, above40_solver], ignore_index=True)
    all_solver_df = all_solver_df.drop_duplicates().reset_index(drop=True)
    all_solver_df.to_csv(STD_DIR / "std_all_cases_with_solver.csv", index=False)

    _split_and_save(all_df, all_solver_df, thr=0.0004, suffix="40")
    _split_and_save(all_df, all_solver_df, thr=0.0005, suffix="50")


if __name__ == "__main__":
    main()
