#!/usr/bin/env python
"""
Create comparison plots from LF outputs on std_below40/std_above40 subsets.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
IN_BELOW = ROOT / "std_below40_with_solver.csv"
IN_ABOVE = ROOT / "std_above40_with_solver.csv"
OUT_RHO = ROOT / "rho_hist_std_compare.png"
OUT_KEFF = ROOT / "solver_minus_mc_keff_std_compare.png"


def _ok(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["status"] == "ok"].copy() if "status" in df.columns else df.copy()


def main() -> None:
    df_below = _ok(pd.read_csv(IN_BELOW))
    df_above = _ok(pd.read_csv(IN_ABOVE))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df_below["delta_pcm"].dropna(), bins=50, alpha=0.6, edgecolor="black", label=f"keff_std < 4e-4 (N={len(df_below)})")
    ax.hist(df_above["delta_pcm"].dropna(), bins=50, alpha=0.6, edgecolor="black", label=f"keff_std > 4e-4 (N={len(df_above)})")
    ax.set_xlabel("Delta rho [pcm]  (LF - MC)")
    ax.set_ylabel("Count")
    ax.set_title("Baseline LF reactivity error by keff_std subset")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_RHO, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist((df_below["solver_keff"] - df_below["keff"]).dropna(), bins=50, alpha=0.6, edgecolor="black", label=f"keff_std < 4e-4 (N={len(df_below)})")
    ax.hist((df_above["solver_keff"] - df_above["keff"]).dropna(), bins=50, alpha=0.6, edgecolor="black", label=f"keff_std > 4e-4 (N={len(df_above)})")
    ax.set_xlabel("solver_keff - MC keff")
    ax.set_ylabel("Count")
    ax.set_title("Baseline LF keff residuals by keff_std subset")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_KEFF, dpi=160)
    plt.close(fig)

    print(f"Saved {OUT_RHO}")
    print(f"Saved {OUT_KEFF}")


if __name__ == "__main__":
    main()
