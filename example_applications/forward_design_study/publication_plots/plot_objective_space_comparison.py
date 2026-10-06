"""
Objective-space comparison: HF-only LHS baseline vs PEDS shortlist.

Two-panel figure:
  (a) Clean (k_eff, PPF) footprint — no arrows
  (b) PPF ECDF comparing the two feasible cohorts

Produces objective_space_comparison.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PLOT_DIR = Path(__file__).resolve().parent
INV_DIR = PLOT_DIR.parent

BASELINE_CSV = INV_DIR / "lhs_openmc_baseline" / "study_scalar_summary.csv"
# Alternative already-filtered file:
#   INV_DIR / "lhs_openmc_baseline" / "lhs_feasible_ranked.csv"
PEDS_CSV = INV_DIR / "lhs_peds_baseline" / "lhs_peds_hf_comparison.csv"

OUT_PNG = PLOT_DIR / "objective_space_comparison.png"

KEFF_TARGET = 1.0
FEAS_PCM = 1500.0

# Publication-sized text
FS_LABEL = 16
FS_TICK = 14
FS_LEGEND = 12
FS_ANNOT = 13
FS_PANEL = 17


def delta_rho_pcm_vs_target(keff: np.ndarray, k_target: float = KEFF_TARGET) -> np.ndarray:
    """|(k - k_t) / (k * k_t)| * 1e5  — reactivity difference in pcm."""
    keff = np.asarray(keff, dtype=float)
    return np.abs(keff - k_target) / (keff * k_target) * 1e5


def keff_bounds_from_pcm(pcm: float = FEAS_PCM, k_target: float = KEFF_TARGET) -> tuple[float, float]:
    """k bounds such that |delta_rho_pcm vs target| <= pcm."""
    dr = pcm / 1e5
    k_lo = k_target / (1.0 + dr * k_target)
    k_hi = k_target / (1.0 - dr * k_target)
    return float(k_lo), float(k_hi)


def load_baseline_feasible(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "ppf" not in df.columns:
        if "radial_power_peaking_factor" in df.columns:
            df = df.rename(columns={"radial_power_peaking_factor": "ppf"})
        else:
            raise KeyError(f"No ppf column in {path}; columns={list(df.columns)}")
    if "delta_rho_pcm" not in df.columns:
        df["delta_rho_pcm"] = delta_rho_pcm_vs_target(df["keff"].to_numpy())
    return df.loc[df["delta_rho_pcm"].abs() <= FEAS_PCM].copy()


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def main() -> None:
    baseline = load_baseline_feasible(BASELINE_CSV)
    peds = pd.read_csv(PEDS_CSV)

    verified = peds.loc[peds["delta_rho_pcm_hf"].abs() <= FEAS_PCM].copy()
    dropped = peds.loc[peds["delta_rho_pcm_hf"].abs() > FEAS_PCM].copy()

    k_lo, k_hi = keff_bounds_from_pcm(FEAS_PCM, KEFF_TARGET)
    best = verified.loc[verified["ppf_hf"].idxmin()]
    base_best_ppf = float(baseline["ppf"].min())
    peds_best_ppf = float(best["ppf_hf"])

    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(13.2, 5.8),
        gridspec_kw={"width_ratios": [1.15, 1.0]},
    )

    # ── (a) Clean objective-space footprint ───────────────────────────────
    ax_a.axvspan(k_lo, k_hi, color="0.88", alpha=0.65, zorder=0)
    ax_a.axvline(KEFF_TARGET, color="0.35", linestyle="--", linewidth=1.4, zorder=1)

    ax_a.scatter(
        baseline["keff"],
        baseline["ppf"],
        s=32,
        c="0.55",
        alpha=0.75,
        edgecolors="none",
        zorder=2,
    )
    if len(dropped):
        ax_a.scatter(
            dropped["keff_hf"],
            dropped["ppf_hf"],
            s=55,
            marker="x",
            c="#c44e52",
            linewidths=1.6,
            alpha=0.85,
            zorder=3,
        )
    ax_a.scatter(
        verified["keff_hf"],
        verified["ppf_hf"],
        s=64,
        c="#1f77b4",
        edgecolors="white",
        linewidths=0.6,
        zorder=4,
    )
    ax_a.scatter(
        [best["keff_hf"]],
        [best["ppf_hf"]],
        s=140,
        facecolors="none",
        edgecolors="#1f77b4",
        linewidths=2.2,
        zorder=5,
    )
    ax_a.annotate(
        f"best PEDS→HF\nPPF={best['ppf_hf']:.3f}\n"
        rf"$k_{{\mathrm{{eff}}}}$={best['keff_hf']:.4f}",
        xy=(best["keff_hf"], best["ppf_hf"]),
        xytext=(14, 18),
        textcoords="offset points",
        fontsize=FS_ANNOT,
        ha="left",
        va="bottom",
        color="#1f77b4",
        arrowprops=dict(arrowstyle="-", color="#1f77b4", lw=1.0),
        zorder=6,
    )

    ax_a.set_xlabel(r"$k_{\mathrm{eff}}$", fontsize=FS_LABEL)
    ax_a.set_ylabel("PPF", fontsize=FS_LABEL)
    ax_a.tick_params(labelsize=FS_TICK)
    ax_a.set_title("(a) Objective space (HF-evaluated)", fontsize=FS_PANEL, pad=8)

    handles_a = [
        Line2D(
            [0], [0], marker="o", color="none",
            markerfacecolor="0.55", markersize=8,
            label=f"HF LHS baseline, feasible (n={len(baseline)})",
        ),
        Line2D(
            [0], [0], marker="o", color="none",
            markerfacecolor="#1f77b4", markeredgecolor="white",
            markersize=9,
            label=f"PEDS shortlist, HF-confirmed (n={len(verified)})",
        ),
        Line2D(
            [0], [0], marker="x", color="#c44e52",
            linestyle="none", markersize=9, markeredgewidth=1.6,
            label=f"PEDS shortlist, failed HF gate (n={len(dropped)})",
        ),
        Line2D(
            [0], [0], color="0.35", linestyle="--", linewidth=1.4,
            label=r"$k_{\mathrm{eff}}=1$ (target)",
        ),
        plt.Rectangle(
            (0, 0), 1, 1, fc="0.88", ec="none", alpha=0.65,
            label=r"$\pm 1500\,\mathrm{pcm}$ gate",
        ),
    ]
    ax_a.legend(handles=handles_a, loc="upper right", fontsize=FS_LEGEND, framealpha=0.95)

    # ── (b) PPF ECDF comparison ───────────────────────────────────────────
    x_b, y_b = ecdf(baseline["ppf"].to_numpy())
    x_p, y_p = ecdf(verified["ppf_hf"].to_numpy())

    ax_b.step(x_b, y_b, where="post", color="0.45", linewidth=2.0, label="HF LHS baseline")
    ax_b.step(x_p, y_p, where="post", color="#1f77b4", linewidth=2.2, label="PEDS→HF confirmed")

    # mark minima
    ax_b.axvline(base_best_ppf, color="0.45", linestyle=":", linewidth=1.3, alpha=0.9)
    ax_b.axvline(peds_best_ppf, color="#1f77b4", linestyle=":", linewidth=1.3, alpha=0.9)
    ax_b.scatter([base_best_ppf], [1.0 / len(baseline)], s=50, c="0.45", zorder=3)
    ax_b.scatter([peds_best_ppf], [1.0 / len(verified)], s=55, c="#1f77b4", zorder=3)

    ymin, ymax = -0.02, 1.05
    ax_b.set_ylim(ymin, ymax)
    ax_b.set_xlabel("PPF", fontsize=FS_LABEL)
    ax_b.set_ylabel("Empirical CDF", fontsize=FS_LABEL)
    ax_b.tick_params(labelsize=FS_TICK)
    ax_b.set_title("(b) Feasible-cohort PPF distribution", fontsize=FS_PANEL, pad=8)
    ax_b.legend(loc="lower right", fontsize=FS_LEGEND, framealpha=0.95)

    # corner callout comparing best PPFs
    ax_b.text(
        0.03,
        0.97,
        f"min PPF (baseline) = {base_best_ppf:.3f}\n"
        f"min PPF (PEDS→HF)  = {peds_best_ppf:.3f}",
        transform=ax_b.transAxes,
        ha="left",
        va="top",
        fontsize=FS_ANNOT,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="0.7", alpha=0.95),
    )

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PNG}")
    print(f"  baseline feasible: {len(baseline)}  (min PPF={base_best_ppf:.4f})")
    print(f"  PEDS verified: {len(verified)}  dropped: {len(dropped)}  (min PPF={peds_best_ppf:.4f})")


if __name__ == "__main__":
    main()
