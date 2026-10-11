"""
Plot PEDS-vs-HF parity for keff and PPF.

Inputs:
- CSV with paired PEDS/HF columns (`keff_peds`, `keff_hf`, `ppf_diffusion`, `ppf_hf`).

Outputs:
- publication figure `peds_hf_parity_ens5k_kefffirst.png` (or custom `--out`).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PLOT_DIR = Path(__file__).resolve().parent
ROOT = PLOT_DIR.parent
DEFAULT_CSV = ROOT / "reference_results" / "ens5k_kefffirst_08092342" / "lhs_peds_hf_comparison.csv"
DEFAULT_OUT = PLOT_DIR / "peds_hf_parity_ens5k_kefffirst.png"
AGREE_PCM = 650.0

if not DEFAULT_CSV.exists():
    # Fallback to the active cleaned results location.
    DEFAULT_CSV = ROOT / "results" / "ens5k_kefffirst_08092342" / "lhs_peds_hf_comparison.csv"


def identity_band_keff(x: np.ndarray, pcm: float = AGREE_PCM) -> tuple[np.ndarray, np.ndarray]:
    dr = pcm / 1e5
    x = np.asarray(x, dtype=float)
    y_lo = x / (1.0 + dr * x)
    y_hi = x / (1.0 - dr * x)
    return y_lo, y_hi


def plot_parity(df: pd.DataFrame, out_png: Path) -> None:
    keff_p = df["keff_peds"].to_numpy(dtype=float)
    keff_h = df["keff_hf"].to_numpy(dtype=float)
    ppf_p = df["ppf_diffusion"].to_numpy(dtype=float)
    ppf_h = df["ppf_hf"].to_numpy(dtype=float)
    ppf_err = ppf_p - ppf_h

    fig, (ax_k, ax_p) = plt.subplots(1, 2, figsize=(13.2, 6.2))

    k_min = min(keff_p.min(), keff_h.min())
    k_max = max(keff_p.max(), keff_h.max())
    pad = 0.04 * (k_max - k_min)
    k_lo, k_hi = k_min - pad, k_max + pad
    xx = np.linspace(k_lo, k_hi, 400)
    y_lo, y_hi = identity_band_keff(xx, AGREE_PCM)

    ax_k.fill_between(xx, y_lo, y_hi, color="0.82", alpha=0.7, zorder=0, label=rf"$\pm{int(AGREE_PCM)}\,\mathrm{{pcm}}$")
    ax_k.plot(xx, xx, color="0.25", linestyle="--", linewidth=1.4, zorder=1, label="1:1")
    ax_k.scatter(keff_h, keff_p, s=55, c="#1f77b4", edgecolors="none", alpha=0.85, zorder=2)
    ax_k.set_xlim(k_lo, k_hi)
    ax_k.set_ylim(k_lo, k_hi)
    ax_k.set_aspect("equal", adjustable="box")
    ax_k.set_xlabel(r"$k_{\mathrm{eff}}$ (HF)", fontsize=21)
    ax_k.set_ylabel(r"$k_{\mathrm{eff}}$ (PEDS)", fontsize=21)
    ax_k.tick_params(labelsize=18)
    ax_k.legend(loc="lower right", fontsize=17, framealpha=0.92)
    ax_k.set_title(r"$k_{\mathrm{eff}}$ parity", fontsize=22)

    p_min = min(ppf_p.min(), ppf_h.min())
    p_max = max(ppf_p.max(), ppf_h.max())
    ppad = 0.05 * (p_max - p_min)
    p_lo, p_hi = p_min - ppad, p_max + ppad
    xp = np.linspace(p_lo, p_hi, 100)

    over = ppf_err >= 0
    under = ~over
    ax_p.plot(xp, xp, color="0.25", linestyle="--", linewidth=1.4, zorder=1, label="1:1")
    ax_p.scatter(ppf_h[over], ppf_p[over], s=55, c="#d62728", edgecolors="none", alpha=0.85, zorder=2,
                 label="over-prediction (PEDS > HF)")
    ax_p.scatter(ppf_h[under], ppf_p[under], s=55, c="#2ca02c", edgecolors="none", alpha=0.85, zorder=2,
                 label="under-prediction (PEDS < HF)")
    ax_p.set_xlim(p_lo, p_hi)
    ax_p.set_ylim(p_lo, p_hi)
    ax_p.set_aspect("equal", adjustable="box")
    ax_p.set_xlabel("PPF (HF)", fontsize=21)
    ax_p.set_ylabel("PPF (diffusion, PEDS)", fontsize=21)
    ax_p.tick_params(labelsize=18)
    ax_p.legend(loc="lower right", fontsize=17, framealpha=0.92)
    ax_p.set_title("PPF parity", fontsize=22)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--peds-csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--expect-n", type=int, default=30)
    args = ap.parse_args()

    df = pd.read_csv(args.peds_csv)
    if args.expect_n > 0 and len(df) != args.expect_n:
        raise AssertionError(f"expected {args.expect_n} rows, got {len(df)} from {args.peds_csv}")
    plot_parity(df, args.out)


if __name__ == "__main__":
    main()
