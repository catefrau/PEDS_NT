"""
PEDS vs HF parity plots for the 30 shortlisted LHS designs.

Produces (defaults):
  - peds_hf_parity.png                 (k_eff and PPF; PEDS vs HF only)
  - peds_diffonly_hf_parity.png        (same layout, PEDS + diffusion-only vs HF)

Paths and output names can be overridden via CLI so new studies do not
overwrite publication figures, e.g.:

  python plot_peds_hf_parity.py \\
    --peds-csv .../lhs_peds_hf_comparison.csv \\
    --diffonly-csv .../diffusion_only_recheck_slim.csv \\
    --out-peds .../peds_hf_parity_ens5k_kefffirst.png \\
    --out-diffonly .../peds_diffonly_hf_parity_ens5k_kefffirst.png

Publication-sized fonts throughout.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PLOT_DIR = Path(__file__).resolve().parent
INV_DIR = PLOT_DIR.parent

PEDS_CSV = INV_DIR / "lhs_peds_baseline" / "lhs_peds_hf_comparison.csv"
DIFFONLY_SLIM_CSV = (
    INV_DIR / "lhs_peds_baseline" / "diffusion_only_check" / "diffusion_only_recheck_slim.csv"
)
OUT_PNG = PLOT_DIR / "peds_hf_parity.png"
OUT_PNG_WITH_DIFFONLY = PLOT_DIR / "peds_diffonly_hf_parity.png"

AGREE_PCM = 650.0  # relative-reactivity band around the k_eff identity line

# Publication-sized text (large for thesis figures; n=30 so no correlation callout)
FS_TITLE = 22
FS_LABEL = 21
FS_TICK = 18
FS_LEGEND = 17
FS_ANNOT = 18

COLOR_PEDS = "#1f77b4"
COLOR_DIFF = "#ff7f0e"


def identity_band_keff(x: np.ndarray, pcm: float = AGREE_PCM) -> tuple[np.ndarray, np.ndarray]:
    """
    Band around the 1:1 line for |Δρ(x, y)| <= pcm.
    With HF on x and prediction on y:
      |(x - y)/(x*y)| * 1e5 <= pcm  =>  y = x / (1 ± dr*x).
    """
    dr = pcm / 1e5
    x = np.asarray(x, dtype=float)
    y_lo = x / (1.0 + dr * x)
    y_hi = x / (1.0 - dr * x)
    return y_lo, y_hi


def compute_delta_rho_pcm(k_a: np.ndarray, k_b: np.ndarray) -> np.ndarray:
    k_a = np.asarray(k_a, dtype=float)
    k_b = np.asarray(k_b, dtype=float)
    out = np.full(k_a.shape, np.nan, dtype=float)
    mask = np.isfinite(k_a) & np.isfinite(k_b) & (k_a > 0) & (k_b > 0)
    out[mask] = np.abs(k_a[mask] - k_b[mask]) / (k_a[mask] * k_b[mask]) * 1e5
    return out


def plot_peds_only(df: pd.DataFrame, out_png: Path) -> None:
    keff_p = df["keff_peds"].to_numpy(dtype=float)
    keff_h = df["keff_hf"].to_numpy(dtype=float)
    ppf_p = df["ppf_diffusion"].to_numpy(dtype=float)
    ppf_h = df["ppf_hf"].to_numpy(dtype=float)
    ppf_err = ppf_p - ppf_h  # signed: >0 over-prediction

    fig, (ax_k, ax_p) = plt.subplots(1, 2, figsize=(13.2, 6.2))

    # ── Left: k_eff parity ────────────────────────────────────────────────
    k_min = min(keff_p.min(), keff_h.min())
    k_max = max(keff_p.max(), keff_h.max())
    pad = 0.04 * (k_max - k_min)
    k_lo, k_hi = k_min - pad, k_max + pad
    xx = np.linspace(k_lo, k_hi, 400)
    y_lo, y_hi = identity_band_keff(xx, AGREE_PCM)

    ax_k.fill_between(
        xx, y_lo, y_hi, color="0.82", alpha=0.7, zorder=0,
        label=rf"$\pm{int(AGREE_PCM)}\,\mathrm{{pcm}}$",
    )
    ax_k.plot(xx, xx, color="0.25", linestyle="--", linewidth=1.4, zorder=1, label="1:1")
    # Convention: HF on x, prediction on y
    ax_k.scatter(keff_h, keff_p, s=55, c=COLOR_PEDS, edgecolors="none", alpha=0.85, zorder=2)

    ax_k.set_xlim(k_lo, k_hi)
    ax_k.set_ylim(k_lo, k_hi)
    ax_k.set_aspect("equal", adjustable="box")
    ax_k.set_xlabel(r"$k_{\mathrm{eff}}$ (HF)", fontsize=FS_LABEL)
    ax_k.set_ylabel(r"$k_{\mathrm{eff}}$ (PEDS)", fontsize=FS_LABEL)
    ax_k.tick_params(labelsize=FS_TICK)
    ax_k.legend(loc="lower right", fontsize=FS_LEGEND, framealpha=0.92)
    ax_k.set_title(r"$k_{\mathrm{eff}}$ parity", fontsize=FS_TITLE)

    # ── Right: PPF parity ─────────────────────────────────────────────────
    p_min = min(ppf_p.min(), ppf_h.min())
    p_max = max(ppf_p.max(), ppf_h.max())
    ppad = 0.05 * (p_max - p_min)
    p_lo, p_hi = p_min - ppad, p_max + ppad
    xp = np.linspace(p_lo, p_hi, 100)

    over = ppf_err >= 0
    under = ~over

    ax_p.plot(xp, xp, color="0.25", linestyle="--", linewidth=1.4, zorder=1, label="1:1")
    ax_p.scatter(
        ppf_h[over], ppf_p[over],
        s=55, c="#d62728", edgecolors="none", alpha=0.85, zorder=2,
        label="over-prediction (PEDS > HF)",
    )
    ax_p.scatter(
        ppf_h[under], ppf_p[under],
        s=55, c="#2ca02c", edgecolors="none", alpha=0.85, zorder=2,
        label="under-prediction (PEDS < HF)",
    )

    ax_p.set_xlim(p_lo, p_hi)
    ax_p.set_ylim(p_lo, p_hi)
    ax_p.set_aspect("equal", adjustable="box")
    ax_p.set_xlabel("PPF (HF)", fontsize=FS_LABEL)
    ax_p.set_ylabel("PPF (diffusion, PEDS)", fontsize=FS_LABEL)
    ax_p.tick_params(labelsize=FS_TICK)
    ax_p.legend(loc="lower right", fontsize=FS_LEGEND, framealpha=0.92)
    ax_p.set_title("PPF parity", fontsize=FS_TITLE)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")
    print(f"  n={len(df)}")
    print(f"  over-predicted: {int(over.sum())}  under-predicted: {int(under.sum())}")


def plot_peds_and_diffonly(df: pd.DataFrame, diff_df: pd.DataFrame, out_png: Path) -> None:
    """Overlay PEDS and diffusion-only predictions against the same HF values."""
    # Join on PEDS rank so we only keep HF-verified rows that also have diff-only.
    left = df[["rank", "keff_peds", "keff_hf", "ppf_diffusion", "ppf_hf"]].copy()
    left = left.rename(columns={"rank": "rank_peds", "ppf_diffusion": "ppf_peds"})
    right = diff_df[["rank_peds", "keff_diffonly", "ppf_diffonly", "keff_hf", "ppf_hf"]].copy()
    right = right.rename(columns={"keff_hf": "keff_hf_d", "ppf_hf": "ppf_hf_d"})
    m = left.merge(right, on="rank_peds", how="inner")
    m = m[np.isfinite(m["keff_diffonly"]) & np.isfinite(m["keff_hf"])].copy()
    if len(m) == 0:
        raise RuntimeError("No overlapping HF-verified rows with diffusion-only results.")

    keff_p = m["keff_peds"].to_numpy(dtype=float)
    keff_d = m["keff_diffonly"].to_numpy(dtype=float)
    keff_h = m["keff_hf"].to_numpy(dtype=float)
    ppf_p = m["ppf_peds"].to_numpy(dtype=float)
    ppf_d = m["ppf_diffonly"].to_numpy(dtype=float)
    ppf_h = m["ppf_hf"].to_numpy(dtype=float)

    avg_drho_p = float(np.nanmean(compute_delta_rho_pcm(keff_p, keff_h)))
    avg_drho_d = float(np.nanmean(compute_delta_rho_pcm(keff_d, keff_h)))
    avg_dppf_p = float(np.nanmean(np.abs(ppf_p - ppf_h)))
    avg_dppf_d = float(np.nanmean(np.abs(ppf_d - ppf_h)))

    fig, (ax_k, ax_p) = plt.subplots(1, 2, figsize=(13.2, 6.2))

    # ── Left: k_eff parity (PEDS + diffusion-only) ────────────────────────
    # Convention: HF on x, prediction on y
    k_min = min(keff_p.min(), keff_d.min(), keff_h.min())
    k_max = max(keff_p.max(), keff_d.max(), keff_h.max())
    pad = 0.04 * (k_max - k_min)
    k_lo, k_hi = k_min - pad, k_max + pad
    xx = np.linspace(k_lo, k_hi, 400)
    y_lo, y_hi = identity_band_keff(xx, AGREE_PCM)

    ax_k.fill_between(
        xx, y_lo, y_hi, color="0.82", alpha=0.7, zorder=0,
        label=rf"$\pm{int(AGREE_PCM)}\,\mathrm{{pcm}}$",
    )
    ax_k.plot(xx, xx, color="0.25", linestyle="--", linewidth=1.4, zorder=1, label="1:1")
    ax_k.scatter(
        keff_h, keff_d, s=55, c=COLOR_DIFF, marker="s",
        edgecolors="none", alpha=0.80, zorder=2,
        label="diffusion-only",
    )
    ax_k.scatter(
        keff_h, keff_p, s=55, c=COLOR_PEDS, marker="o",
        edgecolors="none", alpha=0.85, zorder=3,
        label="PEDS",
    )

    ax_k.set_xlim(k_lo, k_hi)
    ax_k.set_ylim(k_lo, k_hi)
    ax_k.set_aspect("equal", adjustable="box")
    ax_k.set_xlabel(r"$k_{\mathrm{eff}}$ (HF)", fontsize=FS_LABEL)
    ax_k.set_ylabel(r"$k_{\mathrm{eff}}$ (prediction)", fontsize=FS_LABEL)
    ax_k.tick_params(labelsize=FS_TICK)
    ax_k.legend(loc="lower right", fontsize=FS_LEGEND, framealpha=0.92)
    ax_k.set_title(r"$k_{\mathrm{eff}}$ parity: PEDS vs diffusion-only", fontsize=FS_TITLE)

    # ── Right: PPF parity (PEDS + diffusion-only) ─────────────────────────
    p_min = min(ppf_p.min(), ppf_d.min(), ppf_h.min())
    p_max = max(ppf_p.max(), ppf_d.max(), ppf_h.max())
    ppad = 0.05 * (p_max - p_min)
    p_lo, p_hi = p_min - ppad, p_max + ppad
    xp = np.linspace(p_lo, p_hi, 100)

    ax_p.plot(xp, xp, color="0.25", linestyle="--", linewidth=1.4, zorder=1, label="1:1")
    ax_p.scatter(
        ppf_h, ppf_d, s=55, c=COLOR_DIFF, marker="s",
        edgecolors="none", alpha=0.80, zorder=2,
        label="diffusion-only",
    )
    ax_p.scatter(
        ppf_h, ppf_p, s=55, c=COLOR_PEDS, marker="o",
        edgecolors="none", alpha=0.85, zorder=3,
        label="PEDS",
    )

    ax_p.set_xlim(p_lo, p_hi)
    ax_p.set_ylim(p_lo, p_hi)
    ax_p.set_aspect("equal", adjustable="box")
    ax_p.set_xlabel("PPF (HF)", fontsize=FS_LABEL)
    ax_p.set_ylabel("PPF (prediction)", fontsize=FS_LABEL)
    ax_p.tick_params(labelsize=FS_TICK)
    ax_p.legend(loc="lower right", fontsize=FS_LEGEND, framealpha=0.92)
    ax_p.set_title("PPF parity: PEDS vs diffusion-only", fontsize=FS_TITLE)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_png}")
    print(f"  n={len(m)} HF-verified designs with diffusion-only")
    print(f"  avg |Δρ| pcm: PEDS={avg_drho_p:.1f}  diffusion-only={avg_drho_d:.1f}")
    print(f"  avg |ΔPPF|:   PEDS={avg_dppf_p:.4f}  diffusion-only={avg_dppf_d:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--peds-csv", type=Path, default=PEDS_CSV)
    ap.add_argument("--diffonly-csv", type=Path, default=DIFFONLY_SLIM_CSV)
    ap.add_argument("--out-peds", type=Path, default=OUT_PNG)
    ap.add_argument("--out-diffonly", type=Path, default=OUT_PNG_WITH_DIFFONLY)
    ap.add_argument("--expect-n", type=int, default=30,
                    help="Expected number of HF-verified rows (0 disables check).")
    args = ap.parse_args()

    df = pd.read_csv(args.peds_csv)
    if args.expect_n > 0 and len(df) != args.expect_n:
        raise AssertionError(
            f"expected {args.expect_n} shortlisted designs, got {len(df)} "
            f"from {args.peds_csv}"
        )

    args.out_peds.parent.mkdir(parents=True, exist_ok=True)
    plot_peds_only(df, args.out_peds)

    if not args.diffonly_csv.exists():
        print(f"Skipping diffusion-only overlay: missing {args.diffonly_csv}")
        return
    diff_df = pd.read_csv(args.diffonly_csv)
    plot_peds_and_diffonly(df, diff_df, args.out_diffonly)


if __name__ == "__main__":
    main()
