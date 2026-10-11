"""
Build concise report artifacts from a verified forward-design run directory.

Inputs:
- run directory from `KEFF_SCAN_OUT_DIR` (or `LATEST_RUN.txt`) containing:
  `config_snapshot.json`, `lhs_peds_timing.json`, `lhs_peds_feasible_ranked.csv`,
  and `keff_scan_hf_diagnostics.csv`.

Outputs:
- `forward_design_report.md` (human-readable summary),
- `lhs_peds_hf_comparison.csv` (diagnostics + derived error metrics),
- `keff_scan_final_summary.csv` (compact ranked HF summary).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
OPENMC_BASELINE_FEASIBLE = THIS_DIR / "openmc_baseline" / "lhs_feasible_ranked.csv"


def dpcm(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    out = np.full(np.broadcast(a, b).shape, np.nan)
    aa, bb = np.broadcast_arrays(a, b)
    m = np.isfinite(aa) & np.isfinite(bb) & (aa > 0) & (bb > 0)
    out[m] = np.abs(aa[m] - bb[m]) / (aa[m] * bb[m]) * 1e5
    return out


def fmt_hms(seconds: float) -> str:
    seconds = float(seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}h {m}m {s:.0f}s"


def resolve_out_dir() -> Path:
    env = os.environ.get("KEFF_SCAN_OUT_DIR")
    if env:
        return Path(env)
    marker = THIS_DIR / "LATEST_RUN.txt"
    if marker.exists():
        return Path(marker.read_text().strip())
    raise FileNotFoundError("No KEFF_SCAN_OUT_DIR / LATEST_RUN.txt")


def main():
    out = resolve_out_dir()
    cfg = json.loads((out / "config_snapshot.json").read_text())
    timing = json.loads((out / "lhs_peds_timing.json").read_text())
    ranked = pd.read_csv(out / "lhs_peds_feasible_ranked.csv")
    diag = pd.read_csv(out / "keff_scan_hf_diagnostics.csv")

    hf_feas_pcm = float(os.environ.get("HF_FEAS_PCM", 1500.0))
    tol_peds = float(cfg.get("TOL_PCM", 1000.0))
    n_total = int(timing.get("n_lhs", len(pd.read_csv(out / "lhs_peds_all_evals.csv"))))
    n_feas = int(timing.get("n_feasible", len(ranked)))
    n_verified = len(diag)

    diag = diag.copy()
    diag["delta_rho_pcm_hf"] = dpcm(diag["keff_hf"].to_numpy(), 1.0)
    diag["delta_rho_pcm_peds"] = dpcm(diag["keff_peds"].to_numpy(), 1.0)
    diag["delta_pcm_peds_hf"] = dpcm(diag["keff_peds"].to_numpy(), diag["keff_hf"].to_numpy())
    diag["ppf_err"] = diag["ppf_diffusion"].astype(float) - diag["ppf_hf"].astype(float)

    n_hf_feas = int(np.sum(diag["delta_rho_pcm_hf"] <= hf_feas_pcm))

    base_n = np.nan
    if OPENMC_BASELINE_FEASIBLE.exists():
        base_n = len(pd.read_csv(OPENMC_BASELINE_FEASIBLE))

    final = diag.sort_values(["ppf_hf", "delta_rho_pcm_hf", "rank"]).copy()
    final.insert(0, "rank_hf_ref", np.arange(1, len(final) + 1))
    final_path = out / "keff_scan_final_summary.csv"
    cols = [
        "rank_hf_ref", "rank", "keff_peds", "keff_hf", "delta_rho_pcm_peds", "delta_rho_pcm_hf",
        "delta_pcm_peds_hf", "ppf_diffusion", "ppf_hf", "b4c_r", "cr_frac", "fuel_r", "enrichment", "f_mod", "water_r",
    ]
    final[[c for c in cols if c in final.columns]].to_csv(final_path, index=False)

    hf_elapsed = float("nan")
    hf_timing = out / "keff_scan_hf_timing.txt"
    if hf_timing.exists():
        for line in hf_timing.read_text().splitlines():
            if line.startswith("Elapsed time"):
                try:
                    hf_elapsed = float(line.split(":")[1].split("s")[0].strip())
                except Exception:
                    pass
    if not np.isfinite(hf_elapsed) and "wall_time_hf_s" in diag.columns:
        hf_elapsed = float(pd.to_numeric(diag["wall_time_hf_s"], errors="coerce").fillna(0).sum())

    mae_pcm = float(np.nanmean(diag["delta_pcm_peds_hf"]))
    mae_ppf = float(np.nanmean(np.abs(diag["ppf_err"])))
    mean_ppf_hf = float(np.nanmean(diag["ppf_hf"]))
    med_ppf_hf = float(np.nanmedian(diag["ppf_hf"]))
    mean_dpcm_hf = float(np.nanmean(diag["delta_rho_pcm_hf"]))
    med_dpcm_hf = float(np.nanmedian(diag["delta_rho_pcm_hf"]))

    summary = f"""# Forward Design Report

**Run directory:** `{out}`

## Main numbers

| Item | Value |
|---|---|
| Screen size | {n_total} |
| PEDS-feasible (`<={tol_peds:.0f}` pcm) | {n_feas} ({100*n_feas/max(n_total,1):.2f}%) |
| HF verified | {n_verified} |
| HF-feasible (`<={hf_feas_pcm:.0f}` pcm) | {n_hf_feas}/{n_verified} |
| Mean / median `|Δρ|_HF` (pcm) | {mean_dpcm_hf:.0f} / {med_dpcm_hf:.0f} |
| Mean / median HF PPF | {mean_ppf_hf:.3f} / {med_ppf_hf:.3f} |
| Mean `|Δρ(PEDS, HF)|` (pcm) | {mae_pcm:.0f} |
| Mean `|PPF_PEDS - PPF_HF|` | {mae_ppf:.3f} |
| HF verify elapsed | {fmt_hms(hf_elapsed) if np.isfinite(hf_elapsed) else 'n/a'} |
| OpenMC baseline feasible count | {base_n if np.isfinite(base_n) else 'n/a'} |

## Top 10 HF-feasible (sorted by HF PPF)

| HF rank | PEDS rank | keff_HF | |Δρ|_HF (pcm) | PPF_HF | PPF_PEDS |
|---:|---:|---:|---:|---:|---:|
"""

    hf_ok = final[final["delta_rho_pcm_hf"] <= hf_feas_pcm]
    for i, (_, r) in enumerate(hf_ok.head(10).iterrows(), start=1):
        summary += (
            f"| {i} | {int(r['rank'])} | {float(r['keff_hf']):.5f} | "
            f"{float(r['delta_rho_pcm_hf']):.0f} | {float(r['ppf_hf']):.4f} | {float(r['ppf_diffusion']):.4f} |\\n"
        )

    summary_path = out / "forward_design_report.md"
    summary_path.write_text(summary)
    diag.to_csv(out / "lhs_peds_hf_comparison.csv", index=False)

    print(f"Wrote {summary_path}")
    print(f"Wrote {final_path}")


if __name__ == "__main__":
    main()
