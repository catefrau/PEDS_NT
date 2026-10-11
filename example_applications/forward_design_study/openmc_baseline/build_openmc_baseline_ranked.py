"""
Build OpenMC-only feasible ranking from `study_scalar_summary.csv`.

Inputs:
- scalar summary CSV from OpenMC screening (`--scalar-csv`).

Outputs:
- ranked feasible CSV (`--out`, default `lhs_feasible_ranked.csv`) containing
  keff/PPF metrics and geometry parameters for direct comparison with PEDS runs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCALAR = ROOT / "solvers" / "HF_openMC" / "training_data" / "2_aug" / "seed_0" / "study_scalar_summary.csv"
DEFAULT_OUT = Path(__file__).resolve().parent / "lhs_feasible_ranked.csv"
TARGET_KEFF = 1.0
TOL_PCM = 1500.0


def dpcm(keff, target=TARGET_KEFF):
    keff = np.asarray(keff, dtype=float)
    out = np.full_like(keff, np.nan, dtype=float)
    m = np.isfinite(keff) & (keff > 0)
    out[m] = np.abs(keff[m] - target) / (keff[m] * target) * 1e5
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scalar-csv", type=Path, default=DEFAULT_SCALAR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--tol-pcm", type=float, default=TOL_PCM)
    args = ap.parse_args()

    df = pd.read_csv(args.scalar_csv)
    if "radial_power_peaking_factor" in df.columns and "ppf" not in df.columns:
        df = df.rename(columns={"radial_power_peaking_factor": "ppf"})

    required = ["run_id", "keff", "keff_std", "ppf"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns in {args.scalar_csv}: {miss}")

    out = df.copy()
    out["delta_rho_pcm"] = dpcm(out["keff"].to_numpy(), TARGET_KEFF)
    out["abs_dk"] = np.abs(out["keff"] - TARGET_KEFF)
    out = out[np.isfinite(out["delta_rho_pcm"]) & (out["delta_rho_pcm"] <= args.tol_pcm)].copy()
    out = out.sort_values(["ppf", "delta_rho_pcm", "run_id"]).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1, dtype=int))

    keep = [c for c in [
        "rank", "run_id", "keff", "keff_std", "delta_rho_pcm", "abs_dk", "ppf",
        "b4c_r", "cr_frac", "fuel_r", "enrichment", "f_mod", "water_r",
        "runtime_s", "openmc_runtime_total_s", "converged_flag"
    ] if c in out.columns]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out[keep].to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(out)} feasible rows)")


if __name__ == "__main__":
    main()
