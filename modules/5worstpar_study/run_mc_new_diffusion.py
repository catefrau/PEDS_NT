"""
run_mc_new_diffusion.py
=========================================================================
Re-run the NT diffusion solver using MGXS from the higher-fidelity MC
rerun in MC_new/full_results.csv, and compare against the previous
diffusion keffs stored in PEDS_save/diffusion_true_xs_results.csv.

Outputs (under MC_new/):
  diffusion_from_mc_new.csv  — per-case keffs + deltas vs old run / OpenMC
=========================================================================
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

STUDY_DIR = Path(__file__).resolve().parent
MODULES_DIR = STUDY_DIR.parent
PROJECT_ROOT = MODULES_DIR.parent
MC_DIR = STUDY_DIR / "MC_new"
PEDS_DIR = STUDY_DIR / "PEDS_save"
MC_CSV = MC_DIR / "full_results.csv"
PREV_CSV = PEDS_DIR / "diffusion_true_xs_results.csv"
OUT_CSV = MC_DIR / "diffusion_from_mc_new.csv"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(MODULES_DIR))
sys.path.insert(0, str(STUDY_DIR))

from NTcode_config_data.config_run import GEO_CYL as GEO  # noqa: E402
from solvers.NTdiffusion.diffusion_solver import run_diffusion_solver  # noqa: E402
from run_5worstpar_study import (  # noqa: E402
    PARAM_COLS, update_geo, find_row, xs_tensor_from_mgxs_row,
)


def pcm(kp: float, kr: float) -> float:
    return float((kp - kr) / (kp * kr) * 1e5)


def main():
    print(f"MC MGXS source : {MC_CSV}")
    print(f"Previous run   : {PREV_CSV}")

    mc = pd.read_csv(MC_CSV)
    prev = pd.read_csv(PREV_CSV)
    print(f"MC rows={len(mc)}  previous cases={len(prev)}")

    rows = []
    for _, prev_row in prev.iterrows():
        rank = int(prev_row["rank"])
        idx = int(prev_row["sample_idx"])
        params = np.array([float(prev_row[c]) for c in PARAM_COLS], dtype=np.float64)

        print(f"\n=== rank {rank}  sample_idx={idx}  source={prev_row['source']} ===")
        mc_row = find_row(mc, params)
        geo = update_geo(GEO, params)
        xs = xs_tensor_from_mgxs_row(mc_row, geo)

        print("  Running diffusion with MC_new MGXS …")
        k_new, _, _ = run_diffusion_solver(xs, geo)
        k_new = float(k_new)

        k_old = float(prev_row["keff_diffusion_true_mgxs"])
        k_om_old = float(prev_row["keff_openmc"])
        k_om_new = float(mc_row["keff"])
        k_om_std_new = float(mc_row["keff_std"]) if "keff_std" in mc_row.index else np.nan

        print(f"  keff_openmc_old        = {k_om_old:.8f}")
        print(f"  keff_openmc_mc_new     = {k_om_new:.8f}  (±{k_om_std_new:.6f})")
        print(f"  keff_diff_old_MGXS     = {k_old:.8f}")
        print(f"  keff_diff_mc_new_MGXS  = {k_new:.8f}")
        print(f"  Δkeff (new−old diff)   = {k_new - k_old:+.8f}  ({pcm(k_new, k_old):+.1f} pcm)")

        rows.append({
            "rank": rank,
            "source": prev_row["source"],
            "sample_idx": idx,
            "keff_openmc_old": k_om_old,
            "keff_openmc_mc_new": k_om_new,
            "keff_std_mc_new": k_om_std_new,
            "keff_openmc_mc_new_minus_old": k_om_new - k_om_old,
            "pcm_openmc_mc_new_minus_old": pcm(k_om_new, k_om_old),
            "keff_diffusion_old_mgxs": k_old,
            "keff_diffusion_mc_new_mgxs": k_new,
            "keff_diffusion_mc_new_minus_old": k_new - k_old,
            "pcm_diffusion_mc_new_minus_old": pcm(k_new, k_old),
            "pcm_diffusion_mc_new_minus_openmc_mc_new": pcm(k_new, k_om_new),
            "pcm_diffusion_old_minus_openmc_old": pcm(k_old, k_om_old),
            "keff_peds": float(prev_row["keff_peds"]),
            "keff_peds_before": float(prev_row["keff_peds_before"]),
            **{c: float(v) for c, v in zip(PARAM_COLS, params)},
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}")
    show = [
        "rank", "sample_idx",
        "keff_openmc_old", "keff_openmc_mc_new",
        "keff_diffusion_old_mgxs", "keff_diffusion_mc_new_mgxs",
        "pcm_diffusion_mc_new_minus_old",
        "pcm_diffusion_mc_new_minus_openmc_mc_new",
    ]
    print(out[show].to_string(index=False))


if __name__ == "__main__":
    main()
