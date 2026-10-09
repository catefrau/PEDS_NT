"""
timing_study.py
=========================================================================
Same geometry pool as `true_xs_study.py` (uniquely matched
LHS ↔ MGXS rows), but only the polynomial-regression baseline XS
(`predict_xs`) is used. No MGXS diffusion solves.

Purpose: measure wall time of one `run_diffusion_solver` call (forward
+ adjoint, same entry point as the full true-XS study) in milliseconds.

Inputs are the paths defined in `true_xs_study.py`:
  data/files/dataset_keff_xs.csv
  data/highfidelity/LHS_0.8_newbounds.npz

Outputs written next to this script, under
example_applications/additional_studies/further_checks/exactXS_vs_poly_full/timing_study_polyXS/:
  poly_xs_timing_results.csv
  poly_xs_timing_summary.csv
=========================================================================
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

STUDY_DIR = Path(__file__).resolve().parent
TRUE_XS_DIR = STUDY_DIR.parent


def _find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "solvers").is_dir() and (candidate / "data").is_dir():
            return candidate
    raise RuntimeError(f"Could not find the PEDS_NT root above {start}")


PROJECT_ROOT = _find_project_root(STUDY_DIR)
OUT_DIR = STUDY_DIR

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TRUE_XS_DIR))

from config_and_run.NTcode_config_data.config_run import GEO_CYL as GEO  # noqa: E402
from solvers.NTdiffusion.diffusion_solver import (  # noqa: E402
    run_diffusion_solver, predict_xs,
)
from true_xs_study import (  # noqa: E402
    MGXS_CSV, NPZ_PATH, PARAM_COLS,
    build_full_case_table, update_geo,
)

N_WARMUP = 5


def _ms_stats(arr: np.ndarray, prefix: str) -> dict:
    s = pd.Series(arr.astype(float))
    return {
        f"{prefix}_mean_ms": float(s.mean()),
        f"{prefix}_median_ms": float(s.median()),
        f"{prefix}_std_ms": float(s.std()),
        f"{prefix}_min_ms": float(s.min()),
        f"{prefix}_max_ms": float(s.max()),
        f"{prefix}_p05_ms": float(s.quantile(0.05)),
        f"{prefix}_p95_ms": float(s.quantile(0.95)),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"NPZ source  : {NPZ_PATH}")
    print(f"MGXS source : {MGXS_CSV}  (matching only; not used as XS)")
    print(f"Output dir  : {OUT_DIR}")
    print(f"mesh_size   : {GEO.mesh_size}")
    print(f"warmup      : {N_WARMUP} cases excluded from stats")

    npz = np.load(NPZ_PATH, allow_pickle=True)
    raw_all = np.asarray(npz["params_raw"], dtype=np.float64)
    keff_all = np.asarray(npz["keffs"], dtype=np.float64)

    mgxs_df = pd.read_csv(MGXS_CSV)
    case_table = build_full_case_table(mgxs_df, raw_all)
    n_cases = len(case_table)

    # Warmup: first N_WARMUP geometries (compile / cache / BLAS init).
    for w in range(min(N_WARMUP, n_cases)):
        idx = int(case_table.iloc[w]["sample_idx"])
        geo = update_geo(GEO, raw_all[idx])
        xs_poly = predict_xs(geo)
        run_diffusion_solver(xs_poly, geo)
    print(f"Warmup of {min(N_WARMUP, n_cases)} cases done.")

    t_loop0 = time.perf_counter()
    rows = []
    n_fail = 0
    for i, case in enumerate(case_table.itertuples(index=False), start=1):
        idx = int(case.sample_idx)
        params = raw_all[idx]
        try:
            geo = update_geo(GEO, params)

            t_pred0 = time.perf_counter()
            xs_poly = predict_xs(geo)
            t_pred_ms = (time.perf_counter() - t_pred0) * 1e3

            t_sol0 = time.perf_counter()
            k_poly, _, _ = run_diffusion_solver(xs_poly, geo)
            t_sol_ms = (time.perf_counter() - t_sol0) * 1e3
            k_poly_f = float(k_poly)
        except Exception as exc:  # noqa: BLE001
            n_fail += 1
            print(f"  [warn] case {i}/{n_cases} sample_idx={idx} failed: {exc}")
            continue

        rows.append({
            "mgxs_csv_row": int(case.mgxs_csv_row),
            "sample_idx": idx,
            "keff_openmc": float(keff_all[idx]),
            "keff_diffusion_poly_xs": k_poly_f,
            "predict_xs_ms": t_pred_ms,
            "solver_fwd_adj_ms": t_sol_ms,
            "predict_plus_solver_ms": t_pred_ms + t_sol_ms,
            **{c: float(v) for c, v in zip(PARAM_COLS, params)},
        })

        if i % 200 == 0 or i == n_cases:
            elapsed = time.perf_counter() - t_loop0
            print(f"  [{i}/{n_cases}] done  ({elapsed:.1f}s elapsed, {n_fail} failures)")

    results_df = pd.DataFrame(rows)
    results_path = OUT_DIR / "poly_xs_timing_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nWrote {results_path}  ({len(results_df)} rows, {n_fail} failed)")

    solver = results_df["solver_fwd_adj_ms"].to_numpy(dtype=float)
    pred = results_df["predict_xs_ms"].to_numpy(dtype=float)
    both = results_df["predict_plus_solver_ms"].to_numpy(dtype=float)

    summary = {
        "n_cases": len(results_df),
        "n_warmup_excluded_from_loop_but_rerun": N_WARMUP,
        "mesh_size": float(GEO.mesh_size),
        "loop_wall_s": float(time.perf_counter() - t_loop0),
    }
    summary.update(_ms_stats(solver, "solver_fwd_adj"))
    summary.update(_ms_stats(pred, "predict_xs"))
    summary.update(_ms_stats(both, "predict_plus_solver"))

    summary_df = pd.DataFrame([summary])
    summary_path = OUT_DIR / "poly_xs_timing_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")
    print("\nTiming summary (ms):")
    print(summary_df.T.to_string(header=False))
    print(
        f"\n>>> mean one solver run (fwd+adj, poly XS): "
        f"{summary['solver_fwd_adj_mean_ms']:.3f} ms"
        f"  (median {summary['solver_fwd_adj_median_ms']:.3f} ms)"
    )


if __name__ == "__main__":
    main()
