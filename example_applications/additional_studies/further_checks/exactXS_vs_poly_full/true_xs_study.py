"""
true_xs_study.py
=========================================================================

Question being answered: does the keff discrepancy vs. the OpenMC target
persist even when the diffusion solver is fed the *exact*, geometry-matched
multi-group cross-sections (MGXS) extracted directly from OpenMC, instead of
the polynomial-regression-predicted XS used as the PEDS pre-training
baseline? The previous study (`diff_mesh_1/`) answered this on a
hand-picked pool of 60 "worst" + "normal" test-set error cases. This script
reruns the exact same diffusion comparison (true MGXS vs. poly-regression
XS, both vs. the OpenMC target keff) over *every* geometry for which a
true-MGXS row can be uniquely matched to an LHS sample — i.e. the full
population, not just the 60-case subsample.

For each matched case:
  * run diffusion with the true, geometry-extracted OpenMC MGXS
  * run diffusion with the poly-regression-predicted XS (same fixed
    regression model used as the PEDS epoch-0 baseline everywhere else)
  * record keff's and pcm discrepancies vs. the OpenMC target

Outputs (written next to this script, under
example_applications/additional_studies/further_checks/exactXS_vs_poly_full/):
  full_true_xs_results.csv       one row per matched case
  full_true_xs_summary_stats.csv aggregate + correlation statistics
=========================================================================
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scistats

STUDY_DIR = Path(__file__).resolve().parent


def _find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "solvers").is_dir() and (candidate / "data").is_dir():
            return candidate
    raise RuntimeError(f"Could not find the PEDS_NT root above {start}")


PROJECT_ROOT = _find_project_root(STUDY_DIR)
# True MGXS rows now live in the merged cross-section table (the old
# modules/FILES/older_datasets/2jul_full.csv was folded into this file).
MGXS_CSV = PROJECT_ROOT / "data" / "files" / "dataset_keff_xs.csv"
NPZ_PATH = PROJECT_ROOT / "data" / "highfidelity" / "LHS_0.8_newbounds.npz"
OUT_DIR = STUDY_DIR

sys.path.insert(0, str(PROJECT_ROOT))

from config_and_run.NTcode_config_data.config_run import GEO_CYL as GEO  # noqa: E402
from config_and_run.NTcode_config_data.config_def import (  # noqa: E402
    GeometryConfig, BoundarySpec, MatProperties,
)
from solvers.NTdiffusion.diffusion_solver import (  # noqa: E402
    run_diffusion_solver, xs_layout, fn_xs_per_region, _enforce_chi, predict_xs,
)

PARAM_COLS = [
    "r0_b4c_rod_outer_radius",
    "r0_b4c_rod_cr_fraction",
    "r1_fuel_annulus_outer_radius",
    "r1_fuel_annulus_enrichment",
    "r1_fuel_annulus_f_mod",
    "r2_water_outer_radius",
]
REGION_PREFIXES = ("b4c_rod", "fuel_annulus", "water")


def pcm(kp: float, kr: float) -> float:
    return float((kp - kr) / (kp * kr) * 1e5)


def update_geo(geo: GeometryConfig, params_raw: np.ndarray) -> GeometryConfig:
    return GeometryConfig(
        G=geo.G,
        regions=geo.regions,
        boundaries=(
            BoundarySpec(name="CR_outer", radius=float(params_raw[0])),
            BoundarySpec(name="core_outer", radius=float(params_raw[2])),
            BoundarySpec(name="moderator_outer", radius=float(params_raw[5])),
        ),
        geometry=geo.geometry,
        mat_properties=MatProperties(
            cr_fraction=float(params_raw[1]),
            enrichment=float(params_raw[3]),
            f_mod=float(params_raw[4]),
        ),
        bc=geo.bc,
        mesh_size=geo.mesh_size,
    )


def match_params_to_sample_idx(params: np.ndarray, raw_all: np.ndarray) -> int | None:
    mask = np.ones(len(raw_all), dtype=bool)
    for j, v in enumerate(params):
        mask &= np.isclose(raw_all[:, j], float(v), rtol=1e-4, atol=1e-4)
    hits = np.where(mask)[0]
    if len(hits) == 1:
        return int(hits[0])
    return None


def xs_tensor_from_mgxs_row(row: pd.Series, geo: GeometryConfig) -> np.ndarray:
    G = geo.G
    n = fn_xs_per_region(G)
    lay = xs_layout(G)
    xs_dict = {}
    for mat in geo.regions:
        m = mat.region_ID
        for g in range(G):
            for qty in ("diffusion-coefficient", "absorption", "nu-fission", "chi"):
                xs_dict[f"{m}_{qty}_g{g+1}"] = float(row[f"{m}_{qty}_g{g+1}"])
        for k in range(G ** 2):
            xs_dict[f"{m}_scatter matrix_g{k+1}"] = float(row[f"{m}_scatter matrix_g{k+1}"])
    xs_dict = _enforce_chi(xs_dict, geo)

    xs_tensor = np.zeros((len(geo.regions), n), dtype=np.float64)
    for mat in geo.regions:
        vec = np.zeros(n, dtype=np.float64)
        m = mat.region_ID
        vec[lay["D"]] = [xs_dict[f"{m}_diffusion-coefficient_g{g+1}"] for g in range(G)]
        vec[lay["Sigma_a"]] = [xs_dict[f"{m}_absorption_g{g+1}"] for g in range(G)]
        vec[lay["nuSigma_f"]] = [xs_dict[f"{m}_nu-fission_g{g+1}"] for g in range(G)]
        vec[lay["chi"]] = [xs_dict[f"{m}_chi_g{g+1}"] for g in range(G)]
        vec[lay["Sigma_s"]] = [xs_dict[f"{m}_scatter matrix_g{k+1}"] for k in range(G ** 2)]
        xs_tensor[mat.region_index] = vec
    return xs_tensor


def build_full_case_table(mgxs_df: pd.DataFrame, raw_all: np.ndarray) -> pd.DataFrame:
    """Match every MGXS-csv row to a unique LHS sample_idx (dedup on sample_idx)."""
    records = []
    for csv_row_idx, row in mgxs_df.iterrows():
        params = np.array([float(row[c]) for c in PARAM_COLS], dtype=np.float64)
        idx = match_params_to_sample_idx(params, raw_all)
        if idx is None:
            continue
        records.append(dict(mgxs_csv_row=int(csv_row_idx), sample_idx=int(idx)))
    table = pd.DataFrame(records)
    n_before = len(table)
    table = table.drop_duplicates("sample_idx", keep="first").reset_index(drop=True)
    print(f"Matched {n_before} MGXS rows -> {len(table)} unique geometries "
          f"(out of {len(mgxs_df)} MGXS rows / {len(raw_all)} LHS samples).")
    return table


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"MGXS source : {MGXS_CSV}")
    print(f"NPZ source  : {NPZ_PATH}")
    print(f"Output dir  : {OUT_DIR}")

    npz = np.load(NPZ_PATH, allow_pickle=True)
    raw_all = np.asarray(npz["params_raw"], dtype=np.float64)
    keff_all = np.asarray(npz["keffs"], dtype=np.float64)

    mgxs_df = pd.read_csv(MGXS_CSV)
    case_table = build_full_case_table(mgxs_df, raw_all)

    t0 = time.time()
    result_rows = []
    n_fail = 0
    for i, case in enumerate(case_table.itertuples(index=False), start=1):
        idx = int(case.sample_idx)
        row = mgxs_df.iloc[int(case.mgxs_csv_row)]
        params = raw_all[idx]
        keff_openmc = float(keff_all[idx])

        try:
            geo = update_geo(GEO, params)
            xs_true = xs_tensor_from_mgxs_row(row, geo)
            xs_poly = predict_xs(geo)

            k_true, _, _ = run_diffusion_solver(xs_true, geo)
            k_poly, _, _ = run_diffusion_solver(xs_poly, geo)
            k_true_f, k_poly_f = float(k_true), float(k_poly)
        except Exception as exc:  # noqa: BLE001
            n_fail += 1
            print(f"  [warn] case {i}/{len(case_table)} sample_idx={idx} failed: {exc}")
            continue

        result_rows.append({
            "mgxs_csv_row": int(case.mgxs_csv_row),
            "sample_idx": idx,
            "keff_openmc": keff_openmc,
            "keff_openmc_csv": float(row["keff"]) if "keff" in row.index else np.nan,
            "keff_std_openmc": float(row["keff_std"]) if "keff_std" in row.index else np.nan,
            "keff_diffusion_true_mgxs": k_true_f,
            "keff_diffusion_poly_xs": k_poly_f,
            "pcm_true_mgxs_minus_openmc": pcm(k_true_f, keff_openmc),
            "pcm_poly_xs_minus_openmc": pcm(k_poly_f, keff_openmc),
            "pcm_poly_xs_minus_true_mgxs": pcm(k_poly_f, k_true_f),
            "abs_pcm_true_mgxs_minus_openmc": abs(pcm(k_true_f, keff_openmc)),
            "abs_pcm_poly_xs_minus_openmc": abs(pcm(k_poly_f, keff_openmc)),
            "abs_pcm_poly_xs_minus_true_mgxs": abs(pcm(k_poly_f, k_true_f)),
            **{c: float(v) for c, v in zip(PARAM_COLS, params)},
        })

        if i % 200 == 0 or i == len(case_table):
            elapsed = time.time() - t0
            print(f"  [{i}/{len(case_table)}] done  ({elapsed:.1f}s elapsed, {n_fail} failures)")

    results_df = pd.DataFrame(result_rows)
    results_path = OUT_DIR / "full_true_xs_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nWrote {results_path}  ({len(results_df)} rows, {n_fail} failed cases skipped)")

    # ── Aggregate + correlation statistics ─────────────────────────────────
    def _stats_block(col: str) -> dict:
        s = results_df[col].astype(float)
        return {
            f"{col}_mean": float(s.mean()),
            f"{col}_median": float(s.median()),
            f"{col}_std": float(s.std()),
            f"{col}_min": float(s.min()),
            f"{col}_max": float(s.max()),
            f"{col}_p05": float(s.quantile(0.05)),
            f"{col}_p95": float(s.quantile(0.95)),
        }

    summary = {"n_cases": len(results_df)}
    for col in [
        "pcm_true_mgxs_minus_openmc",
        "pcm_poly_xs_minus_openmc",
        "pcm_poly_xs_minus_true_mgxs",
        "abs_pcm_true_mgxs_minus_openmc",
        "abs_pcm_poly_xs_minus_openmc",
        "abs_pcm_poly_xs_minus_true_mgxs",
    ]:
        summary.update(_stats_block(col))

    x_true = results_df["pcm_true_mgxs_minus_openmc"].to_numpy(dtype=float)
    x_poly = results_df["pcm_poly_xs_minus_openmc"].to_numpy(dtype=float)
    k_true = results_df["keff_diffusion_true_mgxs"].to_numpy(dtype=float)
    k_poly = results_df["keff_diffusion_poly_xs"].to_numpy(dtype=float)
    k_ref = results_df["keff_openmc"].to_numpy(dtype=float)

    pear_r, pear_p = scistats.pearsonr(x_true, x_poly)
    spear_r, spear_p = scistats.spearmanr(x_true, x_poly)
    pear_k_true, _ = scistats.pearsonr(k_true, k_ref)
    pear_k_poly, _ = scistats.pearsonr(k_poly, k_ref)

    summary.update({
        "pearson_r_pcm_true_vs_pcm_poly": float(pear_r),
        "pearson_p_pcm_true_vs_pcm_poly": float(pear_p),
        "spearman_r_pcm_true_vs_pcm_poly": float(spear_r),
        "spearman_p_pcm_true_vs_pcm_poly": float(spear_p),
        "pearson_r_keff_true_mgxs_vs_openmc": float(pear_k_true),
        "pearson_r_keff_poly_xs_vs_openmc": float(pear_k_poly),
        "rmse_pcm_true_mgxs_minus_openmc": float(np.sqrt(np.mean(x_true ** 2))),
        "rmse_pcm_poly_xs_minus_openmc": float(np.sqrt(np.mean(x_poly ** 2))),
        "frac_same_sign_pcm_true_and_poly": float(np.mean(np.sign(x_true) == np.sign(x_poly))),
    })

    # Correlation of the true-MGXS discrepancy with each geometry parameter,
    # to see which design variables drive the diffusion-approximation error.
    for p in PARAM_COLS:
        pvals = results_df[p].to_numpy(dtype=float)
        r_param, _ = scistats.pearsonr(pvals, x_true)
        summary[f"pearson_r_abs_pcm_true_vs_{p}"] = float(
            scistats.pearsonr(pvals, np.abs(x_true))[0]
        )
        summary[f"pearson_r_pcm_true_vs_{p}"] = float(r_param)

    summary_df = pd.DataFrame([summary])
    summary_path = OUT_DIR / "full_true_xs_summary_stats.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")
    print("\nSummary:")
    print(summary_df.T.to_string(header=False))


if __name__ == "__main__":
    main()
