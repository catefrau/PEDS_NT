"""
run_5worstpar_study.py
=========================================================================
Extract the 5 diverse high-|Δρ| geometries identified from
1000reference testset evaluations, pull their OpenMC MGXS from
modules/FILES/older_datasets/2jul_full.csv, and run the NT diffusion
solver with those *true* MGXS (no polynomial-regression predict_xs).

Outputs (under modules/5worstpar_study/diff_mesh_1/):
    cases_params_mgxs.csv   — parameters + MGXS (+ OpenMC keff meta)
    diffusion_true_xs_results.csv — keff from diffusion with true MGXS
    flux_plots/             — forward/adjoint group fluxes per case
=========================================================================
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STUDY_DIR = Path(__file__).resolve().parent          # modules/5worstpar_study
MODULES_DIR = STUDY_DIR.parent                       # modules
PROJECT_ROOT = MODULES_DIR.parent
OUT_DIR = STUDY_DIR / "diff_mesh_1"
FLUX_DIR = OUT_DIR / "flux_plots"
MGXS_CSV = MODULES_DIR / "FILES" / "older_datasets" / "2jul_full.csv"
ORIG_KEFF_CSV = (
    MODULES_DIR / "RUNS" / "1000reference" / "testset_results"
    / "representative_train1000_seed1_keff_comparison.csv"
)
ALT_KEFF_CSV = (
    MODULES_DIR / "RUNS" / "1000reference" / "testset_results_alt"
    / "rep_train1000_seed3_keff_comparison.csv"
)

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(MODULES_DIR))

from NTcode_config_data.config_run import GEO_CYL as GEO  # noqa: E402
from NTcode_config_data.config_def import (  # noqa: E402
    GeometryConfig, BoundarySpec, MatProperties,
)
from solvers.NTdiffusion.diffusion_solver import (  # noqa: E402
    run_diffusion_solver, xs_layout, fn_xs_per_region, _enforce_chi, predict_xs,
    _plot_fluxes, normalize_group_fluxes,
)

PARAM_COLS = [
    "r0_b4c_rod_outer_radius",
    "r0_b4c_rod_cr_fraction",
    "r1_fuel_annulus_outer_radius",
    "r1_fuel_annulus_enrichment",
    "r1_fuel_annulus_f_mod",
    "r2_water_outer_radius",
]

# Diverse worst cases from representative seed comparisons
# (orig = testset_results seed1, alt = testset_results_alt seed3).
CASES = [
    dict(rank=1, source="orig", sample_idx=1605,
         keff_peds=0.91600507, delta_rho_pcm=3368.766481,
         signed_delta_rho_pcm=-3368.766481),
    dict(rank=2, source="orig", sample_idx=527,
         keff_peds=0.88376373, delta_rho_pcm=2829.271683,
         signed_delta_rho_pcm=2829.271683),
    dict(rank=3, source="alt", sample_idx=171,
         keff_peds=1.10897160, delta_rho_pcm=2640.075030,
         signed_delta_rho_pcm=2640.075030),
    dict(rank=4, source="orig", sample_idx=760,
         keff_peds=0.88295060, delta_rho_pcm=2876.501497,
         signed_delta_rho_pcm=-2876.501497),
    dict(rank=5, source="alt", sample_idx=663,
         keff_peds=0.84996420, delta_rho_pcm=2058.709060,
         signed_delta_rho_pcm=2058.709060),
]

REGION_PREFIXES = ("b4c_rod", "fuel_annulus", "water")


def update_geo(geo: GeometryConfig, params_raw: np.ndarray) -> GeometryConfig:
    """Same mapping as PEDS.update_geo."""
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


def _mgxs_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for prefix in REGION_PREFIXES:
        cols.extend([c for c in df.columns if c.startswith(f"{prefix}_")])
    # Keep only XS / chi (exclude material name strings if any)
    keep = []
    for c in cols:
        if any(tok in c for tok in (
            "diffusion-coefficient", "absorption", "nu-fission",
            "scatter matrix", "chi",
        )):
            keep.append(c)
    return keep


def find_row(df: pd.DataFrame, params: np.ndarray) -> pd.Series:
    mask = np.ones(len(df), dtype=bool)
    for c, v in zip(PARAM_COLS, params):
        mask &= np.isclose(df[c].astype(float).to_numpy(), float(v), rtol=1e-4, atol=1e-4)
    hits = df.loc[mask]
    if hits.empty:
        raise ValueError(f"No MGXS row matching params={params}")
    if len(hits) > 1:
        # Prefer exact keff match later; take first for now
        print(f"  [warn] {len(hits)} duplicate param matches — using first (csv index {hits.index[0]})")
    return hits.iloc[0]


def xs_tensor_from_mgxs_row(row: pd.Series, geo: GeometryConfig) -> np.ndarray:
    """
    Pack OpenMC MGXS columns into the solver's (N_regions, XS_PER_REGION) tensor.
    Column naming matches predict_xs: '{region_ID}_{quantity}_g{k}'.
    """
    G = geo.G
    n = fn_xs_per_region(G)
    lay = xs_layout(G)

    xs_dict = {}
    for mat in geo.regions:
        m = mat.region_ID
        for g in range(G):
            for qty in ("diffusion-coefficient", "absorption", "nu-fission", "chi"):
                key = f"{m}_{qty}_g{g+1}"
                if key not in row.index:
                    raise KeyError(f"Missing MGXS column: {key}")
                xs_dict[key] = float(row[key])
        for k in range(G ** 2):
            key = f"{m}_scatter matrix_g{k+1}"
            if key not in row.index:
                raise KeyError(f"Missing MGXS column: {key}")
            xs_dict[key] = float(row[key])

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


def load_epoch0_keff_peds(source: str, sample_idx: int) -> float:
    """keff_peds at epoch 0 from the matching representative comparison CSV.

    Falls back to an already-written study results CSV when the original
    representative comparison files are unavailable.
    """
    path = ORIG_KEFF_CSV if source == "orig" else ALT_KEFF_CSV
    if path.is_file():
        df = pd.read_csv(path)
        e0 = df[df["epoch"] == df["epoch"].min()]
        hit = e0[e0["sample_idx"] == int(sample_idx)]
        if len(hit) != 1:
            raise ValueError(
                f"Expected one epoch-0 row for source={source} sample_idx={sample_idx} "
                f"in {path}, found {len(hit)}"
            )
        return float(hit.iloc[0]["keff_peds"])

    fallback = OUT_DIR / "diffusion_true_xs_results.csv"
    if fallback.is_file():
        prev = pd.read_csv(fallback)
        hit = prev[
            (prev["source"] == source) & (prev["sample_idx"] == int(sample_idx))
        ]
        if len(hit) == 1 and "keff_peds_before" in hit.columns:
            print(
                f"  [warn] {path.name} missing — using keff_peds_before from {fallback.name}"
            )
            return float(hit.iloc[0]["keff_peds_before"])

    raise FileNotFoundError(
        f"Cannot load epoch-0 keff_peds: missing {path} "
        f"(and no usable fallback in {fallback})"
    )


def _cell_centres(geo: GeometryConfig) -> np.ndarray:
    """Radial cell-centre grid matching run_diffusion_solver mesh."""
    R = geo.boundaries[-1].radius
    I = int(R / geo.mesh_size)
    return np.array([(i + 0.5) * geo.mesh_size for i in range(I)])


def plot_case_fluxes(
    geo: GeometryConfig,
    phi_fwd_true: np.ndarray,
    phi_adj_true: np.ndarray,
    phi_fwd_poly: np.ndarray,
    out_path: Path,
) -> None:
    """Plot true-MGXS fluxes with poly-regression fluxes as overlay."""
    x = _cell_centres(geo)
    _plot_fluxes(
        x=x,
        geo=geo,
        phi_fwd_norm=normalize_group_fluxes(np.asarray(phi_fwd_true)),
        phi_adj_norm=normalize_group_fluxes(np.asarray(phi_adj_true)),
        an_fwd_groups=normalize_group_fluxes(np.asarray(phi_fwd_poly)),
        an_adj_groups=None,
        plot_output=str(out_path),
    )
    plt.close("all")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FLUX_DIR.mkdir(parents=True, exist_ok=True)
    print(f"MGXS source : {MGXS_CSV}")
    print(f"Output dir  : {OUT_DIR}")
    print(f"Flux plots  : {FLUX_DIR}")

    df = pd.read_csv(MGXS_CSV)
    mgxs_cols = _mgxs_columns(df)
    print(f"Loaded {len(df)} rows, {len(mgxs_cols)} MGXS columns")

    # Load npz targets for canonical params / keff
    npz = np.load(
        PROJECT_ROOT / "data" / "highfidelity" / "LHS_0.8_newbounds.npz",
        allow_pickle=True,
    )
    raw_all = np.asarray(npz["params_raw"], dtype=np.float64)
    keff_all = np.asarray(npz["keffs"], dtype=np.float64)

    case_rows = []
    result_rows = []

    for case in CASES:
        idx = int(case["sample_idx"])
        params = raw_all[idx]
        keff_openmc = float(keff_all[idx])
        keff_peds_before = load_epoch0_keff_peds(case["source"], idx)
        print(f"\n=== rank {case['rank']}  sample_idx={idx}  source={case['source']} ===")
        print(f"  params_raw = {params}")
        print(f"  keff_openmc (npz) = {keff_openmc:.8f}")
        print(f"  keff_peds_before (epoch 0) = {keff_peds_before:.8f}")

        row = find_row(df, params)
        geo = update_geo(GEO, params)
        xs_true = xs_tensor_from_mgxs_row(row, geo)

        # Optional poly baseline for comparison (same geometry).
        xs_poly = predict_xs(geo)

        print("  Running diffusion solver with TRUE OpenMC MGXS …")
        k_true, phi_fwd_true, phi_adj_true = run_diffusion_solver(xs_true, geo)
        print(f"  keff_diffusion_true_mgxs = {float(k_true):.8f}")

        print("  Running diffusion solver with poly-reg predicted XS …")
        k_poly, phi_fwd_poly, _ = run_diffusion_solver(xs_poly, geo)
        print(f"  keff_diffusion_poly_xs   = {float(k_poly):.8f}")

        flux_path = (
            FLUX_DIR
            / f"rank{case['rank']}_sample{idx}_source{case['source']}_fluxes.png"
        )
        print(f"  Plotting fluxes → {flux_path}")
        plot_case_fluxes(geo, phi_fwd_true, phi_adj_true, phi_fwd_poly, flux_path)

        # Build export row: meta + params + MGXS
        export = {
            "rank": case["rank"],
            "source": case["source"],
            "sample_idx": idx,
            "mgxs_csv": MGXS_CSV.name,
            "mgxs_csv_row": int(row.name),
            "keff_openmc": keff_openmc,
            "keff_openmc_csv": float(row["keff"]) if "keff" in row.index else np.nan,
            "keff_std_openmc": float(row["keff_std"]) if "keff_std" in row.index else np.nan,
            "keff_peds": case["keff_peds"],
            "keff_peds_before": keff_peds_before,
            "delta_rho_pcm_peds_vs_openmc": case["delta_rho_pcm"],
            "signed_delta_rho_pcm_peds_vs_openmc": case["signed_delta_rho_pcm"],
        }
        for c, v in zip(PARAM_COLS, params):
            export[c] = float(v)
        for c in mgxs_cols:
            export[c] = float(row[c])
        case_rows.append(export)

        # Reactivity diffs vs OpenMC target
        def pcm(kp, kr):
            return float((kp - kr) / (kp * kr) * 1e5)

        k_true_f = float(k_true)
        result_rows.append({
            "rank": case["rank"],
            "source": case["source"],
            "sample_idx": idx,
            "keff_openmc": keff_openmc,
            "keff_peds": case["keff_peds"],
            "keff_peds_before": keff_peds_before,
            "keff_diffusion_true_mgxs": k_true_f,
            "keff_diffusion_poly_xs": float(k_poly),
            "keff_diffusion_true_mgxs_minus_peds_before": k_true_f - keff_peds_before,
            "pcm_true_mgxs_minus_peds_before": pcm(k_true_f, keff_peds_before),
            "pcm_peds_minus_openmc": case["signed_delta_rho_pcm"],
            "pcm_true_mgxs_minus_openmc": pcm(k_true_f, keff_openmc),
            "pcm_poly_xs_minus_openmc": pcm(float(k_poly), keff_openmc),
            "pcm_true_mgxs_minus_peds": pcm(k_true_f, case["keff_peds"]),
            **{c: float(v) for c, v in zip(PARAM_COLS, params)},
        })

    cases_path = OUT_DIR / "cases_params_mgxs.csv"
    results_path = OUT_DIR / "diffusion_true_xs_results.csv"
    pd.DataFrame(case_rows).to_csv(cases_path, index=False)
    pd.DataFrame(result_rows).to_csv(results_path, index=False)

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║                     STUDY COMPLETE                       ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Wrote {cases_path}")
    print(f"  Wrote {results_path}")
    print(f"  Wrote flux plots under {FLUX_DIR}")
    print("\n  Summary (keff):")
    res = pd.DataFrame(result_rows)
    cols = ["rank", "sample_idx", "keff_openmc", "keff_diffusion_true_mgxs",
            "keff_peds_before", "keff_diffusion_true_mgxs_minus_peds_before",
            "keff_peds", "pcm_true_mgxs_minus_peds_before"]
    print(res[cols].to_string(index=False))


if __name__ == "__main__":
    main()
