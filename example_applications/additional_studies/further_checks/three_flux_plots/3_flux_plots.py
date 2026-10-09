"""
3_flux_plots.py
=========================================================================
For the geometries still listed in ``mesh_conv_study/selected_cases.csv``
(the precise_param_strat seed-1 worst test cases), produce per-case flux
plots comparing three profiles on the same axes:

  1. Diffusion + baseline XS          (poly-regression ``predict_xs``)
  2. Diffusion + PEDS-corrected XS    (same seed-1 checkpoint)
  3. OpenMC reference flux            (``solvers/HF_openMC/MC_solver.run_mc``)

Legend lists method × energy group only (no figure title).

Phases (run via the companion Slurm script, from this folder):
  --phase diffusion   jax-env: baseline + PEDS fluxes → plot_data NPZ
  --phase openmc      mc-env:  OpenMC fluxes + final PNGs
  --phase replot      PNGs from plot_data only

Kept on disk: plot_data/*.npz (the arrays the figures are drawn from),
flux_plots/*.png, and three_flux_summary.csv. OpenMC work directories
are temporary and removed after the flux is stored.
=========================================================================
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STUDY_DIR = Path(__file__).resolve().parent


def _find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "models").is_dir() and (candidate / "solvers").is_dir():
            return candidate
    raise RuntimeError(f"Could not find the PEDS_NT root above {start}")


PROJECT_ROOT = _find_project_root(STUDY_DIR)
CHECKS_DIR = STUDY_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
ANALYSIS_DIR = MODELS_DIR / "PEDS_subdivision" / "analysis"
CONFIG_RUN_DIR = PROJECT_ROOT / "config_and_run"
SELECTED_CSV = CHECKS_DIR / "mesh_conv_study" / "selected_cases.csv"
OUT_DIR = STUDY_DIR
PLOT_DATA_DIR = OUT_DIR / "plot_data"
PLOT_DIR = OUT_DIR / "flux_plots"
CASES_CSV = OUT_DIR / "selected_cases.csv"

# Same checkpoint that defined these worst cases (precise_param_strat, seed 1).
PEDS_RUN_DIR = (
    PROJECT_ROOT / "example_applications" / "additional_studies"
    / "forward_design" / "pretrained_models" / "train_1000_seed_1"
)
NPZ_PATH = PROJECT_ROOT / "data" / "highfidelity" / "17jul_0.8_1.2.npz"
TRAIN_SIZE = 1000
MODEL_SEED = 1

DEFAULT_XS_XML = (
    "/global/scratch/users/caterinafrau/openmc_data/"
    "endfb-viii.0-hdf5/cross_sections.xml"
)

N_CASES = 5
PARAM_COLS = [
    "r0_b4c_rod_outer_radius",
    "r0_b4c_rod_cr_fraction",
    "r1_fuel_annulus_outer_radius",
    "r1_fuel_annulus_enrichment",
    "r1_fuel_annulus_f_mod",
    "r2_water_outer_radius",
]

for _root in (
    PROJECT_ROOT, PROJECT_ROOT / "solvers", MODELS_DIR, CONFIG_RUN_DIR, ANALYSIS_DIR,
):
    _root_s = str(_root)
    if _root_s not in sys.path:
        sys.path.insert(0, _root_s)


def _pcm(kp: float, kr: float) -> float:
    return float((kp - kr) / (kp * kr) * 1e5)


def plot_data_path(rank: int, sample_idx: int) -> Path:
    return PLOT_DATA_DIR / f"rank{rank}_sample{sample_idx}.npz"


def select_cases(n: int = N_CASES) -> pd.DataFrame:
    """Top-ranked geometries from the mesh-convergence case list."""
    if not SELECTED_CSV.is_file():
        raise FileNotFoundError(SELECTED_CSV)
    df = pd.read_csv(SELECTED_CSV).sort_values("rank")
    picked = df.head(n).copy()
    if picked.empty:
        raise RuntimeError(f"No cases in {SELECTED_CSV}")
    if len(picked) < n:
        print(f"  [warn] only {len(picked)} cases in {SELECTED_CSV.name} (asked for {n})")
    return picked.reset_index(drop=True)


def diffusion_cell_centres(geo) -> np.ndarray:
    R = geo.boundaries[-1].radius
    I = int(R / geo.mesh_size)
    return np.array([(i + 0.5) * geo.mesh_size for i in range(I)], dtype=np.float64)


def unpack_phi(phi_fwd_padded: np.ndarray, geo) -> np.ndarray:
    """Unpack padded NT solver flux → shape (G, I)."""
    R = geo.boundaries[-1].radius
    I = int(R / geo.mesh_size)
    G = geo.G
    flat = np.asarray(phi_fwd_padded, dtype=np.float64)
    # Already (G, I)?
    if flat.ndim == 2 and flat.shape == (G, I):
        return flat
    phi = np.zeros((G, I), dtype=np.float64)
    for g in range(G):
        phi[g, :] = flat[g * (I + 1): g * (I + 1) + I]
    return phi


def normalize_flux(phi_GI: np.ndarray) -> np.ndarray:
    """Normalise by global max (same convention as prior diffusion study plots)."""
    m = float(np.max(phi_GI))
    if m <= 0:
        return phi_GI
    return phi_GI / m


def build_title(
    params: np.ndarray,
    rank: int,
    sample_idx: int,
    study_source: str,
    keff_base: float | None = None,
    keff_peds: float | None = None,
    keff_omc: float | None = None,
) -> str:
    b4c_r, cr_frac, fuel_r, enrich, f_mod, water_r = [float(x) for x in params]
    line1 = (
        f"CR r = {b4c_r:.2f} cm   |   absorption fraction = {cr_frac:.2f}   |   "
        f"fuel r = {fuel_r:.1f} cm"
    )
    line2 = (
        f"fuel enrich = {enrich:.1f}%   |   moderator fraction = {f_mod:.2f}   |   "
        f"water r = {water_r:.1f} cm"
    )
    keff_bits = []
    if keff_base is not None:
        keff_bits.append(f"$k_{{\\mathrm{{eff}}}}$ base = {keff_base:.3f}")
    if keff_peds is not None:
        keff_bits.append(f"$k_{{\\mathrm{{eff}}}}$ PEDS = {keff_peds:.3f}")
    if keff_omc is not None:
        keff_bits.append(f"$k_{{\\mathrm{{eff}}}}$ OpenMC = {keff_omc:.3f}")
    if keff_peds is not None and keff_omc is not None:
        dpcm = _pcm(float(keff_peds), float(keff_omc))
        keff_bits.append(rf"$\Delta\rho$ = {dpcm:+.0f} pcm")
    lines = [line1, line2]
    if keff_bits:
        lines.append("   |   ".join(keff_bits))
    return "\n".join(lines)


def _even_radius_sample(r: np.ndarray, phi_GI: np.ndarray,
                        spacing_cm: float = 1.5):
    """Interpolate OpenMC flux onto a uniform radial marker grid.

    Returns (r_mark, phi_mark) with phi_mark shape (G, n_mark). Using
    interpolation (instead of selecting mesh indices) keeps dots evenly
    spaced even when OpenMC's cylindrical mesh is non-uniform.
    """
    r = np.asarray(r, dtype=float)
    phi = np.asarray(phi_GI, dtype=float)
    if len(r) < 2:
        return r, phi
    r_mark = np.arange(float(r[0]), float(r[-1]) + 0.5 * spacing_cm, spacing_cm)
    if r_mark[-1] < float(r[-1]) - 1e-9:
        r_mark = np.append(r_mark, float(r[-1]))
    # Guard against duplicate last bin from arange overshoot.
    r_mark = np.unique(np.clip(r_mark, float(r[0]), float(r[-1])))
    phi_mark = np.vstack([
        np.interp(r_mark, r, phi[g]) for g in range(phi.shape[0])
    ])
    return r_mark, phi_mark


def plot_three_fluxes(
    params: np.ndarray,
    rank: int,
    sample_idx: int,
    study_source: str,
    r_base: np.ndarray,
    phi_base: np.ndarray,
    r_peds: np.ndarray,
    phi_peds: np.ndarray,
    r_omc: np.ndarray,
    phi_omc: np.ndarray,
    out_path: Path,
    keff_base: float | None = None,
    keff_peds: float | None = None,
    keff_omc: float | None = None,
) -> None:
    """
    phi_* : shape (2, n_r) with index 0 = fast, index 1 = thermal.
    Legend lists method × group only (no title).
    """
    phi_base_n = normalize_flux(phi_base)
    phi_peds_n = normalize_flux(phi_peds)
    phi_omc_n = normalize_flux(phi_omc)

    # Same hue families; OpenMC only slightly darker than PEDS.
    # OpenMC: small markers on a uniform radial grid (interpolated).
    # PEDS: dash-dot (-.-.-) so overlaps still show underlying colour.
    lw = 5.0
    peds_ls = (0, (5.0, 1.6, 1.2, 1.6))  # dash · dash · …
    r_omc_mark, phi_omc_mark = _even_radius_sample(r_omc, phi_omc_n, spacing_cm=1.0)
    styles = {
        "fast": {
            "base": dict(color="#8EC8FF", ls="-", lw=lw),
            "peds": dict(color="#1565C0", ls=peds_ls, lw=lw + 0.3),
            "omc": dict(color="#0D47A1", ls="none", marker="o",
                        ms=6.5, mew=0.0),
        },
        "thermal": {
            "base": dict(color="#FFCC80", ls="-", lw=lw),
            "peds": dict(color="#E53935", ls=peds_ls, lw=lw + 0.3),
            "omc": dict(color="#C62828", ls="none", marker="o",
                        ms=6.5, mew=0.0),
        },
    }

    # Match publication readability from parity plots on a much wider canvas.
    fs_ax = 32
    fs_tick = 30
    fs_leg = 28

    fig, ax = plt.subplots(figsize=(13.5, 8.2), layout="constrained")

    ax.plot(r_base, phi_base_n[0], label="Baseline (poly-reg) — Fast",
            **styles["fast"]["base"])
    ax.plot(r_peds, phi_peds_n[0], label="PEDS corrected — Fast",
            **styles["fast"]["peds"])
    ax.plot(r_omc_mark, phi_omc_mark[0], label="OpenMC — Fast",
            **styles["fast"]["omc"])

    ax.plot(r_base, phi_base_n[1], label="Baseline (poly-reg) — Thermal",
            **styles["thermal"]["base"])
    ax.plot(r_peds, phi_peds_n[1], label="PEDS corrected — Thermal",
            **styles["thermal"]["peds"])
    ax.plot(r_omc_mark, phi_omc_mark[1], label="OpenMC — Thermal",
            **styles["thermal"]["omc"])

    # Region interfaces (lines only — no text labels).
    for r_int in (float(params[0]), float(params[2])):
        ax.axvline(r_int, color="gray", ls=":", lw=2.4, alpha=0.9)

    ax.set_xlabel("r  (cm)", fontsize=fs_ax)
    ax.set_ylabel(r"Normalised flux  $\phi(r)$", fontsize=fs_ax)
    ax.tick_params(labelsize=fs_tick, width=1.5, length=8)
    leg = ax.legend(fontsize=fs_leg, framealpha=0.92, loc="best", ncol=1,
                    borderpad=0.6, labelspacing=0.45, handlelength=3.0)
    for line in leg.get_lines():
        line.set_linewidth(4.5)
        line.set_markersize(10.0)
    ax.grid(True, alpha=0.35)
    ax.set_xlim([0.0, max(float(r_base[-1]), float(r_omc[-1]))])
    ax.set_ylim([0.0, None])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] → {out_path}")


# ── Phase: diffusion (baseline + PEDS) ───────────────────────────────────────

def phase_diffusion(cases: pd.DataFrame) -> None:
    import jax
    import jax.numpy as jnp
    from flax import nnx

    from evaluate_test_metrics import (
        get_run_eval_context,
        load_checkpoint,
        build_model_from_metadata,
        get_or_build_norm_stats,
        _read_indices_from_split_log,
        load_dataset_arrays,
    )
    from NTcode_config_data.config_def import (
        GeometryConfig, BoundarySpec, MatProperties,
    )
    from solvers.NTdiffusion.diffusion_solver import (
        run_diffusion_solver, predict_xs,
    )

    # The seed-1 training snapshot imports plot_functions.xs_heatmap at load
    # time. That package is gone; the flux study never calls it.
    if "plot_functions.xs_heatmap" not in sys.modules:
        import types
        plot_functions = types.ModuleType("plot_functions")
        xs_heatmap = types.ModuleType("plot_functions.xs_heatmap")
        xs_heatmap.plot_xs_subplots = lambda *args, **kwargs: None
        plot_functions.xs_heatmap = xs_heatmap
        sys.modules["plot_functions"] = plot_functions
        sys.modules["plot_functions.xs_heatmap"] = xs_heatmap

    import PEDS_subdivision.logging_csv as logging_csv
    import PEDS_subdivision.plotting as plotting
    if not hasattr(logging_csv, "log_xs_history_samples"):
        logging_csv.log_xs_history_samples = lambda *args, **kwargs: None
    if not hasattr(logging_csv, "val_log_path"):
        logging_csv.val_log_path = None
    if not hasattr(plotting, "_plot_xs_heatmap"):
        plotting._plot_xs_heatmap = lambda *args, **kwargs: None

    run_dir = str(PEDS_RUN_DIR)
    print(f"PEDS run dir : {run_dir}")
    ckpt_path = os.path.join(run_dir, "checkpoints", "best_model.pkl")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(ckpt_path)

    ctx = get_run_eval_context(run_dir)
    train_size, seed = TRAIN_SIZE, MODEL_SEED
    train_idx, _val_idx, _test_idx, split_source = _read_indices_from_split_log(
        run_dir, train_size, seed,
    )
    print(f"  Split source: {split_source} ({len(train_idx)} train)")

    _geoms, _keffs, rawparams_all = load_dataset_arrays(run_dir)
    logs_root = os.path.dirname(run_dir)
    norm_stats = get_or_build_norm_stats(
        LOGS_ROOT=logs_root, train_size=train_size, seed=seed,
        rawparams=rawparams_all, train_idx=train_idx, ctx=ctx,
    )

    state, meta = load_checkpoint(ckpt_path)
    model = build_model_from_metadata(ctx, meta, seed_for_init=seed)
    nnx.update(model, jax.tree_util.tree_map(jnp.asarray, state))
    print(f"  Loaded checkpoint epoch={meta.get('epoch')}  "
          f"val_mean_pcm={meta.get('val_mean_pcm')}")

    # Bounds for geom normalisation (same as training).
    with np.load(NPZ_PATH, allow_pickle=True) as d:
        bounds_lo = np.asarray(d["bounds_lo"], dtype=np.float32)
        bounds_hi = np.asarray(d["bounds_hi"], dtype=np.float32)
        raw_all = np.asarray(d["params_raw"], dtype=np.float64)
        keff_all = np.asarray(d["keffs"], dtype=np.float64)

    span = bounds_hi - bounds_lo
    geo_template = ctx.GEO

    PLOT_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for _, case in cases.iterrows():
        idx = int(case["sample_idx"])
        rank = int(case["rank"])
        study_source = str(case["study_source"])
        if idx < 0 or idx >= len(raw_all):
            raise IndexError(f"sample_idx={idx} out of range for {NPZ_PATH}")
        csv_p = np.array([case[col] for col in PARAM_COLS], dtype=np.float64)
        if not np.allclose(csv_p, raw_all[idx], rtol=1e-4, atol=1e-4):
            raise RuntimeError(
                f"Param mismatch for sample_idx={idx}: "
                f"{SELECTED_CSV.name} vs {NPZ_PATH.name}"
            )
        params = raw_all[idx].astype(np.float32)
        keff_openmc_npz = float(keff_all[idx])

        print(f"\n=== [diffusion] rank={rank} sample_idx={idx} "
              f"source={study_source} ===")

        # Update geometry for this sample.
        geo = GeometryConfig(
            G=geo_template.G,
            regions=geo_template.regions,
            boundaries=(
                BoundarySpec(name="CR_outer", radius=float(params[0])),
                BoundarySpec(name="core_outer", radius=float(params[2])),
                BoundarySpec(name="moderator_outer", radius=float(params[5])),
            ),
            geometry=geo_template.geometry,
            mat_properties=MatProperties(
                cr_fraction=float(params[1]),
                enrichment=float(params[3]),
                f_mod=float(params[4]),
            ),
            bc=geo_template.bc,
            mesh_size=geo_template.mesh_size,
        )

        # 1) Baseline poly-reg XS → diffusion
        xs_base = predict_xs(geo)
        k_base, phi_base_raw, _ = run_diffusion_solver(xs_base, geo)
        phi_base = np.asarray(phi_base_raw, dtype=np.float64)
        if phi_base.ndim == 1:
            phi_base = unpack_phi(phi_base, geo)
        r_base = diffusion_cell_centres(geo)
        k_base_f = float(k_base)

        # 2) PEDS-corrected XS → diffusion
        geoms = ((params - bounds_lo) / span).astype(np.float32)[None, :]
        phi_features = ctx.compute_phi_features(params[None, :])
        phi_norm = ((phi_features - norm_stats["phi_mean"])
                    / norm_stats["phi_std"]).astype(np.float32)
        xs_baselines = ctx.compute_batch_baselines(params[None, :], ctx.GEO)
        xs_final = model.compute_xs(
            jnp.array(geoms),
            jnp.array(xs_baselines),
            jnp.array(phi_norm),
            jnp.array(norm_stats["log_xs_mean"]),
            jnp.array(norm_stats["log_xs_std"]),
        )
        k_peds, phi_peds_pad, _adj, _F = ctx._run_NT_solver(
            xs_final[0], params, np.array([idx], dtype=np.int32),
        )
        phi_peds = unpack_phi(phi_peds_pad, geo)
        r_peds = diffusion_cell_centres(geo)
        k_peds_f = float(k_peds)

        keff_peds_ref = float(case["keff_peds"])
        d_ref = _pcm(k_peds_f, keff_peds_ref)
        print(f"  k_base={k_base_f:.6f}  k_PEDS={k_peds_f:.6f}  "
              f"k_OpenMC(npz)={keff_openmc_npz:.6f}")
        print(f"  pcm_PEDS_vs_OpenMC = {_pcm(k_peds_f, keff_openmc_npz):+.1f}")
        print(f"  pcm(recalc PEDS − case keff_peds) = {d_ref:+.1f}")
        if abs(d_ref) > 50.0:
            raise RuntimeError(
                f"PEDS keff for sample_idx={idx} differs from "
                f"selected_cases.csv by {d_ref:+.1f} pcm"
            )

        npz_path = plot_data_path(rank, idx)
        np.savez_compressed(
            npz_path,
            sample_idx=idx,
            rank=rank,
            study_source=np.array(study_source),
            params=params.astype(np.float64),
            r_base=r_base,
            phi_base=phi_base,
            keff_base=k_base_f,
            r_peds=r_peds,
            phi_peds=phi_peds,
            keff_peds=k_peds_f,
            keff_openmc_npz=keff_openmc_npz,
        )
        print(f"  [saved] {npz_path}")

    print(f"\nDiffusion fluxes saved under {PLOT_DATA_DIR}")


# ── Phase: OpenMC + plots ────────────────────────────────────────────────────

def phase_openmc(cases: pd.DataFrame) -> None:
    from solvers.HF_openMC.design_api import (
        ensure_openmc_data, build_cfg_from_params,
    )
    from solvers.HF_openMC.MC_solver import run_mc

    ensure_openmc_data(os.environ.get("OPENMC_CROSS_SECTIONS", DEFAULT_XS_XML))
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DATA_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for _, case in cases.iterrows():
        idx = int(case["sample_idx"])
        rank = int(case["rank"])
        study_source = str(case["study_source"])
        npz_path = plot_data_path(rank, idx)
        if not npz_path.is_file():
            raise FileNotFoundError(
                f"Missing plot data {npz_path}. Run --phase diffusion first."
            )
        with np.load(npz_path, allow_pickle=True) as data:
            params = np.asarray(data["params"], dtype=np.float64)
            saved = {k: np.array(data[k]) for k in data.files}

        print(f"\n=== [openmc] rank={rank} sample_idx={idx} ===")
        work_dir = tempfile.mkdtemp(prefix=f"three_flux_rank{rank}_")
        t0 = time.time()
        try:
            cfg = build_cfg_from_params(params, work_dir=work_dir, verbose=True)
            keff, flux_data, r_centers, _lib, _cells, _conv = run_mc(cfg)
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
        elapsed = time.time() - t0

        # OpenMC EnergyFilter with bins (0, 0.625, 20e6): col0=thermal, col1=fast.
        # Remap to PEDS/diffusion convention: row0=fast, row1=thermal.
        flux = np.asarray(flux_data, dtype=np.float64)  # (n_bins, G)
        if flux.shape[1] != 2:
            raise ValueError(f"Expected G=2 flux, got shape {flux.shape}")
        phi_omc = np.vstack([flux[:, 1], flux[:, 0]])  # (2, n_bins)
        r_omc = np.asarray(r_centers, dtype=np.float64)
        k_omc = float(keff.nominal_value)
        k_omc_std = float(keff.std_dev)

        print(f"  k_OpenMC = {k_omc:.6f} ± {k_omc_std:.6f}  ({elapsed:.1f}s)")

        saved.update(
            r_omc=r_omc,
            phi_omc=phi_omc,
            keff_omc=np.float64(k_omc),
            keff_omc_std=np.float64(k_omc_std),
        )
        np.savez_compressed(npz_path, **saved)

        plot_path = PLOT_DIR / f"rank{rank}_sample{idx}_three_fluxes.png"
        plot_three_fluxes(
            params=params,
            rank=rank,
            sample_idx=idx,
            study_source=study_source,
            r_base=np.asarray(saved["r_base"]),
            phi_base=np.asarray(saved["phi_base"]),
            r_peds=np.asarray(saved["r_peds"]),
            phi_peds=np.asarray(saved["phi_peds"]),
            r_omc=r_omc,
            phi_omc=phi_omc,
            out_path=plot_path,
            keff_base=float(saved["keff_base"]),
            keff_peds=float(saved["keff_peds"]),
            keff_omc=k_omc,
        )

        summary_rows.append({
            "rank": rank,
            "sample_idx": idx,
            "study_source": study_source,
            "keff_base": float(saved["keff_base"]),
            "keff_peds": float(saved["keff_peds"]),
            "keff_openmc": k_omc,
            "keff_openmc_std": k_omc_std,
            "pcm_base_vs_openmc": _pcm(float(saved["keff_base"]), k_omc),
            "pcm_peds_vs_openmc": _pcm(float(saved["keff_peds"]), k_omc),
            **{c: float(v) for c, v in zip(PARAM_COLS, params)},
        })

    out_csv = OUT_DIR / "three_flux_summary.csv"
    pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")
    print(f"Plots in {PLOT_DIR}")


def phase_replot(cases: pd.DataFrame) -> None:
    """Regenerate PNGs from plot_data NPZs (diffusion + OpenMC arrays)."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    for _, case in cases.iterrows():
        idx = int(case["sample_idx"])
        rank = int(case["rank"])
        study_source = str(case["study_source"])
        npz_path = plot_data_path(rank, idx)
        if not npz_path.is_file():
            raise FileNotFoundError(f"Missing plot data {npz_path}")
        d = np.load(npz_path, allow_pickle=True)
        if "r_omc" not in d.files:
            raise FileNotFoundError(
                f"{npz_path.name} has no OpenMC flux. Run --phase openmc first."
            )
        plot_path = PLOT_DIR / f"rank{rank}_sample{idx}_three_fluxes.png"
        plot_three_fluxes(
            params=np.asarray(d["params"], dtype=np.float64),
            rank=rank,
            sample_idx=idx,
            study_source=study_source,
            r_base=np.asarray(d["r_base"]),
            phi_base=np.asarray(d["phi_base"]),
            r_peds=np.asarray(d["r_peds"]),
            phi_peds=np.asarray(d["phi_peds"]),
            r_omc=np.asarray(d["r_omc"]),
            phi_omc=np.asarray(d["phi_omc"]),
            out_path=plot_path,
            keff_base=float(d["keff_base"]),
            keff_peds=float(d["keff_peds"]),
            keff_omc=float(d["keff_omc"]),
        )
    print(f"\nReplotted {len(cases)} figures → {PLOT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Three-flux plots for the mesh-convergence cases.",
    )
    parser.add_argument(
        "--phase",
        choices=["diffusion", "openmc", "replot", "all"],
        required=True,
        help="diffusion=baseline+PEDS; openmc=HF+plots; replot=PNGs from NPZs",
    )
    parser.add_argument("--n-cases", type=int, default=N_CASES)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = select_cases(args.n_cases)
    cases.to_csv(CASES_CSV, index=False)
    print(f"Selected {len(cases)} cases from {SELECTED_CSV.name} → {CASES_CSV}")
    print(cases[["rank", "sample_idx", "study_source", "abs_pcm"]].to_string(index=False))

    if args.phase in ("diffusion", "all"):
        phase_diffusion(cases)
    if args.phase in ("openmc", "all"):
        phase_openmc(cases)
    if args.phase == "replot":
        phase_replot(cases)


if __name__ == "__main__":
    main()
