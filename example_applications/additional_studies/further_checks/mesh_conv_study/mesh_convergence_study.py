"""
run_mesh_convergence_study.py
=========================================================================
Mesh-size convergence study for the diffusion eigenvalue solve.

Why not vary mesh inside the trained PEDS model?
  PEDS was trained at mesh_size=1 cm. The JAX callback pads flux vectors to
  N_FLAT_MAX sized for that mesh, so finer meshes break the padded path.
  Regional XS (poly-regression and NN-corrected) are mesh-independent, so
  the proper approach is:

    1. Select worst PEDS test geometries (same pool as run_5worstpar_study).
    2. Extract XS once at the trained mesh (mesh=1):
         - poly  : predict_xs(geo)
         - peds  : NN-corrected XS from the pretrained checkpoint
    3. Re-run the diffusion eigenvalue solve only, sweeping mesh_size,
       recording keff and wall time for each (xs_source, mesh) pair.
    4. Plot keff and solve time vs mesh size.

Outputs (under --out-dir):
  selected_cases.csv
  xs_tensors.npz          # per-case poly + peds XS arrays
  mesh_convergence.csv    # keff + timing table
  plots/keff_vs_mesh.png
  plots/time_vs_mesh.png
  plots/keff_vs_mesh_per_case.png
=========================================================================
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STUDY_DIR = Path(__file__).resolve().parent
MODULES_DIR = STUDY_DIR.parent.parent  # modules/
PROJECT_ROOT = MODULES_DIR.parent
INVERSE_DESIGN_DIR = MODULES_DIR / "inverse_design"
KEFF_SWEEP_DIR = INVERSE_DESIGN_DIR / "keff_sweep"
OLDER_CASES_DIR = INVERSE_DESIGN_DIR / "older_cases"

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault(
    "XLA_FLAGS",
    f"--xla_force_host_platform_device_count={os.cpu_count()}",
)

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(MODULES_DIR))
sys.path.insert(0, str(INVERSE_DESIGN_DIR))
sys.path.insert(0, str(OLDER_CASES_DIR))
sys.path.insert(0, str(KEFF_SWEEP_DIR))
sys.path.insert(0, str(STUDY_DIR))

import jax
import jax.numpy as jnp
from flax import nnx

from NTcode_config_data.config_run import GEO_CYL as GEO
from NTcode_config_data.config_def import GeometryConfig, BoundarySpec, MatProperties
from solvers.NTdiffusion.diffusion_solver import (
    predict_xs,
    build_xs_callables,
    bc_to_coeffs,
    GEOMETRY_CODE,
    is_homogeneous,
)
from solvers.NTdiffusion.core.MG1D_eigenvalue_nregions import DiffusionEigenvalue_MG

from evaluate_test_metrics import (
    get_run_eval_context,
    _read_indices_from_split_log,
    load_dataset_arrays,
    get_or_build_norm_stats,
    load_checkpoint,
    build_model_from_metadata,
)
from PEDS_subdivision import context as peds_context
from PEDS_subdivision.context import update_geo as peds_update_geo

DEFAULT_MESHES = [2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.01]
DEFAULT_PEDS_RUN = MODULES_DIR / "pretrained_models" / "train_1000_seed_1"
# Match the pretrained / precise_param_strat seed_1 training dataset.
DEFAULT_NPZ_PATH = PROJECT_ROOT / "data" / "highfidelity" / "17jul_0.8_1.2.npz"
DEFAULT_KEFF_CMP = (
    MODULES_DIR / "RUNS" / "precise_param_strat" / "testset_results"
    / "run_train1000_seed1_keff_comparison.csv"
)

PARAM_COLS = [
    "r0_b4c_rod_outer_radius",
    "r0_b4c_rod_cr_fraction",
    "r1_fuel_annulus_outer_radius",
    "r1_fuel_annulus_enrichment",
    "r1_fuel_annulus_f_mod",
    "r2_water_outer_radius",
]


def update_geo(geo: GeometryConfig, params_raw: np.ndarray, mesh_size: float | None = None) -> GeometryConfig:
    """Rebuild GeometryConfig from raw params; optionally override mesh_size."""
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
        mesh_size=float(geo.mesh_size if mesh_size is None else mesh_size),
    )


def pcm(kp: float, kr: float) -> float:
    if kp <= 0 or kr <= 0 or not np.isfinite(kp) or not np.isfinite(kr):
        return float("nan")
    return float((kp - kr) / (kp * kr) * 1e5)


def _signed_pcm(row) -> float:
    return float(
        (float(row["keff_peds"]) - float(row["keff_openmc"]))
        / (float(row["keff_peds"]) * float(row["keff_openmc"])) * 1e5
    )


def build_geometry_pool_from_keff_cmp(keff_cmp_csv: Path, study_source: str) -> pd.DataFrame:
    """Final-epoch test rows from a PEDS keff_comparison.csv (one model/seed)."""
    if not keff_cmp_csv.is_file():
        raise FileNotFoundError(f"Missing keff comparison CSV: {keff_cmp_csv}")
    df = pd.read_csv(keff_cmp_csv)
    if "epoch" not in df.columns:
        raise ValueError(f"{keff_cmp_csv} has no 'epoch' column")
    sub = df[df["epoch"] == df["epoch"].max()].copy()
    rows = []
    for _, row in sub.iterrows():
        signed = float(row["signed_delta_rho_pcm"]) if "signed_delta_rho_pcm" in row.index \
            else _signed_pcm(row)
        abs_pcm = float(row["delta_rho_pcm"])
        rows.append(dict(
            study_source=study_source,
            sample_idx=int(row["sample_idx"]),
            keff_openmc=float(row["keff_openmc"]),
            keff_peds=float(row["keff_peds"]),
            delta_rho_pcm=abs_pcm,
            signed_delta_rho_pcm=signed,
            abs_pcm=abs_pcm,
            **{c: float(row[c]) for c in PARAM_COLS},
        ))
    if not rows:
        raise RuntimeError(f"No final-epoch rows in {keff_cmp_csv}")
    pool = pd.DataFrame(rows).sort_values("abs_pcm", ascending=False)
    return pool.drop_duplicates("sample_idx", keep="first").reset_index(drop=True)


def select_study_cases(pool: pd.DataFrame, n_worst: int) -> list[dict]:
    worst = pool.nlargest(n_worst, "abs_pcm").copy()
    cases: list[dict] = []
    for rank, (_, row) in enumerate(worst.iterrows(), start=1):
        cases.append(dict(
            case_group="worst",
            rank=rank,
            sample_idx=int(row["sample_idx"]),
            study_source=str(row["study_source"]),
            keff_peds=float(row["keff_peds"]),
            keff_openmc=float(row["keff_openmc"]),
            delta_rho_pcm=float(row["delta_rho_pcm"]),
            signed_delta_rho_pcm=float(row["signed_delta_rho_pcm"]),
            abs_pcm=float(row["abs_pcm"]),
            **{c: float(row[c]) for c in PARAM_COLS},
        ))
    return cases


def solve_keff_forward(xs_tensor: np.ndarray, geo: GeometryConfig) -> tuple[float, float, int]:
    """Forward eigenvalue solve only (keff). Returns (keff, wall_s, n_cells)."""
    R = float(geo.boundaries[-1].radius)
    I = int(R / geo.mesh_size)
    gcode = GEOMETRY_CODE[geo.geometry]
    bc = bc_to_coeffs(geo.bc)
    r_div = [b.radius for b in geo.boundaries[:-1]] if not is_homogeneous(geo) else []
    D_fn, Sa_fn, nF_fn, Ss_fn, chi_fn = build_xs_callables(np.asarray(xs_tensor), geo)

    t0 = time.perf_counter()
    k_fwd, _phi_fwd, _x = DiffusionEigenvalue_MG(
        R, I, geo.G, r_div,
        D_fn, Sa_fn, nF_fn, Ss_fn, chi_fn,
        bc, gcode,
    )
    wall_s = time.perf_counter() - t0
    return float(k_fwd), float(wall_s), int(I)


def _phi_features_from_flux(geo_template, params_raw, phi_fwd_padded):
    geo_i = peds_update_geo(geo_template, params_raw)
    R = geo_i.boundaries[-1].radius
    I = int(R / geo_i.mesh_size)
    Delta_r = geo_i.mesh_size
    G = geo_i.G
    region_radii = [b.radius for b in geo_i.boundaries]
    feats = []
    for g in range(G):
        phi_g = np.asarray(
            phi_fwd_padded[g * (I + 1): g * (I + 1) + I], dtype=np.float64
        )
        r_prev = 0.0
        for r_reg in region_radii:
            centres = (np.arange(I) + 0.5) * Delta_r
            mask = (centres >= r_prev) & (centres < r_reg)
            feats.append(float(np.mean(phi_g[mask])) if mask.any() else 0.0)
            r_prev = r_reg
    return np.asarray(feats, dtype=np.float32)


def load_peds_model(run_dir: Path, train_size: int, seed: int):
    ckpt_path = run_dir / "checkpoints" / "best_model.pkl"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")

    ctx = get_run_eval_context(str(run_dir))
    train_idx, _val_idx, _test_idx, split_source = _read_indices_from_split_log(
        str(run_dir), train_size, seed,
    )
    print(f"  Split source: {split_source} ({len(train_idx)} train samples)")

    _geoms, _keffs, rawparams = load_dataset_arrays(str(run_dir))
    logs_root = str(run_dir.parent)
    norm_stats = get_or_build_norm_stats(
        LOGS_ROOT=logs_root, train_size=train_size, seed=seed,
        rawparams=rawparams, train_idx=train_idx, ctx=ctx,
    )

    state, meta = load_checkpoint(str(ckpt_path))
    model = build_model_from_metadata(ctx, meta, seed_for_init=seed)
    nnx.update(model, jax.tree_util.tree_map(jnp.asarray, state))
    print(
        f"  Loaded checkpoint: epoch={meta.get('epoch')}, "
        f"val_mean_pcm={meta.get('val_mean_pcm')}"
    )
    return ctx, model, norm_stats


def extract_xs_for_case(
    ctx,
    model,
    norm_stats,
    bounds_lo: np.ndarray,
    bounds_hi: np.ndarray,
    params_raw: np.ndarray,
    cache_sid: int,
):
    """Extract poly-regression and PEDS-corrected XS for one geometry."""
    params = np.asarray(params_raw, dtype=np.float32)
    geo = peds_update_geo(ctx.GEO, params)

    xs_poly = np.array(predict_xs(geo), dtype=np.float32)
    xs_poly = np.maximum(xs_poly, 1e-6)
    xs_poly = np.where(peds_context.XS_MASK, xs_poly, 0.0).astype(np.float32)

    span = (bounds_hi - bounds_lo).astype(np.float32)
    geom = ((params - bounds_lo) / span).astype(np.float32)

    # Baseline flux features at trained mesh
    _k, phi_fwd, _adj, _F = ctx._run_NT_solver(
        xs_poly, params, np.array([cache_sid], dtype=np.int32),
    )
    phi_feats = _phi_features_from_flux(ctx.GEO, params, phi_fwd)
    phi_norm = (
        (phi_feats - norm_stats["phi_mean"]) / norm_stats["phi_std"]
    ).astype(np.float32)

    xs_peds = model.compute_xs(
        jnp.array(geom[None, :]),
        jnp.array(xs_poly[None, ...]),
        jnp.array(phi_norm[None, :]),
        jnp.array(norm_stats["log_xs_mean"]),
        jnp.array(norm_stats["log_xs_std"]),
    )
    xs_peds = np.asarray(xs_peds[0], dtype=np.float32)
    return xs_poly, xs_peds, float(_k)


def plot_results(df: pd.DataFrame, out_dir: Path) -> None:
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    # Aggregate across cases: mean ± std of keff and time
    fig, ax = plt.subplots(figsize=(8, 5))
    for xs_src, style in [("peds", "-o"), ("poly", "--s")]:
        sub = df[df["xs_source"] == xs_src]
        grp = sub.groupby("mesh_size", sort=True)
        m = grp["keff"].mean()
        s = grp["keff"].std().fillna(0.0)
        ax.errorbar(
            m.index.to_numpy(), m.to_numpy(), yerr=s.to_numpy(),
            fmt=style, capsize=3, label=f"{xs_src} (mean±std over cases)",
        )
    # Coarse → fine left-to-right on a log axis
    def _mesh_xlim(ax_):
        lo, hi = min(df["mesh_size"]), max(df["mesh_size"])
        ax_.set_xlim(hi * 1.2, lo / 1.2)

    ax.set_xscale("log")
    _mesh_xlim(ax)
    ax.set_xlabel("mesh size (cm)")
    ax.set_ylabel(r"$k_{\mathrm{eff}}$")
    ax.set_title("Mesh convergence of diffusion $k_{\\mathrm{eff}}$")
    ax.grid(True, which="both", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "keff_vs_mesh.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for xs_src, style in [("peds", "-o"), ("poly", "--s")]:
        sub = df[df["xs_source"] == xs_src]
        grp = sub.groupby("mesh_size", sort=True)
        m = grp["wall_s"].mean()
        s = grp["wall_s"].std().fillna(0.0)
        ax.errorbar(
            m.index.to_numpy(), m.to_numpy(), yerr=s.to_numpy(),
            fmt=style, capsize=3, label=f"{xs_src} (mean±std)",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    _mesh_xlim(ax)
    ax.set_xlabel("mesh size (cm)")
    ax.set_ylabel("forward-solve wall time (s)")
    ax.set_title("Solve cost vs mesh size")
    ax.grid(True, which="both", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "time_vs_mesh.png", dpi=150)
    plt.close(fig)

    # Per-case keff curves (peds only — cleaner)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sub = df[df["xs_source"] == "peds"].sort_values(["sample_idx", "mesh_size"])
    for sid, g in sub.groupby("sample_idx"):
        rank = int(g["rank"].iloc[0])
        ax.plot(
            g["mesh_size"].to_numpy(), g["keff"].to_numpy(),
            "-o", ms=4, lw=1.2, label=f"rank{rank} idx{sid}",
        )
    ax.set_xscale("log")
    _mesh_xlim(ax)
    ax.set_xlabel("mesh size (cm)")
    ax.set_ylabel(r"$k_{\mathrm{eff}}$ (PEDS XS)")
    ax.set_title("Per-case mesh convergence (PEDS-corrected XS)")
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(fontsize=8, ncol=2, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(plots / "keff_vs_mesh_per_case.png", dpi=150)
    plt.close(fig)

    # Δkeff relative to finest mesh (pcm) — convergence metric
    fig, ax = plt.subplots(figsize=(8, 5))
    rows = []
    for (xs_src, sid), g in df.groupby(["xs_source", "sample_idx"]):
        g = g.sort_values("mesh_size")
        finest = g.loc[g["mesh_size"].idxmin()]
        for _, r in g.iterrows():
            rows.append({
                "xs_source": xs_src,
                "sample_idx": sid,
                "mesh_size": r["mesh_size"],
                "dpcm_vs_finest": pcm(float(r["keff"]), float(finest["keff"])),
            })
    ddf = pd.DataFrame(rows)
    for xs_src, style in [("peds", "-o"), ("poly", "--s")]:
        sub = ddf[ddf["xs_source"] == xs_src]
        grp = sub.groupby("mesh_size", sort=True)["dpcm_vs_finest"]
        m = grp.apply(lambda s: np.nanmean(np.abs(s)))
        ax.plot(m.index.to_numpy(), m.to_numpy(), style, label=f"{xs_src} mean |Δρ| vs finest")
    ax.set_xscale("log")
    ax.set_yscale("log")
    _mesh_xlim(ax)
    ax.set_xlabel("mesh size (cm)")
    ax.set_ylabel(r"mean $|\Delta\rho|$ vs finest mesh (pcm)")
    ax.set_title("Mesh discretisation error vs finest mesh")
    ax.grid(True, which="both", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "dpcm_vs_finest_mesh.png", dpi=150)
    plt.close(fig)

    print(f"  Wrote plots under {plots}")


def plot_error_evolutions(df: pd.DataFrame, out_dir: Path) -> None:
    """Signed Δρ vs OpenMC and vs converged mesh (meshes ≤ 1 cm), with means."""
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["signed_pcm_vs_openmc"] = df.apply(
        lambda r: pcm(float(r["keff"]), float(r["keff_openmc"])), axis=1,
    )

    # Full mesh set for vs-OpenMC
    meshes_all = sorted(df["mesh_size"].unique())
    xlim_all = (max(meshes_all) * 1.25, min(meshes_all) / 1.25)

    for xs_source in ["peds", "poly"]:
        sub = df[df["xs_source"] == xs_source].sort_values(["rank", "mesh_size"])
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(9.5, 5.8))
        for rank, g in sub.groupby("rank", sort=True):
            g = g.sort_values("mesh_size")
            k_ref = float(g["keff_openmc"].iloc[0])
            ax.plot(
                g["mesh_size"].to_numpy(), g["signed_pcm_vs_openmc"].to_numpy(),
                "-o", ms=4.5, lw=1.2, alpha=0.85,
                label=f"rank {int(rank)}, $k_{{\\mathrm{{HF}}}}$={k_ref:.3f}",
            )
        mean_signed = sub.groupby("mesh_size")["signed_pcm_vs_openmc"].mean().sort_index()
        mean_abs = sub.groupby("mesh_size")["signed_pcm_vs_openmc"].apply(
            lambda s: float(np.mean(np.abs(s)))
        ).sort_index()
        ax.plot(mean_signed.index, mean_signed.values, "-k", lw=2.6, marker="D",
                ms=6, label="mean (signed)", zorder=5)
        ax.plot(mean_abs.index, mean_abs.values, "--", color="0.25", lw=2.2,
                marker="s", ms=5.5, label=r"mean $|\Delta\rho|$", zorder=5)
        ax.axhline(0.0, color="gray", ls=":", lw=1.0)
        ax.set_xscale("log")
        ax.set_xlim(*xlim_all)
        ax.set_xlabel("mesh size (cm)")
        ax.set_ylabel(r"signed $\Delta\rho$ vs OpenMC (pcm)")
        ax.set_title(f"Signed reactivity error vs HF target\nXS source = {xs_source}")
        ax.grid(True, which="both", alpha=0.35)
        ax.legend(fontsize=10, ncol=2, framealpha=0.92)
        fig.tight_layout()
        fig.savefig(plots / f"signed_pcm_vs_openmc_{xs_source}.png", dpi=160)
        plt.close(fig)

    # vs converged, meshes ≤ 1
    sub1 = df[df["mesh_size"] <= 1.0 + 1e-12].copy()
    rows = []
    for (xs_src, sid), g in sub1.groupby(["xs_source", "sample_idx"]):
        g = g.sort_values("mesh_size")
        finest = g.loc[g["mesh_size"].idxmin()]
        k_conv = float(finest["keff"])
        for _, r in g.iterrows():
            dpcm = pcm(float(r["keff"]), k_conv)
            rows.append({
                "xs_source": xs_src,
                "sample_idx": int(sid),
                "rank": int(r["rank"]),
                "mesh_size": float(r["mesh_size"]),
                "signed_pcm_vs_converged": dpcm,
                "abs_pcm_vs_converged": abs(dpcm),
            })
    cdf = pd.DataFrame(rows)
    cdf.to_csv(out_dir / "pcm_vs_converged_by_mesh.csv", index=False)
    df[["rank", "sample_idx", "xs_source", "mesh_size", "keff", "keff_openmc",
        "signed_pcm_vs_openmc", "keff_peds_ref", "abs_pcm_peds_vs_openmc"]].sort_values(
        ["xs_source", "rank", "mesh_size"]
    ).to_csv(out_dir / "signed_pcm_vs_openmc_by_mesh.csv", index=False)

    meshes = sorted(cdf["mesh_size"].unique())
    xlim = (max(meshes) * 1.25, min(meshes) / 1.25)
    for xs_source in ["peds", "poly"]:
        sub = cdf[cdf["xs_source"] == xs_source]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(9.5, 5.8))
        for sid, g in sub.groupby("sample_idx"):
            g = g.sort_values("mesh_size")
            rank = int(g["rank"].iloc[0])
            ax.plot(
                g["mesh_size"].to_numpy(), g["signed_pcm_vs_converged"].to_numpy(),
                "-o", ms=4.5, lw=1.2, alpha=0.85, label=f"rank{rank} idx{sid}",
            )
        mean_signed = sub.groupby("mesh_size")["signed_pcm_vs_converged"].mean().sort_index()
        mean_abs = sub.groupby("mesh_size")["abs_pcm_vs_converged"].mean().sort_index()
        ax.plot(mean_signed.index, mean_signed.values, "-k", lw=2.6, marker="D",
                ms=6, label="mean (signed)", zorder=5)
        ax.plot(mean_abs.index, mean_abs.values, "--", color="0.25", lw=2.2,
                marker="s", ms=5.5, label=r"mean $|\Delta\rho|$", zorder=5)
        ax.axhline(0.0, color="gray", ls=":", lw=1.0)
        ax.set_xscale("log")
        ax.set_xlim(*xlim)
        ax.set_xlabel("mesh size (cm)")
        ax.set_ylabel(r"signed $\Delta\rho$ vs converged mesh (pcm)")
        ax.set_title(
            f"Mesh discretisation error vs finest mesh (0.01 cm)\n"
            f"XS source = {xs_source}  |  meshes ≤ 1 cm"
        )
        ax.grid(True, which="both", alpha=0.35)
        ax.legend(fontsize=8, ncol=2, framealpha=0.92)
        fig.tight_layout()
        fig.savefig(plots / f"pcm_vs_converged_{xs_source}.png", dpi=160)
        plt.close(fig)

    print(f"  Wrote error-evolution plots under {plots}")


def main():
    parser = argparse.ArgumentParser(description="Diffusion mesh-size convergence study.")
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output directory (default: mesh_conv_<timestamp>)",
    )
    parser.add_argument("--n-worst", type=int, default=10)
    parser.add_argument(
        "--meshes", type=float, nargs="+", default=DEFAULT_MESHES,
        help="Mesh sizes in cm (coarse → fine recommended)",
    )
    parser.add_argument(
        "--peds-run-dir", type=Path, default=DEFAULT_PEDS_RUN,
        help="Pretrained PEDS run directory with checkpoints/best_model.pkl",
    )
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--model-seed", type=int, default=1)
    parser.add_argument(
        "--xs-sources", nargs="+", default=["poly", "peds"],
        choices=["poly", "peds"],
        help="Which XS sets to mesh-sweep",
    )
    parser.add_argument(
        "--keff-cmp-csv", type=Path, default=DEFAULT_KEFF_CMP,
        help="Final-epoch test keff_comparison.csv used to pick worst cases",
    )
    parser.add_argument(
        "--npz-path", type=Path, default=DEFAULT_NPZ_PATH,
        help="HF dataset NPZ whose sample_idx matches the keff comparison CSV",
    )
    parser.add_argument(
        "--study-source-name", type=str, default="precise_param_strat_seed1",
        help="Label written into selected_cases.csv",
    )
    args = parser.parse_args()

    stamp = datetime.now().strftime("%m%d%H%M")
    out_dir = args.out_dir or (STUDY_DIR / f"mesh_conv_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    meshes = [float(m) for m in args.meshes]
    print(f"Started at   : {datetime.now().isoformat(timespec='seconds')}")
    print(f"Output dir   : {out_dir}")
    print(f"PEDS run     : {args.peds_run_dir}")
    print(f"keff_cmp_csv : {args.keff_cmp_csv}")
    print(f"npz_path    : {args.npz_path}")
    print(f"n_worst      : {args.n_worst}")
    print(f"meshes (cm)  : {meshes}")
    print(f"xs_sources   : {args.xs_sources}")

    # ── 1. Select worst cases from this model's own test metrics ─────────────
    npz = np.load(args.npz_path, allow_pickle=True)
    raw_all = np.asarray(npz["params_raw"], dtype=np.float64)
    keff_all = np.asarray(npz["keffs"], dtype=np.float64)

    pool = build_geometry_pool_from_keff_cmp(args.keff_cmp_csv, args.study_source_name)
    print(f"Final-epoch test pool: {len(pool)} geometries from {args.keff_cmp_csv.name}")
    cases = select_study_cases(pool, args.n_worst)
    selected_path = out_dir / "selected_cases.csv"
    pd.DataFrame(cases).to_csv(selected_path, index=False)
    print(f"Wrote {selected_path}  ({len(cases)} cases)")
    print("Top worst:")
    for c in cases:
        print(
            f"  rank{c['rank']:2d}  idx={c['sample_idx']:4d}  "
            f"|Δρ|={c['abs_pcm']:.1f} pcm  k_peds={c['keff_peds']:.6f}  "
            f"k_hf={c['keff_openmc']:.6f}"
        )

    # Sanity: CSV params vs NPZ[sample_idx]
    for c in cases:
        idx = int(c["sample_idx"])
        if idx < 0 or idx >= len(raw_all):
            raise IndexError(
                f"sample_idx={idx} out of range for {args.npz_path} "
                f"(n={len(raw_all)})"
            )
        csv_p = np.array([c[col] for col in PARAM_COLS], dtype=np.float64)
        npz_p = raw_all[idx]
        if not np.allclose(csv_p, npz_p, rtol=1e-4, atol=1e-4):
            raise RuntimeError(
                f"Param mismatch for sample_idx={idx}: CSV vs NPZ. "
                "Wrong --npz-path for this keff comparison CSV?"
            )

    # ── 2. Load PEDS (if needed) and dataset bounds ──────────────────────────
    need_peds = "peds" in args.xs_sources
    ctx = model = norm_stats = None
    bounds_lo = bounds_hi = None
    if need_peds:
        print("\nLoading PEDS model …")
        ctx, model, norm_stats = load_peds_model(
            args.peds_run_dir, args.train_size, args.model_seed,
        )
        from evaluate_test_metrics import _detect_data_filepath, _DEFAULT_DATA_FILEPATH
        dataset_path = _detect_data_filepath(str(args.peds_run_dir)) or _DEFAULT_DATA_FILEPATH
        with np.load(dataset_path, allow_pickle=True) as d:
            bounds_lo = np.array(d["bounds_lo"], dtype=np.float32)
            bounds_hi = np.array(d["bounds_hi"], dtype=np.float32)
        print(f"  Dataset for bounds: {dataset_path}")
        print(f"  Trained mesh_size : {ctx.GEO.mesh_size} cm")

    # ── 3. Extract XS once per case ──────────────────────────────────────────
    xs_store: dict[str, np.ndarray] = {}
    extract_rows = []
    print("\nExtracting XS tensors …")
    for i, case in enumerate(cases, start=1):
        idx = int(case["sample_idx"])
        params = raw_all[idx]
        print(
            f"  [{i}/{len(cases)}] rank={case['rank']} sample_idx={idx} "
            f"|Δρ|_PEDS={case['abs_pcm']:.1f} pcm"
        )

        geo_train = update_geo(GEO, params, mesh_size=float(GEO.mesh_size))
        xs_poly = np.array(predict_xs(geo_train), dtype=np.float32)
        xs_poly = np.maximum(xs_poly, 1e-6)
        xs_store[f"poly_{idx}"] = xs_poly

        keff_openmc = float(case.get("keff_openmc", keff_all[idx]))
        row = {
            "rank": case["rank"],
            "case_group": case["case_group"],
            "sample_idx": idx,
            "study_source": case["study_source"],
            "keff_openmc": keff_openmc,
            "keff_peds_ref": float(case["keff_peds"]),
            "abs_pcm_peds_vs_openmc": float(case["abs_pcm"]),
            **{c: float(v) for c, v in zip(PARAM_COLS, params)},
            "xs_poly_shape": str(xs_poly.shape),
        }

        if need_peds:
            xs_poly_m, xs_peds, k_base = extract_xs_for_case(
                ctx, model, norm_stats, bounds_lo, bounds_hi, params,
                cache_sid=idx,
            )
            # Prefer masked poly from the PEDS path for consistency
            xs_store[f"poly_{idx}"] = xs_poly_m
            xs_store[f"peds_{idx}"] = xs_peds
            row["keff_baseline_at_train_mesh"] = k_base
            row["xs_peds_shape"] = str(xs_peds.shape)
            # Sanity: solve once at train mesh with PEDS XS
            k_check, t_check, I_check = solve_keff_forward(xs_peds, geo_train)
            row["keff_peds_xs_at_train_mesh"] = k_check
            row["check_wall_s"] = t_check
            row["check_n_cells"] = I_check
            dpcm_match = pcm(k_check, float(case["keff_peds"]))
            row["pcm_recalc_minus_ref"] = dpcm_match
            print(
                f"    poly+PEDS XS extracted; keff(PEDS XS, mesh={GEO.mesh_size})="
                f"{k_check:.6f}  ref={case['keff_peds']:.6f}  "
                f"Δρ(recalc-ref)={dpcm_match:.1f} pcm"
            )
            if abs(dpcm_match) > 50.0:
                print(
                    "    [WARN] mesh=1 recalc differs from keff_peds_ref by "
                    f">{abs(dpcm_match):.1f} pcm — model/CSV mismatch?"
                )

        extract_rows.append(row)

    np.savez_compressed(out_dir / "xs_tensors.npz", **xs_store)
    pd.DataFrame(extract_rows).to_csv(out_dir / "xs_extract_summary.csv", index=False)
    print(f"Wrote {out_dir / 'xs_tensors.npz'}")

    # ── 4. Mesh sweep ────────────────────────────────────────────────────────
    result_rows = []
    n_jobs = len(cases) * len(args.xs_sources) * len(meshes)
    done = 0
    print(f"\nMesh sweep: {n_jobs} solves …")
    t_all0 = time.perf_counter()

    for case in cases:
        idx = int(case["sample_idx"])
        params = raw_all[idx]
        keff_openmc = float(case.get("keff_openmc", keff_all[idx]))
        for xs_src in args.xs_sources:
            key = f"{xs_src}_{idx}"
            if key not in xs_store:
                print(f"  [skip] missing XS {key}")
                continue
            xs = xs_store[key]
            for mesh in meshes:
                done += 1
                geo_m = update_geo(GEO, params, mesh_size=mesh)
                R = float(geo_m.boundaries[-1].radius)
                try:
                    k, wall_s, I = solve_keff_forward(xs, geo_m)
                    status = "ok"
                    err = ""
                except Exception as exc:
                    k, wall_s, I = float("nan"), float("nan"), int(R / mesh)
                    status = "error"
                    err = str(exc)
                    print(f"  [ERROR] idx={idx} xs={xs_src} mesh={mesh}: {exc}")

                result_rows.append({
                    "rank": case["rank"],
                    "case_group": case["case_group"],
                    "sample_idx": idx,
                    "study_source": case["study_source"],
                    "xs_source": xs_src,
                    "mesh_size": mesh,
                    "n_cells": I,
                    "R_cm": R,
                    "keff": k,
                    "wall_s": wall_s,
                    "keff_openmc": keff_openmc,
                    "keff_peds_ref": float(case["keff_peds"]),
                    "abs_pcm_peds_vs_openmc": float(case["abs_pcm"]),
                    "pcm_vs_openmc": pcm(k, keff_openmc),
                    "status": status,
                    "error": err,
                    **{c: float(v) for c, v in zip(PARAM_COLS, params)},
                })
                print(
                    f"  [{done:3d}/{n_jobs}] idx={idx} xs={xs_src:4s} "
                    f"mesh={mesh:<5g} I={I:5d}  keff={k:.6f}  t={wall_s:.3f}s",
                    flush=True,
                )

    results_df = pd.DataFrame(result_rows)
    results_path = out_dir / "mesh_convergence.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nWrote {results_path}")
    print(f"Total mesh-sweep wall: {time.perf_counter() - t_all0:.1f}s")

    # ── 5. Summary vs finest mesh ────────────────────────────────────────────
    summary_rows = []
    for (xs_src, sid), g in results_df[results_df["status"] == "ok"].groupby(
        ["xs_source", "sample_idx"]
    ):
        g = g.sort_values("mesh_size")
        finest = g.loc[g["mesh_size"].idxmin()]
        for _, r in g.iterrows():
            summary_rows.append({
                "xs_source": xs_src,
                "sample_idx": int(sid),
                "rank": int(r["rank"]),
                "mesh_size": float(r["mesh_size"]),
                "keff": float(r["keff"]),
                "keff_finest": float(finest["keff"]),
                "dpcm_vs_finest": pcm(float(r["keff"]), float(finest["keff"])),
                "wall_s": float(r["wall_s"]),
                "n_cells": int(r["n_cells"]),
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "mesh_vs_finest_summary.csv", index=False)

    # Highlight training mesh (1 cm) vs finest
    train_mesh = float(GEO.mesh_size)
    highlight = summary_df[np.isclose(summary_df["mesh_size"], train_mesh)]
    if len(highlight):
        print("\n|Δρ| at training mesh vs finest mesh:")
        for xs_src, sub in highlight.groupby("xs_source"):
            print(
                f"  {xs_src}: mean={sub['dpcm_vs_finest'].abs().mean():.2f} pcm  "
                f"max={sub['dpcm_vs_finest'].abs().max():.2f} pcm"
            )

    plot_results(results_df[results_df["status"] == "ok"].copy(), out_dir)
    plot_error_evolutions(results_df[results_df["status"] == "ok"].copy(), out_dir)

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║              MESH CONVERGENCE STUDY COMPLETE             ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Results → {out_dir}")


if __name__ == "__main__":
    main()
