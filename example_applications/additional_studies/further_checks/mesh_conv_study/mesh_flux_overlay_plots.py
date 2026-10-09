"""
run_mesh_flux_overlay_plots.py
=========================================================================
For the top-N worst cases of mesh_conv_08091523, overlay diffusion flux
profiles (PEDS-corrected XS, several mesh sizes from 1 → 0.01 cm) with the
OpenMC reference flux from ``solvers/HF_openMC/MC_solver.run_mc``.

Phases (via companion Slurm script):
  --phase diffusion   jax-env: solve diffusion at each mesh → intermediate NPZ
  --phase openmc      mc-env:  OpenMC fluxes + final PNGs
  --phase replot      regenerate PNGs from saved intermediates

Outputs under
  mesh_conv_08091523/flux_mesh_overlay/
=========================================================================
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

STUDY_DIR = Path(__file__).resolve().parent
MODULES_DIR = STUDY_DIR.parent.parent
PROJECT_ROOT = MODULES_DIR.parent

MESH_STUDY_DIR = STUDY_DIR / "mesh_conv_08091523"
SELECTED_CSV = MESH_STUDY_DIR / "selected_cases.csv"
XS_NPZ = MESH_STUDY_DIR / "xs_tensors.npz"
OUT_DIR = MESH_STUDY_DIR / "flux_mesh_overlay"
INTERMEDIATE_DIR = OUT_DIR / "intermediate"
PLOT_DIR = OUT_DIR / "flux_plots"

DEFAULT_MESHES = [1.0, 0.5, 0.2, 0.1, 0.01]
N_CASES = 5
PARAM_COLS = [
    "r0_b4c_rod_outer_radius",
    "r0_b4c_rod_cr_fraction",
    "r1_fuel_annulus_outer_radius",
    "r1_fuel_annulus_enrichment",
    "r1_fuel_annulus_f_mod",
    "r2_water_outer_radius",
]
DEFAULT_XS_XML = (
    "/global/scratch/users/caterinafrau/openmc_data/"
    "endfb-viii.0-hdf5/cross_sections.xml"
)

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "solvers"))
sys.path.insert(0, str(MODULES_DIR))


def _pcm(kp: float, kr: float) -> float:
    return float((kp - kr) / (kp * kr) * 1e5)


def select_top_cases(n: int = N_CASES) -> pd.DataFrame:
    df = pd.read_csv(SELECTED_CSV)
    worst = df[df["case_group"] == "worst"].sort_values("rank").head(n).copy()
    if len(worst) < n:
        raise RuntimeError(f"Only found {len(worst)} worst cases (wanted {n})")
    return worst.reset_index(drop=True)


def update_geo(geo_template, params, mesh_size: float):
    from NTcode_config_data.config_def import (
        GeometryConfig, BoundarySpec, MatProperties,
    )
    return GeometryConfig(
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
        mesh_size=float(mesh_size),
    )


def diffusion_cell_centres(geo) -> np.ndarray:
    R = geo.boundaries[-1].radius
    I = int(R / geo.mesh_size)
    return np.array([(i + 0.5) * geo.mesh_size for i in range(I)], dtype=np.float64)


def as_phi_GI(phi_fwd, geo) -> np.ndarray:
    """Ensure flux array shape (G, I)."""
    phi = np.asarray(phi_fwd, dtype=np.float64)
    R = geo.boundaries[-1].radius
    I = int(R / geo.mesh_size)
    G = geo.G
    if phi.ndim == 2 and phi.shape == (G, I):
        return phi
    flat = phi.ravel()
    out = np.zeros((G, I), dtype=np.float64)
    for g in range(G):
        out[g, :] = flat[g * (I + 1): g * (I + 1) + I]
    return out


def normalize_flux(phi_GI: np.ndarray) -> np.ndarray:
    m = float(np.max(phi_GI))
    if m <= 0:
        return phi_GI
    return phi_GI / m


def _even_radius_sample(r: np.ndarray, phi_GI: np.ndarray, spacing_cm: float = 1.0):
    r = np.asarray(r, dtype=float)
    phi = np.asarray(phi_GI, dtype=float)
    if len(r) < 2:
        return r, phi
    r_mark = np.arange(float(r[0]), float(r[-1]) + 0.5 * spacing_cm, spacing_cm)
    if r_mark[-1] < float(r[-1]) - 1e-9:
        r_mark = np.append(r_mark, float(r[-1]))
    r_mark = np.unique(np.clip(r_mark, float(r[0]), float(r[-1])))
    phi_mark = np.vstack([np.interp(r_mark, r, phi[g]) for g in range(phi.shape[0])])
    return r_mark, phi_mark


def build_title(params, rank, sample_idx, keffs_by_mesh, keff_omc=None) -> str:
    b4c_r, cr_frac, fuel_r, enrich, f_mod, water_r = [float(x) for x in params]
    line1 = (
        f"rank {rank}  |  sample {sample_idx}  |  "
        f"CR r={b4c_r:.2f} cm  |  CR frac={cr_frac:.2f}  |  fuel r={fuel_r:.1f} cm"
    )
    line2 = (
        f"enrich={enrich:.1f}%  |  f_mod={f_mod:.2f}  |  water r={water_r:.1f} cm"
    )
    # Show mesh=1 and OpenMC keffs if available
    bits = []
    if 1.0 in keffs_by_mesh:
        bits.append(rf"$k_{{\mathrm{{diff}}}}(1\,\mathrm{{cm}})={keffs_by_mesh[1.0]:.4f}$")
    if keff_omc is not None:
        bits.append(rf"$k_{{\mathrm{{OpenMC}}}}={keff_omc:.4f}$")
        if 1.0 in keffs_by_mesh:
            bits.append(rf"$\Delta\rho={_pcm(keffs_by_mesh[1.0], keff_omc):+.0f}$ pcm")
    lines = [line1, line2]
    if bits:
        lines.append("   |   ".join(bits))
    return "\n".join(lines)


def plot_mesh_flux_overlay(
    params: np.ndarray,
    rank: int,
    sample_idx: int,
    mesh_profiles: list[dict],
    r_omc: np.ndarray,
    phi_omc: np.ndarray,
    out_path: Path,
    keff_omc: float | None = None,
) -> None:
    """
    mesh_profiles: list of dicts with keys mesh_size, r, phi (G,I), keff
    One figure with two panels (fast / thermal). Mesh sizes as a colour
    gradient; OpenMC as markers.
    """
    # Independent peak normalisation so every curve (each mesh + OpenMC)
    # reaches 1 — absolute tally/solver units differ and must not be mixed.
    meshes = [float(p["mesh_size"]) for p in mesh_profiles]
    # Log-scaled colour: fine=dark, coarse=light
    log_m = np.log10(np.asarray(meshes))
    log_lo, log_hi = float(log_m.min()), float(log_m.max())
    if abs(log_hi - log_lo) < 1e-12:
        fracs = np.zeros(len(meshes))
    else:
        # fine (small mesh) → 1.0 (dark), coarse → 0.15 (light)
        fracs = 1.0 - (log_m - log_lo) / (log_hi - log_lo)
        fracs = 0.2 + 0.75 * fracs
    cmap_fast = cm.Blues
    cmap_therm = cm.Reds

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2), layout="constrained", sharey=True)
    group_names = ["Fast", "Thermal"]
    cmaps = [cmap_fast, cmap_therm]

    phi_omc_n = normalize_flux(phi_omc)
    r_omc_mark, phi_omc_mark = _even_radius_sample(
        np.asarray(r_omc, dtype=float), phi_omc_n, spacing_cm=1.2,
    )

    keffs = {float(p["mesh_size"]): float(p["keff"]) for p in mesh_profiles}

    for g, (ax, gname, cmap) in enumerate(zip(axes, group_names, cmaps)):
        for p, frac in zip(mesh_profiles, fracs):
            mesh = float(p["mesh_size"])
            r = np.asarray(p["r"], dtype=float)
            phi_n = normalize_flux(p["phi"])
            color = cmap(float(frac))
            ax.plot(
                r, phi_n[g], "-", lw=2.2, color=color,
                label=f"diff mesh={mesh:g} cm",
            )
        ax.plot(
            r_omc_mark, phi_omc_mark[g], "o", ms=5.5, color="k",
            markerfacecolor="none", markeredgewidth=1.6,
            label="OpenMC",
        )
        for r_int in (float(params[0]), float(params[2])):
            ax.axvline(r_int, color="gray", ls=":", lw=1.4, alpha=0.85)
        ax.set_title(gname, fontsize=16)
        ax.set_xlabel("r  (cm)", fontsize=14)
        ax.grid(True, alpha=0.35)
        ax.set_xlim([0.0, max(float(mesh_profiles[0]["r"][-1]), float(r_omc[-1]))])
        ax.set_ylim([0.0, None])
        ax.tick_params(labelsize=12)
        ax.legend(fontsize=9, framealpha=0.92, loc="best")

    axes[0].set_ylabel(r"Normalised flux  $\phi(r)$  (peak = 1)", fontsize=14)
    fig.suptitle(
        build_title(params, rank, sample_idx, keffs, keff_omc=keff_omc),
        fontsize=13,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] → {out_path}")


# ── Phase: diffusion at several meshes ───────────────────────────────────────

def phase_diffusion(cases: pd.DataFrame, meshes: list[float]) -> None:
    from NTcode_config_data.config_run import GEO_CYL as GEO
    from solvers.NTdiffusion.diffusion_solver import run_diffusion_solver

    if not XS_NPZ.is_file():
        raise FileNotFoundError(f"Missing saved XS tensors: {XS_NPZ}")
    xs_store = np.load(XS_NPZ)

    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for _, case in cases.iterrows():
        idx = int(case["sample_idx"])
        rank = int(case["rank"])
        params = np.array([float(case[c]) for c in PARAM_COLS], dtype=np.float64)
        key = f"peds_{idx}"
        if key not in xs_store.files:
            raise KeyError(f"{key} not in {XS_NPZ}")
        xs = np.asarray(xs_store[key], dtype=np.float64)

        print(f"\n=== [diffusion] rank={rank} sample_idx={idx} ===")
        profiles = []
        for mesh in meshes:
            geo = update_geo(GEO, params, mesh_size=mesh)
            t0 = time.perf_counter()
            k, phi_fwd, _ = run_diffusion_solver(xs, geo)
            wall = time.perf_counter() - t0
            phi = as_phi_GI(phi_fwd, geo)
            r = diffusion_cell_centres(geo)
            print(
                f"  mesh={mesh:<5g}  I={len(r):5d}  keff={float(k):.6f}  "
                f"t={wall:.2f}s"
            )
            profiles.append({
                "mesh_size": float(mesh),
                "r": r,
                "phi": phi,
                "keff": float(k),
                "wall_s": wall,
                "n_cells": int(len(r)),
            })

        npz_path = INTERMEDIATE_DIR / f"rank{rank}_sample{idx}_diffusion_meshes.npz"
        payload = {
            "sample_idx": idx,
            "rank": rank,
            "params": params,
            "meshes": np.asarray(meshes, dtype=np.float64),
            "keffs": np.asarray([p["keff"] for p in profiles], dtype=np.float64),
        }
        for p in profiles:
            tag = f"m{p['mesh_size']:g}".replace(".", "p")
            payload[f"r_{tag}"] = p["r"]
            payload[f"phi_{tag}"] = p["phi"]
            payload[f"keff_{tag}"] = p["keff"]
        np.savez_compressed(npz_path, **payload)
        print(f"  [saved] {npz_path}")

        row = {
            "rank": rank,
            "sample_idx": idx,
            "keff_peds_ref": float(case["keff_peds"]),
            "keff_openmc_csv": float(case["keff_openmc"]),
            "diffusion_npz": str(npz_path),
            **{c: float(v) for c, v in zip(PARAM_COLS, params)},
        }
        for p in profiles:
            row[f"keff_mesh_{p['mesh_size']:g}"] = p["keff"]
            row[f"wall_s_mesh_{p['mesh_size']:g}"] = p["wall_s"]
        summary_rows.append(row)

    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "diffusion_phase_summary.csv", index=False)
    print(f"\nWrote {OUT_DIR / 'diffusion_phase_summary.csv'}")


def _load_mesh_profiles(diff_npz: Path, meshes: list[float]) -> list[dict]:
    d = np.load(diff_npz, allow_pickle=True)
    profiles = []
    for mesh in meshes:
        tag = f"m{float(mesh):g}".replace(".", "p")
        profiles.append({
            "mesh_size": float(mesh),
            "r": np.asarray(d[f"r_{tag}"]),
            "phi": np.asarray(d[f"phi_{tag}"]),
            "keff": float(d[f"keff_{tag}"]),
        })
    return profiles


# ── Phase: OpenMC + plots ────────────────────────────────────────────────────

def phase_openmc(cases: pd.DataFrame, meshes: list[float]) -> None:
    from solvers.HF_openMC.design_api import (
        ensure_openmc_data, build_cfg_from_params,
    )
    from solvers.HF_openMC.MC_solver import run_mc

    ensure_openmc_data(os.environ.get("OPENMC_CROSS_SECTIONS", DEFAULT_XS_XML))
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    openmc_root = OUT_DIR / "openmc_runs"
    openmc_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for _, case in cases.iterrows():
        idx = int(case["sample_idx"])
        rank = int(case["rank"])
        diff_npz = INTERMEDIATE_DIR / f"rank{rank}_sample{idx}_diffusion_meshes.npz"
        if not diff_npz.is_file():
            raise FileNotFoundError(
                f"Missing {diff_npz}. Run --phase diffusion first."
            )
        d = np.load(diff_npz, allow_pickle=True)
        params = np.asarray(d["params"], dtype=np.float64)
        profiles = _load_mesh_profiles(diff_npz, meshes)

        print(f"\n=== [openmc] rank={rank} sample_idx={idx} ===")
        work_dir = str(openmc_root / f"rank{rank}_sample{idx}")
        t0 = time.time()
        cfg = build_cfg_from_params(params, work_dir=work_dir, verbose=True)
        keff, flux_data, r_centers, _lib, _cells, _conv = run_mc(cfg)
        elapsed = time.time() - t0

        flux = np.asarray(flux_data, dtype=np.float64)  # (n_bins, G)
        if flux.shape[1] != 2:
            raise ValueError(f"Expected G=2 flux, got shape {flux.shape}")
        # OpenMC EnergyFilter: col0=thermal, col1=fast → row0=fast, row1=thermal
        phi_omc = np.vstack([flux[:, 1], flux[:, 0]])
        r_omc = np.asarray(r_centers, dtype=np.float64)
        k_omc = float(keff.nominal_value)
        k_omc_std = float(keff.std_dev)
        print(f"  k_OpenMC = {k_omc:.6f} ± {k_omc_std:.6f}  ({elapsed:.1f}s)")

        omc_npz = INTERMEDIATE_DIR / f"rank{rank}_sample{idx}_openmc.npz"
        np.savez_compressed(
            omc_npz,
            r_omc=r_omc,
            phi_omc=phi_omc,
            keff_omc=k_omc,
            keff_omc_std=k_omc_std,
            elapsed_s=elapsed,
        )

        plot_path = PLOT_DIR / f"worst_rank{rank}_sample{idx}_mesh_flux_overlay.png"
        plot_mesh_flux_overlay(
            params=params,
            rank=rank,
            sample_idx=idx,
            mesh_profiles=profiles,
            r_omc=r_omc,
            phi_omc=phi_omc,
            out_path=plot_path,
            keff_omc=k_omc,
        )

        row = {
            "rank": rank,
            "sample_idx": idx,
            "keff_openmc": k_omc,
            "keff_openmc_std": k_omc_std,
            "openmc_elapsed_s": elapsed,
            "plot_path": str(plot_path),
            **{c: float(v) for c, v in zip(PARAM_COLS, params)},
        }
        for p in profiles:
            row[f"keff_mesh_{p['mesh_size']:g}"] = p["keff"]
            row[f"pcm_mesh_{p['mesh_size']:g}_vs_openmc"] = _pcm(p["keff"], k_omc)
        summary_rows.append(row)

    out_csv = OUT_DIR / "flux_overlay_summary.csv"
    pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")
    print(f"Plots in {PLOT_DIR}")


def phase_replot(cases: pd.DataFrame, meshes: list[float]) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    for _, case in cases.iterrows():
        idx = int(case["sample_idx"])
        rank = int(case["rank"])
        diff_npz = INTERMEDIATE_DIR / f"rank{rank}_sample{idx}_diffusion_meshes.npz"
        omc_npz = INTERMEDIATE_DIR / f"rank{rank}_sample{idx}_openmc.npz"
        if not diff_npz.is_file() or not omc_npz.is_file():
            raise FileNotFoundError(f"Missing intermediates for rank={rank} idx={idx}")
        d = np.load(diff_npz, allow_pickle=True)
        o = np.load(omc_npz, allow_pickle=True)
        plot_path = PLOT_DIR / f"worst_rank{rank}_sample{idx}_mesh_flux_overlay.png"
        plot_mesh_flux_overlay(
            params=np.asarray(d["params"], dtype=np.float64),
            rank=rank,
            sample_idx=idx,
            mesh_profiles=_load_mesh_profiles(diff_npz, meshes),
            r_omc=np.asarray(o["r_omc"]),
            phi_omc=np.asarray(o["phi_omc"]),
            out_path=plot_path,
            keff_omc=float(o["keff_omc"]),
        )
    print(f"\nReplotted {len(cases)} figures → {PLOT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mesh-refined diffusion flux overlays + OpenMC reference.",
    )
    parser.add_argument(
        "--phase",
        choices=["diffusion", "openmc", "replot"],
        required=True,
    )
    parser.add_argument("--n-cases", type=int, default=N_CASES)
    parser.add_argument(
        "--meshes", type=float, nargs="+", default=DEFAULT_MESHES,
        help="Diffusion mesh sizes (cm), coarse→fine",
    )
    args = parser.parse_args()
    meshes = [float(m) for m in args.meshes]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = select_top_cases(args.n_cases)
    cases.to_csv(OUT_DIR / "selected_cases.csv", index=False)
    print(f"Cases ({len(cases)}):")
    print(cases[["rank", "sample_idx", "abs_pcm", "keff_peds", "keff_openmc"]].to_string(index=False))
    print(f"Meshes (cm): {meshes}")

    if args.phase == "diffusion":
        phase_diffusion(cases, meshes)
    elif args.phase == "openmc":
        phase_openmc(cases, meshes)
    else:
        phase_replot(cases, meshes)


if __name__ == "__main__":
    main()
