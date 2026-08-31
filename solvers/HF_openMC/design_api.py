"""
Thin Stage-2 facing API for the HF OpenMC solver.

Converts the 6-vector params_raw used by PEDS into an MCConfig, runs OpenMC,
and computes a mesh-cell power peaking factor (PPF) from HF flux + nu-fission.

Must be run under ``mc-env`` with OPENMC_CROSS_SECTIONS set to the ENDF/B-VIII.0
HDF5 library (see ``ensure_openmc_data()``).
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

DEFAULT_XS_XML = (
    "/global/scratch/users/caterinafrau/openmc_data/"
    "endfb-viii.0-hdf5/cross_sections.xml"
)


def ensure_openmc_data(xs_xml: Optional[str] = None) -> str:
    """Set OPENMC_CROSS_SECTIONS before importing/running OpenMC."""
    path = xs_xml or os.environ.get("OPENMC_CROSS_SECTIONS", DEFAULT_XS_XML)
    if not os.path.exists(path):
        raise FileNotFoundError(f"OpenMC cross_sections.xml not found: {path}")
    os.environ["OPENMC_CROSS_SECTIONS"] = path
    return path


def params_raw_to_sample(params_raw) -> dict:
    """
    Map PEDS params_raw [b4c_r, cr_frac, fuel_r, enrichment, f_mod, water_r]
    onto the r{i}_{name}_{field} keys expected by ``apply_sample``.
    """
    b4c_r, cr_frac, fuel_r, enrichment, f_mod, water_r = [float(x) for x in params_raw]
    return {
        "r0_b4c_rod_outer_radius": b4c_r,
        "r0_b4c_rod_cr_fraction": cr_frac,
        "r1_fuel_annulus_outer_radius": fuel_r,
        "r1_fuel_annulus_enrichment": enrichment,
        "r1_fuel_annulus_f_mod": f_mod,
        "r2_water_outer_radius": water_r,
    }


def build_cfg_from_params(params_raw, work_dir: Optional[str] = None,
                          base_cfg=None, particles: Optional[int] = None,
                          batches: Optional[int] = None,
                          inactive: Optional[int] = None,
                          verbose: bool = True):
    """
    Build an MCConfig for one design point.

    Paths for CSV/plots are redirected under ``work_dir`` so Stage-2 runs do
    not clobber training-data outputs.
    """
    from .config_run import CFG_ROD
    from .MC_solver import apply_sample, _replace

    cfg = base_cfg if base_cfg is not None else CFG_ROD
    sample = params_raw_to_sample(params_raw)
    cfg = apply_sample(cfg, sample)

    if work_dir is not None:
        os.makedirs(work_dir, exist_ok=True)
        sets = cfg.settings
        sets = _replace(
            sets,
            xs_output_path=os.path.join(work_dir, "hf_results.csv"),
            diagnostics_output_path=os.path.join(work_dir, "hf_diagnostics.csv"),
            plot_output=os.path.join(work_dir, "hf_fluxes.png"),
            convergence_output_dir=os.path.join(work_dir, "convergence"),
            openmc_work_dir=os.path.join(work_dir, "openmc_run"),
            verbose=verbose,
        )
        if particles is not None:
            sets = _replace(sets, particles=int(particles))
        if batches is not None:
            sets = _replace(sets, batches=int(batches))
        if inactive is not None:
            sets = _replace(sets, inactive=int(inactive))
        cfg = _replace(cfg, settings=sets)
    return cfg


def _region_of_r(r: float, region_radii) -> int:
    for j, R in enumerate(region_radii):
        if r <= R:
            return j
    return len(region_radii) - 1


def compute_ppf_openmc(flux_data: np.ndarray, r_centers: np.ndarray,
                       xs_dict: dict, cfg, cells) -> float:
    """
    Peak-to-average power peaking factor from OpenMC mesh flux + MGXS nu-fission.

        P_i ∝ V_i * sum_g φ_{g,i} * νΣ_{f,g,r(i)}

    ``flux_data`` from ``run_mc`` is already volume-normalised (tally/volume),
    so multiplying by cell volume recovers the volume-integrated power weight.
    Mean is taken over cells with P > 0 (fission-producing).
    """
    from .MC_solver import _spatial_volumes, _mesh_from_sp_tally  # noqa: F401 — volumes via geometry

    G = cfg.energy_groups.G
    region_radii = [r.outer_radius for r in cfg.regions]
    n_spatial = flux_data.shape[0]

    # Reconstruct approximate cylindrical ring volumes from mesh spacing.
    # r_centers are mid-bin; infer edges by midpoints between centres.
    r = np.asarray(r_centers, dtype=float)
    edges = np.zeros(n_spatial + 1)
    edges[1:-1] = 0.5 * (r[:-1] + r[1:])
    edges[0] = 0.0
    edges[-1] = region_radii[-1]
    if cfg.geometry == "cylindrical":
        V = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    elif cfg.geometry == "spherical":
        V = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    else:
        V = edges[1:] - edges[:-1]

    # nu-fission per region, shape [n_regions, G]
    nuf = np.zeros((len(cells), G), dtype=float)
    for i, cell in enumerate(cells):
        for g in range(G):
            key = f"{cell.name}_nu-fission_g{g + 1}"
            nuf[i, g] = float(xs_dict.get(key, 0.0))

    P = np.zeros(n_spatial, dtype=float)
    for i in range(n_spatial):
        reg = _region_of_r(float(r[i]), region_radii)
        P[i] = V[i] * float(np.sum(flux_data[i, :] * nuf[reg, :]))

    positive = P[P > 0.0]
    if positive.size == 0:
        return float("nan")
    return float(positive.max() / positive.mean())


def verify_design(params_raw, work_dir: str, xs_xml: Optional[str] = None,
                  particles: Optional[int] = None, batches: Optional[int] = None,
                  inactive: Optional[int] = None, verbose: bool = True,
                  compute_ppf: bool = True) -> dict:
    """
    Run OpenMC on one design and return reference keff (+ optional HF PPF).

    Returns a plain dict suitable for JSON/CSV logging.
    """
    ensure_openmc_data(xs_xml)
    from .MC_solver import run_mc, extract_mgxs_vector

    cfg = build_cfg_from_params(
        params_raw, work_dir=work_dir,
        particles=particles, batches=batches, inactive=inactive,
        verbose=verbose,
    )
    keff, flux_data, r_centers, lib, cells, conv = run_mc(cfg)
    xs_dict = extract_mgxs_vector(lib, cells)

    out = {
        "keff_hf": float(keff.nominal_value),
        "keff_hf_std": float(keff.std_dev),
        "ppf_hf": float("nan"),
        "b4c_r": float(params_raw[0]),
        "cr_frac": float(params_raw[1]),
        "fuel_r": float(params_raw[2]),
        "enrichment": float(params_raw[3]),
        "f_mod": float(params_raw[4]),
        "water_r": float(params_raw[5]),
        "work_dir": os.path.abspath(work_dir),
    }
    if compute_ppf:
        out["ppf_hf"] = compute_ppf_openmc(flux_data, r_centers, xs_dict, cfg, cells)
    if conv is not None:
        out["keff_std_converged"] = bool(conv.get("keff_std_converged_flag", False))
    return out
