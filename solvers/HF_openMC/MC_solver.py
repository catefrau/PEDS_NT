# mc_solver.py
#
# Config-driven OpenMC runner for 1-D multigroup diffusion training data.
# Supports N concentric regions (centre → outside) for spherical, cylindrical,
# and slab geometries.  Mirrors the NamedTuple style of diffusion_solver.py.
#
# Compatible with OpenMC Python API ≥ 0.14
# ──────────────────────────────────────────────────────────────────────────────
import glob
import os
import warnings
from contextlib import contextmanager
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import dataclasses
import shutil

import openmc
import openmc.mgxs as mgxs
import openmc.cmfd

try:
    from .config_def import RegionSpec, EnergyGroupSpec, SolverSettings, MCConfig
    from .config_run import CFG_ROD as CFG
except ImportError:  # running as a flat script from this directory
    from config_def import RegionSpec, EnergyGroupSpec, SolverSettings, MCConfig
    from config_run import CFG_ROD as CFG

# Suppress OpenMC ID churn when running in loops
warnings.filterwarnings('ignore', category=openmc.IDWarning)


# ──────────────────────────────────────────────────────────────────────────────
#  Convenience check
# ──────────────────────────────────────────────────────────────────────────────

def _is_homogeneous(cfg: MCConfig) -> bool:
    return len(cfg.regions) == 1


# ══════════════════════════════════════════════════════════════════════════════
#  MATERIAL BUILDERS
# ══════════════════════════════════════════════════════════════════════════════

def _build_fuel(region: RegionSpec) -> openmc.Material:
    """UO₂-like fuel: U235 + U238 mixture at 10.5 g/cm³."""
    e    = region.enrichment / 100.0
    fuel = openmc.Material(name=region.name)
    fuel.add_nuclide('U235', e,       percent_type='ao')
    fuel.add_nuclide('U238', 1.0 - e, percent_type='ao')
    fuel.set_density('g/cm3', 10.5)
    return fuel


def _build_water(region: RegionSpec) -> openmc.Material:
    """H₂O moderator at 1.0 g/cm³."""
    water = openmc.Material(name=region.name)
    water.add_nuclide('H1',  2/3, percent_type='ao')
    water.add_nuclide('O16', 1/3, percent_type='ao')
    water.set_density('g/cm3', 1.0)
    return water


def _build_b4c(region: RegionSpec) -> openmc.Material:
    """
    Boron-carbide control-rod material.
    cr_fraction scales between pure B4C (1.0) and a diluted absorber (< 1.0).
    Density scales linearly with cr_fraction from 2.52 g/cm³ (full B4C) to
    ~0 (pure void).  Natural boron isotopics are used.
    """
    f    = max(0.0, min(1.0, region.cr_fraction))
    b4c  = openmc.Material(name=region.name)
    # B4C: 4 boron atoms + 1 carbon per formula unit
    # Natural B: 19.9% B10, 80.1% B11
    b4c.add_nuclide('B10',  4 * 0.199 / 5.0 * f, percent_type='ao')
    b4c.add_nuclide('B11',  4 * 0.801 / 5.0 * f, percent_type='ao')
    b4c.add_nuclide('C12',          1.0 / 5.0 * f, percent_type='ao')
    if f < 1.0:
        # Fill remainder with light structural steel (Fe) to maintain density
        b4c.add_nuclide('Fe56', (1.0 - f), percent_type='ao')
    b4c.set_density('g/cm3', 2.52 * f + 7.87 * (1.0 - f))
    return b4c


def _build_mix(region: RegionSpec) -> openmc.Material:
    """
    Homogeneous U + water mixture for lattice-cell calculations.
    Atom fractions are computed from mass densities, preserving the physical
    number density of each species.
    """
    e     = region.enrichment / 100.0
    f_mod = region.f_mod
    NA    = 6.022e23
    rho_U, rho_W = 10.5, 1.0

    M_U    = e * 235.0 + (1.0 - e) * 238.0
    n_U235 = (1.0 - f_mod) * rho_U / M_U * NA * e
    n_U238 = (1.0 - f_mod) * rho_U / M_U * NA * (1.0 - e)
    n_H    = f_mod * rho_W / 18.015 * NA * 2.0
    n_O    = f_mod * rho_W / 18.015 * NA
    N_tot  = n_U235 + n_U238 + n_H + n_O

    mix = openmc.Material(name=region.name)
    mix.set_density('g/cm3', (1.0 - f_mod) * rho_U + f_mod * rho_W)
    mix.add_nuclide('U235', n_U235 / N_tot, percent_type='ao')
    mix.add_nuclide('U238', n_U238 / N_tot, percent_type='ao')
    mix.add_nuclide('H1',   n_H    / N_tot, percent_type='ao')
    mix.add_nuclide('O16',  n_O    / N_tot, percent_type='ao')
    return mix


def _build_void(region: RegionSpec) -> openmc.Material:
    """Near-zero-density void (OpenMC requires at least one nuclide)."""
    v = openmc.Material(name=region.name)
    v.add_nuclide('He4', 1.0, percent_type='ao')
    v.set_density('g/cm3', 1e-10)
    return v


def build_region_material(region: RegionSpec) -> openmc.Material:
    """
    Dispatcher: build an openmc.Material for a RegionSpec based on its
    material preset.
    """
    preset = region.material.lower()
    if   preset == 'fuel':   return _build_fuel(region)
    elif preset == 'water':  return _build_water(region)
    elif preset == 'b4c':    return _build_b4c(region)
    elif preset == 'mix':    return _build_mix(region)
    elif preset == 'void':   return _build_void(region)
    elif preset == 'custom':
        if region.custom_material is None:
            raise ValueError(
                f"RegionSpec '{region.name}' has material='custom' but "
                "custom_material is None.  Pass an openmc.Material instance."
            )
        # Rename to match the region so CSV column prefixes are consistent
        mat      = region.custom_material
        mat.name = region.name
        return mat
    else:
        raise ValueError(
            f"Unknown material preset '{region.material}' in region "
            f"'{region.name}'.  Choose from: "
            "'fuel', 'water', 'b4c', 'mix', 'void', 'custom'."
        )


# ══════════════════════════════════════════════════════════════════════════════
#  GEOMETRY BUILDER  — N concentric regions
# ══════════════════════════════════════════════════════════════════════════════

def build_geometry(cfg: MCConfig):
    """
    Build an openmc.Geometry for N concentric zones.

    Returns
    -------
    geometry : openmc.Geometry
    cells    : list[openmc.Cell], same order as cfg.regions (centre → outside)
    mats     : list[openmc.Material], same order
    """
    regions  = cfg.regions
    gtype    = cfg.geometry
    H        = cfg.settings.axial_half_height

    mats  = [build_region_material(r) for r in regions]
    cells = []

    if gtype == 'spherical':
        # ── N concentric spheres ────────────────────────────────────────────
        surfaces = []
        for i, r in enumerate(regions):
            bc = 'vacuum' if i == len(regions) - 1 else 'transmission'
            surfaces.append(openmc.Sphere(r=r.outer_radius, boundary_type=bc))

        for i, (r, mat) in enumerate(zip(regions, mats)):
            region_expr = -surfaces[i]
            if i > 0: # setting the border between regions
                region_expr = +surfaces[i - 1] & -surfaces[i]
            cells.append(openmc.Cell(fill=mat, region=region_expr, name=r.name))

    elif gtype == 'cylindrical':
        # ── N concentric ZCylinders + reflective axial planes ───────────────
        z_bot = openmc.ZPlane(z0=-H, boundary_type='reflective')
        z_top = openmc.ZPlane(z0=+H, boundary_type='reflective')
        axial = +z_bot & -z_top

        surfaces = []
        for i, r in enumerate(regions):
            bc = 'vacuum' if i == len(regions) - 1 else 'transmission'
            surfaces.append(openmc.ZCylinder(r=r.outer_radius, boundary_type=bc))

        for i, (r, mat) in enumerate(zip(regions, mats)):
            region_expr = -surfaces[i] & axial
            if i > 0:
                region_expr = +surfaces[i - 1] & -surfaces[i] & axial
            cells.append(openmc.Cell(fill=mat, region=region_expr, name=r.name))

    elif gtype == 'slab':
        # ── N symmetric slab layers, transport along X ──────────────────────
        # outer_radius of each RegionSpec is its half-thickness from the origin.
        # Transverse (Y, Z) planes are reflective.
        y_bot = openmc.YPlane(y0=-H, boundary_type='reflective')
        y_top = openmc.YPlane(y0=+H, boundary_type='reflective')
        z_bot = openmc.ZPlane(z0=-H, boundary_type='reflective')
        z_top = openmc.ZPlane(z0=+H, boundary_type='reflective')
        transverse = +y_bot & -y_top & +z_bot & -z_top

        # Positive-x surfaces only; negative-x by symmetry
        surfs_pos = []
        for i, r in enumerate(regions):
            bc = 'vacuum' if i == len(regions) - 1 else 'transmission'
            surfs_pos.append(openmc.XPlane(x0=+r.outer_radius, boundary_type=bc))

        # Mirror surfaces on the negative side (vacuum only on the outer one)
        surfs_neg = []
        for i, r in enumerate(regions):
            bc = 'vacuum' if i == len(regions) - 1 else 'transmission'
            surfs_neg.append(openmc.XPlane(x0=-r.outer_radius, boundary_type=bc))

        for i, (r, mat) in enumerate(zip(regions, mats)):
            pos_inner = surfs_pos[i - 1] if i > 0 else None
            neg_inner = surfs_neg[i - 1] if i > 0 else None

            # Right half of slab layer
            right = (-surfs_pos[i]  & transverse
                     if pos_inner is None
                     else +pos_inner & -surfs_pos[i] & transverse)
            # Left half of slab layer (mirror)
            left  = (+surfs_neg[i]  & transverse
                     if neg_inner is None
                     else +surfs_neg[i] & -neg_inner & transverse)

            cells.append(openmc.Cell(
                fill=mat, region=(left | right), name=r.name))

    else:
        raise ValueError(
            f"Unknown geometry type '{gtype}'. "
            "Choose 'spherical', 'cylindrical', or 'slab'."
        )

    geometry = openmc.Geometry(cells)
    return geometry, cells, mats

def build_cmfd_mesh2(cfg: MCConfig) -> openmc.cmfd.CMFDMesh:
    R = cfg.regions[-1].outer_radius
    H = cfg.settings.axial_half_height
    nx, ny, nz = cfg.settings.cmfd_mesh_dim     # <-- new, dedicated field

    Rin = R / np.sqrt(2.0)   # inscribed square, guarantees mesh stays inside model

    cmfd_mesh = openmc.cmfd.CMFDMesh()
    cmfd_mesh.lower_left  = (-Rin, -Rin, -H)
    cmfd_mesh.upper_right = ( Rin,  Rin,  H)
    cmfd_mesh.dimension   = (nx, ny, nz)

    # ── Coremap: mark every mesh cell as active (1) by default ──────────
    # CMFDMesh.map expects a flat list, length nx*ny*nz, ordered
    # x-fastest (same convention as mesh.dimension). All active is safe
    # as long as every cell in the box actually falls inside your outer
    # region (guaranteed here since Rin < R by construction).
    cmfd_mesh.map = [1] * (nx * ny * nz)

    return cmfd_mesh

def build_cmfd_mesh(cfg: MCConfig) -> openmc.cmfd.CMFDMesh:
    if cfg.geometry != 'cylindrical':
        raise ValueError("This CMFD mesh builder is for cylindrical geometry.")

    R = cfg.regions[-1].outer_radius
    H = cfg.settings.axial_half_height
    nx, ny, nz = cfg.settings.cmfd_mesh_dim

    Rin = R / np.sqrt(2.0)   # inscribed square, guarantees mesh stays inside model

    cmfd_mesh = openmc.cmfd.CMFDMesh()
    cmfd_mesh.lower_left  = (-Rin, -Rin, -H)
    cmfd_mesh.upper_right = ( Rin,  Rin,  H)
    cmfd_mesh.dimension   = (nx, ny, nz)

    x = np.linspace(-Rin, Rin, nx + 1)
    y = np.linspace(-Rin, Rin, ny + 1)

    def cell_intersects_circle(x0, x1, y0, y1, R):
        dx = 0.0 if x0 <= 0.0 <= x1 else min(abs(x0), abs(x1))
        dy = 0.0 if y0 <= 0.0 <= y1 else min(abs(y0), abs(y1))
        return (dx*dx + dy*dy) < (R*R)

    coremap = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                active = cell_intersects_circle(x[i], x[i+1], y[j], y[j+1], R)
                coremap.append(1 if active else 0)

    if not any(coremap):
        raise ValueError("CMFD map has no active cells.")

    cmfd_mesh.map = coremap
    return cmfd_mesh

# ══════════════════════════════════════════════════════════════════════════════
#  MGXS LIBRARY
# ══════════════════════════════════════════════════════════════════════════════

def build_mgxs_library(geometry: openmc.Geometry,
                        cells: list,
                        cfg: MCConfig) -> mgxs.Library:
    """
    Build a G-group MGXS library for all cell domains in the config.

    XS types tallied:
        diffusion-coefficient, absorption, nu-fission, scatter matrix, chi
    """
    eg    = cfg.energy_groups
    groups = mgxs.EnergyGroups(list(eg.boundaries))

    lib               = mgxs.Library(geometry)
    lib.energy_groups = groups
    lib.mgxs_types    = [
        'diffusion-coefficient',
        'absorption',
        'nu-fission',
        'scatter matrix',
        'chi',
    ]
    lib.by_nuclide  = False
    lib.domain_type = 'cell'
    lib.domains     = list(cells)
    lib.build_library()
    return lib


# ══════════════════════════════════════════════════════════════════════════════
#  FLUX MESH TALLY
# ══════════════════════════════════════════════════════════════════════════════

def _cylindrical_flux_grid(cfg: MCConfig) -> np.ndarray:
    """
    Non-uniform radial grid for cylindrical geometry:
      • inner half of the total radius gets 2/3 of all bins (finer near centre)
      • outer half gets 1/3 of bins
    This captures steep gradients near a central control rod or pin.
    """
    R      = cfg.regions[-1].outer_radius
    mesh_size = cfg.mesh_size
    n         = int(R / mesh_size)     # uniform bins, matches diffusion solver
    r_mid  = R / 2.0
    n_fine = int(round(n * 2 / 3))
    n_coar = n - n_fine

    fine   = np.linspace(0.0,   r_mid, n_fine + 1)
    coarse = np.linspace(r_mid, R,     n_coar + 1)
    return np.unique(np.concatenate([fine, coarse]))   # (n+1,) edges


def build_flux_mesh_tally(cfg: MCConfig):
    """
    Returns a flux tally + mesh spanning the whole domain.

    Energy bins match cfg.energy_groups.boundaries so that the flux array
    comes out with exactly G energy groups.

    For cylindrical geometry the radial grid is non-uniform (finer near centre).
    For spherical and slab the grid is uniform.
    """
    R     = cfg.regions[-1].outer_radius
    mesh_size  = cfg.mesh_size   # ✅ add this field to SolverSettings
    n_spatial  = int(R / mesh_size)       # matches diffusion: I = int(R/mesh_size)
    H     = cfg.settings.axial_half_height
    energy_bins    = list(cfg.energy_groups.boundaries)
    gtype = cfg.geometry

    if gtype == 'spherical':
        r_grid = np.linspace(0.0, R, n_spatial + 1)
        mesh   = openmc.SphericalMesh(r_grid=r_grid)

    elif gtype == 'cylindrical':
        r_grid = _cylindrical_flux_grid(cfg)
        n_spatial = len(r_grid) - 1
        mesh   = openmc.CylindricalMesh(
            r_grid   = r_grid,
            phi_grid = [0.0, 2.0 * np.pi],
            z_grid   = [-H, H],
        )

    elif gtype == 'slab':
        mesh             = openmc.RegularMesh()
        mesh.dimension   = [n_spatial, 1, 1]
        mesh.lower_left  = [0.0, -H, -H]
        mesh.upper_right = [R,    H,  H]

    else:
        raise ValueError(f"Unknown geometry type '{gtype}'.")

    mesh_filter   = openmc.MeshFilter(mesh)
    energy_filter = openmc.EnergyFilter(energy_bins)

    tally         = openmc.Tally(name='flux_spatial')
    tally.filters = [mesh_filter, energy_filter]
    tally.scores  = ['flux']

    return tally, mesh


def build_power_mesh_tally(mesh):
    """
    Build a mesh tally on the same spatial mesh for recoverable fission energy.
    """
    mesh_filter = openmc.MeshFilter(mesh)
    tally = openmc.Tally(name='power_spatial')
    tally.filters = [mesh_filter]
    tally.scores = ['kappa-fission']
    return tally

def _run_tag(cfg: MCConfig) -> str:
    return os.path.splitext(os.path.basename(cfg.settings.plot_output))[0]


def _clean_scalar_dict(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.floating, float)):
            out[k] = None if (np.isnan(v) or np.isinf(v)) else float(v)
        elif isinstance(v, (np.integer, int)):
            out[k] = int(v)
        elif isinstance(v, (np.bool_, bool)):
            out[k] = bool(v)
        else:
            out[k] = v
    return out


def build_entropy_mesh(cfg: MCConfig) -> openmc.RegularMesh:
    R = cfg.regions[-1].outer_radius
    H = cfg.settings.axial_half_height
    nx, ny, nz = cfg.settings.entropy_mesh_dim

    mesh = openmc.RegularMesh()

    if cfg.geometry in ('cylindrical', 'spherical'):
        mesh.lower_left = (-R, -R, -H)
        mesh.upper_right = (R, R, H)
    elif cfg.geometry == 'slab':
        mesh.lower_left = (-R, -H, -H)
        mesh.upper_right = (R, H, H)
    else:
        raise ValueError(f"Unknown geometry type '{cfg.geometry}'.")

    mesh.dimension = (nx, ny, nz)
    return mesh


# Convergence diagnostic thresholds (see analyze_statepoint docstring).
_PCM = 1e-5                          # 1 pcm = 1e-5 in Δk (k ≈ 1)
_ENTROPY_PLATEAU_REL_RANGE = 5e-3    # 0.5% relative entropy range in tail window
_K_WINDOW_STABILITY_SIGMA = 2.0      # window-mean shift vs batch-noise limit


def analyze_statepoint(sp: openmc.StatePoint,
                       final_keff_std: float,
                       final_keff_nom: float,
                       cfg: MCConfig,
                       window_batches: int = 20) -> tuple[dict, dict]:
    """
    Summarise k/entropy convergence from a statepoint.

    converged_flag requires (all must pass):
      • keff_std below keff_trigger_std (default 5 pcm)
      • entropy relative range in tail window < 0.5%
      • |Δk| between consecutive batch windows < 2× batch-to-batch noise
    """
    try:
        k_raw = sp.k_generation
        k = np.asarray(k_raw if k_raw is not None else [], dtype=float)
    except (KeyError, AttributeError, OSError):
        k = np.array([], dtype=float)

    try:
        H_raw = sp.entropy
        H = np.asarray(H_raw if H_raw is not None else [], dtype=float)
    except (KeyError, AttributeError, OSError):
        H = np.array([], dtype=float)

    try:
        gpb = int(sp.generations_per_batch)
    except (KeyError, AttributeError, OSError):
        gpb = 1

    n_inactive = int(sp.n_inactive)
    inactive_gens = n_inactive * gpb
    active_k = k[inactive_gens:] if k.size > inactive_gens else np.array([])

    keff_thresh = (
        cfg.settings.keff_trigger_std
        if cfg.settings.keff_trigger_std is not None
        else 5 * _PCM
    )

    summary = {
        'n_batches': int(sp.n_batches),
        'n_inactive': n_inactive,
        'generations_per_batch': gpb,
        'k_generation_count': int(k.size),
        'entropy_count': int(H.size),
        'entropy_present': bool(H.size > 0),
        'cmfd_on': bool(cfg.settings.cmfd_on),
        'keff_rel_std': float(final_keff_std / final_keff_nom) if final_keff_nom != 0 else np.nan,
        'keff_trigger_std_pcm': float(keff_thresh / _PCM),
    }

    if cfg.settings.cmfd_on:
        try:
            cmfd_dom = sp.cmfd_dominance
            if cmfd_dom is not None:
                cmfd_dom = np.asarray(cmfd_dom, dtype=float)
        except (KeyError, AttributeError, OSError):
            pass

    if active_k.size > 0:
        kwin = max(1, min(window_batches * gpb, active_k.size))
        summary.update({
            'k_active_mean': float(active_k.mean()),
            'k_active_std_sample': float(active_k.std(ddof=1)) if active_k.size > 1 else 0.0,
            'k_last_window_mean': float(active_k[-kwin:].mean()),
        })

        if active_k.size >= 2 * kwin:
            prev_mean = float(active_k[-2 * kwin:-kwin].mean())
            shift = summary['k_last_window_mean'] - prev_mean
            batch_noise = (
                summary['k_active_std_sample'] / np.sqrt(kwin)
                if summary['k_active_std_sample'] > 0 else np.nan
            )
            summary['k_prev_window_mean'] = prev_mean
            summary['k_window_shift'] = float(shift)
            summary['k_window_shift_over_batch_noise'] = (
                float(abs(shift) / batch_noise) if batch_noise > 0 else np.nan
            )
            summary['k_window_stable_flag'] = bool(
                np.isnan(summary['k_window_shift_over_batch_noise'])
                or summary['k_window_shift_over_batch_noise'] < _K_WINDOW_STABILITY_SIGMA
            )
        else:
            summary['k_prev_window_mean'] = np.nan
            summary['k_window_shift'] = np.nan
            summary['k_window_shift_over_batch_noise'] = np.nan
            summary['k_window_stable_flag'] = False
    else:
        summary['k_active_mean'] = np.nan
        summary['k_active_std_sample'] = np.nan
        summary['k_last_window_mean'] = np.nan
        summary['k_prev_window_mean'] = np.nan
        summary['k_window_shift'] = np.nan
        summary['k_window_shift_over_batch_noise'] = np.nan
        summary['k_window_stable_flag'] = False

    if H.size > 0:
        hwin = max(1, min(window_batches * gpb, H.size))
        H_last = H[-hwin:]
        summary.update({
            'entropy_last': float(H[-1]),
            'entropy_mean_last': float(H_last.mean()),
            'entropy_rel_range_last': float(
                (H_last.max() - H_last.min()) / max(abs(H_last.mean()), 1e-12)
            ),
            'entropy_slope_last': float(np.polyfit(np.arange(hwin), H_last, 1)[0]) if hwin >= 2 else 0.0,
        })
        summary['entropy_plateau_flag'] = bool(
            summary['entropy_rel_range_last'] < _ENTROPY_PLATEAU_REL_RANGE
        )
    else:
        summary['entropy_last'] = np.nan
        summary['entropy_mean_last'] = np.nan
        summary['entropy_rel_range_last'] = np.nan
        summary['entropy_slope_last'] = np.nan
        summary['entropy_plateau_flag'] = False

    summary['keff_std_converged_flag'] = bool(final_keff_std < keff_thresh)
    summary['converged_flag'] = bool(
        summary['keff_std_converged_flag']
        and summary.get('entropy_plateau_flag', False)
        and summary.get('k_window_stable_flag', False)
    )

    history = {
        'k_generation': k.tolist(),
        'entropy': H.tolist(),
    }

    return _clean_scalar_dict(summary), history


def plot_shannon_entropy(generations, entropy, out_path: str,
                         n_inactive: int | None = None,
                         title: str | None = None):
    """
    Plot Shannon entropy vs generation/batch and save to out_path.
    Marks the inactive→active transition when n_inactive is given.
    """
    gens = np.asarray(generations, dtype=float)
    H = np.asarray(entropy, dtype=float)
    if H.size == 0:
        return None

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(gens, H, color='#1f4e79', lw=1.4, label='Shannon entropy')

    if n_inactive is not None and n_inactive > 0:
        ax.axvline(n_inactive, color='#c45c26', ls='--', lw=1.2,
                   label=f'inactive → active ({n_inactive})')

    ax.set_xlabel('Generation / batch')
    ax.set_ylabel('Shannon entropy')
    ax.set_title(title or 'Shannon entropy evolution')
    ax.grid(True, alpha=0.35)
    ax.legend(loc='best', fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def save_convergence_artifacts(cfg, sp_path, summary, history, tail=200):
    out_dir = cfg.settings.convergence_output_dir
    os.makedirs(out_dir, exist_ok=True)

    tag = _run_tag(cfg)

    k_hist = history.get('k_generation', [])
    h_hist = history.get('entropy', [])

    compact_history = {
        'tail_window_size': tail,
        'k_generation_tail': k_hist[-tail:] if len(k_hist) > tail else k_hist,
        'entropy_tail': h_hist[-tail:] if len(h_hist) > tail else h_hist,
        # Full cycle histories (for Shannon-entropy / k evolution plots)
        'k_generation': k_hist,
        'entropy': h_hist,
    }

    conv_json_path = os.path.join(out_dir, f'{tag}_convergence.json')
    payload = {**summary, **compact_history}
    with open(conv_json_path, 'w') as fh:
        json.dump(payload, fh, indent=2)

    # Dedicated Shannon-entropy CSV: one row per generation/cycle
    entropy_csv_path = os.path.join(out_dir, f'{tag}_shannon_entropy.csv')
    pd.DataFrame({
        'generation': np.arange(1, len(h_hist) + 1),
        'shannon_entropy': h_hist,
        'k_generation': k_hist[:len(h_hist)] if len(k_hist) >= len(h_hist)
                        else k_hist + [np.nan] * (len(h_hist) - len(k_hist)),
    }).to_csv(entropy_csv_path, index=False)

    entropy_plot_path = os.path.join(out_dir, f'{tag}_shannon_entropy.png')
    plot_shannon_entropy(
        generations=np.arange(1, len(h_hist) + 1),
        entropy=h_hist,
        out_path=entropy_plot_path,
        n_inactive=summary.get('n_inactive', cfg.settings.inactive),
        title=f'Shannon entropy — {tag}',
    )

    saved_statepoint = sp_path
    if cfg.settings.copy_statepoint:
        saved_statepoint = os.path.join(out_dir, f'{tag}_statepoint.h5')
        if os.path.abspath(sp_path) != os.path.abspath(saved_statepoint):
            shutil.copy2(sp_path, saved_statepoint)

    summary = dict(summary)
    summary['convergence_json_path'] = conv_json_path
    summary['shannon_entropy_csv_path'] = entropy_csv_path
    summary['shannon_entropy_plot_path'] = entropy_plot_path
    summary['statepoint_path'] = saved_statepoint
    return _clean_scalar_dict(summary)

# ══════════════════════════════════════════════════════════════════════════════
#  MGXS EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_mgxs_vector(lib: mgxs.Library, cells: list) -> dict:
    """
    Flatten all MGXS values into a dict keyed
    '{region_name}_{xs_type}_g{i}' (1-indexed).

    The scatter matrix is stored as 'scatter matrix_g{k}' where k runs over
    all G² elements in row-major order (from_group × to_group).
    """
    results = {}
    for cell in cells:
        for xs_type in lib.mgxs_types:
            xs     = lib.get_mgxs(cell, xs_type)
            values = xs.get_xs(value='mean').flatten()
            for i, v in enumerate(values):
                results[f"{cell.name}_{xs_type}_g{i + 1}"] = float(v)
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG-TO-COLUMN MAPPING 
# ══════════════════════════════════════════════════════════════════════════════

def get_knob_columns(cfg: MCConfig) -> list:
    """
    Return the CSV column names that are design knobs for this config —
    i.e. the parameters that *could* be swept to generate training data.

    These names mirror exactly what _build_spec_row writes, so they serve
    as the canonical 'input_cols' list for regression_core.py.

    Rules (same logic as _build_spec_row):
      • outer_radius   → always a potential knob for every region
      • enrichment     → only for material in ('fuel', 'mix')
      • f_mod          → only for material == 'mix'
      • cr_fraction    → only for material == 'b4c'
    """
    knobs = []
    for i, r in enumerate(cfg.regions):
        pfx = f"r{i}_{r.name}"
        knobs.append(f"{pfx}_outer_radius")
        if r.material in ('fuel', 'mix'):
            knobs.append(f"{pfx}_enrichment")
        if r.material == 'mix':
            knobs.append(f"{pfx}_f_mod")
        if r.material == 'b4c':
            knobs.append(f"{pfx}_cr_fraction")
    return knobs


def get_fixed_columns(cfg: MCConfig) -> list:
    """
    Return the CSV column names that are categorical / fixed descriptors
    (geometry label, group count, material preset per region).
    These are never used as regression inputs or outputs.
    """
    fixed = ['geometry', 'G']
    for i, r in enumerate(cfg.regions):
        fixed.append(f"r{i}_{r.name}_material")
    return fixed


def _replace(obj, **kwargs):
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.replace(obj, **kwargs)   # dataclass
    elif hasattr(obj, '_replace'):
        return obj._replace(**kwargs)               # NamedTuple
    else:
        raise TypeError(...)


def apply_sample(base_cfg: MCConfig, sample: dict) -> MCConfig:
    """
    Return a perturbed MCConfig by injecting a flat sample dict into the
    correct RegionSpec fields.  Works for *any* MCConfig — no hardcoded
    region names or counts.

    Key naming convention (mirrors get_knob_columns / _build_spec_row):
        r{i}_{region.name}_outer_radius
        r{i}_{region.name}_enrichment    (only if material in 'fuel','mix')
        r{i}_{region.name}_f_mod         (only if material == 'mix')
        r{i}_{region.name}_cr_fraction   (only if material == 'b4c')

    Any key in `sample` that does not match a known knob is silently ignored,
    so you can safely pass a superset dict.

    Parameters
    ----------
    base_cfg : MCConfig
        The nominal configuration to perturb.
    sample : dict
        Flat dict of {column_name: float} as produced by the LHS sampler
        (keys must follow the naming convention above).

    Returns
    -------
    MCConfig  — a new config with the sampled values applied.
                All other fields are unchanged from base_cfg.
    """
    new_regions = []
    for i, r in enumerate(base_cfg.regions):
        pfx = f"r{i}_{r.name}"

        # Collect only the knobs that are valid for this region's material
        kwargs = {}
        kwargs['outer_radius'] = sample.get(f"{pfx}_outer_radius",
                                             r.outer_radius)
        if r.material in ('fuel', 'mix'):
            kwargs['enrichment'] = sample.get(f"{pfx}_enrichment",
                                              r.enrichment)
        if r.material == 'mix':
            kwargs['f_mod']      = sample.get(f"{pfx}_f_mod", r.f_mod)
        if r.material == 'b4c':
            kwargs['cr_fraction'] = sample.get(f"{pfx}_cr_fraction",
                                               r.cr_fraction)

        new_regions.append(_replace(r, **kwargs))   

    return _replace(base_cfg, regions=tuple(new_regions))

# ══════════════════════════════════════════════════════════════════════════════
#  CSV SAVE
# ══════════════════════════════════════════════════════════════════════════════

def _build_spec_row(cfg: MCConfig) -> dict:
    """
    Build the config-descriptor part of the CSV row.
    One column per config scalar + per-region knobs.
    """
    row = {
        'geometry': cfg.geometry,
        'G':        cfg.energy_groups.G,
    }
    for i, r in enumerate(cfg.regions):
        pfx = f"r{i}_{r.name}"
        row[f"{pfx}_outer_radius"] = r.outer_radius
        row[f"{pfx}_material"]     = r.material
        # Include the knob that is active for this preset
        if r.material in ('fuel', 'mix'):
            row[f"{pfx}_enrichment"] = r.enrichment
        if r.material == 'mix':
            row[f"{pfx}_f_mod"] = r.f_mod
        if r.material == 'b4c':
            row[f"{pfx}_cr_fraction"] = r.cr_fraction
    return row

def _meta_path(csv_path: str) -> str:
    """Derive the JSON path from the CSV path."""
    base, _ = os.path.splitext(csv_path)
    return base + '_meta.json'


def save_meta(cfg: MCConfig, xs_dict: dict,  extra_exclude: list[str] | None = None):
    """
    Write a JSON sidecar file that describes the column structure of the CSV.

    This file is the single source of truth consumed by regression_core.py
    to automatically identify input / chi / output columns without any
    hardcoding.  It is (re-)written whenever the CSV header is created, so
    it always matches the CSV produced by the current config.

    Schema
    ------
    {
      "geometry"    : str,           # e.g. "spherical"
      "G"           : int,           # number of energy groups
      "input_cols"  : [str, ...],    # design knobs (potential swept params)
      "fixed_cols"  : [str, ...],    # categorical descriptors, never regressed
      "chi_cols"    : [str, ...],    # fixed-physics XS columns, excluded from Y
      "exclude_cols": [str, ...],    # keff, keff_std — neither input nor XS output
      "xs_cols"     : [str, ...]     # all XS columns written by extract_mgxs_vector
    }

    regression_core.py uses this to build:
      input_cols  → X matrix (filtered to those with non-zero variance)
      output_cols → Y matrix  = xs_cols  − chi_cols
    """
    all_xs   = list(xs_dict.keys())
    chi_cols = [c for c in all_xs if '_chi_' in c]

    exclude_cols = ['keff', 'keff_std']
    if extra_exclude:
        exclude_cols.extend(extra_exclude)

    meta = {
        'geometry': cfg.geometry,
        'G': cfg.energy_groups.G,
        'input_cols': get_knob_columns(cfg),
        'fixed_cols': get_fixed_columns(cfg),
        'chi_cols': chi_cols,
        'exclude_cols': exclude_cols,
        'xs_cols': all_xs,
    }

    path = _meta_path(cfg.settings.xs_output_path)
    with open(path, 'w') as fh:
        json.dump(meta, fh, indent=2)

    if cfg.settings.verbose:
        print(f"  [meta saved]  → {path}")
        print(f"    input_cols  : {meta['input_cols']}")
        print(f"    chi_cols    : {meta['chi_cols']}")
        print(f"    xs_cols     : {len(meta['xs_cols'])} columns")


DIAG_COLS = [
    'keff_rel_std', 'keff_std_converged_flag',
    'k_window_shift_over_batch_noise', 'k_window_stable_flag',
    'entropy_rel_range_last', 'entropy_slope_last',
    'entropy_plateau_flag', 'converged_flag',
    'cmfd_on', 'convergence_json_path',
    'openmc_runtime_total_s', 'openmc_runtime_transport_s',
    'openmc_runtime_inactive_s', 'openmc_runtime_active_s',
    'radial_power_peaking_factor', 'fuel_avg_power_density',
    'fuel_max_power_density', 'flux_peak_factor_full',
    'fuel_bin_count', 'study_metrics_path', 'power_tally_sidecar_path',
]

def save_results(cfg: MCConfig, keff, xs_dict: dict, conv_row: dict | None = None):
    spec_row = _build_spec_row(cfg)          # 6 sweep params
    keff_row = {'keff': keff.nominal_value, 'keff_std': keff.std_dev}
    conv_row = conv_row or {}

    # ── MGXS file: sweep params + keff + MGXS only ──────────────────────
    mgxs_row = {**spec_row, **keff_row, **xs_dict}
    mgxs_path = cfg.settings.xs_output_path
    write_header = not os.path.exists(mgxs_path)
    os.makedirs(os.path.dirname(mgxs_path) or '.', exist_ok=True)
    pd.DataFrame([mgxs_row]).to_csv(mgxs_path, mode='a', index=False, header=write_header)

    # ── Diagnostics file: sweep params + keff + trimmed diagnostics ─────
    diag_selected = {k: conv_row.get(k) for k in DIAG_COLS}
    diag_row = {**spec_row, **keff_row, **diag_selected}
    diag_path = cfg.settings.diagnostics_output_path
    write_header_diag = not os.path.exists(diag_path)
    os.makedirs(os.path.dirname(diag_path) or '.', exist_ok=True)
    pd.DataFrame([diag_row]).to_csv(diag_path, mode='a', index=False, header=write_header_diag)

    if cfg.settings.verbose:
        label = 'created' if write_header else 'appended'
        print(f"  [MGXS CSV {label}] → {mgxs_path}")
        label_d = 'created' if write_header_diag else 'appended'
        print(f"  [Diagnostics CSV {label_d}] → {diag_path}")

    if write_header:
        save_meta(cfg, xs_dict, extra_exclude=list(diag_selected.keys()))

def persist_mc_results(cfg: MCConfig, keff, lib, cells, conv_summary: dict | None = None):
    """Append one run's CSV rows after a completed (possibly retried) OpenMC solve."""
    xs_dict = extract_mgxs_vector(lib, cells)
    save_results(cfg, keff, xs_dict, conv_summary)

# ══════════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

# Pairs of colours: (light shade for group g, darker shade for its marker)
_GROUP_PALETTE = [
    ("#E07B39", "#994d17"),   # orange — group 1 (fast)
    ("#2E86AB", "#1a5070"),   # blue   — group 2 (thermal)
    ("#3BB273", "#1f6b3e"),   # green  — group 3
    ("#9B59B6", "#5e2d7d"),   # purple — group 4
]

_GEOM_LABEL = {
    'spherical':   'Sphere',
    'cylindrical': 'Cylinder',
    'slab':        'Slab',
}


def _group_labels(G: int) -> list:
    if G == 1:
        return ["Group 1"]
    if G == 2:
        return ["Thermal (g1)", "Fast (g2)"]
    return [f"Group {g+1}" for g in range(G)]


def plot_fluxes(r_centers: np.ndarray, flux_data: np.ndarray, cfg: MCConfig):
    """
    Plot normalised group flux profiles and save to cfg.settings.plot_output.

    flux_data : shape (n_bins, G), column g = flux in group g+1
                (as returned by run_mc)
    """
    G      = cfg.energy_groups.G
    gtype  = cfg.geometry
    path   = cfg.settings.plot_output
    homo   = _is_homogeneous(cfg)

    # Normalise to peak total flux
    total = flux_data.sum(axis=1)
    norm  = total.max() if total.max() > 0 else 1.0
    """ fast_group   = flux_data[:, -1]          # fast = last column (high energy)
    norm         = fast_group[0] if fast_group[0] > 0 else 1.0   # first spatial bin of fast """
    phi_n = flux_data / norm

    labels  = _group_labels(G)
    geom_str = _GEOM_LABEL.get(gtype, gtype)
    homo_str = "homogeneous" if homo else "heterogeneous"

    fig, ax = plt.subplots(figsize=(9, 5))

    for g in range(G):
        c = _GROUP_PALETTE[g % len(_GROUP_PALETTE)]
        ax.plot(r_centers, phi_n[:, g], '-', color=c[0], lw=2.5,
                label=f"OpenMC — {labels[g]}")

    # One vertical dashed line per region interface (all except the outer edge)
    for r in cfg.regions[:-1]:
        ax.axvline(r.outer_radius, color='gray', ls=':', lw=1.3,
                   label=f"'{r.name}' outer  @ {r.outer_radius:.1f} cm")

    # Title carries all key problem info
    enrich_tags = []
    for r in cfg.regions:
        if r.material in ('fuel', 'mix'):
            enrich_tags.append(f"{r.name}:{r.enrichment:.1f}%")
    enrich_str = '  enrich ' + ', '.join(enrich_tags) if enrich_tags else ''

    title = (f"OpenMC {G}G Flux  |  {homo_str}  |  {geom_str}"
             f"  |  {len(cfg.regions)} regions{enrich_str}")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("r  (cm)", fontsize=12)
    ax.set_ylabel("Normalised Flux  φ(r) / φ_max", fontsize=12)
    ax.legend(fontsize=8, framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.35)
    ax.set_xlim([r_centers[0], r_centers[-1]])
    ax.set_ylim([0, None])

    plt.tight_layout()
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    plt.savefig(path, dpi=150)
    if cfg.settings.verbose:
        print(f"  [flux plot saved] → {path}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  PRINT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def print_config(cfg: MCConfig):
    """Print a human-readable summary of the MCConfig."""
    if not cfg.settings.verbose:
        return

    R    = cfg.regions[-1].outer_radius
    homo = _is_homogeneous(cfg)
    eg   = cfg.energy_groups
    sets = cfg.settings

    print("╔══════════════════════════════════════════════════════════╗")
    print("║           OPENMC SOLVER — PROBLEM SETUP                  ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Geometry    : {_GEOM_LABEL.get(cfg.geometry, cfg.geometry)}")
    print(f"  Mode        : {'homogeneous (single zone)' if homo else 'heterogeneous'}")
    print(f"  Energy grps : {eg.G}  |  boundaries: {list(eg.boundaries)} eV")
    print(f"  Outer radius: {R:.2f} cm")
    print(f"  MC settings : {sets.particles} particles × {sets.batches} batches"
          f"  ({sets.inactive} inactive)")
    n_spatial = int(R / cfg.mesh_size)
    print(f"  Flux bins   : ~{n_spatial}  (mesh_size={cfg.mesh_size} cm)"
          f"  {'(non-uniform near centre)' if cfg.geometry == 'cylindrical' else ''}")

    print(f"\n  ── Regions (centre → outside) ──")
    for i, r in enumerate(cfg.regions):
        knobs = f"enrich={r.enrichment:.2f}%"
        if r.material == 'mix':
            knobs += f"  f_mod={r.f_mod:.3f}"
        elif r.material == 'b4c':
            knobs = f"cr_frac={r.cr_fraction:.3f}"
        elif r.material == 'custom':
            knobs = "custom openmc.Material"
        print(f"    [{i}] '{r.name}'  preset='{r.material}'  "
              f"r_out={r.outer_radius:.2f} cm  |  {knobs}")

    print()


def print_results(keff, flux_data: np.ndarray, r_centers: np.ndarray,
                  cfg: MCConfig):
    """Print a structured post-run summary."""
    if not cfg.settings.verbose:
        return

    G = cfg.energy_groups.G
    labels = _group_labels(G)

    print("╔══════════════════════════════════════════════════════════╗")
    print("║                      RESULTS                             ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  k_eff   = {keff.nominal_value:.6f} ± {keff.std_dev:.6f}")

    total = flux_data.sum(axis=1)
    print(f"  Max total flux = {total.max():.4e}  (at r = {r_centers[total.argmax()]:.2f} cm)")

    for r in cfg.regions[:-1]:
        idx = np.argmin(np.abs(r_centers - r.outer_radius))
        print(f"\n  ── Interface '{r.name}' outer @ r ≈ {r_centers[idx]:.3f} cm ──")
        for g in range(G):
            print(f"    φ_g{g+1} ({labels[g]}) = {flux_data[idx, g]:.4e}")
        if G == 2 and flux_data[idx, 1] > 0:
            print(f"    Fast/Thermal ratio = {flux_data[idx, 1]/ flux_data[idx, 0] :.4f}")

    print()


# ══════════════════════════════════════════════════════════════════════════════
#  OPENMC WORKING DIRECTORY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _statepoint_batch_list(max_batches: int, mid_fraction: float = 0.75) -> list[int]:
    """
    Batch numbers at which OpenMC writes statepoint HDF5 files.

    By default saves two checkpoints per run: one at 75% of the scheduled
    maximum batches and one at the final batch (e.g. 600 and 800).
    """
    mid_batch = max(1, int(round(mid_fraction * max_batches)))
    return sorted({mid_batch, max_batches})


def _resolve_openmc_work_dir(cfg: MCConfig) -> str:
    """Return the absolute directory where OpenMC XML and statepoint files are written."""
    if cfg.settings.openmc_work_dir:
        return os.path.abspath(cfg.settings.openmc_work_dir)
    plot_dir = os.path.dirname(cfg.settings.plot_output) or '.'
    return os.path.abspath(os.path.join(plot_dir, 'openmc_runs', _run_tag(cfg)))


_OPENMC_ARTIFACT_PATTERNS = (
    'model.xml',
    'geometry.xml',
    'materials.xml',
    'settings.xml',
    'tallies.xml',
    'summary.h5',
    'statepoint.*.h5',
)


@contextmanager
def _openmc_cwd(run_dir: str):
    run_dir = os.path.abspath(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    prev = os.getcwd()
    try:
        os.chdir(run_dir)
        yield run_dir
    finally:
        os.chdir(prev)


def _cleanup_openmc_artifacts():
    """Remove stale OpenMC inputs/outputs so CMFD cannot reuse an old model.xml."""
    for pattern in _OPENMC_ARTIFACT_PATTERNS:
        for path in glob.glob(pattern):
            try:
                os.remove(path)
            except OSError:
                pass


def _mesh_from_sp_tally(sp: openmc.StatePoint, tally: openmc.Tally):
    mesh_filter = tally.find_filter(openmc.MeshFilter)
    mesh = mesh_filter.mesh
    if mesh.id in sp.meshes:
        return sp.meshes[mesh.id]
    return mesh


def _spatial_volumes(mesh, geometry: str) -> np.ndarray:
    if geometry == 'spherical':
        return mesh.volumes.flatten()
    return mesh.volumes[:, 0, 0].flatten()


def _extract_flux_data(sp: openmc.StatePoint, tally_name: str,
                       G: int, geometry: str) -> tuple[np.ndarray, object]:
    """
    Reshape the flux tally using the mesh stored in the statepoint (authoritative).
    """
    tally = sp.get_tally(name=tally_name)
    raw = np.asarray(tally.mean).ravel()
    mesh = _mesh_from_sp_tally(sp, tally)

    if raw.size % G != 0:
        raise ValueError(
            f"flux tally size {raw.size} is not divisible by G={G}"
        )
    n_spatial = raw.size // G
    vols = _spatial_volumes(mesh, geometry)
    if vols.size != n_spatial:
        raise ValueError(
            f"flux mesh has {vols.size} spatial bins but tally has {n_spatial}"
        )

    flux_data = raw.reshape(n_spatial, G) / vols[:, np.newaxis]
    return flux_data, mesh


def _extract_spatial_tally_density(sp: openmc.StatePoint, tally_name: str,
                                   geometry: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, object]:
    """
    Extract per-bin mean/std and convert to volume-normalized density.
    """
    tally = sp.get_tally(name=tally_name)
    mean_raw = np.asarray(tally.mean).ravel()
    std_raw = np.asarray(tally.std_dev).ravel()
    mesh = _mesh_from_sp_tally(sp, tally)
    vols = _spatial_volumes(mesh, geometry)

    if mean_raw.size != vols.size:
        raise ValueError(
            f"{tally_name} tally has {mean_raw.size} bins but mesh has {vols.size}"
        )

    mean_density = mean_raw / vols
    std_density = std_raw / vols
    return mean_density, std_density, vols, mesh


def _r_centers_from_mesh(mesh, cfg: MCConfig, n_spatial: int) -> np.ndarray:
    if cfg.geometry in ('cylindrical', 'spherical'):
        r_edges = np.asarray(mesh.r_grid, dtype=float)
    else:
        R = cfg.regions[-1].outer_radius
        r_edges = np.linspace(0.0, R, n_spatial + 1)
    return 0.5 * (r_edges[:-1] + r_edges[1:])


def _r_edges_from_mesh(mesh, cfg: MCConfig, n_spatial: int) -> np.ndarray:
    if cfg.geometry in ('cylindrical', 'spherical'):
        return np.asarray(mesh.r_grid, dtype=float)
    R = cfg.regions[-1].outer_radius
    return np.linspace(0.0, R, n_spatial + 1)


def _is_fuel_bearing_region(region: RegionSpec) -> bool:
    material = region.material.lower()
    if material in ('fuel', 'mix'):
        return True
    # Heuristic for custom materials where fuel may be encoded in region name.
    return material == 'custom' and ('fuel' in region.name.lower() or 'uran' in region.name.lower())


def _fuel_mask_from_regions(cfg: MCConfig, r_centers: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(r_centers, dtype=bool)
    r_inner = 0.0
    for region in cfg.regions:
        r_outer = float(region.outer_radius)
        if _is_fuel_bearing_region(region):
            in_region = (r_centers >= r_inner) & (r_centers < r_outer)
            mask = mask | in_region
        r_inner = r_outer

    if mask.size > 0:
        # Ensure the final spatial bin edge inclusion for the outermost fuel shell.
        for region in reversed(cfg.regions):
            if _is_fuel_bearing_region(region):
                mask = mask | np.isclose(r_centers, float(region.outer_radius))
                break
    return mask


def _safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den > 0 else np.nan


def _save_power_sidecar(cfg: MCConfig,
                        r_edges: np.ndarray,
                        r_centers: np.ndarray,
                        volumes: np.ndarray,
                        power_mean_density: np.ndarray,
                        power_std_density: np.ndarray,
                        fuel_mask: np.ndarray):
    out_dir = cfg.settings.convergence_output_dir
    os.makedirs(out_dir, exist_ok=True)
    tag = _run_tag(cfg)
    path = os.path.join(out_dir, f'{tag}_power_tally_raw.npz')
    np.savez(
        path,
        geometry=np.array([cfg.geometry]),
        r_edges=r_edges,
        r_centers=r_centers,
        volumes=volumes,
        power_mean_density=power_mean_density,
        power_std_density=power_std_density,
        fuel_mask=fuel_mask.astype(np.int8),
    )
    return path


def _save_study_metrics_sidecar(cfg: MCConfig,
                                keff,
                                flux_data: np.ndarray,
                                r_centers: np.ndarray,
                                flux_peak_factor_full: float,
                                radial_power_peaking_factor: float,
                                fuel_avg_power_density: float,
                                fuel_max_power_density: float,
                                fuel_mask: np.ndarray,
                                power_raw_path: str,
                                runtime_scalars: dict | None = None):
    out_dir = cfg.settings.convergence_output_dir
    os.makedirs(out_dir, exist_ok=True)
    tag = _run_tag(cfg)
    path = os.path.join(out_dir, f'{tag}_study_metrics.json')

    payload = {
        'keff': float(keff.nominal_value),
        'keff_std': float(keff.std_dev),
        'r_centers': r_centers.tolist(),
        'flux_data': flux_data.tolist(),
        'flux_peak_factor_full': float(flux_peak_factor_full),
        'radial_power_peaking_factor': float(radial_power_peaking_factor),
        'fuel_avg_power_density': float(fuel_avg_power_density),
        'fuel_max_power_density': float(fuel_max_power_density),
        'fuel_mask': fuel_mask.astype(bool).tolist(),
        'power_tally_sidecar_path': power_raw_path,
        'statepoint_copy_enabled': bool(cfg.settings.copy_statepoint),
    }
    if runtime_scalars:
        payload.update(runtime_scalars)
    with open(path, 'w') as fh:
        json.dump(payload, fh, indent=2)
    return path

def estimate_keff_pilot(cfg: MCConfig,
                         particles: int = 1000,
                         batches: int = 20,
                         inactive: int = 10):
    """
    Cheap eigenvalue-only run used to screen keff before committing to the
    full MGXS + flux run. No tallies other than the built-in keff estimator,
    so this is typically 10-50x faster than a full run.

    Runs inside {openmc_work_dir}/pilot so parallel Slurm array tasks do not
    clobber each other's XML/HDF5 files in the project root.
    """
    openmc.reset_auto_ids()

    geometry, cells, mats = build_geometry(cfg)
    materials_col = openmc.Materials(mats)

    settings = openmc.Settings()
    settings.batches   = batches
    settings.inactive  = inactive
    settings.particles = particles
    settings.run_mode  = 'eigenvalue'
    settings.statepoint = {'batches': [batches]}

    run_dir = _resolve_openmc_work_dir(cfg)
    pilot_dir = os.path.join(run_dir, 'pilot')

    with _openmc_cwd(pilot_dir):
        _cleanup_openmc_artifacts()
        model = openmc.model.Model(geometry, materials_col, settings, openmc.Tallies())
        sp_path = model.run(output=False)
        abs_sp = os.path.abspath(sp_path)

    with openmc.StatePoint(abs_sp) as sp:
        keff = sp.keff

    shutil.rmtree(pilot_dir, ignore_errors=True)

    return keff

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN RUN FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def run_mc(cfg: MCConfig, persist_results: bool = True):
    """
    Execute a complete OpenMC eigenvalue run as described by cfg.

    Parameters
    ----------
    persist_results : if True, append keff/XS/diagnostics rows to CSV files.
                      Set False for intermediate std-retry attempts.

    Returns
    -------
    keff      : UFloat  — k-effective with uncertainty (uncertainties package)
    flux_data : np.ndarray, shape (n_spatial_bins, G)
                  column g = volume-normalised flux in energy group g+1
    r_centers : np.ndarray, shape (n_spatial_bins,)
                  spatial bin centres in cm
    lib       : mgxs.Library  — loaded from statepoint (MGXS values accessible)
    cells     : list[openmc.Cell]  — same order as cfg.regions
    """
    openmc.reset_auto_ids()
    print_config(cfg)

    sets = cfg.settings

    # ── 1. Geometry & materials ────────────────────────────────────────────────
    geometry, cells, mats = build_geometry(cfg)
    materials_col = openmc.Materials(mats)

    # ── 2. MGXS library ───────────────────────────────────────────────────────
    lib = build_mgxs_library(geometry, cells, cfg)

    # ── 3. Flux + power mesh tallies ──────────────────────────────────────────
    flux_tally, flux_mesh = build_flux_mesh_tally(cfg)
    power_tally = build_power_mesh_tally(flux_mesh)

    # ── 4. Tallies collection ─────────────────────────────────────────────────
    tallies = openmc.Tallies()
    lib.add_to_tallies(tallies, merge=True)
    tallies.append(flux_tally)
    tallies.append(power_tally)

    # ── 5. MC settings ────────────────────────────────────────────────────────
    settings          = openmc.Settings()
    settings.batches  = sets.batches
    settings.inactive = sets.inactive
    settings.particles = sets.particles
    settings.generations_per_batch = sets.generations_per_batch
    settings.entropy_mesh = build_entropy_mesh(cfg)

    # ── Convergence trigger (advance until target std or max batches) ─────────
    end_batch_for_statepoints = sets.batches
    if sets.keff_trigger_std is not None:
        settings.trigger_active = True
        settings.trigger_max_batches = sets.trigger_max_batches or sets.batches
        settings.keff_trigger = {'type': 'std_dev', 'threshold': sets.keff_trigger_std}
        end_batch_for_statepoints = settings.trigger_max_batches

    sp_batches = _statepoint_batch_list(
        end_batch_for_statepoints,
        mid_fraction=sets.statepoint_mid_fraction,
    )
    settings.statepoint = {'batches': sp_batches}
    if sets.verbose:
        print(f"  [statepoints] batches {sp_batches}")

    if sets.verbose:
        print(f"  [OpenMC run] {sets.particles} p × {sets.batches} batches …")

    run_dir = _resolve_openmc_work_dir(cfg)
    if sets.verbose:
        print(f"  [OpenMC work dir] {os.path.abspath(run_dir)}")

    # ── CMFD acceleration ─────────────────────────────────────────────────
    with _openmc_cwd(run_dir):
        _cleanup_openmc_artifacts()

        if sets.cmfd_on:
            cmfd = openmc.cmfd.CMFDRun()
            cmfd.mesh = build_cmfd_mesh(cfg)
            cmfd.tally_begin = sets.cmfd_tally_begin
            cmfd.solver_begin = sets.cmfd_solver_begin
            cmfd.display = {'dominance': True}
            cmfd.feedback = True

            if sets.verbose:
                print("  [CMFD] enabled — mesh dim:", sets.cmfd_mesh_dim)
                print("  [CMFD] radii:", [r.outer_radius for r in cfg.regions])

            model = openmc.model.Model(geometry, materials_col, settings, tallies)
            model.export_to_xml()

            cmfd.run(output=False)

            sp_files = sorted(glob.glob("statepoint.*.h5"), key=os.path.getmtime)
            if not sp_files:
                raise FileNotFoundError(
                    f"CMFD run finished but no statepoint.*.h5 file was found in {run_dir}."
                )

            sp_path = sp_files[-1]

        else:
            model = openmc.Model(geometry, materials_col, settings, tallies)
            sp_path = model.run(output=False)

    sp_path = os.path.join(run_dir, os.path.basename(sp_path))

    # ── 7. Extract results ────────────────────────────────────────────────────
    G = cfg.energy_groups.G

    with openmc.StatePoint(sp_path) as sp:
        keff = sp.keff
        conv_summary, conv_history = analyze_statepoint(
            sp,
            final_keff_std=keff.std_dev,
            final_keff_nom=keff.nominal_value,
            cfg=cfg,
            window_batches=20,
        )
        runtime = getattr(sp, 'runtime', {}) or {}
        rt_total = runtime.get('total')
        rt_transport = runtime.get('transport')
        rt_inactive = runtime.get('inactive batches')
        rt_active = runtime.get('active batches')
        if rt_active is None and rt_transport is not None and rt_inactive is not None:
            rt_active = max(0.0, float(rt_transport) - float(rt_inactive))

        conv_summary = dict(conv_summary)
        conv_summary.update({
            'openmc_runtime_total_s': float(rt_total) if rt_total is not None else np.nan,
            'openmc_runtime_transport_s': float(rt_transport) if rt_transport is not None else np.nan,
            'openmc_runtime_inactive_s': float(rt_inactive) if rt_inactive is not None else np.nan,
            'openmc_runtime_active_s': float(rt_active) if rt_active is not None else np.nan,
        })

        if sets.verbose:
            print(f"  k-eff = {keff.nominal_value:.6f} ± {keff.std_dev:.6f}")
            for row in sp.global_tallies:
                if 'leakage' in row['name'].decode().lower():
                    print(f"  {row['name'].decode():<20} "
                        f"= {row['mean']:.5f} ± {row['std_dev']:.5f}")
            print(f"  Wall time: total={float(rt_total):.2f}s  "
                  f"transport={float(rt_transport):.2f}s  "
                  f"inactive={float(rt_inactive):.2f}s"
                  if (rt_total is not None and rt_transport is not None and rt_inactive is not None)
                  else "  Wall time: runtime breakdown unavailable in statepoint.")

        lib.load_from_statepoint(sp)
        conv_summary = save_convergence_artifacts(cfg, sp_path, conv_summary, conv_history)

        flux_data, sp_mesh = _extract_flux_data(sp, 'flux_spatial', G, cfg.geometry)
        power_mean_density, power_std_density, power_volumes, power_mesh = _extract_spatial_tally_density(
            sp, 'power_spatial', cfg.geometry
        )

    # ── 8. Bin centres ────────────────────────────────────────────────────────
    n_spatial = flux_data.shape[0]
    r_centers = _r_centers_from_mesh(sp_mesh, cfg, n_spatial)
    r_edges = _r_edges_from_mesh(power_mesh, cfg, n_spatial)

    fuel_mask = _fuel_mask_from_regions(cfg, r_centers)
    fuel_volumes = power_volumes[fuel_mask]
    fuel_power_density = power_mean_density[fuel_mask]
    fuel_max_power_density = float(np.max(fuel_power_density)) if fuel_power_density.size else np.nan
    fuel_avg_power_density = (
        float(np.sum(fuel_power_density * fuel_volumes) / np.sum(fuel_volumes))
        if fuel_power_density.size and np.sum(fuel_volumes) > 0
        else np.nan
    )
    radial_power_peaking_factor = _safe_ratio(fuel_max_power_density, fuel_avg_power_density)

    phi_total = flux_data.sum(axis=1)
    flux_peak_factor_full = _safe_ratio(float(phi_total.max()), float(phi_total.mean()))

    power_raw_path = _save_power_sidecar(
        cfg=cfg,
        r_edges=r_edges,
        r_centers=r_centers,
        volumes=power_volumes,
        power_mean_density=power_mean_density,
        power_std_density=power_std_density,
        fuel_mask=fuel_mask,
    )
    study_metrics_path = _save_study_metrics_sidecar(
        cfg=cfg,
        keff=keff,
        flux_data=flux_data,
        r_centers=r_centers,
        flux_peak_factor_full=flux_peak_factor_full,
        radial_power_peaking_factor=radial_power_peaking_factor,
        fuel_avg_power_density=fuel_avg_power_density,
        fuel_max_power_density=fuel_max_power_density,
        fuel_mask=fuel_mask,
        power_raw_path=power_raw_path,
        runtime_scalars={
            'openmc_runtime_total_s': conv_summary.get('openmc_runtime_total_s'),
            'openmc_runtime_transport_s': conv_summary.get('openmc_runtime_transport_s'),
            'openmc_runtime_inactive_s': conv_summary.get('openmc_runtime_inactive_s'),
            'openmc_runtime_active_s': conv_summary.get('openmc_runtime_active_s'),
        },
    )

    conv_summary = dict(conv_summary)
    conv_summary.update({
        'radial_power_peaking_factor': radial_power_peaking_factor,
        'fuel_avg_power_density': fuel_avg_power_density,
        'fuel_max_power_density': fuel_max_power_density,
        'flux_peak_factor_full': flux_peak_factor_full,
        'fuel_bin_count': int(np.sum(fuel_mask)),
        'study_metrics_path': study_metrics_path,
        'power_tally_sidecar_path': power_raw_path,
    })

    # ── 9. MGXS extraction & CSV ─────────────────────────────────────────────
    xs_dict = extract_mgxs_vector(lib, cells)
    if persist_results:
        save_results(cfg, keff, xs_dict, conv_summary)

    # ── 10. Results print & plot ──────────────────────────────────────────────
    print_results(keff, flux_data, r_centers, cfg)
    if cfg.settings.verbose:
        print(f"  PPF (fuel-only, radial) = {radial_power_peaking_factor:.6f}")
        print(f"  Flux peak factor (all bins) = {flux_peak_factor_full:.6f}")
        print(f"  [study metrics] → {study_metrics_path}")
        print(f"  [raw power sidecar] → {power_raw_path}")
    #plot_fluxes(r_centers, flux_data, cfg)

    return keff, flux_data, r_centers, lib, cells, conv_summary



# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    # ── Pick which config to run ──────────────────────────────────────────────
    cfg = CFG  

    keff, flux_data, r_centers, lib, cells, _conv = run_mc(cfg)
