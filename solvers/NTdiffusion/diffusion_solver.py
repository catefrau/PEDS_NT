# diffusion_solver.py
# Solver entry-point: takes a GeometryConfig, predicts XS, runs the MG diffusion
# eigenvalue problem, and prints/plots results.
#
# Output: k_eff, group flux shapes (forward & adjoint), flux-ratio diagnostics.

import os
import joblib
import json
import numpy as np
import pandas as pd
import time
from typing import NamedTuple, Optional
from pathlib import Path
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from .core.MG1D_eigenvalue_nregions import DiffusionEigenvalue_MG, DiffusionEigenvalue_MG_adjoint
from modules.NTcode_config_data.config_def import GeometryConfig, MaterialSpec, BoundarySpec, BoundaryCondition, MatProperties
from modules.NTcode_config_data.config_run import GEO_CYL as GEO

# ══════════════════════════════════════════════════════════════════════════════
#  PATHS DEFINITION  
# ══════════════════════════════════════════════════════════════════════════════

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent

DATA_FOLDER = PROJECT_ROOT / "modules" / "reg_and_data" / "inputs" / "CR_1000samples"
DATA_PATH = DATA_FOLDER / "1000_clean.csv"
REG_MODEL_PATH = DATA_FOLDER / "polyreg_model"

PLOT_OUTPUT = PROJECT_ROOT / "LOGS" / "fluxes_plots" / "fixed_CR_fluxes.png"

META_PATH = REG_MODEL_PATH / "xs_model_meta.json"

_KNOB_EXTRACTORS = {
    'outer_radius' : lambda geo, i: geo.boundaries[i].radius,
    'enrichment'   : lambda geo, i: (geo.mat_properties.enrichment
                                     if geo.mat_properties.enrichment is not None
                                     else 0.0),
    'f_mod'        : lambda geo, i: (geo.mat_properties.f_mod
                                     if geo.mat_properties.f_mod is not None
                                     else 0.0),
    'cr_fraction'  : lambda geo, i: (geo.mat_properties.cr_fraction
                                     if hasattr(geo.mat_properties, 'cr_fraction')
                                     and geo.mat_properties.cr_fraction is not None
                                     else 0.0),
}

# ── Load column schema from sidecar (written by MC_solver.save_meta()) ────────
if os.path.exists(META_PATH):
    with open(META_PATH) as fh:
        _meta = json.load(fh)

    _swept     = _meta.get('swept_cols', [])
    _all_knobs = _meta.get('input_cols', [])

    # Use swept_cols if declared, otherwise fall back to all knobs
    _INPUT_COLS = _meta.get('input_cols', [])    
    _CHI_COLS   = _meta.get('chi_cols', [])
    _XS_COLS = _meta.get('output_cols') or _meta.get('xs_cols', [])
    print(f"\n[meta] Loaded from {META_PATH}")
    #print(f"  geometry={_meta['geometry']} | G={_meta['G']}")
    #print(f"  input_cols = {_INPUT_COLS}")
else:
    # ── Fallback: hardcoded for pre-meta CSV files ────────────────────────────
    print("[meta] No sidecar found — using hardcoded column names (legacy mode)")
    _INPUT_COLS = ['enrichment', 'dim_inner', 'f_mod']
    _CHI_COLS   = []
    _XS_COLS    = []
    
# TODO also change the exclude columns
# more important X_new = _regression_inputs(geo), 
# which also hardcodes the columns names. 

def precompute_geometry(geo: GeometryConfig) -> dict:
    """
    Convert a GeometryConfig into a plain dict of Python/NumPy values.
    """
    R        = geo.boundaries[-1].radius
    I        = int(R / geo.mesh_size)
    Delta_r  = geo.mesh_size
    geometry_code = GEOMETRY_CODE[geo.geometry]

    # Radial grid: cell edges at r_i = i * Delta_r
    r_edges = np.array([i * Delta_r for i in range(I + 2)])  # length I+2

    # Surface areas S[i] and volumes V[i] for each geometry
    if geometry_code == 0:       # slab
        S = np.ones(I + 1)
        V = np.ones(I)
    elif geometry_code == 1:     # cylindrical
        S = r_edges[:-1]                              # S[i] = r_i
        V = 0.5 * (r_edges[1:-1]**2 - r_edges[:-2]**2) / Delta_r
    elif geometry_code == 2:     # spherical
        S = r_edges[:-1]**2
        V = (r_edges[1:-1]**3 - r_edges[:-2]**3) / (3 * Delta_r)

    # Map each cell index → region index using boundary radii
    region_of_cell = []
    for i in range(I):
        r_centre = (i + 0.5) * Delta_r
        reg = 0
        for j, bspec in enumerate(geo.boundaries):
            if r_centre <= bspec.radius:
                reg = j
                break
        region_of_cell.append(reg)

    BC = bc_to_coeffs(geo.bc)   # [A, B, C] — already defined in your file

    return {
        'G'              : geo.G,
        'I'              : I,
        'Delta_r'        : Delta_r,
        'S'              : S,
        'V'              : V,
        'BC'             : BC,
        'region_of_cell' : region_of_cell,
    }

# ── Geometry string → solver integer ──────────────────────────────────────────
GEOMETRY_CODE = {'slab': 0, 'cylindrical': 1, 'spherical': 2}

def bc_to_coeffs(bc: BoundaryCondition) -> list:
    """
    Convert a BoundaryCondition into the [A, B, C] coefficient list that
    the diffusion solver expects (see PDF eq. 26 and §2.3.1).

        A·φ(R) + B·(dφ/dr)|_R = C
    """
    t = bc.bc_type.lower()

    if t == 'reflective':
        return [0.0, 1.0, 0.0]

    elif t == 'vacuum':
        return [1.0, 0.0, 0.0]

    elif t == 'dirichlet':
        return [1.0, 0.0, float(bc.phi_val)]

    elif t == 'albedo':
        if bc.alpha is None:
            raise ValueError("BoundaryCondition with bc_type='albedo' requires alpha.")
        D = bc.D_val if bc.D_val is not None else 1.0   # fallback; warn user
        alpha = bc.alpha
        A = (1.0 - alpha) / (4.0 * (1.0 + alpha))
        B = D / 2.0
        return [A, B, 0.0]

    elif t == 'partial_current':
        if bc.Jin is None:
            raise ValueError("BoundaryCondition with bc_type='partial_current' requires Jin.")
        D = bc.D_val if bc.D_val is not None else 1.0
        A = 0.25
        B = D / 2.0
        C = float(bc.Jin)
        return [A, B, C]

    else:
        raise ValueError(
            f"Unknown bc_type '{bc.bc_type}'. "
            "Choose from: 'reflective', 'vacuum', 'dirichlet', 'albedo', 'partial_current'."
        )


def is_homogeneous(geo: GeometryConfig) -> bool:
    """True when a single material fills the whole domain."""
    return len(geo.regions) == 1

# ══════════════════════════════════════════════════════════════════════════════
#  XS TENSOR LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
# For G energy groups, one region's XS are stored as a 1-D vector of length
# XS_PER_REGION = 4*G + G²:
#
#   [0   : G  )   D           diffusion coefficient   (G values)
#   [G   : 2G )   Sigma_a     macroscopic absorption  (G values)
#   [2G  : 3G )   nuSigma_f   ν × fission xs          (G values)
#   [3G  : 3G+G²) Sigma_s     scatter matrix, row-major
#                              Sigma_s[i,j] = scatter FROM group i TO group j
#   [3G+G²: 4G+G²) chi        fission spectrum        (G values)
#
# Example G=2  →  12 values per region:
#   [0]  D_g1          [1]  D_g2
#   [2]  Σ_a_g1        [3]  Σ_a_g2
#   [4]  νΣ_f_g1       [5]  νΣ_f_g2
#   [6]  S[0,0]  g1→g1   [7]  S[0,1]  g1→g2  (downscatter)
#   [8]  S[1,0]  g2→g1   [9]  S[1,1]  g2→g2
#   [10] χ_g1           [11] χ_g2

def xs_layout(G: int) -> dict:
    """Index slices for each XS type, given G groups."""
    return {
        'D':         slice(0,           G),
        'Sigma_a':   slice(G,           2*G),
        'nuSigma_f': slice(2*G,         3*G),
        'Sigma_s':   slice(3*G,         3*G + G**2),
        'chi':       slice(3*G + G**2,  4*G + G**2),
    }

def fn_xs_per_region(G: int) -> int:
    return 4*G + G**2


# ══════════════════════════════════════════════════════════════════════════════
#  REGRESSION MODELS  (loaded once at import time)
# ══════════════════════════════════════════════════════════════════════════════
scaler = joblib.load(REG_MODEL_PATH / 'xs_scaler.pkl')
poly   = joblib.load(REG_MODEL_PATH / 'xs_poly_transformer.pkl')
reg    = joblib.load(REG_MODEL_PATH / 'xs_regression_model.pkl')


# ── Hardcoded input columns for the current heterogeneous training data ────────
# TODO: generalise when training data generation is updated for new geometries /
#       new mat_properties knobs.  For now we map the three regression inputs
#       directly from geo.mat_properties and geo.boundaries.

# ── Column name → GeometryConfig value extractor ─────────────────────────────
# Column names written by MC_solver follow the pattern:
#   r{i}_{region_name}_{knob}
# where knob is one of: outer_radius, enrichment, f_mod, cr_fraction
# We parse the knob suffix and extract the matching value from geo.


def _parse_knob(col_name: str) -> tuple[int, str]:
    """
    Parse 'r{i}_{region_name}_{knob}' → (region_index, knob_name).
    e.g. 'r0_uranyl_fuel_outer_radius' → (0, 'outer_radius')
         'r1_water_reflector_enrichment' → (1, 'enrichment')
    """
    # Strip the leading r{i}_ prefix
    parts = col_name.split('_')
        # ── Try prefixed format: r{i}_... ────────────────────────────────
    if parts[0].startswith('r') and parts[0][1:].isdigit():
        region_idx = int(parts[0][1:])   # 'r0' → 0
        # Try 2-word suffix first, then 1-word
        for length in (2, 1):
            knob = '_'.join(parts[-length:])
            # Known multi-word knobs: 'outer_radius', 'cr_fraction', 'f_mod'        
            if knob in _KNOB_EXTRACTORS:
                return region_idx, knob
        raise ValueError(
            f"Cannot parse knob from prefixed column '{col_name}'. "
            f"Known knobs: {list(_KNOB_EXTRACTORS.keys())}"
        )

    # ── Fallback: global (unprefixed) column name → region 0 ─────────
    # Map legacy/homogeneous column names to the unified knob vocabulary
    _GLOBAL_KNOB_ALIASES = {
        'dim_inner'  : 'outer_radius',
        'enrichment' : 'enrichment',
        'f_mod'      : 'f_mod',
        'cr_fraction': 'cr_fraction',
    }
    if col_name in _GLOBAL_KNOB_ALIASES:
        return 0, _GLOBAL_KNOB_ALIASES[col_name]

    raise ValueError(
        f"Cannot parse knob from column '{col_name}'. "
        f"Expected 'r{{i}}_..._{{knob}}' or one of {list(_GLOBAL_KNOB_ALIASES.keys())}."
    )


def _regression_inputs(geo: GeometryConfig) -> np.ndarray:
    """
    Build the (1, n_inputs) array fed to the polynomial regression.
    Column order is determined by _INPUT_COLS, loaded from the meta sidecar.
    No hardcoded feature names — works for any geometry and material knobs.
    """
    values = []
    for col in _INPUT_COLS:
        region_idx, knob = _parse_knob(col)
        extractor = _KNOB_EXTRACTORS[knob]
        values.append(extractor(geo, region_idx))

    return np.array([values])   # shape (1, n_inputs)

def _enforce_chi(xs_dict: dict, geo: GeometryConfig) -> dict:
    """
    Overwrite chi values in xs_dict so that:
      • chi sums to 1.0 in every material that has nuSigma_f > 0 in any group,
        with chi = 1 in the FASTEST group that has fission and 0 elsewhere.
      • chi is identically 0 in purely absorbing / scattering regions.

    This ensures the training-data convention (chi_g1=1, rest=0 for fissile
    regions) is always respected, regardless of what the regression predicted.
    """
    G = geo.G
    for mat in geo.regions:
        m = mat.region_ID
        # Collect nuSigma_f values for this material
        nu_vals = [xs_dict.get(f'{m}_nu-fission_g{g+1}', 0.0) for g in range(G)]

        if any(v > 0.0 for v in nu_vals):
            # Fissile material: chi = 1 in the fastest (first) group, 0 elsewhere
            for g in range(G):
                xs_dict[f'{m}_chi_g{g+1}'] = 1.0 if g == 0 else 0.0
        else:
            # Non-fissile material: all chi = 0
            for g in range(G):
                xs_dict[f'{m}_chi_g{g+1}'] = 0.0

    return xs_dict


def predict_xs(geo: GeometryConfig) -> np.ndarray:
    """
    Predict all cross-sections from geo and package them into an xs_tensor.

    Parameters
    ----------
    geo : GeometryConfig
        Complete problem specification.

    Returns
    -------
    xs_tensor : np.ndarray, shape (N_regions, XS_PER_REGION)
        Row i contains the XS vector for geo.regions[i].
    """
    G   = geo.G
    n   = fn_xs_per_region(G)
    lay = xs_layout(G)

    # ── Step 1: polynomial regression ─────────────────────────────────────────
    X_new     = _regression_inputs(geo) # WATCHOUT - HARDCODED!!!
    X_scaled  = scaler.transform(X_new)
    X_poly    = poly.transform(X_scaled)
    xs_values = reg.predict(X_poly)[0]   # shape: (n_output_cols,)


    # ── Step 2: map predicted values to a named dictionary ────────────────────
    # Column order mirrors the CSV used during training.
    # TODO: load from geo or a config file once training data is generalised.
    if _XS_COLS:
        # Fast path: use xs_cols from sidecar (excludes chi automatically)
        chi_set    = set(_CHI_COLS)
        output_cols = [c for c in _XS_COLS if c not in chi_set]
        #print(f"THE OUTPUT COLS ARE {output_cols}")
        #print("PRINTING STUFF \n")
        """ for name, val in zip(output_cols, xs_values):
            print(f"  {name:<45s} = {val:.6f}") """
    else:
        # Fallback: derive from CSV header (legacy behaviour)
        df_cols = pd.read_csv(DATA_PATH, nrows=0).columns.tolist()
        chi_cols_local = [c for c in df_cols if 'chi' in c]
        exclude = set(_INPUT_COLS) | {'geometry', 'G', 'keff', 'keff_std'} \
                | set(chi_cols_local)
        output_cols = [c for c in df_cols if c not in exclude]

    xs_dict = dict(zip(output_cols, xs_values))

    # ── Step 3: enforce physics-consistent chi ────────────────────────────────
    # Inject predicted nuSigma_f into xs_dict first so _enforce_chi can read them
    for mat in geo.regions:
        m = mat.region_ID
        for g in range(G):
            key = f'{m}_nu-fission_g{g+1}'
            # already in xs_dict from regression — no action needed here
            pass

    xs_dict = _enforce_chi(xs_dict, geo)

    # ── Step 4: enforce neutron balance for scatter term with poor R² ─────────
    # (kept from original; adapt the column names if your training data changes)
    """ if 'core_scatter matrix_g3' in xs_dict:
        xs_dict['core_scatter matrix_g3'] = (
            xs_dict.get('core_total_g2', 0.0)
            - xs_dict.get('core_absorption_g2', 0.0)
            - xs_dict.get('core_scatter matrix_g2', 0.0)
            - xs_dict.get('core_scatter matrix_g4', 0.0)
        ) """

    # ── Step 5: pack into tensor (N_regions, XS_PER_REGION) ─────────────────
    xs_tensor = np.zeros((len(geo.regions), n))

    for mat in geo.regions:
        vec = np.zeros(n)
        m   = mat.region_ID

        vec[lay['D']]         = [xs_dict.get(f'{m}_diffusion-coefficient_g{g+1}', 0.0) for g in range(G)]
        vec[lay['Sigma_a']]   = [xs_dict.get(f'{m}_absorption_g{g+1}',            0.0) for g in range(G)]
        vec[lay['nuSigma_f']] = [xs_dict.get(f'{m}_nu-fission_g{g+1}',            0.0) for g in range(G)]
        vec[lay['chi']]       = [xs_dict.get(f'{m}_chi_g{g+1}',                   0.0) for g in range(G)]

        # Scatter matrix: row-major S[from_g, to_g]
        scatter_flat = [xs_dict.get(f'{m}_scatter matrix_g{k+1}', 0.0) for k in range(G**2)]
        vec[lay['Sigma_s']] = scatter_flat

        xs_tensor[mat.region_index] = vec

    return xs_tensor


# ══════════════════════════════════════════════════════════════════════════════
#  CALLABLE BUILDERS FOR THE DIFFUSION SOLVER
# ══════════════════════════════════════════════════════════════════════════════

def build_xs_callables(xs_tensor: np.ndarray, geo: GeometryConfig):
    """
    Convert xs_tensor → 5 callable functions expected by DiffusionEigenvalue_MG.

    Region lookup: for a given r, walks geo.boundaries in order and returns the
    first zone whose outer radius >= r.  Falls back to the outermost zone.
    """
    G   = geo.G
    lay = xs_layout(G)

    def _region_vec(r: float) -> np.ndarray:
        for i, bspec in enumerate(geo.boundaries):
            if r <= bspec.radius:
                return xs_tensor[i]
        return xs_tensor[-1]   # fallback

    D_fn         = lambda r, *_: _region_vec(r)[lay['D']]
    Sigma_a_fn   = lambda r, *_: _region_vec(r)[lay['Sigma_a']]
    nuSigma_f_fn = lambda r, *_: _region_vec(r)[lay['nuSigma_f']]
    chi_fn       = lambda r, *_: _region_vec(r)[lay['chi']]

    #print(f"\n --------\n WHAT IS THIS SIGMA PROBLEM {lay['Sigma_s']}")
    #print(f"\n --------\n AND THE OTHERS? THATS D {lay['D']}")
    #print(f"\n --------\n AND THE OTHERS? THATS NUSI {lay['nuSigma_f']}")

    
    def Sigma_s_fn(r, *_):
        s_flat = _region_vec(r)[lay['Sigma_s']]   # (G²,)
        return s_flat.reshape(G, G)               # (G, G)

    return D_fn, Sigma_a_fn, nuSigma_f_fn, Sigma_s_fn, chi_fn


# ══════════════════════════════════════════════════════════════════════════════
#  POST-PROCESSING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def normalize_group_fluxes(phi: np.ndarray) -> np.ndarray:
    """Normalise so that group-1 flux at the centre (index 0) equals 1."""
    max_phi = phi.max()
    return phi / max_phi


_GROUP_COLORS = [
    ("skyblue",     "dodgerblue",   "cornflowerblue", "royalblue"),
    ("lightsalmon", "orangered",    "tomato",         "firebrick"),
    ("lightgreen",  "seagreen",     "mediumseagreen", "darkgreen"),
    ("plum",        "darkorchid",   "mediumorchid",   "purple"),
]

_GEOM_LABEL = {'slab': 'Slab', 'cylindrical': 'Cylinder', 'spherical': 'Sphere'}


def _plot_fluxes(x, geo: GeometryConfig,
                 phi_fwd_norm, phi_adj_norm,
                 an_fwd_groups=None, an_adj_groups=None,
                 plot_output=None):
    """
    Plot forward (and optionally adjoint + analytical) group fluxes.
    Title and legend are built from GeometryConfig so all problem
    details appear automatically.
    """
    G     = geo.G
    R     = geo.boundaries[-1].radius
    homo  = is_homogeneous(geo)

    group_labels = (["Fast", "Thermal"] if G == 2
                    else ["One group"] if G == 1
                    else [f"Group {g+1}" for g in range(G)])

    geom_str  = _GEOM_LABEL.get(geo.geometry, geo.geometry)
    homo_str  = "homogeneous" if homo else "heterogeneous"
    group_str = f"{G}G"
    enrich    = geo.mat_properties.enrichment
    bc_str    = geo.bc.bc_type.capitalize()

    # Zone interface lines
    interfaces = [(b.name, b.radius) for b in geo.boundaries[:-1]] if not homo else []

    fig, ax = plt.subplots(figsize=(9, 5))
    every = max(1, len(x) // 30)

    for g, g_label in enumerate(group_labels):
        c = _GROUP_COLORS[g % len(_GROUP_COLORS)]
        ax.plot(x, phi_fwd_norm[g], "-",  color=c[0], lw=2.5,
                label=f"Fwd - {g_label}")
        ax.plot(x, phi_adj_norm[g], "--", color=c[1], lw=2.5,
                label=f"Num Adj  - {g_label}  φ*")                
        if an_fwd_groups is not None:
            ax.plot(x, an_fwd_groups[g], "o", color=c[2], ms=4,
                    markevery=(0, every), label=f"Baseline (regression) - {g_label}")  # was Analytic Fwd 
        if an_adj_groups is not None:
            ax.plot(x, an_adj_groups[g], "s", color=c[3], ms=4,
                    markevery=(every//2, every), label=f"Analytic Adj - {g_label} φ*")

    for name, r_int in interfaces:
        ax.axvline(r_int, color="gray", ls=":", lw=1.5,
                   label=f"{name} @ r = {r_int:.2f} cm")

    enrich_tag = f"  |  enrich = {enrich:.1f}%" if enrich is not None else ""
    title = (f"{group_str}  |  {homo_str}  |  {geom_str}"
             f"  |  BC: {bc_str}{enrich_tag}")
    if an_fwd_groups is None:
        title += "  (no analytical solution)"

    ax.set_xlabel("r  (cm)", fontsize=12)
    ax.set_ylabel("Normalised Flux  φ(r)", fontsize=12)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.35)
    ax.set_xlim([x[0], x[-1]])
    ax.set_ylim([0, None])

    plt.tight_layout()
    os.makedirs(os.path.dirname(plot_output) or ".", exist_ok=True)
    plt.savefig(plot_output, dpi=150)
    print(f"  [Plot saved] → {plot_output}")


def _print_xs_summary(xs_tensor: np.ndarray, geo: GeometryConfig):
    """Print the XS tensor, one block per material."""
    G   = geo.G
    lay = xs_layout(G)

    print("\n  ┌─────────────────────────────────────────┐")
    print("  │           XS Tensor Summary             │")
    print("  └─────────────────────────────────────────┘")
    for mat in geo.regions:
        vec = xs_tensor[mat.region_index]
        print(f"\n  ▸ Material [{mat.region_index}]  '{mat.region_ID}'")
        D_vals  = vec[lay['D']]
        Sa_vals = vec[lay['Sigma_a']]
        nF_vals = vec[lay['nuSigma_f']]
        chi_v   = vec[lay['chi']]
        Ss_mat  = vec[lay['Sigma_s']].reshape(G, G)

        for g in range(G):
            print(f"      Group {g+1}:  D={D_vals[g]:.4f}  Σ_a={Sa_vals[g]:.4f}"
                  f"  νΣ_f={nF_vals[g]:.4f}  χ={chi_v[g]:.4f}")
        print(f"      Scatter matrix (from→to):\n{Ss_mat}")


def _print_config(geo: GeometryConfig):
    """Print a human-readable summary of the GeometryConfig."""
    R    = geo.boundaries[-1].radius
    I    = int(R / geo.mesh_size)
    homo = is_homogeneous(geo)
    mp   = geo.mat_properties
    bc   = geo.bc

    print("╔══════════════════════════════════════════════════════════╗")
    print("║              DIFFUSION SOLVER — PROBLEM SETUP            ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Geometry      : {_GEOM_LABEL.get(geo.geometry, geo.geometry)}")
    print(f"  Energy groups : {geo.G}")
    print(f"  Mode          : {'homogeneous (single zone)' if homo else 'heterogeneous'}")
    print(f"  Outer radius  : {R:.2f} cm")
    print(f"  Mesh size     : {geo.mesh_size} cm  →  I = {I} cells")

    print(f"\n  ── Material zones (centre → outside) ──")
    for mat, bspec in zip(geo.regions, geo.boundaries):
        ht = f"  height={bspec.height} cm" if bspec.height is not None else ""
        print(f"    [{mat.region_index}] '{mat.region_ID}'  →  "
              f"boundary '{bspec.name}' @ r = {bspec.radius:.2f} cm{ht}")

    print(f"\n  ── Material properties ──")
    if mp.enrichment is not None:
        print(f"    Enrichment         : {mp.enrichment:.2f} atom %")
    if mp.f_mod is not None:
        print(f"    Moderator fraction : {mp.f_mod:.4f}")
    if mp.cr_fraction is not None:
        print(f"    Plutonium fraction : {mp.cr_fraction:.4f} atom %")

    print(f"\n  ── Boundary condition at r = R ──")
    print(f"    Type  : {bc.bc_type}")
    if bc.bc_type == 'albedo':
        print(f"    Alpha : {bc.alpha}")
    if bc.bc_type == 'partial_current':
        print(f"    J_in  : {bc.Jin}")
    if bc.bc_type == 'dirichlet':
        print(f"    φ(R)  : {bc.phi_val}")
    if bc.D_val is not None:
        print(f"    D_val : {bc.D_val}")
    print(f"    [A, B, C] = {bc_to_coeffs(bc)}")
    print()


def _print_results(k_fwd, k_adj, phi_fwd_norm, phi_adj_norm,
                   x, geo: GeometryConfig, elapsed: float):
    """Print a structured results block."""
    G    = geo.G
    R    = geo.boundaries[-1].radius
    homo = is_homogeneous(geo)

    print("╔══════════════════════════════════════════════════════════╗")
    print("║                      RESULTS                             ║")
    print("╚══════════════════════════════════════════════════════════╝")
    minutes = int(elapsed // 60)
    seconds = elapsed % 60
    print(f"  Run time  : {elapsed:.2f} s  ({minutes}m {seconds:.2f}s)")
    print(f"  k_eff     : {k_fwd:.6f}  (forward)   |  {k_adj:.6f}  (adjoint)")

    if not homo:
        # Print diagnostics at each interface
        for bspec in geo.boundaries[:-1]:
            r_int = bspec.radius
            idx   = np.argmin(np.abs(x - r_int))
            print(f"\n  ── Interface '{bspec.name}' @ r ≈ {x[idx]:.3f} cm ──")
            for g in range(G):
                g_label = (["Fast","Thermal"] if G==2 else [f"G{g+1}"])[min(g,1) if G==2 else 0]
                print(f"    φ_fwd_g{g+1}({g_label}) = {phi_fwd_norm[g, idx]:.6e}"
                      f"  |  φ_adj_g{g+1} = {phi_adj_norm[g, idx]:.6e}")
            if G == 2:
                r_th2fast_fwd = phi_fwd_norm[0,idx] / phi_fwd_norm[1,idx]
                r_th2fast_adj = phi_adj_norm[0,idx] / phi_adj_norm[1,idx]
                print(f"    Fast/Thermal ratio  fwd={r_th2fast_fwd:.6f}  |  adj={r_th2fast_adj:.6f}")

    # Centre diagnostics
    print(f"\n  ── Centre (r = 0) ──")
    for g in range(G):
        print(f"    φ_fwd_g{g+1} = {phi_fwd_norm[g, 0]:.6e}"
              f"  |  φ_adj_g{g+1} = {phi_adj_norm[g, 0]:.6e}")

    # Outer edge
    print(f"\n  ── Outer edge (r = {x[-1]:.2f} cm) ──")
    for g in range(G):
        print(f"    φ_fwd_g{g+1} = {phi_fwd_norm[g, -1]:.6e}"
              f"  |  φ_adj_g{g+1} = {phi_adj_norm[g, -1]:.6e}")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def get_xs_basedon_geo(geo: GeometryConfig,):
    # ── 1. Print problem configuration ────────────────────────────────────────
    #_print_config(geo)

    # ── 2. Derived scalars ────────────────────────────────────────────────────
    R             = geo.boundaries[-1].radius
    I             = int(R / geo.mesh_size)
    geometry_code = GEOMETRY_CODE[geo.geometry]
    BC_coeffs     = bc_to_coeffs(geo.bc)

    # r_div passed to legacy solver: first boundary if hetero, else R (homogeneous)
    #r_div = geo.boundaries[0].radius if not is_homogeneous(geo) else R
    # up to [-1] because the last radius is R and is already stored 
    r_divisions = [b.radius for b in geo.boundaries[:-1]] if not is_homogeneous(geo) else []

    # ── 3. Predict XS ─────────────────────────────────────────────────────────
    #print("  Predicting XS via polynomial regression for the baseline …")
    xs_tensor = predict_xs(geo)  # HERE I WILL ADD THE NN CONTRIB
    #print(f"  Predicted XS tensor shape: {xs_tensor.shape}  (N_regions={len(geo.regions)}, XS_per_region={xs_tensor.shape[1]})")
    #rint(f"  XS layout (index slices): {xs_layout(geo.G)}")
    #print(f"the tensor is {xs_tensor}")
    # NN_xs_tensor = NN(geo) -> add weighted contribution to xs_tensor
    #_print_xs_summary(xs_tensor, geo)
    return xs_tensor

def run_diffusion_solver(xs_tensor, geo: GeometryConfig, plot_output="plots/fluxes.png"):
    start_time = time.time()
    #_print_config(geo)
    R             = geo.boundaries[-1].radius
    I             = int(R / geo.mesh_size)
    geometry_code = GEOMETRY_CODE[geo.geometry]
    BC_coeffs     = bc_to_coeffs(geo.bc)
    r_divisions = [b.radius for b in geo.boundaries[:-1]] if not is_homogeneous(geo) else []

    # ── 4. Build XS callables ─────────────────────────────────────────────────
    D_fn, Sigma_a_fn, nuSigma_f_fn, Sigma_s_fn, chi_fn = \
        build_xs_callables(xs_tensor, geo)

    # ── 5. Forward & adjoint eigenvalue solves ────────────────────────────────
    #print("\n  Running forward eigenvalue solve …")
    k_fwd, phi_fwd, x = DiffusionEigenvalue_MG(
        R, I, geo.G, r_divisions,
        D_fn, Sigma_a_fn, nuSigma_f_fn, Sigma_s_fn, chi_fn,
        BC_coeffs, geometry_code
    )

    #print("  Running adjoint eigenvalue solve …")
    k_adj, phi_adj, x = DiffusionEigenvalue_MG_adjoint(
        R, I, geo.G, r_divisions,
        D_fn, Sigma_a_fn, nuSigma_f_fn, Sigma_s_fn, chi_fn,
        BC_coeffs, geometry_code
    )

    # ── Rescale to volume-integrated normalization ──────────────────
    # phi_fwd_unit has shape [G, I], phi values at cell centers
    # Rescale so that sum_g sum_i phi[g,i] * V[i] = 1
    # This gives physically meaningful flux magnitudes for the VJP.
    R = geo.boundaries[-1].radius
    I = int(R / geo.mesh_size)
    Delta_r = geo.mesh_size

    geometry_code = GEOMETRY_CODE[geo.geometry]
    edges = np.arange(I + 1) * Delta_r

    if geometry_code == 1:   # cylindrical
        V = np.pi * (edges[1:]**2 - edges[:-1]**2)   # shape [I]
    elif geometry_code == 2: # spherical
        V = (4/3) * np.pi * (edges[1:]**3 - edges[:-1]**3)
    else:                    # slab
        V = np.ones(I) * Delta_r

    # Volume-weighted norm: sum over all groups and cells
    vol_norm_fwd = np.sum(phi_fwd * V[np.newaxis, :])  # scalar
    vol_norm_adj = np.sum(phi_adj * V[np.newaxis, :])  # scalar

    phi_fwd = phi_fwd / vol_norm_fwd   # now sum(phi * V) = 1
    phi_adj = phi_adj / vol_norm_adj   # same for adjoint

    """ # ── 6. Normalise and post-process ─────────────────────────────────────────
    phi_fwd_norm = normalize_group_fluxes(phi_fwd)
    phi_adj_norm = normalize_group_fluxes(phi_adj) """

    elapsed = time.time() - start_time
    #_print_results(k_fwd, k_adj, phi_fwd_norm, phi_adj_norm, x, geo, elapsed)

    # ── 7. Plot ───────────────────────────────────────────────────────────────
    #_plot_fluxes(x, geo, phi_fwd_norm, phi_adj_norm, plot_output=PLOT_OUTPUT)

    return k_fwd, phi_fwd, phi_adj

if __name__ == '__main__':
    xs_tensor = get_xs_basedon_geo(GEO)
    k, phi_fwd, phi_adj = run_diffusion_solver(xs_tensor, GEO)
