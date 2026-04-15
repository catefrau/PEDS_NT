# LF_solver.py
# Solver entry-point: takes a GeometryConfig, predicts XS, runs the MG diffusion
# eigenvalue problem, and prints/plots results.
#
# Output: k_eff, group flux shapes (forward & adjoint), flux-ratio diagnostics.

import os
import sys
import joblib
import numpy as np
import pandas as pd
import time
from typing import NamedTuple, Optional
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from solvers.NTdiffusion.core.MG1D_eigenvalue import DiffusionEigenvalue_MG, DiffusionEigenvalue_MG_adjoint
from config_def import GeometryConfig, MaterialSpec, BoundarySpec, BoundaryCondition, MatProperties
from config_run import GEO_HOM as GEO

# ══════════════════════════════════════════════════════════════════════════════
#  PATHS DEFINITION  
# ══════════════════════════════════════════════════════════════════════════════

DATA_PATH   = 'polyreg/hom_MCdf_XSk.csv'
REG_MODEL_PATH = 'polyreg/hom_regre_files'
_INPUT_COLS = ['enrichment', 'dim_inner', 'f_mod']
# TODO also change the exclude columns
# more important X_new = _regression_inputs(geo), 
# which also hardcodes the columns names. 


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

def xs_per_region(G: int) -> int:
    return 4*G + G**2


# ══════════════════════════════════════════════════════════════════════════════
#  REGRESSION MODELS  (loaded once at import time)
# ══════════════════════════════════════════════════════════════════════════════
scaler = joblib.load(f'{REG_MODEL_PATH}/xs_scaler.pkl')
poly   = joblib.load(f'{REG_MODEL_PATH}/xs_poly_transformer.pkl')
reg    = joblib.load(f'{REG_MODEL_PATH}/xs_regression_model.pkl')


# ── Hardcoded input columns for the current heterogeneous training data ────────
# TODO: generalise when training data generation is updated for new geometries /
#       new mat_properties knobs.  For now we map the three regression inputs
#       directly from geo.mat_properties and geo.boundaries.

def _regression_inputs(geo: GeometryConfig) -> np.ndarray:
    """
    Build the (1, n_inputs) array fed to the polynomial regression.
    Reads from geo.mat_properties and geo.boundaries.

    Hardcoded to the current 3-feature model:
        enrichment, dim_inner (core radius), thickness (moderator shell width).
    """
    mp = geo.mat_properties

    if is_homogeneous(geo):
        # Homogeneous case: single zone — dim_inner = 0, thickness = R
        R = geo.boundaries[-1].radius
        enrichment = mp.enrichment if mp.enrichment is not None else 0.0
        f_mod = mp.moderator_fraction  if mp.moderator_fraction  is not None else 0.0
        return np.array([[enrichment, R, f_mod]])
    else:
        enrichment = mp.enrichment if mp.enrichment is not None else 0.0
        dim_inner  = geo.boundaries[0].radius               # first zone outer edge = core radius
        thickness  = geo.boundaries[-1].radius - dim_inner  # remaining shell
        return np.array([[enrichment, dim_inner, thickness]])


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
    n   = xs_per_region(G)
    lay = xs_layout(G)

    # ── Step 1: polynomial regression ─────────────────────────────────────────
    X_new     = _regression_inputs(geo) # WATCHOUT - HARDCODED!!!
    X_scaled  = scaler.transform(X_new)
    X_poly    = poly.transform(X_scaled)
    xs_values = reg.predict(X_poly)[0]   # shape: (n_output_cols,)

    # ── Step 2: map predicted values to a named dictionary ────────────────────
    # Column order mirrors the CSV used during training.
    # TODO: load from geo or a config file once training data is generalised.
    df_cols     = pd.read_csv(DATA_PATH, nrows=0).columns.tolist()
    chi_cols    = [c for c in df_cols if 'chi' in c]
    exclude     = _INPUT_COLS +  ['mode', 'geometry_type', 'thickness', 'keff', 'keff_std'] + chi_cols
    output_cols = [c for c in df_cols if c not in exclude]
    print(f"  Output columns : {output_cols}")
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

    def Sigma_s_fn(r, *_):
        s_flat = _region_vec(r)[lay['Sigma_s']]   # (G²,)
        return s_flat.reshape(G, G)               # (G, G)

    return D_fn, Sigma_a_fn, nuSigma_f_fn, Sigma_s_fn, chi_fn


# ══════════════════════════════════════════════════════════════════════════════
#  POST-PROCESSING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def normalize_group_fluxes(phi: np.ndarray) -> np.ndarray:
    """Normalise so that group-1 flux at the centre (index 0) equals 1."""
    return phi / phi[0, 0]


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
    every = max(1, len(x) // 14)

    for g, g_label in enumerate(group_labels):
        c = _GROUP_COLORS[g % len(_GROUP_COLORS)]
        ax.plot(x, phi_fwd_norm[g], "-",  color=c[0], lw=2.5,
                label=f"Fwd - {g_label}")
        ax.plot(x, phi_adj_norm[g], "--", color=c[1], lw=2.5,
                label=f"Num Adj  - {g_label}  φ*")                
        if an_fwd_groups is not None:
            ax.plot(x, an_fwd_groups[g], "o", color=c[2], ms=4,
                    markevery=(0, every), label=f"Analytic Fwd - {g_label}")
        if an_adj_groups is not None:
            ax.plot(x, an_adj_groups[g], "s", color=c[3], ms=4,
                    markevery=(every//2, every), label=f"Analytic Adj - {g_label} φ*")

    for name, r_int in interfaces:
        ax.axvline(r_int, color="gray", ls=":", lw=1.5,
                   label=f"{name} @ r = {r_int} cm")

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
    if mp.moderator_fraction is not None:
        print(f"    Moderator fraction : {mp.moderator_fraction:.4f}")
    if mp.plutonium_fraction is not None:
        print(f"    Plutonium fraction : {mp.plutonium_fraction:.4f} atom %")

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
                r_th2fast_fwd = phi_fwd_norm[1,idx]/phi_fwd_norm[0,idx]
                r_th2fast_adj = phi_adj_norm[1,idx]/phi_adj_norm[0,idx]
                print(f"    Thermal/Fast ratio  fwd={r_th2fast_fwd:.6f}  |  adj={r_th2fast_adj:.6f}")

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
    _print_config(geo)

    # ── 2. Derived scalars ────────────────────────────────────────────────────
    R             = geo.boundaries[-1].radius
    I             = int(R / geo.mesh_size)
    geometry_code = GEOMETRY_CODE[geo.geometry]
    BC_coeffs     = bc_to_coeffs(geo.bc)

    # r_div passed to legacy solver: first boundary if hetero, else R (homogeneous)
    r_div = geo.boundaries[0].radius if not is_homogeneous(geo) else R

    # ── 3. Predict XS ─────────────────────────────────────────────────────────
    print("  Predicting XS via polynomial regression for the baseline …")
    xs_tensor = predict_xs(geo)  # HERE I WILL ADD THE NN CONTRIB
    print("  The NN created XS will be added as a comtribution …")
    # NN_xs_tensor = NN(geo) -> add weighted contribution to xs_tensor
    _print_xs_summary(xs_tensor, geo)
    return xs_tensor

def run_diffusion_solver(xs_tensor, geo: GeometryConfig, plot_output="plots/fluxes.png"):
    start_time = time.time()
    _print_config(geo)
    R             = geo.boundaries[-1].radius
    I             = int(R / geo.mesh_size)
    geometry_code = GEOMETRY_CODE[geo.geometry]
    BC_coeffs     = bc_to_coeffs(geo.bc)
    r_div = geo.boundaries[0].radius if not is_homogeneous(geo) else R

    # ── 4. Build XS callables ─────────────────────────────────────────────────
    D_fn, Sigma_a_fn, nuSigma_f_fn, Sigma_s_fn, chi_fn = \
        build_xs_callables(xs_tensor, geo)

    # ── 5. Forward & adjoint eigenvalue solves ────────────────────────────────
    print("\n  Running forward eigenvalue solve …")
    k_fwd, phi_fwd, x = DiffusionEigenvalue_MG(
        R, I, geo.G, r_div,
        D_fn, Sigma_a_fn, nuSigma_f_fn, Sigma_s_fn, chi_fn,
        BC_coeffs, geometry_code
    )

    print("  Running adjoint eigenvalue solve …")
    k_adj, phi_adj, x = DiffusionEigenvalue_MG_adjoint(
        R, I, geo.G, r_div,
        D_fn, Sigma_a_fn, nuSigma_f_fn, Sigma_s_fn, chi_fn,
        BC_coeffs, geometry_code
    )

    # ── 6. Normalise and post-process ─────────────────────────────────────────
    phi_fwd_norm = normalize_group_fluxes(phi_fwd)
    phi_adj_norm = normalize_group_fluxes(phi_adj)

    elapsed = time.time() - start_time
    _print_results(k_fwd, k_adj, phi_fwd_norm, phi_adj_norm, x, geo, elapsed)

    # ── 7. Plot ───────────────────────────────────────────────────────────────
    #_plot_fluxes(x, geo, phi_fwd_norm, phi_adj_norm,plot_output="plots/fluxes.png")

    return k_fwd, phi_fwd_norm, phi_adj_norm

if __name__ == '__main__':
    xs_tensor = get_xs_basedon_geo(GEO)
    k, phi_fwd, phi_adj = run_diffusion_solver(xs_tensor, GEO)