import os
from collections import OrderedDict

import jax
import jax.numpy as jnp
import numpy as np

from matrix_JAX_optimized import Aphi_Fphi_vjp
from NTcode_config_data.config_def import GeometryConfig, BoundarySpec, MatProperties
from NTcode_config_data.config_run import GEO_CYL as GEO
from solvers.NTdiffusion.core.MG1D_eigenvalue_nregions import diffusion_setup
from solvers.NTdiffusion.diffusion_solver import (
    run_diffusion_solver,
    is_homogeneous,
    precompute_geometry,
    xs_layout,
    build_xs_callables,
    bc_to_coeffs,
    GEOMETRY_CODE,
)
from PEDS_core.timing_utils import timer

# ── precomputed geometry constants ───────────────────────────────────────────
SLAY     = xs_layout(GEO.G)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
_DATA_FILEPATH = os.path.join(PROJECT_ROOT, "data", "highfidelity", "1000_clean.npz")
if not os.path.exists(_DATA_FILEPATH):
    raise FileNotFoundError(f"Could not find data file: {_DATA_FILEPATH}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────
def update_geo(geo: GeometryConfig, params_raw: np.ndarray) -> GeometryConfig:
    """Rebuild a GeometryConfig from a [6] raw-parameter vector."""
    new_boundaries = (
        BoundarySpec(name='CR_outer',        radius=float(params_raw[0])),
        BoundarySpec(name='core_outer',      radius=float(params_raw[2])),
        BoundarySpec(name='moderator_outer', radius=float(params_raw[5])),
    )
    new_mat = MatProperties(
        cr_fraction=float(params_raw[1]),
        enrichment =float(params_raw[3]),
        f_mod      =float(params_raw[4]),
    )
    return GeometryConfig(
        G              = geo.G,
        regions        = geo.regions,
        boundaries     = new_boundaries,
        geometry       = geo.geometry,
        mat_properties = new_mat,
        bc             = geo.bc,
        mesh_size      = geo.mesh_size,
    )


# ── max flat flux vector size (needed for padded pure_callback) ──────────────
def _compute_n_flat_max_from_file(filepath: str, geo: GeometryConfig) -> int:
    data = np.load(filepath, allow_pickle=True)
    rawparams = np.array(data['params_raw'], dtype=np.float32)
    n_flat_max = 0
    for i in range(len(rawparams)):
        geo_i  = update_geo(geo, rawparams[i])
        R      = geo_i.boundaries[-1].radius
        I      = int(R / geo_i.mesh_size)
        n_flat_max = max(n_flat_max, geo_i.G * (I + 1))
    return n_flat_max

N_FLAT_MAX = _compute_n_flat_max_from_file(_DATA_FILEPATH, GEO)

class LRUCache(OrderedDict):
    def __init__(self, maxsize):
        super().__init__()
        self.maxsize = maxsize
    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if len(self) > self.maxsize:
            self.popitem(last=False)

_GEO_DATA_CACHE  = LRUCache(maxsize=50)
_PRESOLVE_CACHE: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Physics solver + custom VJP  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────
def _run_NT_solver(xs_tensor, params_raw_single, sample_id):
    """NumPy/SciPy forward solve. Returns (k, phi_fwd_padded, phi_adj_padded, Fphi_padded)."""
    geo_i    = update_geo(GEO, np.array(params_raw_single))
    geo_data = precompute_geometry(geo_i)
    sid      = int(sample_id[0])
    _GEO_DATA_CACHE[sid] = geo_data

    # ── use cached solve if available ────────────────────────────────────────
    if sid in _PRESOLVE_CACHE:
        k, phi_fwd_padded, phi_adj_padded, geo_data_pre = _PRESOLVE_CACHE.pop(sid)
        _GEO_DATA_CACHE[sid] = geo_data_pre
        R      = geo_i.boundaries[-1].radius
        I      = int(R / geo_i.mesh_size)
        N_flat = geo_i.G * (I + 1)
        r_div  = [b.radius for b in geo_i.boundaries[:-1]] if not is_homogeneous(geo_i) else []
        BC     = bc_to_coeffs(geo_i.bc)
        gcode  = GEOMETRY_CODE[geo_i.geometry]
        D_fn, Sa_fn, nF_fn, Ss_fn, chi_fn = build_xs_callables(np.array(xs_tensor), geo_i)
        _, _, F_real = diffusion_setup(R, I, geo_i.G, r_div, D_fn, Sa_fn, nF_fn, Ss_fn, chi_fn, BC, gcode)
        Fphi_full = F_real @ phi_fwd_padded[:N_flat].astype(np.float64)
        Fphi_padded = np.zeros(N_FLAT_MAX, dtype=np.float32)
        Fphi_padded[:N_flat] = Fphi_full
        return np.float32(k), phi_fwd_padded, phi_adj_padded, Fphi_padded

    # ── full eigenvalue solve ─────────────────────────────────────────────────
    R      = geo_i.boundaries[-1].radius
    I      = int(R / geo_i.mesh_size)
    N_flat = geo_i.G * (I + 1)

    with timer("  solver: eigenvalue solve (fwd)", verbose=False):
        k, phi_fwd, phi_adj = run_diffusion_solver(xs_tensor, geo_i)

    phi_fwd_flat = np.zeros(N_flat, dtype=np.float64)
    phi_adj_flat = np.zeros(N_flat, dtype=np.float64)
    for g in range(geo_i.G):
        phi_fwd_flat[g*(I+1): g*(I+1)+I] = phi_fwd[g, :]
        phi_adj_flat[g*(I+1): g*(I+1)+I] = phi_adj[g, :]

    r_div  = [b.radius for b in geo_i.boundaries[:-1]] if not is_homogeneous(geo_i) else []
    BC     = bc_to_coeffs(geo_i.bc)
    gcode  = GEOMETRY_CODE[geo_i.geometry]
    D_fn, Sa_fn, nF_fn, Ss_fn, chi_fn = build_xs_callables(np.array(xs_tensor), geo_i)
    _, A_real, F_real = diffusion_setup(R, I, geo_i.G, r_div, D_fn, Sa_fn, nF_fn, Ss_fn, chi_fn, BC, gcode)

    Fphi     = F_real @ phi_fwd_flat
    biorth   = phi_adj_flat @ Fphi
    phi_adj_flat *= 1.0 / biorth   # enforce ⟨φ†, Fφ⟩ = 1
    #print(f"bwd sanity: phiadj·F·phi = {float(phi_adj_flat @ Fphi):.6f}  (should be ~1.0)")


    phi_fwd_padded = np.zeros(N_FLAT_MAX, dtype=np.float32)
    phi_adj_padded = np.zeros(N_FLAT_MAX, dtype=np.float32)
    Fphi_padded    = np.zeros(N_FLAT_MAX, dtype=np.float32)
    phi_fwd_padded[:N_flat] = phi_fwd_flat[:N_flat]
    phi_adj_padded[:N_flat] = phi_adj_flat[:N_flat]
    Fphi_padded[:N_flat]    = (F_real @ phi_fwd_flat)[:N_flat]

    return np.float32(k), phi_fwd_padded, phi_adj_padded, Fphi_padded


def _NTdiff_fwd(xs_tensor, params_raw_single, sample_id):
    geo_i  = update_geo(GEO, np.array(params_raw_single))
    R      = geo_i.boundaries[-1].radius
    I      = int(R / geo_i.mesh_size)

    keff, phi_fwd_padded, phi_adj_padded, Fphi_padded = jax.pure_callback(
        _run_NT_solver,
        (
            jax.ShapeDtypeStruct((),             jnp.float32),
            jax.ShapeDtypeStruct((N_FLAT_MAX,),  jnp.float32),
            jax.ShapeDtypeStruct((N_FLAT_MAX,),  jnp.float32),
            jax.ShapeDtypeStruct((N_FLAT_MAX,),  jnp.float32),
        ),
        xs_tensor, params_raw_single, sample_id,
    )
    residuals = (xs_tensor, keff, phi_fwd_padded, phi_adj_padded, Fphi_padded,
                 params_raw_single, sample_id)
    return keff, residuals


def _NTdiff_bwd(residuals, g):
    xs_tensor, k, phi_fwd_padded, phi_adj_padded, Fphi_padded, params_raw_single, sample_id = residuals
    geo_i  = update_geo(GEO, np.array(params_raw_single))
    R      = geo_i.boundaries[-1].radius
    I      = int(R / geo_i.mesh_size)
    N_flat = int(geo_i.G * (I + 1))

    phi_fwd  = phi_fwd_padded[:N_flat]
    phi_adj  = phi_adj_padded[:N_flat]
    Fphi     = Fphi_padded[:N_flat]
    geo_data = _GEO_DATA_CACHE[int(sample_id[0])]

    v_A = phi_adj
    v_F = -(1.0 / k) * phi_adj
    with timer("  bwd: vjp A@phi, F@phi -> numerator", verbose=False):
        numerator   = Aphi_Fphi_vjp(xs_tensor, geo_data, SLAY, phi_fwd, v_A, v_F)
    denominator = (1.0 / k**2) * (phi_adj @ Fphi)
    dk_dp       = -numerator / denominator
    return (g * dk_dp, None, None)


@jax.custom_vjp
def NTdiff_solver(xs_tensor, params_raw_single, sample_id):
    keff, _ = _NTdiff_fwd(xs_tensor, params_raw_single, sample_id)
    return keff

NTdiff_solver.defvjp(_NTdiff_fwd, _NTdiff_bwd)


def _solve_sample_worker(args):
    """Runs in a subprocess. Returns (i, k, phi_fwd, phi_adj, geo_data, epoch)."""
    import os
    os.environ["JAX_PLATFORMS"] = "cpu"
    i, xs_np, params_np, sample_id_int, epoch = args
    geo_i    = update_geo(GEO, params_np)
    geo_data = precompute_geometry(geo_i)
    k, phi_fwd, phi_adj, Fphi = _run_NT_solver(xs_np, params_np, np.array([sample_id_int]))
    return i, k, phi_fwd, phi_adj, geo_data, epoch


