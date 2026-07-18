"""Shared runtime context for the PEDS training pipeline.

This module is the single place that holds all *derived* constants and
shared runtime state that the split-out implementation modules
(`physics_solver`, `data_loading`, `logging_csv`, `plotting`) need but that
depend on the user-editable knobs kept in ``PEDS.py``.

Design (dependency injection — avoids circular imports):
    1. ``PEDS.py`` defines the editable knobs (SECTION 0).
    2. ``PEDS.py`` calls :func:`init` at import time (module scope, *not* under
       ``if __name__ == "__main__"``) passing those knobs in.
    3. :func:`init` computes the derived constants / caches and stores them as
       module-level attributes here.
    4. The implementation modules ``import PEDS_subdivision.context as context``
       and read ``context.GEO``, ``context.N_FLAT_MAX``, ``context.LOG_DIR``, …

Because :func:`init` runs at module-import scope, it also runs in
``spawn`` worker subprocesses (which re-import ``PEDS.py``) and when
``evaluate_test_metrics.py`` ``exec``s a run's ``code_snapshot_*.py`` — so the
context is fully initialised everywhere the pipeline runs.
"""

from collections import OrderedDict

import jax
import jax.numpy as jnp
import numpy as np

from NTcode_config_data.config_def import GeometryConfig, BoundarySpec, MatProperties
from NTcode_config_data.config_run import GEO_CYL as GEO
from solvers.NTdiffusion.diffusion_solver import (
    predict_xs, precompute_geometry, xs_layout, bc_to_coeffs,
)

# ─────────────────────────────────────────────────────────────────────────────
# Derived constants / shared state — populated by init()
# ─────────────────────────────────────────────────────────────────────────────
_INITIALIZED = False
_DATA_FILEPATH = None

# Run configuration mirrored from PEDS.py knobs (set every init call)
LOG_DIR = None
XS_DIR = None
PARAM_NAMES = None
PARAM_STRAT_BINS = None
LOG_RATIO_CLIP_LO = None
LOG_RATIO_CLIP_HI = None
BATCH_SIZE = None

# Precomputed geometry / solver constants (set once per data file)
GEO_DATA = None
SLAY = None
N_FLAT_MAX = None
I_MAX_SCAN = None
_DELTA_R_CONST = None
_BC_CONST = None
_vjp_single = None
_Aphi_Fphi_vjp_batch = None
XS_MASK = None
XS_MASK_J = None

# Shared runtime caches (created once, mutated in place)
_GEO_DATA_CACHE = None
_PRESOLVE_CACHE = None


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers (foundational — kept here so init() can use them without
# creating a circular import against physics_solver)
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


class LRUCache(OrderedDict):
    def __init__(self, maxsize):
        super().__init__()
        self.maxsize = maxsize
    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if len(self) > self.maxsize:
            self.popitem(last=False)


def _make_vjp_single(I_max, G, Delta_r, BC, lay):
    """
    Returns a closure that computes Aphi_Fphi_vjp_padded for ONE sample,
    with the static geometry constants (I_max, G, Delta_r, BC, lay) baked in.
    This lets jax.vmap work cleanly: every in_axes argument is per-sample.

    phi2d / vA2d / vF2d must be [G, I_max+1] — see _NT_batch_bwd for unpacking.
    """
    from matrix_JAX_optimized import Aphi_Fphi_vjp_padded
    def _single(xs, S_pad, V_pad, roc_pad, phi2d, vA2d, vF2d, I_valid):
        return Aphi_Fphi_vjp_padded(
            xs, S_pad, V_pad, roc_pad,
            phi2d, vA2d, vF2d,
            I_valid, I_max, G, Delta_r, BC, lay,
        )
    return _single


def build_xs_mask(filepath: str, geo: GeometryConfig) -> np.ndarray:
    """
    Returns shape [nregions, xsperregion] with 1.0 where XS can be nonzero,
    0.0 where physics requires exactly zero.
    Built from the polynomial-regression XS averaged over training data,
    or simply from a known reference sample.
    """
    data = np.load(filepath, allow_pickle=True)
    rawparams = np.array(data['params_raw'], dtype=np.float32)
    ref_params = rawparams[0]  # or average over a few
    ref_xs = predict_xs(update_geo(geo, ref_params))  # shape [3, 12]
    mask = (np.abs(ref_xs) > 1e-10).astype(np.float32)
    return mask  # shape [3, 12]


# ─────────────────────────────────────────────────────────────────────────────
# Initialisation
# ─────────────────────────────────────────────────────────────────────────────
def init(*, data_filepath, log_dir, xs_dir, param_names, param_strat_bins,
         log_ratio_clip_lo, log_ratio_clip_hi, batch_size):
    """Populate the shared context from the knobs defined in PEDS.py.

    The light-weight run configuration (paths, param names, clip bounds) is
    refreshed on every call. The heavy derived constants (which require loading
    the dataset and JIT-compiling the batched VJP) are computed only the first
    time for a given ``data_filepath``.
    """
    global LOG_DIR, XS_DIR, PARAM_NAMES, PARAM_STRAT_BINS
    global LOG_RATIO_CLIP_LO, LOG_RATIO_CLIP_HI, BATCH_SIZE
    global _DATA_FILEPATH, _INITIALIZED
    global GEO_DATA, SLAY, N_FLAT_MAX, I_MAX_SCAN, _DELTA_R_CONST, _BC_CONST
    global _vjp_single, _Aphi_Fphi_vjp_batch, XS_MASK, XS_MASK_J
    global _GEO_DATA_CACHE, _PRESOLVE_CACHE

    LOG_DIR = log_dir
    XS_DIR = xs_dir
    PARAM_NAMES = param_names
    PARAM_STRAT_BINS = param_strat_bins
    LOG_RATIO_CLIP_LO = log_ratio_clip_lo
    LOG_RATIO_CLIP_HI = log_ratio_clip_hi
    BATCH_SIZE = batch_size

    if _INITIALIZED and _DATA_FILEPATH == data_filepath:
        return

    _DATA_FILEPATH = data_filepath

    # ── precomputed geometry constants ───────────────────────────────────────
    GEO_DATA = precompute_geometry(GEO)
    SLAY = xs_layout(GEO.G)

    # ── max flat flux vector size (needed for padded pure_callback) ──────────
    N_FLAT_MAX = _compute_n_flat_max_from_file(data_filepath, GEO)

    # ── Batch-backward constants (derived once, fixed for the whole run) ─────
    # I_MAX_SCAN is the maximum mesh-cell count across the whole dataset.
    # Padded arrays in the batched VJP always have this size, so XLA compiles
    # one kernel for all batches (no recompilation per batch).
    I_MAX_SCAN = N_FLAT_MAX // GEO.G - 1          # e.g. 160//2 - 1 = 79
    _DELTA_R_CONST = float(GEO.mesh_size)         # same for all samples
    _BC_CONST = bc_to_coeffs(GEO.bc)              # same for all samples

    # JIT + vmap over the batch dimension — compiled once, reused every backward
    _vjp_single = _make_vjp_single(I_MAX_SCAN, GEO.G, _DELTA_R_CONST, _BC_CONST, SLAY)
    _Aphi_Fphi_vjp_batch = jax.jit(jax.vmap(_vjp_single))

    XS_MASK = build_xs_mask(data_filepath, GEO)   # [3, 12], fixed for the whole run
    XS_MASK_J = jnp.array(XS_MASK)                # JAX version

    # ── shared runtime caches ────────────────────────────────────────────────
    _GEO_DATA_CACHE = LRUCache(maxsize=batch_size)
    _PRESOLVE_CACHE = {}

    _INITIALIZED = True
