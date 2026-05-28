"""
=============================================================================
PEDS Training Script
=============================================================================
 
Architecture overview:
  problem geometry (p vector) 
        │
        ▼
  GeneratorNN  ──►   XS tensor (num_regions x G x XS types)
        │
        ▼
  Diffusion solver  ──►  k-eff (scalar per sample)
        │
        ▼
  MSE loss vs k-eff from OpenMC
        │
        ▼
  Backprop through solver into NN weights

=============================================================================
"""
import multiprocessing
import os

multiprocessing.set_start_method("spawn", force=True)
os.environ["JAX_PLATFORMS"] = "cpu"   # force CPU

import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np
import pandas as pd
import optax
from flax import nnx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
from contextlib import contextmanager
from collections import defaultdict
import csv
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
ctx = mp.get_context("spawn")   # explicit spawn context

import functools
import threading
_csv_lock = threading.Lock()

from matrix_JAX_optimized import diffusion_setup_jax, Aphi_Fphi_scan
from config_def import GeometryConfig, MaterialSpec, BoundarySpec, BoundaryCondition, MatProperties
from config_run import GEO_CYL as GEO
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from plot_functions.xs_heatmap import plot_xs_heatmap, plot_xs_subplots
from solvers.NTdiffusion.core.MG1D_eigenvalue_nregions import diffusion_setup, region_index
from solvers.NTdiffusion.diffusion_solver import (get_xs_basedon_geo, run_diffusion_solver, is_homogeneous,
    predict_xs, precompute_geometry, xs_layout, build_xs_callables, fn_xs_per_region, bc_to_coeffs, GEOMETRY_CODE)


# ─────────────────────────────────────────────
# SECTION 0: variables initiation and global constants
# ─────────────────────────────────────────────
train_size    = 30
test_size     = 5
batch_size    = 30
epochs        = 40
N_WORKERS = min(
    int(os.environ.get("SLURM_CPUS_PER_TASK", 4)),
    batch_size   # no point having more workers than samples to solve
)
print(f"Using {N_WORKERS} parallel workers for solver")
executor = ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=ctx)
print(f"SLURM_CPUS_PER_TASK = {os.environ.get('SLURM_CPUS_PER_TASK', 'NOT SET')}", flush=True)


# Redirect all print output to a log file
loggy_file = open("training_log2.txt", "a", buffering=1)  # buffering=1 = write every line immediately
sys.stdout = loggy_file
sys.stderr = loggy_file
#print("JAX devices:", jax.devices())
#print("Backend:", jax.default_backend())

# predict the XS with the polynomial reg built in the other code
# acts as a first guess???
# TODO remove this and thefunction used 
XS_BASELINE = get_xs_basedon_geo(GEO)  # HERE I WILL ADD THE NN CONTRIB
XS_BASELINE = jnp.array(XS_BASELINE, dtype=jnp.float32)   # convert to JAX
# ----------------------
GEO_DATA = precompute_geometry(GEO)
# Side channel to store the geometry information for each run — indexed by sample position in batch
_GEO_DATA_CACHE = {}
SLAY      = xs_layout(GEO.G) # Index slices for each XS type, given G groups

# Fixed reference scale for the NN output — computed ONCE from the default geometry.
# The NN learns log-ratio offsets relative to this scale; exp(0) * XS_SCALE = XS_SCALE at init.
XS_SCALE = jnp.array(get_xs_basedon_geo(GEO), dtype=jnp.float32)  # shape (3, 12), constant

_XS_CACHE: dict = {}
HEATMAP_INTERVAL = 2
XS_HEATMAP_SNAPSHOTS = []
XS_HEATMAP_LABELS = []
_xs_baseline_ref = [None]        # list-box so inner assignment doesn't shadow
_snap_final      = None

from collections import OrderedDict

# ─────────────────────────────────────────────
# SECTION 0: Solutions for better runspeed and memory efficiency
# ─────────────────────────────────────────────
MAX_CACHE_SIZE = 256   # tune based on your RAM

class LRUCache(OrderedDict):
    def __init__(self, maxsize):
        super().__init__()
        self.maxsize = maxsize
    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if len(self) > self.maxsize:
            self.popitem(last=False)   # evict oldest entry

_GEO_DATA_CACHE = LRUCache(maxsize=MAX_CACHE_SIZE)

_PRESOLVE_CACHE: dict = {}   # stores (k, phi_fwd, phi_adj) from parallel pre-solve
def _solve_sample_worker(args):
    """
    Runs in a separate process — no JAX, pure NumPy/SciPy.
    args: (i, xs_np, params_np, sample_id_int)
    Returns: (i, k, phi_fwd, phi_adj, geo_data)
    """
    t0 = time.perf_counter()
    pid = os.getpid()

    i, xs_np, params_np, sample_id_int, epoch = args
    geo_i    = update_geo(GEO, params_np)
    geo_data = precompute_geometry(geo_i)
    
    k, phi_fwd, phi_adj = _run_NT_solver(
        xs_np,
        params_np,
        np.array([sample_id_int])
    )
    elapsed = time.perf_counter() - t0
    #print(f"  worker pid={pid} sample={i} took {elapsed:.2f}s", flush=True)
    return i, k, phi_fwd, phi_adj, geo_data, epoch


_EXECUTOR = ProcessPoolExecutor(max_workers=N_WORKERS)

def _run_one_sample(args):
    """Top-level (picklable) wrapper for one sample."""
    xs_np, params_np, sample_id_int = args
    k, phi_fwd, phi_adj = _run_NT_solver(
        xs_np,
        jnp.array(params_np),
        jnp.array([sample_id_int])
    )
    return k, phi_fwd, phi_adj

def _run_batch_parallel(xs_batch_np, params_batch_np, batch_size):
    """Run all samples in the batch concurrently."""
    args = [
        (xs_batch_np[i], params_batch_np[i], i)
        for i in range(batch_size)
    ]
    results = list(_EXECUTOR.map(_run_one_sample, args))
    return results

def _run_batch_parallel_callback(xs_batch, params_batch):
    # xs_batch and params_batch are NOW real numpy arrays (not tracers!)
    batch_size = xs_batch.shape[0]
    args = [(i, xs_batch[i], params_batch[i], i) for i in range(batch_size)]
    results = list(EXECUTOR.map(solve_sample_worker, args))
    keffs = np.array([r[1] for r in results], dtype=np.float32)
    return keffs

# ─────────────────────────────────────────────
# SECTION 1: Physics Solver
# ─────────────────────────────────────────────
def _run_NT_solver(xs_tensor, params_raw_single, sample_id):
    """ NumPy function — Neutron diffusion solver for the steady-state NT equation.
    GEO is captured from the cached _GEO_DATA_CACHE for the sample."""
    
    geo_i    = update_geo(GEO, np.array(params_raw_single))
    geo_data = precompute_geometry(geo_i)       # per-sample geometry
    
    sid = int(sample_id[0])
    _GEO_DATA_CACHE[sid] = geo_data

    if sid in _PRESOLVE_CACHE:
        k, phi_fwd, phi_adj, geo_data_pre = _PRESOLVE_CACHE.pop(sid)
        _GEO_DATA_CACHE[sid] = geo_data_pre
        return k, phi_fwd, phi_adj

    R      = geo_i.boundaries[-1].radius
    I      = int(R / geo_i.mesh_size)
    G      = geo_i.G

    with timer("  solver: eigenvalue solve (fwd)", verbose=False):
        k, phi_fwd, phi_adj= run_diffusion_solver(xs_tensor, geo_i)
    
    #print(f" DOUBLECHECKINGGGG [solver] k = {float(k):.6f}, ||phi_fwd||={float(jnp.linalg.norm(phi_fwd)):.6e}, " 
    #      f"||phi_adj||={float(jnp.linalg.norm(phi_adj)):.6e}, ")
    
    # flatten phi: from [G, I] to [G*(I+1)] to match matrix size
    N_flat = geo_i.G * (I + 1)
    phi_fwd_flat = np.zeros(N_flat, dtype=np.float64)
    phi_adj_flat = np.zeros(N_flat, dtype=np.float64)
    for g in range(geo_i.G):
        phi_fwd_flat[g*(I+1) : g*(I+1)+I] = phi_fwd[g, :]
        phi_adj_flat[g*(I+1) : g*(I+1)+I] = phi_adj[g, :]

    # ── Biorthonormalization ──────────────────────────────────────────
    # Enforce ⟨φ†, F·φ⟩ = k²  so the VJP denominator = 1.0 exactly.
    # This removes the arbitrary scaling ambiguity from inverse_power.

    r_divisions   = [b.radius for b in geo_i.boundaries[:-1]] \
                    if not is_homogeneous(geo_i) else []
    BC_coeffs     = bc_to_coeffs(geo_i.bc)               # [A, B, C]
    geometry_code = GEOMETRY_CODE[geo_i.geometry]         # 0/1/2

    D_fn, Sa_fn, nF_fn, Ss_fn, chi_fn = build_xs_callables(np.array(xs_tensor), geo_i)
    _, A_real, F_real = diffusion_setup(R, I, geo_i.G, r_divisions,
                                D_fn, Sa_fn, nF_fn, Ss_fn, chi_fn,
                                BC_coeffs, geometry_code)
    Fphi = F_real @ phi_fwd_flat
    biorth = phi_adj_flat @ Fphi   # scalar ⟨φ†, F·φ⟩

    # Step 3: rescale φ† so that ⟨φ†, F·φ⟩ = k²
    # This makes denominator = (1/k²) * k² = 1.0
    phi_adj_flat *= 1.0 / biorth 
    #print(f"bwd sanity: phiadj·F·phi = {float(phi_adj_flat @ Fphi):.6f}  (should be ~1.0)")

    return np.float32(k), phi_fwd_flat.astype(np.float32), \
           phi_adj_flat.astype(np.float32)


# this is the function that runs when gradients are being computed
def _NTdiff_fwd(xs_tensor, params_raw_single, sample_id):
    """Forward pass: run the solver and save the residuals (input, output) for the backward.
    Args:
        XS_tensor: [batch, num_regions x M] neutron cross-sections (NN prediction), 
        to be assigned to specific cells in A,F matrixes 
    Returns:
        k_fwd: scalar, dominant eigenvalue from forward solve
        phi: [batch, N]  — steady-state neutron flux evolution over space
    """
    geo_i    = update_geo(GEO, np.array(params_raw_single))

    R      = geo_i.boundaries[-1].radius
    I      = int(R / geo_i.mesh_size)

    N_flat = geo_i.G * (I + 1)   # total size of flux vector

    keff, phi_fwd, phi_adj = jax.pure_callback(
        _run_NT_solver,
        (
            jax.ShapeDtypeStruct((),        jnp.float32),
            jax.ShapeDtypeStruct((N_flat,), jnp.float32),
            jax.ShapeDtypeStruct((N_flat,), jnp.float32),
        ),
        xs_tensor,  params_raw_single, sample_id
    )  
    
    geo_data = _GEO_DATA_CACHE[int(sample_id[0])]
    _, Fphi = Aphi_Fphi_scan(xs_tensor, geo_data, SLAY, phi_fwd)

    residuals = (xs_tensor, keff, phi_fwd, phi_adj, Fphi, sample_id)        
    # primal must match exactly what NTdiff_solver returns
    return keff, residuals


def _NTdiff_bwd(residuals, g):
    """
    Custom backward pass.
    Args:
        residuals: the residuals saved by _NTdiff_fwd
        g:   scalar, dL/dk arriving from the loss function
             (= k - k_ref  when loss is MSE)
    Returns:
        dL/d(xs_tensor), shape [batch, M]
    """
    xs_tensor, k, phi_fwd, phi_adj, Fphi, sample_id = residuals
    geo_data = _GEO_DATA_CACHE[int(sample_id[0])]
    dL_dk = g  

    # --- define the two matrix-vector functions ---
    # TODO phi_fwd is captured from outer scope (treated as constant)    
    def AF_phi(xs):
        with timer("  (Matrixes building part)", verbose=False):
            return Aphi_Fphi_scan(xs, geo_data, SLAY, phi_fwd)  # returns (Aphi, Fphi)
        
    with timer("  bwd: vjp A@phi, F@phi", verbose=False):
        _, vjp_fn = jax.vjp(AF_phi, xs_tensor)
        numerator, = vjp_fn((phi_adj, -(1.0 / k) * phi_adj))

    with timer("  bwd: denominator", verbose=False):
        phiadj_F_phi = phi_adj @ Fphi  # this is a scalar (dot product of two vectors)
        denominator =  (1/k**2) * phiadj_F_phi
    
    dk_dp = - numerator / denominator
    dL_dxs_tensor = dL_dk * dk_dp

    return (dL_dxs_tensor, None, None)  # only return gradients for xs_tensor; the other two inputs have no gradients

# this function represents what the solver does when called outside of differentiation context.
@jax.custom_vjp
def NTdiff_solver(xs_tensor, params_raw_single, sample_id):
    keff, _ = _NTdiff_fwd(xs_tensor, params_raw_single, sample_id) 
    return keff    

NTdiff_solver.defvjp(_NTdiff_fwd, _NTdiff_bwd)

# ─────────────────────────────────────────────
# SECTION 2: Utils: timings, geometry updates, sanity checks
# ─────────────────────────────────────────────

# Global registry — stores all measured times
_TIMINGS = defaultdict(list)

@contextmanager
def timer(label: str, verbose: bool = True):
    """
    Context manager that measures wall time and stores it.
    Usage:  with timer("forward pass"):
                result = model(x)
    """
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    _TIMINGS[label].append(elapsed)
    if verbose:
        print(f"  ⏱  {label:<35s} {elapsed*1000:.1f} ms")

def print_timing_report():
    """Print a summary of all measured times after training."""
    print("\n" + "="*60)
    print(f"{'TIMING REPORT':^60}")
    print("="*60)
    print(f"{'Step':<35s} {'calls':>6s} {'total(s)':>10s} {'mean(ms)':>10s} {'min(ms)':>10s}")
    print("-"*60)
    for label, times in sorted(_TIMINGS.items()):
        total   = sum(times)
        mean_ms = (total / len(times)) * 1000
        min_ms  = min(times) * 1000
        print(f"{label:<35s} {len(times):>6d} {total:>10.2f} {mean_ms:>10.1f} {min_ms:>10.1f}")
    print("="*60)


def compute_batch_baselines(params_raw: np.ndarray, geo: GeometryConfig) -> jnp.ndarray:
    """
    params_raw: [batch, 6] — un-normalized geometry parameters
    Returns: [batch, 3, 12] — regression-predicted XS for each sample
    """
    baselines = []
    for i in range(params_raw.shape[0]):
        # Build a modified GEO for this sample's raw parameters vector
        geo_i = update_geo(geo, params_raw[i]) #
        xs_i  = predict_xs(geo_i)        # regression function
        baselines.append(xs_i)

    return jnp.array(np.stack(baselines), dtype=jnp.float32)  # [batch, 3, 12]

def update_geo(geo: GeometryConfig, params_raw: np.ndarray) -> GeometryConfig:
    """
    Build a new GeometryConfig from a single raw parameter vector.
    params_raw: [6] — [b4c_outer_r, cr_fraction, fuel_outer_r, enrichment, f_mod, water_outer_r]
    """
    new_boundaries = (
        BoundarySpec(name='CR_outer',         radius=float(params_raw[0])),
        BoundarySpec(name='core_outer',       radius=float(params_raw[2])),
        BoundarySpec(name='moderator_outer',  radius=float(params_raw[5])),
    )
    new_mat = MatProperties(
        cr_fraction = float(params_raw[1]),
        enrichment  = float(params_raw[3]),
        f_mod       = float(params_raw[4]),
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

def physics_sanity_check(xs_baseline, params_raw_single, sample_id):
    """
    Check gradient signs against physical intuition.
    Run these BEFORE finite differences — they're faster to interpret.
    """
    xs = jnp.array(xs_baseline, dtype=jnp.float32)
    _, vjp_fn = jax.vjp(NTdiff_solver, xs, params_raw_single, sample_id)
    grad = np.array(vjp_fn(jnp.ones(()))[0])   # shape (3, 12)

    lay = xs_layout(2)   # G=2
    print("=== Physics Sanity Check ===")

    for reg_idx, reg_name in enumerate(['b4c_rod', 'fuel_annulus', 'water']):
        d_keff_d_D      = grad[reg_idx, lay['D']]
        d_keff_d_Siga   = grad[reg_idx, lay['Sigma_a']]
        d_keff_d_nuSigf = grad[reg_idx, lay['nuSigma_f']]

        print(f"\nRegion: {reg_name}")
        print(f"  ∂keff/∂D         = {d_keff_d_D}")
        print(f"  ∂keff/∂Σ_a       = {d_keff_d_Siga}")
        print(f"  ∂keff/∂νΣ_f      = {d_keff_d_nuSigf}")

def check_eigenvalue_residual_vs_solver(xs_tensor, geo_i, phi_fwd_flat, k):
    """
    Build A and B using the REAL diffusion_setup from MG1D_eigenvalue_nregions,
    then check if Aphi_Fphi_scan gives the same result.
    """

    R = geo_i.boundaries[-1].radius
    I = int(R / geo_i.mesh_size)
    G = geo_i.G
    BC_coeffs = bc_to_coeffs(geo_i.bc)
    geometry_code = GEOMETRY_CODE[geo_i.geometry]
    r_divisions = [b.radius for b in geo_i.boundaries[:-1]]

    # Build XS callables the same way the solver does
    D_fn, Sigma_a_fn, nuSigma_f_fn, Sigma_s_fn, chi_fn = \
        build_xs_callables(np.array(xs_tensor), geo_i)

    # Get the REAL A and B matrices from the solver
    _, A_real, B_real = diffusion_setup(
        R, I, G, r_divisions,
        D_fn, Sigma_a_fn, nuSigma_f_fn, Sigma_s_fn, chi_fn,
        BC_coeffs, geometry_code
    )

    phi = np.array(phi_fwd_flat)  # convert from JAX to numpy

    # Reference: what the REAL solver says A·φ and F·φ are
    Aphi_real = A_real @ phi
    Fphi_real = B_real @ phi

    # What Aphi_Fphi_scan computes
    geo_data = _GEO_DATA_CACHE[0]
    Aphi_scan, Fphi_scan = Aphi_Fphi_scan(xs_tensor, geo_data, SLAY, jnp.array(phi))

    diff_A = np.abs(np.array(Aphi_scan) - Aphi_real)
    diff_F = np.abs(np.array(Fphi_scan) - Fphi_real)

    print("=== Scan vs Real Solver Matrix Check ===")
    print(f"  ||A_scan·φ - A_real·φ||_inf  = {diff_A.max():.3e}  at flat idx {diff_A.argmax()}")
    print(f"  ||F_scan·φ - F_real·φ||_inf  = {diff_F.max():.3e}  at flat idx {diff_F.argmax()}")

    # Per-group breakdown
    diff_A_2d = diff_A.reshape(G, I+1)
    diff_F_2d = diff_F.reshape(G, I+1)
    print("\n  Per-group A mismatch:")
    for g in range(G):
        i_w = int(diff_A_2d[g].argmax())
        print(f"    Group {g+1}: max={diff_A_2d[g,i_w]:.3e} at cell i={i_w}  "
              f"(i=0:{i_w==0}, i=I:{i_w==I})")

    print("\n  Per-group F mismatch:")
    for g in range(G):
        i_w = int(diff_F_2d[g].argmax())
        print(f"    Group {g+1}: max={diff_F_2d[g,i_w]:.3e} at cell i={i_w}  "
              f"(i=0:{i_w==0}, i=I:{i_w==I})")

    # Also check eigenvalue residual with the REAL matrices
    residual_real = Aphi_real - (1.0/k) * Fphi_real
    print(f"\n  ||A_real·φ - (1/k)·F_real·φ||_inf = {np.abs(residual_real).max():.3e}")
    print(f"  (should be ≈ solver tolerance ~1e-8)")


def diagnose_scan_vs_real_cellwise(xs_tensor, geo_i, phi_fwd_flat, k):
    """
    For each cell where A_scan·φ ≠ A_real·φ, decompose the error
    into which term is responsible: diagonal, left off-diag, right off-diag,
    scatter-in, or scatter-out.
    """
    from solvers.NTdiffusion.core.MG1D_eigenvalue_nregions import diffusion_setup, create_grid

    R = geo_i.boundaries[-1].radius
    I = int(R / geo_i.mesh_size)
    G = geo_i.G
    Delta_r = geo_i.mesh_size
    BC_coeffs = bc_to_coeffs(geo_i.bc)
    geometry_code = GEOMETRY_CODE[geo_i.geometry]
    r_divisions = [b.radius for b in geo_i.boundaries[:-1]]

    D_fn, Sa_fn, nF_fn, Ss_fn, chi_fn = build_xs_callables(np.array(xs_tensor), geo_i)
    _, A_real, _ = diffusion_setup(R, I, G, r_divisions,
                                    D_fn, Sa_fn, nF_fn, Ss_fn, chi_fn,
                                    BC_coeffs, geometry_code)

    phi = np.array(phi_fwd_flat)

    phi2d_check = phi.reshape(G, I+1)
    print(f"\n=== Flux Sanity Check ===")
    print(f"  phi_fwd_flat norm       = {np.linalg.norm(phi):.6e}")
    print(f"  phi2d[g=0, i=0:5]      = {phi2d_check[0, :5]}")
    print(f"  phi2d[g=1, i=0:5]      = {phi2d_check[1, :5]}")
    print(f"  phi2d[g=0, ghost i=I]  = {phi2d_check[0, I]:.6e}  (expect 0.0)")
    print(f"  phi2d[g=1, ghost i=I]  = {phi2d_check[1, I]:.6e}  (expect 0.0)")
    print(f"  max(phi)               = {phi.max():.6e}")
    print(f"  min(phi)               = {phi.min():.6e}")

    Aphi_real = A_real @ phi

    geo_data = _GEO_DATA_CACHE[0]
    Aphi_scan = np.array(Aphi_Fphi_scan(xs_tensor, geo_data, SLAY, jnp.array(phi))[0])

    diff_2d     = (Aphi_scan - Aphi_real).reshape(G, I+1)
    Aphi_real_2d = Aphi_real.reshape(G, I+1)
    phi2d        = phi.reshape(G, I+1)

    # Recompute geometry arrays exactly as the real solver does
    _, centers, edges = create_grid(R, I)
    import jax.numpy as jnp2
    S = 2.0 * np.pi * edges          # cylindrical surface areas
    V = np.pi * (edges[1:]**2 - edges[:-1]**2)  # cylindrical volumes

    region_of_cell = geo_data['region_of_cell']

    print(f"\n=== Cell-by-Cell A-matrix Decomposition ===")
    print(f"  Showing cells where |error| > 1e-6\n")

    for g in range(G):
        print(f"\n  --- Group {g+1} ---")
        for i in range(I):
            err = abs(diff_2d[g, i])
            if err < 1e-3:
                continue

            reg   = int(region_of_cell[i])
            reg_n = int(region_of_cell[min(i+1, I-1)])
            xs    = np.array(xs_tensor)

            D_g      = float(xs[reg,   SLAY['D']][g])
            D_g_next = float(xs[reg_n, SLAY['D']][g])
            Dplus    = 2*D_g*D_g_next / (D_g + D_g_next + 1e-30)
            Dminus   = 0.0
            if i > 0:
                reg_p = int(region_of_cell[i-1])
                D_g_prev = float(xs[reg_p, SLAY['D']][g])
                Dminus = 2*D_g_prev*D_g / (D_g_prev + D_g + 1e-30)

            Sig_a    = float(xs[reg, SLAY['Sigma_a']][g])
            Sig_s_mat = xs[reg, SLAY['Sigma_s']].reshape(G, G)
            Sig_s_out = sum(Sig_s_mat[g, gp] for gp in range(G) if gp != g)

            # Manual reconstruction of each term
            diag_leak  = Dplus * S[i+1] / (Delta_r * V[i])
            diag_total = diag_leak + Sig_a + Sig_s_out

            term_diag   = diag_total * phi2d[g, i]
            term_right  = -Dplus * S[i+1] / (Delta_r * V[i]) * phi2d[g, i+1]
            term_left   = Dminus * S[i] / (Delta_r * V[i]) * (phi2d[g,i] - phi2d[g, i-1]) if i > 0 else 0.0
            term_scatin = -sum(Sig_s_mat[gp, g] * phi2d[gp, i]
                               for gp in range(G) if gp != g)

            manual_sum  = term_diag + term_right + term_left + term_scatin

            print(f"  cell i={i:3d}  reg={reg}  scan={Aphi_scan[g*(I+1)+i]:.6e}"
                  f"  real={Aphi_real_2d[g,i]:.6e}  err={diff_2d[g,i]:.3e}")
            print(f"    manual reconstruction = {manual_sum:.6e}")
            print(f"    terms: diag={term_diag:.4e}  right={term_right:.4e}"
                  f"  left={term_left:.4e}  scatin={term_scatin:.4e}")
            print(f"    Dplus={Dplus:.4f}  Dminus={Dminus:.4f}"
                  f"  S[i]={S[i]:.4f}  S[i+1]={S[i+1]:.4f}  V[i]={V[i]:.4f}")
            print("finitooooooooooooo")


def finite_difference_gradient_check(xs_tensor, params_raw_single, sample_id,
                                      rel_eps=1e-3, abs_eps_floor=1e-6):
    """
    Gradient check with per-component relative epsilon:
        ε_i = rel_eps * |xs_i|  (but at least abs_eps_floor)
    
    This keeps the perturbation at ~0.1% of each XS value,
    avoiding both truncation error (ε too large) and
    cancellation error (ε too small for float32).
    """
    # Get k, phi_fwd, phi_adj ONCE — freeze them for both analytic and FD
    xs = jnp.array(xs_tensor, dtype=jnp.float32)
    flat_xs = xs.flatten()
    n = flat_xs.shape[0]        
    _ = NTdiff_solver(xs, params_raw_single, sample_id)  # warms cache
    geo_data = _GEO_DATA_CACHE[int(sample_id[0])]

    # Retrieve the frozen fluxes from the last solver call
    _, (_, k_frozen, phi_fwd_frozen, phi_adj_frozen, _, _) = \
        _NTdiff_fwd(xs, params_raw_single, sample_id)

    # The same scalar function used in _NTdiff_bwd — phi frozen as constants
    def scalar_fn(xs_):
        Aphi, Fphi = Aphi_Fphi_scan(xs_, geo_data, SLAY, phi_fwd_frozen)
        return jnp.dot(phi_adj_frozen, Aphi) - (1.0 / k_frozen) * jnp.dot(phi_adj_frozen, Fphi)

    # --- Analytic gradient via custom VJP ---
    _, vjp_fn = jax.vjp(NTdiff_solver, xs, params_raw_single, sample_id)
    grad_analytic = np.array(vjp_fn(jnp.ones(()))[0]).flatten()

    # --- Numerical gradient with per-component ε ---
    grad_fd     = np.zeros(n)
    eps_used    = np.zeros(n)  # log what ε was actually used
    skipped     = []

    for i in range(n):
        xi = float(flat_xs[i])
        print(f"FD grad check: component {i}/{n}, value={xi:.4e}", end="")

        # Skip truly zero components (non-physical — no gradient possible)
        if abs(xi) < 1e-10:
            skipped.append(i)
            continue

        # Relative epsilon, floored to avoid float32 cancellation
        eps_i = max(rel_eps * abs(xi), abs_eps_floor)
        eps_used[i] = eps_i

        xs_plus  = flat_xs.at[i].add(+eps_i).reshape(xs.shape)
        xs_minus = flat_xs.at[i].add(-eps_i).reshape(xs.shape)

        k_plus  = float(scalar_fn(xs_plus))
        k_minus = float(scalar_fn(xs_minus))

        grad_fd[i] = (k_plus - k_minus) / (2.0 * eps_i)

    # --- Compare only non-skipped components ---
    active = np.array([i for i in range(n) if i not in skipped])
    ga = grad_analytic[active]
    gf = grad_fd[active]

    abs_err = np.abs(ga - gf)
    rel_err = abs_err / (np.abs(gf) + 1e-30)

    # --- Pretty print per-component table ---
    lay = xs_layout(GEO.G)  # for label lookup
    xs_names = []
    for reg in range(xs.shape[0]):
        for name, sl in lay.items():
            for g in range(GEO.G):
                xs_names.append(f"reg{reg}_{name}_g{g+1}")


    def make_xs_label(i, xs_shape, lay, G):
        """Convert flat index i → human-readable XS name."""
        n_per_reg = xs_shape[1]
        reg = i // n_per_reg
        offset = i % n_per_reg
        for name, sl in lay.items():
            indices = list(range(*sl.indices(n_per_reg)))
            if offset in indices:
                g = indices.index(offset)
                return f"reg{reg}_{name}_g{g+1}"
        return f"reg{reg}_idx{offset}"

    print(f"\n{'='*65}")
    print(f"  Gradient Check — rel_eps={rel_eps:.0e}, floor={abs_eps_floor:.0e}")
    print(f"{'='*65}")
    print(f"  {'Component':<25} {'analytic':>12} {'FD':>12} {'|err|':>10} {'rel_err':>10}")
    print(f"  {'-'*65}")
    for j, i in enumerate(active):
        ref_scale = max(abs(a), abs(f))
        if ref_scale < 1e-8:
            flag = ""                  # both are negligible noise — genuinely skip
        elif abs(f) < 1e-8 and abs(a) >= 1e-8:
            flag = " FD_UNRESOLVED"   # float32 can't resolve it — not a scan bug
        elif abs(a) < 1e-8 and abs(f) >= 1e-8:
            flag = " AD_ZERO !"       # AD says zero, FD disagrees — real concern
        elif rel > 0.05:
            flag = " !"               # normal large relative error
        else:
            flag = ""

        label = make_xs_label(i, xs.shape, lay, GEO.G)
        print(f"  {label:<25} {ga[j]:>12.4e} {gf[j]:>12.4e} "
            f"{abs_err[j]:>10.2e} {rel_err[j]:>10.2e}{flag}")

    print(f"\n  Skipped (zero XS): {skipped}")
    print(f"  Max  |rel err| : {rel_err.max():.3e}")
    print(f"  Mean |rel err| : {rel_err.mean():.3e}")
    print(f"  Components > 5% error: {(rel_err > 0.05).sum()} / {len(active)}")
    print(f"  (target: < 5% for float32 with converged solver)")

    return grad_analytic.reshape(xs.shape), grad_fd.reshape(xs.shape), rel_err
    

def diagnose_scan_autodiff(xs_tensor, geo_data, phi_fwd, phi_adj, k):
    """
    FD vs vjp check on AphiFphiscan directly.
    
    Epsilon strategy:
      eps_i = max(rel_eps * |xs_i|, abs_floor_factor * eps_machine)
    
    - rel_eps: relative step size (default 1e-2 for float32; use 1e-4 for float64)
    - abs_floor_factor: floor = this * machine_epsilon of the array dtype.
      Avoids the fixed 1e-6 which was too large for small XS and too small
      relative to float32's actual noise floor.
    """
    print("\n" + "="*65)
    print("=== Scan Autodiff Diagnostic (FD vs vjp on scan directly) ===")
    print("="*65)

    # --- Determine machine epsilon from array dtype ---
    rel_eps = 1e-2
    abs_floor_factor= 1e2
    xs_dtype = jnp.array(xs_tensor).dtype
    eps_machine = float(jnp.finfo(xs_dtype).eps)   # ~1.19e-7 for float32
    grad_noise_floor = eps_machine * 1e2  # ~1.19e-5 for float32
    abs_floor = abs_floor_factor * eps_machine       # ~1.19e-5 for float32

    # The scalar function we differentiate — mirrors exactly what _NTdiff_bwd computes
    def scalar_fn(xs):
        Aphi, Fphi = Aphi_Fphi_scan(xs, geo_data, SLAY, phi_fwd)
        return jnp.dot(phi_adj, Aphi) - (1.0 / k) * jnp.dot(phi_adj, Fphi)

    # --- Analytic gradient via jax.grad ---
    grad_analytic = jax.grad(scalar_fn)(xs_tensor)

    # --- Finite difference gradient ---
    xs_np = np.array(xs_tensor)
    grad_fd = np.zeros_like(xs_np)
    eps_used = np.zeros_like(xs_np) 

    for r in range(xs_np.shape[0]):
        for m in range(xs_np.shape[1]):
            val = xs_np[r, m]
            if abs(val) < abs_floor:
                continue  # skip true zeros — FD is meaningless there
            eps_i = max(rel_eps * abs(val), abs_floor)
            eps_used[r, m] = eps_i       

            xs_p = xs_np.copy(); xs_p[r, m] += eps_i
            xs_m = xs_np.copy(); xs_m[r, m] -= eps_i
            fp = float(scalar_fn(jnp.array(xs_p)))
            fm = float(scalar_fn(jnp.array(xs_m)))
            grad_fd[r, m] = (fp - fm) / (2 * eps_i)

    grad_fd = jnp.array(grad_fd)

    # --- Report ---
    G = geo_data['G']
    reg_names = [r['name'] for r in geo_data['regions']] if 'regions' in geo_data \
                else [f'reg{r}' for r in range(xs_np.shape[0])]
    xs_type_names = []
    for name, sl in SLAY.items():
        size = (sl.stop - sl.start) if isinstance(sl, slice) else 1
        for g in range(size):
            xs_type_names.append(f"{name}_g{g+1}")
    
    print(f"  dtype={xs_dtype}, eps_machine={eps_machine:.2e}, abs_floor={abs_floor:.2e}")
    print(f"  rel_eps={rel_eps:.0e}")
    print(f"\n  {'Component':<20} {'xs_val':>10s} {'eps_used':>10s} {'analytic':>12} {'FD':>12} {'rel_err':>10}")
    print(f"  {'-'*68}")

    all_rel = []
    for r, rname in enumerate(reg_names):
        for m, xname in enumerate(xs_type_names):
            a = float(grad_analytic[r, m])
            f = float(grad_fd[r, m])
            """ if (abs(f) < 1e-6 or abs(f) == 0) or (abs(a) < 1e-6 or abs(a) == 0):
                continue
            if abs(xs_np[r, m]) < abs_floor:
                continue """
            rel = abs(a - f) / (abs(f) + 1e-30)
            
            ref_scale = max(abs(a), abs(f))
            if ref_scale <= grad_noise_floor:
                flag = "noise"  # both are float32 noise — skip
                #continue    # <-- add this to exclude from allrel
            elif abs(f) < grad_noise_floor or abs(f) == 0: # and abs(a) >= grad_noise_floor:
                flag = " FD_UNRESOLVED"     # AD sees it, float32 FD cannot
            elif abs(a) < grad_noise_floor or abs(a) == 0: # and abs(f) >= grad_noise_floor:
                flag = " AD_ZERO !"         # AD blind to something FD finds
            elif rel > 0.05:
                flag = " !"
            else:
                flag = ""
                all_rel.append(rel)

                
            eps_i = float(eps_used[r, m])
            xs_val = float(xs_np[r, m])
            print(f"  {rname+xname:20s}  {xs_val:10.3e} {eps_i:10.2e} {a:12.4e} {f:12.4e} {rel:10.2e} {flag}")
            
    if all_rel:
        print(f"\n  Max rel error  : {max(all_rel):.3e}")
        mean_re = sum(all_rel)/len(all_rel)
        print(f"  Mean rel error : {mean_re:.3e} (target < 1e-2 in relevant)")
        if mean_re < 1e-2:
            print("  ✅ Scan autodiff is CORRECT — bug is elsewhere in the pipeline")
        else:
            print("  ❌ Scan autodiff is WRONG — bug is inside Aphi_Fphi_scan")
    print("="*65 + "\n")


def diagnose_full_vjp(xs_tensor, params_raw_single, sample_id, rel_eps=1e-2, abs_floor=1e-5):
    """
    FD vs. custom VJP check on NTdiffsolver end-to-end.
    
    Assumes diagnose_scan_autodiff already passed — so any failure here
    is definitively inside NTdiffbwd (the custom VJP formula), NOT in AphiFphiscan.

    Uses the same epsilon strategy as diagnose_scan_autodiff for consistency.
    """
    print("=" * 65)
    print("Full Custom VJP Diagnostic — FD vs. NTdiffbwd")
    print("=" * 65)

    xs = jnp.array(xs_tensor, dtype=jnp.float32)
    xs_np = np.array(xs)

    # --- Step 1: warm up the cache and get frozen k ---
    # NTdiffsolver uses pure_callback internally, so k comes from the cache
    NTdiff_solver(xs, params_raw_single, sample_id)   # warm cache
    _, (_, k_frozen, phi_fwd_frozen, phi_adj_frozen, Fphi_frozen, _) = _NTdiff_fwd(xs, params_raw_single, sample_id)
    geodata = _GEO_DATA_CACHE[int(sample_id[0])]
    print(f"  frozen k = {k_frozen:.6f}")

    # --- Step 2: analytic gradient via custom VJP (triggers NTdiffbwd) ---
    # jax.vjp with g=1.0 gives dL/dxs where L = k
    _, vjp_fn = jax.vjp(NTdiff_solver, xs, params_raw_single, sample_id)
    grad_analytic = np.array(vjp_fn(jnp.ones(()))[0])  

    # --- Step 3: finite difference on NTdiffsolver directly ---
    # We call the solver as a scalar function: xs -> k
    def scalar_fn1(xs_in):
        Aphi, Fphi = Aphi_Fphi_scan(xs_in, geodata, SLAY, phi_fwd_frozen)
        return float(jnp.dot(phiadj_frozen, Aphi - (1.0 / k_frozen) * Fphi))

    _, Fphi_frozen = Aphi_Fphi_scan(xs, geodata, SLAY, phi_fwd_frozen)
    phi_adj_Fphi = jnp.dot(phi_adj_frozen, Fphi_frozen)
    denom = (1.0 / k_frozen**2) * phi_adj_Fphi 
    
    def scalar_fn(xs):
        Aphi, Fphi = Aphi_Fphi_scan(xs, geodata, SLAY, phi_fwd_frozen)
        num = jnp.dot(phi_adj_frozen, Aphi - (1.0 / k_frozen) * Fphi)
        return float(-num / denom) 
        
    grad_fd = np.zeros_like(xs_np)
    eps_used = np.zeros_like(xs_np)

    for r in range(xs_np.shape[0]):
        for m in range(xs_np.shape[1]):
            val = xs_np[r, m]
            if abs(val) < abs_floor:
                continue  # skip true zeros — FD is meaningless
            epsi = max(rel_eps * abs(val), abs_floor)
            eps_used[r, m] = epsi
            xsp = xs_np.copy(); xsp[r, m] += epsi
            xsm = xs_np.copy(); xsm[r, m] -= epsi
            grad_fd[r, m] = (scalar_fn(jnp.array(xsp)) - scalar_fn(jnp.array(xsm))) / (2 * epsi)

    # --- Step 4: report ---
    xs_type_names = []
    for name, sl in SLAY.items():
        size = sl.stop - sl.start if isinstance(sl, slice) else 1
        for g in range(size):
            xs_type_names.append(f"{name}g{g+1}")

    reg_names = [f"reg{r}" for r in range(xs_np.shape[0])]
    grad_noise_floor = float(jnp.finfo(jnp.float32).eps) * 1e2  # ~1.19e-5

    print(f"\n  {'Component':<22} {'xs val':>10} {'eps':>10} {'analytic':>12} {'FD':>12} {'rel err':>10}")
    print(f"  {'-'*70}")

    all_rel = []
    for r, r_name in enumerate(reg_names):
        for m, x_name in enumerate(xs_type_names):
            a = float(grad_analytic[r, m])
            f = float(grad_fd[r, m])
            ref_scale = max(abs(a), abs(f))
            rel = abs(a - f) / (abs(f) + 1e-30)
            if ref_scale < grad_noise_floor:
                flag = "  noise!"  
            elif abs(f) < grad_noise_floor or abs(f) == 0 and abs(a) >= grad_noise_floor:
                flag = " FD_UNRESOLVED"     # AD sees it, float32 FD cannot
            elif abs(a) < grad_noise_floor or abs(a) == 0 and abs(f) >= grad_noise_floor:
                flag = " AD_ZERO !" 
            elif rel > 0.05:
                flag = "  !"
            else:
                flag = ""
                all_rel.append(rel)

            label = f"{r_name}/{x_name}"
            eps_i = float(eps_used[r, m])
            xs_val = float(xs_np[r, m])
            print(f"  {label:<22} {xs_val:>10.3e} {eps_i:>10.2e} {a:>12.3e} {f:>12.3e} {rel:>10.2e}{flag}")

    print(f"  {'-'*70}")
    if all_rel:
        print(f"  Max rel error  : {max(all_rel):.3e}")
        mean = sum(all_rel)/len(all_rel)
        print(f"  Mean rel error : {mean:.3e} (target < 5e-2 for float32)")
        if mean < 5e-2:
            print("  ✓ Custom VJP is CORRECT — NTdiffbwd formula is valid")
        else:
            print("  ✗ Custom VJP is WRONG — bug is inside NTdiffbwd")
            print("    → Check: denominator (phiadj·F·phi), dAterm/dFterm chain, sign of numerator")
    print("=" * 65)

def log_keff_batch(writer, filehandle, epoch, keff_preds, keff_refs, avg_train_loss, avg_val_loss, sample_id_offset=0):
    """Log per-sample keff predictions vs OpenMC reference to a CSV."""
    with _csv_lock:                          # ← protect the whole loop
        for sidx, (kref, kpred) in enumerate(zip(keff_refs, keff_preds)):
            delta_rho_pcm = abs(kpred - kref) / (kpred * kref) * 1e5
            writer.writerow([epoch + 1, sidx + sample_id_offset, f"{kref:.6f}", 
                f"{kpred:.6f}", f"{delta_rho_pcm:.1f}", avg_train_loss, avg_val_loss])
        filehandle.flush()

val_log_path   = "./LOGS/keff_epoch_log_val.csv"
train_log_path = "./LOGS/keff_epoch_log_train.csv"

val_logfile   = open(val_log_path, "w", newline="", buffering=1)
train_logfile = open(train_log_path, "w", newline="", buffering=1)

val_writer   = csv.writer(val_logfile)
train_writer = csv.writer(train_logfile)

header = ["epoch", "sample_idx", "keff_openmc", "keff_peds", "delta_rho_pcm", "train_loss", "val_loss"]
val_writer.writerow(header)
train_writer.writerow(header)

# Save trunk weights at the epoch when validation loss first improves significantly
# (once, not every epoch):
def save_ewc_anchor(model, fisher_n_samples=10):
    """Compute Fisher diagonal and save current trunk weights as anchor."""
    params_anchor = {
        name: jnp.array(p) 
        for name, p in nnx.state(model.generator).items()
        if 'layers' in name   # trunk only, not heads
    }
    # Fisher diagonal = squared gradient magnitude (importance estimate)
    fisher = {k: jnp.zeros_like(v) for k, v in params_anchor.items()}
    return params_anchor, fisher

# ─────────────────────────────────────────────
# SECTION 2: Neural Network (Generator)
# ─────────────────────────────────────────────

def _hardtanh_correction(x):
    """Clip activations to (1e-16, 160.0) — ensures conductivity stays physical.
    This replaces the final activation layer so the network cannot output
    negative conductivities (which would break the physics solver).
    """
    return 0.5 * jnp.tanh(x)  # it only influences the output for 50%

def _xs_log_scale_activation(x):
    """Maps NN raw output to a log-ratio multiplier.
    exp(0) = 1.0  →  XS_SCALE * 1.0 at init (identity start).
    exp(+2) ≈ 7.4 →  up to ~7x the reference scale.
    exp(-2) ≈ 0.14 → down to ~14% of the reference scale.
    Always positive — no clipping needed; exp() is strictly > 0.
    """
    return x  # raw output; exp() is applied in PEDSModel.__ call__

class GeneratorNN(nnx.Module):
    """MLP that maps a flat geometry vector → a 2D conductivity field.

    Architecture (matching PEDS config m1):
        Input  : [batch, 6]    — 6 config parameters
        Hidden : [32, 32]       — fully-connected ReLU layers  
        Output : [batch, 6]    → reshaped to [batch,6]
                                  with hardtanh_positive activation
                                  so all values ∈ (1e-16, 160)
    """

    def __init__(self, layer_sizes: list, n_regions: int, xs_per_region: int, rngs: nnx.Rngs):
        """
        Args:
            layer_sizes: e.g. [6, 32, 32, 6]  (input → hiddens → output)
            rngs: nnx.Rngs wraps a JAX PRNGKey and hands out sub-keys on demand.
                  Every nnx module that needs randomness (Linear init, Dropout …)
                  asks `rngs` for a fresh key — no manual key splitting needed.
        """
        super().__init__()

        # He (Kaiming) initialisation is recommended for ReLU networks because
        # it keeps activation variance constant across layers.
        he_init = nnx.initializers.kaiming_normal()
        #bias_init = nnx.initializers.constant(0.0)
        zero_init = nnx.initializers.zeros

        # Shared trunk — ALL layers get He init, ALL get ReLU in __call__
        # layer_sizes is now [6, 128, 128, 64], no output dim here
        self.layers = nnx.List([
            nnx.Linear(
                in_features=layer_sizes[i],
                out_features=layer_sizes[i + 1],
                kernel_init=he_init,
                bias_init=nnx.initializers.constant(0.0),
                rngs=rngs,
            )
            for i in range(len(layer_sizes) - 1)
        ])

        trunk_out = layer_sizes[-1]  # 64

        # One independent head per physical region.
        # Zero init → exp(0)*baseline = baseline at epoch 0, same as before.
        # Each head has its own 64×12 weight matrix — gradients for b4c
        # no longer interfere with gradients for fuel or water.
        self.heads = nnx.List([
            nnx.Linear(
                in_features=trunk_out,
                out_features=xs_per_region,   # 12
                kernel_init=zero_init,
                bias_init=zero_init,
                rngs=rngs,
            )
            for _ in range(n_regions)         # 3 heads: b4c, fuel, water
        ])

        # Build layers as a plain Python list — nnx.Module discovers them
        # automatically via its attribute-scanning mechanism.
        """ self.layers = nnx.List([
            nnx.Linear(
                in_features=layer_sizes[i],
                out_features=layer_sizes[i + 1],
                kernel_init=he_init,
                bias_init=bias_init,
                rngs=rngs,          # rngs is passed here so each Linear gets
            )                        # its own unique initialisation key
            for i in range(len(layer_sizes) - 1)
        ]) """
        #  THE OUTPUT OF THE NN IS THE XS_TENSOR, WHICH HAS SIZE (N_regions, XS_per_region)
        self.n_regions    = n_regions     # 3  (b4c_rod, fuel_annulus, water)
        self.xs_per_region = xs_per_region  # 12  (4*G + G² for G=2)
        # self.resolution = int(layer_sizes[-1] ** 0.5)  # output side-length (5)

        """ self.layers[-1] = nnx.Linear(
            in_features=layer_sizes[-2],
            out_features=layer_sizes[-1],
            kernel_init=nnx.initializers.zeros,   # ← zero init
            bias_init=nnx.initializers.zeros,
            rngs=rngs,
        ) """

    def __call__(self, geoms: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        """
        Args:
            geoms:    [batch, 6]  binary pore mask
            training: flag for layers like Dropout (unused here, kept for API parity)

        Returns:
            conductivity_field: [batch,6]  physical conductivity values
        """
        x = geoms
        # Apply ReLU on all but the final layer
        """ for layer in self.layers[:-1]:
            x = nnx.relu(layer(x))        # nnx.relu is just jax.nn.relu, re-exported """
        for layer in self.layers:          # ALL trunk layers get ReLU now
            x = nnx.relu(layer(x))         # x shape stays [batch, 64] after last layer

        #x = self.layers[-1](x)  # Final layer: linear projection → raw log-ratio output
        # No activation here! exp() is applied in PEDSModel.__ call__ after multiplying by XS_SCALE.
        # The NN just outputs unconstrained values; exp() in the caller guarantees positivity.

        # Reshape flat vector [batch, 6] → spatial grid [batch,6]
        #batch_size = geoms.shape[0]
        region_outputs = [head(x) for head in self.heads]   # list of 3 × [batch, 12]
        # print(f"The last layer is {x}, the size of the batch being {batch_size} with a resolution of {self.resolution}")
        #return jnp.reshape(x, (batch_size, self.n_regions, self.xs_per_region))
        return jnp.stack(region_outputs, axis=1)             # [batch, 3, 12]



# ─────────────────────────────────────────────
# SECTION 3: Combined PEDS Model
# ─────────────────────────────────────────────

class PEDSModel(nnx.Module):
    """Wraps GeneratorNN + GaussSolver into a single differentiable forward pass.

    Gradients from the scalar κ loss flow back through the solver (via the
    custom VJP) and then into the NN weights.
    """

    def __init__(self, hidden_sizes: list, n_regions: int, G:int, rngs: nnx.Rngs):
        super().__init__()
        xs_region = fn_xs_per_region(G)            # = 12 for G=2
        output_size = n_regions * xs_region     # = 3 * 12 = 36        
        layer_sizes = [6] + hidden_sizes #+ [output_size]  # 6 geometry inputs → 36 XS outputs
    
        # rngs is stored by nnx and thread through all sub-modules that need it
        self.generator = GeneratorNN(layer_sizes=layer_sizes, n_regions=n_regions,
                                 xs_per_region=xs_region, rngs=rngs)
        self.n_regions = n_regions
        self.G = G

    def compute_xs(self, geoms, xs_baselines):
        """Pure NN forward — called OUTSIDE jax.grad, returns concrete numpy."""
        batch_size = geoms.shape[0]
        geoms_flat = jnp.reshape(geoms, (batch_size, 6))
        xs_log_ratios = self.generator(geoms_flat, training=False)
        xs_final = jnp.exp(xs_log_ratios) * xs_baselines
        return np.array(xs_final)   # ← concrete numpy, exits JAX world

    def __call__(self, geoms: jnp.ndarray, params_raw, xs_baselines, training: bool = False, sample_id_offset=0):
        """
        Args:
            geoms: [batch,6] binary pore geometry (flattened inside)

        Returns:
            keff:               [batch]    effective thermal conductivity
            conductivity_field:  [batch, N, N]  intermediate field (for visualisation)
        """
        batch_size = geoms.shape[0]
        geoms_flat = jnp.reshape(geoms, (batch_size, 6))  # flatten spatial dims

        # NN outputs log-ratios relative to XS_SCALE (the fixed default-geometry reference).
        # exp(0) * XS_SCALE = XS_SCALE at init; the NN freely learns any multiplier during training.
        with timer("forward: NN generated XS", verbose=False):
            xs_log_ratios = self.generator(geoms_flat, training)       # [batch, 3, 12], values near 0 at init
            xs_final = jnp.exp(xs_log_ratios) * xs_baselines              # [batch, 3, 12], always positive
        
        #print("RUNNING a gradients physics check ")
        #physics_sanity_check(xs_baselines[0])        
        #print(f" the baseline starting point XS tensor was {xs_baselines[0]}")
        #print(f"and the final corrected one is {xs_final[0]}")

        with timer("forward: solver loop (all samples)", verbose=False):
            keffs = [] # runs a calc for each of the 10 and then stores
            for i in range(batch_size):
                """ if i == 2:  # print the XS for the first sample in the batch as a sanity check
                    print(f"\nthe LOG RATIOS computed by the NN for this geom are {xs_log_ratios[i]}")
                    print(f"\n AND THE XS PREDICTIONNNNNNNNN for this geom is {xs_final[i]}")
                    print(f"the baseline starting point XS tensor was {xs_baselines[i]}")
                    #print(f"for the id number {i}, the offset is {sample_id_offset}") """

                keff_i = NTdiff_solver(xs_final[i], 
                    jnp.array(params_raw[i], dtype=jnp.float32),       # (6,)   raw geometry
                    jnp.array([i + sample_id_offset], dtype=jnp.int32), )         # (1,)   sample ID)   # pass (3, 12) slice
                keffs.append(keff_i)
            keffs = jnp.stack(keffs)   # [batch]

        return keffs, xs_final
        #return keff, xs_tensor


# ─────────────────────────────────────────────
# SECTION 4: Data Loading
# ─────────────────────────────────────────────

def data_loader(*arrays, batch_size: int):
    """Yield mini-batches from multiple arrays in lock-step.
    Pure-Python generator that slices NumPy arrays and feeds them one batch at a time.
    """
    n_samples = arrays[0].shape[0]
    for start in range(0, n_samples, batch_size):
        yield tuple(arr[start:start + batch_size] for arr in arrays)


def load_data(filepath: str, train_size: int, test_size: int, seed: int = 42):
    """Load geometries specs and keff labels from the .npz dataset created with MC. Returns NumPy arrays 
    """
    data = np.load(filepath, allow_pickle=True)
    print(f"Dataset keys: {list(data.keys())}")
    geoms  = np.array(data['params'],  dtype=np.float32)   # [N,6]
    keffs = np.array(data['keffs'], dtype=np.float32)   # [N]
    rawparams = np.array(data['params_raw'], dtype=np.float32)   # [N]
    print(f"Loaded the data: example geom {geoms[0]} and keff {keffs[0]}")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(geoms)) # randomly shuffle the indices of the dataset to ensure random sampling of training and test sets
    train_idx = idx[:train_size] # index to save array from location 0 to location of training set dimension
    test_idx  = idx[train_size:train_size + test_size] # index for the array that takes the following chunk of data, of dimension test set

    return (geoms[train_idx], keffs[train_idx], rawparams[train_idx]), (geoms[test_idx], keffs[test_idx], rawparams[test_idx])


# ─────────────────────────────────────────────
# SECTION 5: Training Loop
# ─────────────────────────────────────────────

def train(filepath, train_size, test_size, batch_size, epochs, lr_max, lr_min,
    hidden_sizes, n_regions, G, seed):
    if hidden_sizes is None:
        hidden_sizes = [32, 32]   # matches config m1 in the original codebase
    
    # ── 5.1  Data ──────────────────────────────────────────────────────────
    print("Loading data …")
    (train_geoms, train_keffs, train_rawparams), (test_geoms, test_keffs, test_rawparams) = load_data(
        filepath, train_size, test_size, seed
    )   # train_geoms is a numpy array of shape (train_size, 6), while train_keffs is a numpy array of shape (train_size,) 
    
    # Print dataset statistics for sanity check
    print("GEOMS STATSSSSSSSSSSS:")
    print(f"  mean: {train_geoms.mean(0)}")
    print(f"  std:  {train_geoms.std(0)}")
    print(f"  min:  {train_geoms.min(0)}")
    print(f"  max:  {train_geoms.max(0)}")

    print(f"  Train: {train_geoms.shape}  Test: {test_geoms.shape}")
    print(" THE NEW PRINTINGGGGGGGGG")
    print(f"Train keff range: {train_keffs.min():.3f} – {train_keffs.max():.3f}")
    print(f"Test  keff range: {test_keffs.min():.3f}  – {test_keffs.max():.3f}")
    
# ── Initialize EWC state before the epoch loop ──────────────────────
    best_val_loss = float('inf')
    ewc_anchor_flat = None
    ewc_fisher_flat = None
    ewc_saved = False  # Track if the anchor is saved yet
    # ── 5.2  Model & Optimiser ─────────────────────────────────────────────
    # nnx.Rngs(seed) creates a named-key container to get a fresh, unique PRNGKey.
    rngs  = nnx.Rngs(seed)
    model = PEDSModel(hidden_sizes=hidden_sizes, n_regions=n_regions, G=G, rngs=rngs)
    # Cosine decay schedule: learning rate anneals smoothly from lr_max → lr_min.
    # `alpha` is the ratio min/max, so the schedule bottoms out at lr_min.
    lr_schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(0.0, lr_max, transition_steps=5),
            optax.cosine_decay_schedule(init_value=lr_max, decay_steps=epochs - 5, alpha=lr_min / lr_max),
        ],
        boundaries=[5]
    )

    #optimizer = nnx.Optimizer(model, optax.adam(lr_schedule), wrt=nnx.Param)     # nnx.Optimizer couples the model's parameters with an Optax update rule.
    optimizer = nnx.Optimizer(model,
        optax.chain(
            optax.clip_by_global_norm(1.0),   # ← clips any gradient explosion
            optax.adam(lr_schedule)
        ),
        wrt=nnx.Param
    )
    # ── 5.3  Loss / gradient function ─────────────────────────────────────
    def loss_fn(model, geoms, keffs_true, rawparams_batch, xs_baselines_batch):
        # model is passed explicitly so nnx.value_and_grad knows which pytree to differentiate with respect to.
        with timer("loss_fn forward pass", verbose=False):
            keff_pred, _ = model(geoms, rawparams_batch, xs_baselines_batch, training=True) # keff predicted by PEDS model
            print(f"the prediction gives a keff of {keff_pred.primal} (size {keff_pred.size}), \n compared to {keffs_true} (size {keffs_true.size})")
        
        reactivity_residuals = (keff_pred - keffs_true) / (keff_pred * keffs_true) 
        residuals = keff_pred - keffs_true
        BETA_EFF = 650e-5  # pcm 
        
        # Dynamic weight: quadratic in how much the error exceeds β_eff.
        normalized_err = jnp.abs(reactivity_residuals) / BETA_EFF
        #penalising_weight = jnp.maximum(1.0, normalized_err ** 2)
        #weight = jnp.minimum(jnp.maximum(1.0, normalized_err), 4.0)  # linear, capped at 4×
        weight = jnp.where(
            normalized_err <= 1.0,
            0.3,                                    # try to anchor samples that already have low error 
            jnp.minimum(normalized_err, 4.0)             # capped linear for hard samples
        )
        #penalising_weight = jnp.where(jnp.abs(reactivity_residuals) > BETA_EFF, 3.0, 1.0)
        loss_value = jnp.mean(weight * reactivity_residuals ** 2)  
        
        if ewc_saved and ewc_anchor_flat is not None:   
            ewc_lam = 0.05  # Start with 0.05, tune between 0.01-0.1
            # Get current trunk state and flatten it the same way
            full_state = nnx.state(model.generator)
            trunk_state = nnx.State({
                k: v for k, v in full_state.items() 
                if 'layers' in k and 'heads' not in k
            })
            current_flat, _ = jax.tree_util.tree_flatten(trunk_state)
            
            # Compute L2 penalty between current and anchor
            penalty = sum(
                jnp.sum(fisher * (curr - anchor) ** 2)
                for curr, anchor, fisher in zip(current_flat, ewc_anchor_flat, ewc_fisher_flat)
            )
            loss_value = loss_value + ewc_lam * penalty

        return loss_value, keff_pred
    
    # nnx.value_and_grad is the Flax-nnx analogue of jax.value_and_grad.
    # It differentiates `loss_fn` with respect to its FIRST argument (the model),
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)  # returning (loss_value, grad_pytree_of_model_params)

    # ── 5.4  Validation helper ─────────────────────────────────────────────
    def validation_step(geoms_np, keffs_np, rawparams_np):
        """Compute mean squared loss and mean percentage error over the test set.
            Also returns per-sample keff_pred and keff_ref arrays for logging."""

        total_sq  = 0.0
        total_pct = 0.0
        all_keff_pred = []
        all_keff_ref  = []
        for batch_idx, (batch_geoms, batch_keffs, batch_rawparams, batch_xs_baselines) in enumerate(data_loader(geoms_np, keffs_np, rawparams_np, test_xs_baselines, batch_size=batch_size)):
            # the batch index is always 0 because the test set is smaller than the batch size, so we only have one batch containing the whole test set.
            global_start = batch_idx * batch_size
            batch_geoms_jax = jnp.array(batch_geoms)

            # ── Pre-solve in parallel, same as training ────────────────────
            xs_np = model.compute_xs(batch_geoms_jax, jnp.array(batch_xs_baselines))
            args_list = [
                (i, xs_np[i], np.array(batch_rawparams[i]), i + global_start, -1)
                for i in range(len(batch_geoms))
            ]
            results = list(executor.map(_solve_sample_worker, args_list))

            _PRESOLVE_CACHE.clear()
            for i, k, phi_fwd, phi_adj, geo_data, _ in results:
                _PRESOLVE_CACHE[i + global_start] = (k, phi_fwd, phi_adj, geo_data)
            # ──────────────────────────────────────────────────────────────

            keff_pred, _ = model(batch_geoms_jax, batch_rawparams, 
            xs_baselines=batch_xs_baselines, training=False, sample_id_offset=global_start)  
            sq_err  = jnp.mean((keff_pred - batch_keffs) ** 2) # squared error
            pct_err = jnp.mean(jnp.abs(keff_pred - batch_keffs) / jnp.abs(batch_keffs) * 100.0) # percent error
            total_sq  += float(sq_err)
            total_pct += float(pct_err)
            all_keff_pred.extend(np.array(keff_pred).tolist())   # ← collect predictions
            all_keff_ref.extend(np.array(batch_keffs).tolist())  # ← collect references            
        n = len(keffs_np)
        return total_sq / n, total_pct / n, all_keff_pred, all_keff_ref   
        # mean over all test samples

    sample_weights = np.ones(train_size, dtype=np.float32)  # uniform initially

    def weighted_data_loader(arrays, batch_size, weights):
        """Sample batches with probability proportional to weights."""
        n = arrays[0].shape[0]
        probs = weights / weights.sum()
        indices = np.random.choice(n, size=batch_size, replace=False, p=probs)
        return tuple(arr[indices] for arr in arrays), indices

    print("Precomputing XS baselines for all samples (done once)...")
    train_xs_baselines = compute_batch_baselines(train_rawparams, GEO)  # [train_size, 3, 12]
    test_xs_baselines  = compute_batch_baselines(test_rawparams, GEO)   # [test_size,  3, 12]
    print("Done.")

    # ── 5.5  Epoch loop ────────────────────────────────────────────────────
    # ── 5.5  EPOCH LOOOOOOOP ───────────────────────────────────────────────
    # ── 5.5  Epoch loop ────────────────────────────────────────────────────

    train_losses   = []
    val_losses     = []
    val_pct_errors = []
    iterations = 0 
    print(f"\nTraining for {epochs} epochs …")

        # ── Open keff log file ────────────────────────────────────────────────

    count = 0
    for epoch in range(epochs):
        count += 1
        print(f"entering the epoch loop, at epoch {epoch} which is")
        epoch_loss = 0.0
        # these are the parameters used for the training loop
        for batch_idx, (batch_geoms, batch_keffs, batch_rawparams_tr, batch_baselines) in enumerate(data_loader(
            train_geoms, train_keffs, train_rawparams, train_xs_baselines, batch_size=batch_size)):
            
            bp = jnp.array(batch_geoms) # Convert NumPy → JAX arrays once per batch 
            bk = jnp.array(batch_keffs)
            bxs = jnp.array(batch_baselines)  
            # Step 1: get concrete xs from NN (outside trace)
            xs_np = model.compute_xs(bp, bxs)   # concrete numpy [batch, 3, 12]

            # Step 2: parallel solver calls — populates _GEO_DATA_CACHE
            args_list = [
                (i, xs_np[i], np.array(batch_rawparams_tr[i]), i, epoch)
                for i in range(len(batch_geoms))
            ]
            results = list(executor.map(_solve_sample_worker, args_list))
            _PRESOLVE_CACHE.clear()   # ← clear stale entries from previous batch/epoch
            for i, k, phi_fwd, phi_adj, geo_data, result_epoch in results:
                _PRESOLVE_CACHE[i] = (k, phi_fwd, phi_adj, geo_data)

            # Step 3: grad_fn traces normally — pure_callback reads from cache
            with timer("train step: forward+backward+optiupd"):
                with timer("train step: forward+loss+grad"):    
                    (loss, keff_preds_batch), grads = grad_fn(model, bp, bk, batch_rawparams_tr, bxs)

            with timer("train step: optimizer update"):            
                optimizer.update(model, grads)
            
            log_keff_batch(
                train_writer, train_logfile, epoch,
                np.array(keff_preds_batch), np.array(bk),
                float(loss), 0.0,
                sample_id_offset=batch_idx * batch_size
            )
            epoch_loss += float(loss)

            print("Running forward pass + AD \n ")
            """ # Part of training: forward pass + automatic differentiation in one call.
            # `loss` is a scalar; `grads` is a pytree mirroring model's parameter tree.
            with timer("train step: forward+backward+update"):
                with timer("train step: forward+loss+grad"):
                    loss, grads = grad_fn(model, bp, bk, batch_rawparams_tr, bxs) # model is the call to PEDS 
                if epoch % 2 == 0:
                    print(f"for this epoch {epoch}, the loss is {loss}, while the grads pytree structure is presented as follow:")
                    for i, leaf in enumerate(jax.tree.leaves(grads)):
                        print(
                            f"Param {i:2d} | shape: {leaf.shape} | "
                            f"mean grad: {float(jnp.mean(jnp.abs(leaf))):.2e} | "
                            f"max grad : {float(jnp.max(jnp.abs(leaf))):.2e}"
                        ) # this pytree is where the weights and biases are stored 
                        
                grad_state = nnx.state(grads) # print this to visualise the full dict
                epoch_loss += float(loss) 
                Apply Adam update: grads → moment estimates → parameter delta → model update
                #   1. Passes grads through the Optax chain (Adam moment updates, LR scaling)
                #   2. Applies the resulting parameter updates in-place on the model.
                with timer("train step: optimizer update"):
                    optimizer.update(model, grads) """
            
            # still inside the epoch loop 
            # ── Collect heatmap snapshot every HEATMAP_INTERVAL epochs ────────
            if (epoch + 1) % HEATMAP_INTERVAL == 0 or epoch == 0:
                _snap_geo_i       = update_geo(GEO, train_rawparams[0])          # single GeometryConfig
                _snap_baseline    = np.array(predict_xs(_snap_geo_i))            # (3, 12) guaranteed

                _snap_geom_flat   = jnp.array(train_geoms[0:1])                  # (1, 6)
                _snap_log_ratios = np.array(
                    model.generator(_snap_geom_flat, training=False)[0]  # (3, 12), log-ratios
                )
                _snap_final = np.exp(_snap_log_ratios) * np.array(XS_SCALE)  # (3, 12), absolute XS

                XS_HEATMAP_SNAPSHOTS.append(_snap_log_ratios.copy())  # store log-ratios for heatmap
                XS_HEATMAP_LABELS.append(f"Epoch {epoch + 1}")
                
                if _xs_baseline_ref[0] is None:
                    _xs_baseline_ref[0] = _snap_baseline

        # Normalise by dataset size (matches `avg_loss` in training.py)
        avg_train_loss = epoch_loss / train_size
        # Validation (no gradient tracking needed)
        with timer("validation step"):
            avg_val_loss, avg_pct_err, keff_preds, keff_refs = validation_step(test_geoms, test_keffs, test_rawparams)
    
        # Check if validation improved AND we haven't saved the anchor yet
        if avg_val_loss < best_val_loss and not ewc_saved:
            best_val_loss = avg_val_loss
            print(f"Saving EWC anchor at epoch {epoch} with val_loss={avg_val_loss:.4f}")
            
            # Get all trainable parameters from the trunk (flatten to get actual arrays)
            full_state = nnx.state(model.generator)
            # Filter for trunk parameters only
            trunk_state = nnx.State({
                k: v for k, v in full_state.items() 
                if 'layers' in k and 'heads' not in k
            })
            
            # Flatten to get actual JAX arrays
            ewc_anchor_flat, ewc_tree_def = jax.tree_util.tree_flatten(trunk_state)
            ewc_fisher_flat = [jnp.ones_like(arr) for arr in ewc_anchor_flat]
            
            print(f"  Saved {len(ewc_anchor_flat)} trunk parameter arrays")
            for i, arr in enumerate(ewc_anchor_flat[:3]):  # Show first 3
                print(f"    Array {i}: shape={arr.shape}")
            
            ewc_saved = True

        # ── Write per-sample keff values to the log file ─────────────────────
        log_keff_batch(val_writer, val_logfile, epoch, keff_preds, keff_refs, avg_train_loss, avg_val_loss)
   
        print(f"this is batch iteration {iterations + 1} with the average train loss {avg_train_loss:.4f}, the average validation loss {avg_val_loss:.4f} and the average percentage error {avg_pct_err:.4f} \n")
        iterations = iterations+1
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_pct_errors.append(avg_pct_err)
        if (epoch + 1) % 50 == 0:
            current_lr = float(lr_schedule(epoch))
            print(
                f" this message is printed every 50 epochs! \n"
                f"Epoch {epoch+1:4d}/{epochs} \n "
                f"Train MSE: {avg_train_loss:8.3f} \n "
                f"Val MSE: {avg_val_loss:8.3f} \n "
                f"Val%: {avg_pct_err:6.2f}% \n "
                f"LR: {current_lr:.2e}"
            )
        # Periodically clear JAX's compilation cache to avoid memory creep
        jax.clear_caches()
        iterations = 0
    
    final_errors = []
    for i in range(len(keff_refs)):
        k_pred = float(keff_preds[i])   # keff from PEDS at final epoch
        k_ref  = float(keff_refs[i])    # keff from OpenMC
        delta_rho = abs(k_pred - k_ref) / (k_pred * k_ref) * 1e5
        final_errors.append(delta_rho)

    final_errors = np.array(final_errors)

    # final_errors: [50] array of pcm values at last epoch for test set
    param_names = ['b4c_r', 'cr_frac', 'fuel_r', 'enrichment', 'f_mod', 'water_r']
    for j, name in enumerate(param_names):
        corr = np.corrcoef(test_rawparams[:, j], final_errors)[0, 1]
        print(f"  {name:12s}: correlation with error = {corr:+.3f}")
        
    print(f"the total number of iterations is {iterations}.")
    print(f"\n keff log saved to: {val_log_path}")

    # ─────────────────────────────────────────────────────────────
    # check the geometry of the badly converged samples
    # ─────────────────────────────────────────────────────────────
    df = pd.read_csv(val_log_path, header=0)
    # This reads the header properly, columns named automatically
    # Then cast types explicitly:
    df['epoch'] = df['epoch'].astype(int)
    df['delta_rho_pcm'] = df['delta_rho_pcm'].astype(float)
    df['sample_idx'] = df['sample_idx'].astype(int)
    # Get the LAST epoch only (final converged state)
    last_epoch = df['epoch'].max()
    print(f"df shape: {df.shape}")
    print(f"df dtypes:\n{df.dtypes}")
    print(f"df head:\n{df.head()}")
    print(f"last epoch = {last_epoch}, type = {type(last_epoch)}")
    df_final = df[df['epoch'] == last_epoch]
    print(f"df final shape: {df_final.shape}")
    # Sort by worst discrepancy
    df_final_sorted = df_final.sort_values('delta_rho_pcm', ascending=False)
    print("=== TOP 10 WORST SAMPLES AT FINAL EPOCH ===")
    print(df_final_sorted[['sample_idx', 'keff_openmc', 'keff_peds', 'delta_rho_pcm']].head(10).to_string())
    # Extract the bad sample indices
    bad_indices = df_final_sorted['sample_idx'].head(10).values
    print(f"\nBad sample indices: {bad_indices}")

    print("\n=== GEOMETRY OF WORST 10 SAMPLES AT FINAL EPOCH ===")
    feat_names = ['b4c_r', 'cr_frac', 'fuel_r', 'enrich', 'f_mod', 'water_r']
    for idx in bad_indices:
        geom = test_geoms[idx]
        raw  = test_rawparams[idx]
        k_ref = test_keffs[idx]
        delta = float(df_final[df_final['sample_idx']==idx]['delta_rho_pcm'].values[0])
        print(f"\n  Sample {idx} | k_ref={k_ref:.4f} | delta_rho={delta:.1f} pcm")
        for i, name in enumerate(feat_names):
            print(f"    {name}: normalized={geom[i]:.3f}  raw={raw[i]:.4f}")

    loggy_file.close()

    # ── Generate XS heatmap ───────────────────────────────────────────
    print(f"[xs_heatmap] baseline ref is None: {_xs_baseline_ref[0] is None}")
    print(f"[xs_heatmap] number of snapshots collected: {len(XS_HEATMAP_SNAPSHOTS)}")
    print(f"[xs_heatmap] snapshot labels: {XS_HEATMAP_LABELS}")
    print(f"CHECKING IF THE PLOT IS RUNNIG WITH {_xs_baseline_ref[0]} as the baseline and {_snap_final} as the final xs to plot")
    if _xs_baseline_ref[0] is not None and len(XS_HEATMAP_SNAPSHOTS) > 0:
        print(f"\nGenerating XS heatmap with {len(XS_HEATMAP_SNAPSHOTS)} snapshots …")
        print(f"saving it to ")
        plot_xs_heatmap(
            baseline           = _xs_baseline_ref[0],
            snapshots          = XS_HEATMAP_SNAPSHOTS,
            epoch_labels       = XS_HEATMAP_LABELS,
            final_xs           = _snap_final,
            G                  = G,
            save_path          = "./LOGS/figures/xs_heatmap.png",
            plot_interval_note = f"snapshot every {HEATMAP_INTERVAL} epochs",
        )
        plot_xs_subplots(
            baseline           = _xs_baseline_ref[0],
            final_xs           = _snap_final,
            G                  = G,
            save_path          = "./LOGS/figures/xs_subplots.png",
        )
    else:
        print("[xs_heatmap] skipped — no snapshots collected (epochs < HEATMAP_INTERVAL?)")
    return model, train_losses, val_losses, val_pct_errors


# ─────────────────────────────────────────────
# SECTION 6: Visualisation
# ─────────────────────────────────────────────

def visualise_results(
    model,
    test_geoms:   np.ndarray,
    test_keffs:  np.ndarray,
    train_losses: list,
    val_losses:   list,
    val_pct_errs: list,
    sample_idx:   int = 0,
    save_path:    str = "./experiments/coding/figures/MYNT_results.png",
):
    """Four-panel figure:
       (a) pore geometry for one test sample
       (b) generated conductivity field for that sample
       (c) training vs validation MSE loss curve
       (d) validation percentage-error curve
    """
    # ── Run one sample through the model ──────────────────────────────────
    pore_sample   = jnp.array(test_geoms[sample_idx:sample_idx + 1])   # [1,6]
    kappa_true    = float(test_keffs[sample_idx])

    keff_pred_arr, cond_field = model(pore_sample)
    keff_pred  = float(keff_pred_arr[0])
    cond_np     = np.array(cond_field[0])   # [5, 5]
    pore_np     = np.array(test_geoms[sample_idx])  # [5, 5]
    pore_np = pore_np.reshape(5, 5)             # force to (5, 5) for imshow

    pct_err = abs(keff_pred - kappa_true) / abs(kappa_true) * 100

    # ── Layout ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # ── Panel (a): Pore geometry ──────────────────────────────────────────
    ax_pore = fig.add_subplot(gs[0, 0])
    im_p = ax_pore.imshow(pore_np, cmap="gray_r", interpolation="nearest",
                          vmin=0, vmax=1)
    ax_pore.set_title(f"(a) Pore Geometry (sample {sample_idx})", fontsize=12)
    ax_pore.set_xlabel("x");  ax_pore.set_ylabel("y")
    for i in range(pore_np.shape[0]):
        for j in range(pore_np.shape[1]):
            ax_pore.text(j, i, f"{int(pore_np[i,j])}", ha="center", va="center",
                         color="red", fontsize=9)
    fig.colorbar(im_p, ax=ax_pore, fraction=0.046, pad=0.04, label="Pore (1=solid)")

    # ── Panel (b): Generated conductivity field ───────────────────────────
    ax_cond = fig.add_subplot(gs[0, 1])
    im_c = ax_cond.imshow(cond_np, cmap="viridis", interpolation="nearest")
    ax_cond.set_title(
        f"(b) Generated Conductivity Field\n"
        f"κ_pred = {keff_pred:.2f}  |  κ_true = {kappa_true:.2f}  |  err = {pct_err:.1f}%",
        fontsize=11,
    )
    ax_cond.set_xlabel("x");  ax_cond.set_ylabel("y")
    for i in range(cond_np.shape[0]):
        for j in range(cond_np.shape[1]):
            ax_cond.text(j, i, f"{cond_np[i,j]:.1f}", ha="center", va="center",
                         color="white", fontsize=8)
    fig.colorbar(im_c, ax=ax_cond, fraction=0.046, pad=0.04, label="Conductivity")

    epochs_range = np.arange(1, len(train_losses) + 1)

    # ── Panel (c): MSE loss curves ────────────────────────────────────────
    ax_loss = fig.add_subplot(gs[1, 0])
    ax_loss.plot(epochs_range, train_losses, "b-",  lw=1.5, label="Train MSE")
    ax_loss.plot(epochs_range, val_losses,   "r--", lw=1.5, label="Val MSE")
    ax_loss.set_xlabel("Epoch", fontsize=11)
    ax_loss.set_ylabel("Mean Squared Loss", fontsize=11)
    ax_loss.set_title("(c) Training & Validation MSE", fontsize=12)
    ax_loss.set_yscale("log")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    # ── Panel (d): Validation percentage error ────────────────────────────
    ax_pct = fig.add_subplot(gs[1, 1])
    ax_pct.plot(epochs_range, val_pct_errs, "g-", lw=1.5, label="Val % Error")
    ax_pct.axhline(5.0, color="gray", linestyle="--", lw=1, label="5% target")
    ax_pct.set_xlabel("Epoch", fontsize=11)
    ax_pct.set_ylabel("Mean |κ_pred - κ_true| / |κ_true| × 100%", fontsize=10)
    ax_pct.set_title("(d) Validation Percentage Error", fontsize=12)
    ax_pct.legend()
    ax_pct.grid(True, alpha=0.3)

    plt.suptitle("PEDS Run — Single Device, No MPI", fontsize=14, y=1.01)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved to: {save_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # ── Hyperparameters (from config_experiment.py / config_model.py) ─────
    HP = dict(
        filepath      = "../data/highfidelity/MCruns.npz",  # adjust path as needed
        train_size    = train_size,
        test_size     = test_size,
        batch_size    = batch_size,
        epochs        = epochs,
        lr_max        = 5e-5,   # cosine schedule peak learning rate
        lr_min        = 1e-6,   # cosine schedule floor learning rate
        hidden_sizes  = [128, 128, 64],  # matches model config m1
        n_regions    = 3,    # b4c_rod, fuel_annulus, water
        G            = 2,    # energy groups
        seed          = 42,
    )

    # ══════════════════════════════════════════════════════════
    # DEBUG BLOCK 
    # ══════════════════════════════════════════════════════════
    """ print("Running gradient check BEFORE training...")
    # Load just one sample to test with
    (train_geoms, train_keffs, train_rawparams), _ = load_data(
        HP["filepath"], HP["train_size"], HP["test_size"], seed=42)
    
    xs_check     = jnp.array(XS_BASELINE, dtype=jnp.float32)
    params_check = jnp.array(train_rawparams[0], dtype=jnp.float32)
    id_check     = jnp.array([0], dtype=jnp.int32)

    print("Warming up solver cache for sample 0...")
    _ = NTdiff_solver(xs_check, params_check, id_check)   # populates _GEO_DATA_CACHE[0]

    # Now the cache has the geo_data for sample 0
    geo_data_check = _GEO_DATA_CACHE[0]

    # ── Step 1: Eigenvalue residual (math foundation) ─────────────────────
    print("\n--- Step 1: Eigenvalue Residual Check ---")

    geo_i_check = update_geo(GEO, np.array(train_rawparams[0]))
    I_check = int(geo_i_check.boundaries[-1].radius / geo_i_check.mesh_size)

    # Level 1: what does the raw solver return?
    k_check, phi_fwd_raw, phi_adj_raw = run_diffusion_solver(
        np.array(xs_check), geo_i_check
    )
    print(f"\n=== Raw Solver Output (direct call) ===")
    print(f"  k                        = {k_check:.6f}")
    print(f"  ||phi_fwd_raw||          = {np.linalg.norm(phi_fwd_raw):.6e}")
    print(f"  phi_fwd_raw[g=0, i=0:5] = {phi_fwd_raw[0, :5]}")
    print(f"  phi_fwd_raw[g=1, i=0:5] = {phi_fwd_raw[1, :5]}")

    # Level 2: flatten it yourself (no normalization)
    N_flat = geo_i_check.G * (I_check + 1)
    phi_fwd_check_flat = np.zeros(N_flat, dtype=np.float32)
    for g in range(geo_i_check.G):
        phi_fwd_check_flat[g*(I_check+1) : g*(I_check+1)+I_check] = phi_fwd_raw[g, :]

    print(f"\n=== After Manual Flattening ===")
    print(f"  ||phi_fwd_flat||         = {np.linalg.norm(phi_fwd_check_flat):.6e}")
    print(f"  ghost cell g=0 (i=I)    = {phi_fwd_check_flat[I_check]:.6e}  (expect 0.0)")

    # Level 3: also call _run_NT_solver to see if it matches
    _ = NTdiff_solver(xs_check, params_check, id_check)  # warm up cache
    k_nt, phi_nt_flat, _ = _run_NT_solver(np.array(xs_check), params_check, id_check)
    print(f"\n=== _run_NT_solver Output (for comparison) ===")
    print(f"  ||phi_nt_flat||          = {np.linalg.norm(phi_nt_flat):.6e}")
    print(f"  phi_nt_flat[0:5]        = {phi_nt_flat[:5]}")
    print(f"  Match raw flat?         = {np.allclose(phi_fwd_check_flat, phi_nt_flat, atol=1e-5)}")

    ############

    # Then flatten manually for the diagnostic
    N_flat = geo_i_check.G * (I_check + 1)
    phi_fwd_check_flat = np.zeros(N_flat, dtype=np.float32)
    for g in range(geo_i_check.G):
        phi_fwd_check_flat[g*(I_check+1) : g*(I_check+1)+I_check] = phi_fwd_raw[g, :]

    print(f"\n=== After Flattening ===")
    print(f"  ||phi_fwd_flat|| = {np.linalg.norm(phi_fwd_check_flat):.6e}")
    print(f"  Same values at i=0:5? {np.allclose(phi_fwd_raw[0,:5], phi_fwd_check_flat[:5])}")
        
    phi_fwd_jax = jnp.array(phi_fwd_check_flat)
    # After the cache warmup, pass geo_i (not geo_data)
    geo_i_check = update_geo(GEO, np.array(train_rawparams[0]))
    check_eigenvalue_residual_vs_solver(xs_check, geo_i_check, phi_fwd_check_flat, k_check)

    # ── Step 2: Physics sanity check (gradient signs) ─────────────────────
    print("\n--- Step 2: Physics Sanity Check ---")
    #physics_sanity_check(xs_check, params_check, id_check)  # already handles its own vjp call internally

    # ── Step 3: Finite difference gradient check ──────────────────────────
    print("\n--- Step 3.1: check cells assignments ---")
    diagnose_scan_vs_real_cellwise(xs_check, geo_i_check, phi_fwd_check_flat, k_check)

    # ---- for the gradients check -------
    print("\n--- Step 3.2: finite difference gradient check ---")
    check_xs = jnp.array(XS_BASELINE, dtype=jnp.float32)
    check_sample_id = jnp.array([0], dtype=jnp.int32)
    k_check, phifwd_check, phiadj_check = _run_NT_solver(
        check_xs, train_rawparams[0], check_sample_id
    )
    geodata_check = _GEO_DATA_CACHE[0]
    diagnose_scan_autodiff(check_xs, geodata_check,
        jnp.array(phifwd_check), jnp.array(phiadj_check), k_check)
    
    print("--- Step 4: Full Custom VJP Check ---")
    diagnose_full_vjp(xs_check, params_check, id_check)  """
        
    # ══════════════════════════════════════════════════════════

    # ── Train ─────────────────────────────────────────────────────────────
    print("starting the training process... \n")
    model, train_losses, val_losses, val_pct_errs = train(**HP)
    print_timing_report()
    # ── Reload test set for final visualisation ───────────────────────────
    _, (test_geoms, test_keffs, test_rawparams) = load_data(
        HP["filepath"], HP["train_size"], HP["test_size"], HP["seed"]
    )
    executor.shutdown(wait=True)


    # ── Visualise ─────────────────────────────────────────────────────────
 
    """ visualise_results(
        model        = model,
        test_geoms   = test_geoms,
        test_keffs  = test_keffs,
        train_losses = train_losses,
        val_losses   = val_losses,
        val_pct_errs = val_pct_errs,
        sample_idx   = 0,
        save_path    = "./experiments/coding/figures/NT/results.png",
    ) """

    print("\nDone.")
    print(f"  Final train MSE     : {train_losses[-1]:.4f}")
    print(f"  Final val MSE       : {val_losses[-1]:.4f}")
    print(f"  Final val % error   : {val_pct_errs[-1]:.2f}%")
