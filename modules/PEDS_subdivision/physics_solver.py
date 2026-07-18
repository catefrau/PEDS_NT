"""Physics solver + custom VJP (forward & backward calls).

Moved out of PEDS.py unchanged (only shared globals now come from
``PEDS_subdivision.context``). This includes the per-sample and batched
NTdiff solvers, their custom VJPs, and the parallel pre-solve worker.
"""
import os

import jax
import jax.numpy as jnp
import numpy as np

from NTcode_config_data.config_run import GEO_CYL as GEO
from solvers.NTdiffusion.core.MG1D_eigenvalue_nregions import diffusion_setup
from solvers.NTdiffusion.diffusion_solver import (
    run_diffusion_solver, is_homogeneous, precompute_geometry,
    build_xs_callables, bc_to_coeffs, GEOMETRY_CODE,
)
from matrix_JAX_optimized import Aphi_Fphi_vjp
from PEDS_core.timing_utils import timer
from PEDS_subdivision import context
from PEDS_subdivision.context import update_geo


def _run_NT_solver(xs_tensor, params_raw_single, sample_id):
    """NumPy/SciPy forward solve. Returns (k, phi_fwd_padded, phi_adj_padded, Fphi_padded)."""
    geo_i    = update_geo(GEO, np.array(params_raw_single))
    geo_data = precompute_geometry(geo_i)
    sid      = int(sample_id[0])
    context._GEO_DATA_CACHE[sid] = geo_data

    # ── use cached solve if available ────────────────────────────────────────
    if sid in context._PRESOLVE_CACHE:
        k, phi_fwd_padded, phi_adj_padded, geo_data_pre = context._PRESOLVE_CACHE.pop(sid)
        context._GEO_DATA_CACHE[sid] = geo_data_pre
        R      = geo_i.boundaries[-1].radius
        I      = int(R / geo_i.mesh_size)
        N_flat = geo_i.G * (I + 1)
        r_div  = [b.radius for b in geo_i.boundaries[:-1]] if not is_homogeneous(geo_i) else []
        BC     = bc_to_coeffs(geo_i.bc)
        gcode  = GEOMETRY_CODE[geo_i.geometry]
        D_fn, Sa_fn, nF_fn, Ss_fn, chi_fn = build_xs_callables(np.array(xs_tensor), geo_i)
        _, _, F_real = diffusion_setup(R, I, geo_i.G, r_div, D_fn, Sa_fn, nF_fn, Ss_fn, chi_fn, BC, gcode)
        Fphi_full = F_real @ phi_fwd_padded[:N_flat].astype(np.float64)
        Fphi_padded = np.zeros(context.N_FLAT_MAX, dtype=np.float32)
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


    phi_fwd_padded = np.zeros(context.N_FLAT_MAX, dtype=np.float32)
    phi_adj_padded = np.zeros(context.N_FLAT_MAX, dtype=np.float32)
    Fphi_padded    = np.zeros(context.N_FLAT_MAX, dtype=np.float32)
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
            jax.ShapeDtypeStruct((context.N_FLAT_MAX,),  jnp.float32),
            jax.ShapeDtypeStruct((context.N_FLAT_MAX,),  jnp.float32),
            jax.ShapeDtypeStruct((context.N_FLAT_MAX,),  jnp.float32),
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
    geo_data = context._GEO_DATA_CACHE[int(sample_id[0])]

    v_A = phi_adj
    v_F = -(1.0 / k) * phi_adj
    with timer("  bwd: vjp A@phi, F@phi -> numerator", verbose=False):
        numerator   = Aphi_Fphi_vjp(xs_tensor, geo_data, context.SLAY, phi_fwd, v_A, v_F)
    denominator = (1.0 / k**2) * (phi_adj @ Fphi)
    dk_dp       = -numerator / denominator
    return (g * dk_dp, None, None)


@jax.custom_vjp
def NTdiff_solver(xs_tensor, params_raw_single, sample_id):
    keff, _ = _NTdiff_fwd(xs_tensor, params_raw_single, sample_id)
    return keff

NTdiff_solver.defvjp(_NTdiff_fwd, _NTdiff_bwd)


# ─────────────────────────────────────────────────────────────────────────────
# BATCH SOLVER — one custom VJP call per batch instead of one per sample
# ─────────────────────────────────────────────────────────────────────────────

def _NT_batch_fwd(xs_batch, params_raw_batch, sample_ids):
    """
    Forward pass for NTdiff_solver_batch.

    Calls pure_callback(_run_NT_solver) for each sample in the batch.
    At training time each call hits context._PRESOLVE_CACHE (populated by the
    parallel pre-solve step), so the eigenvalue solve itself is NOT repeated.
    All results are stacked and returned as residuals for the backward.

    xs_batch        : [B, n_regions, M]  float32
    params_raw_batch: [B, 6]             float32
    sample_ids      : [B]                int32
    Returns: keffs [B], residuals
    """
    B = xs_batch.shape[0]
    keffs_list, phi_fwd_list, phi_adj_list, Fphi_list = [], [], [], []

    for i in range(B):
        k, phi_fwd, phi_adj, Fphi = jax.pure_callback(
            _run_NT_solver,
            (
                jax.ShapeDtypeStruct((),            jnp.float32),
                jax.ShapeDtypeStruct((context.N_FLAT_MAX,), jnp.float32),
                jax.ShapeDtypeStruct((context.N_FLAT_MAX,), jnp.float32),
                jax.ShapeDtypeStruct((context.N_FLAT_MAX,), jnp.float32),
            ),
            xs_batch[i], params_raw_batch[i], sample_ids[i:i+1],
        )
        keffs_list.append(k)
        phi_fwd_list.append(phi_fwd)
        phi_adj_list.append(phi_adj)
        Fphi_list.append(Fphi)

    keffs     = jnp.stack(keffs_list)      # [B]
    phi_fwd_b = jnp.stack(phi_fwd_list)    # [B, context.N_FLAT_MAX]
    phi_adj_b = jnp.stack(phi_adj_list)    # [B, context.N_FLAT_MAX]
    Fphi_b    = jnp.stack(Fphi_list)       # [B, context.N_FLAT_MAX]

    residuals = (xs_batch, keffs, phi_fwd_b, phi_adj_b, Fphi_b,
                 params_raw_batch, sample_ids)
    return keffs, residuals


def _build_padded_geo_arrays(params_raw_batch_np):
    """
    For each sample in the batch, call precompute_geometry and pack the
    geometry arrays into fixed-size padded arrays of shape [context.I_MAX_SCAN(+1)].

    Padding rules that avoid numerical issues in the batched scan:
      - S_pad  : last valid S value is replicated into padding slots
      - V_pad  : padded with 1.0 so that S/(Delta_r * V) stays finite;
                 the validity mask (valid = i < I_valid) zeroes contributions
      - roc_pad: last valid region index is replicated
    """
    B = params_raw_batch_np.shape[0]
    S_pads   = np.ones( (B, context.I_MAX_SCAN + 1), dtype=np.float32)
    V_pads   = np.ones( (B, context.I_MAX_SCAN),     dtype=np.float32)
    roc_pads = np.zeros((B, context.I_MAX_SCAN),     dtype=np.int32)
    I_vals   = np.zeros(B, dtype=np.int32)

    for idx in range(B):
        geo_i    = update_geo(GEO, params_raw_batch_np[idx])
        gd       = precompute_geometry(geo_i)
        I_i      = gd['I']
        I_vals[idx] = I_i

        S_pads[idx, :I_i+1]  = np.array(gd['S'],              dtype=np.float32)[:I_i+1]
        V_pads[idx, :I_i]    = np.array(gd['V'],              dtype=np.float32)[:I_i]
        roc_np = np.array(gd['region_of_cell'], dtype=np.int32)
        roc_pads[idx, :I_i]  = roc_np
        if I_i < context.I_MAX_SCAN:
            # replicate last valid region into padding to avoid out-of-bound XS reads
            roc_pads[idx, I_i:] = roc_np[I_i - 1]

    return (jnp.array(S_pads), jnp.array(V_pads),
            jnp.array(roc_pads), jnp.array(I_vals))


def _NT_batch_bwd(residuals, g_batch):
    """
    Batched backward pass for NTdiff_solver_batch.

    Instead of calling Aphi_Fphi_vjp once per sample (35 000 separate JAX
    dispatch + compile events), this calls context._Aphi_Fphi_vjp_batch once,
    which is a jit-compiled vmap over the whole batch.

    g_batch: [B] upstream gradient d(loss)/d(keff_i)
    Returns: (grad_xs_batch [B, n_regions, M], None, None)
    """
    xs_batch, k_batch, phi_fwd_batch, phi_adj_batch, Fphi_batch, \
        params_raw_batch, sample_ids = residuals

    B = xs_batch.shape[0]

    # ── padded geometry arrays for all samples in this batch ─────────────────
    params_np = np.array(params_raw_batch)   # bring to CPU once
    S_pad_b, V_pad_b, roc_pad_b, I_vals = _build_padded_geo_arrays(params_np)

    # ── repack flat padded fluxes into [B, G, context.I_MAX_SCAN+1] ──────────────────
    # The flat padded layout is: [g0_cell0..g0_cellI | g1_cell0..g1_cellI | 0 0 ...]
    # with stride I_valid+1 per group.  reshape(G, context.I_MAX_SCAN+1) would be wrong
    # for any sample where I_valid < context.I_MAX_SCAN.  We unpack per-sample on CPU.
    G_val       = GEO.G
    phi_fwd_np  = np.array(phi_fwd_batch)    # [B, context.N_FLAT_MAX]
    phi_adj_np  = np.array(phi_adj_batch)    # [B, context.N_FLAT_MAX]
    k_np        = np.array(k_batch)          # [B]
    I_vals_np   = np.array(I_vals)           # [B]

    phi2d_np = np.zeros((B, G_val, context.I_MAX_SCAN + 1), dtype=np.float32)
    vA2d_np  = np.zeros((B, G_val, context.I_MAX_SCAN + 1), dtype=np.float32)
    vF2d_np  = np.zeros((B, G_val, context.I_MAX_SCAN + 1), dtype=np.float32)

    for b in range(B):
        I_i    = int(I_vals_np[b])
        inv_k  = -1.0 / float(k_np[b])
        for g in range(G_val):
            src = g * (I_i + 1)           # stride in the flat padded vector
            phi2d_np[b, g, :I_i + 1] = phi_fwd_np[b, src:src + I_i + 1]
            vA2d_np [b, g, :I_i + 1] = phi_adj_np[b, src:src + I_i + 1]
            vF2d_np [b, g, :I_i + 1] = inv_k * phi_adj_np[b, src:src + I_i + 1]

    phi2d_j = jnp.array(phi2d_np)   # [B, G, context.I_MAX_SCAN+1]
    vA2d_j  = jnp.array(vA2d_np)
    vF2d_j  = jnp.array(vF2d_np)

    # ── one batched VJP call  →  [B, n_regions, M] ───────────────────────────
    with timer("  bwd: vjp A@phi, F@phi -> numerator", verbose=False):
        numerators = context._Aphi_Fphi_vjp_batch(
            xs_batch,   # [B, n_regions, M]
            S_pad_b,    # [B, context.I_MAX_SCAN+1]
            V_pad_b,    # [B, context.I_MAX_SCAN]
            roc_pad_b,  # [B, context.I_MAX_SCAN]
            phi2d_j,    # [B, G, context.I_MAX_SCAN+1]
            vA2d_j,     # [B, G, context.I_MAX_SCAN+1]
            vF2d_j,     # [B, G, context.I_MAX_SCAN+1]
            I_vals,     # [B]
        )  # → [B, n_regions, M]

    # ── denominator ⟨φ†, Fφ⟩ / k² per sample ────────────────────────────────
    denominators = (1.0 / k_batch**2) * jnp.sum(
        phi_adj_batch * Fphi_batch, axis=1)                   # [B]

    # ── scale by upstream gradient and denominator ────────────────────────────
    scale    = -(g_batch / denominators)                       # [B]
    dk_dxs_b = scale[:, None, None] * numerators              # [B, n_regions, M]

    return (dk_dxs_b, None, None)


@jax.custom_vjp
def NTdiff_solver_batch(xs_batch, params_raw_batch, sample_ids):
    """
    Drop-in replacement for the per-sample NTdiff_solver loop in
    PEDSModel.__call__.  Returns keffs [B].  The backward computes
    d(loss)/d(xs_batch) with a single batched VJP call.
    """
    keffs, _ = _NT_batch_fwd(xs_batch, params_raw_batch, sample_ids)
    return keffs

NTdiff_solver_batch.defvjp(_NT_batch_fwd, _NT_batch_bwd)


def _solve_sample_worker(args):
    """
    Runs in a subprocess.
    Returns (i, k, phi_fwd, phi_adj, geo_data, epoch, worker_elapsed_s).

    worker_elapsed_s is the wall time inside this process, comparable to
    the per-sample backward time shown in the timing report.  The parent
    collects these to 'fwd pre-solve worker sample' in _TIMINGS, giving
    the same call count as 'bwd: vjp A@phi, F@phi -> numerator'.
    """
    import os, time as _time
    os.environ["JAX_PLATFORMS"] = "cpu"
    _t0 = _time.perf_counter()
    i, xs_np, params_np, sample_id_int, epoch = args
    geo_i    = update_geo(GEO, params_np)
    geo_data = precompute_geometry(geo_i)
    k, phi_fwd, phi_adj, Fphi = _run_NT_solver(xs_np, params_np, np.array([sample_id_int]))
    worker_elapsed = _time.perf_counter() - _t0
    return i, k, phi_fwd, phi_adj, geo_data, epoch, worker_elapsed


