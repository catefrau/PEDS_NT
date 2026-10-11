"""
Canonical 5-seed PEDS ensemble screen for forward-design studies.

What this script does:
1) samples LHS geometries in the trained domain,
2) evaluates each geometry with 5 trained PEDS seeds,
3) aggregates ensemble mean/std for keff and diffusion PPF,
4) applies feasibility gate on keff, then ranks feasible designs,
5) writes `keff_scan_top_candidates.csv` for OpenMC verification.

Inputs/config:
- top-level CONFIG constants only.

Outputs:
- run folder under `results/` with eval tables, ranked shortlist,
  timing/config snapshots, and `LATEST_RUN.txt` pointer.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
import types
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from scipy.stats.qmc import LatinHypercube, scale as qmc_scale

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.PEDS_subdivision.analysis.evaluate_test_metrics import (
    _DEFAULT_DATA_FILEPATH,
    _detect_data_filepath,
    _read_indices_from_split_log,
    build_model_from_metadata,
    get_or_build_norm_stats,
    get_run_eval_context,
    load_checkpoint,
    load_dataset_arrays,
)
from PEDS_subdivision.context import update_geo
from solvers.NTdiffusion.diffusion_solver import xs_layout
import models.PEDS_subdivision.analysis.evaluate_test_metrics as _eval_metrics

# Stored training snapshots predate the current PEDS package layout.
# Evaluate with the live model code and the saved checkpoints.
_eval_metrics.USE_CURRENT_PEDS = True

RESULTS_DIR = THIS_DIR / "results"

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_FLAGS", f"--xla_force_host_platform_device_count={os.cpu_count()}")
os.environ.setdefault("XLA_CPU_ENABLE_FAST_MATH", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

# ============================================================================
# CONFIG
# ============================================================================
SEEDS = [0, 1, 2, 3, 4]
RUN_DIR_TEMPLATE = str(PROJECT_ROOT / "modules" / "RUNS" / "precise_param_strat" / "train_1000_seed_{seed}")
TRAIN_SIZE = 1000

N_LHS = 3000
LHS_SEED = 0
TARGET_KEFF = 1.0
TOL_PCM = 1000.0
TOP_K = 50
NN_BATCH = 128

# Choose ranking among feasible points:
# - "keff_first": sort by keff proximity first, then by PPF
# - "ppf_first" : sort by PPF first, then by keff proximity
# Set this here in the config block before running.
RANK_MODE = "keff_first"
if RANK_MODE not in {"ppf_first", "keff_first"}:
    raise ValueError(f"Unsupported RANK_MODE={RANK_MODE!r}")

RUN_TAG = "ens_clean"
OUT_DIR = RESULTS_DIR / f"lhs_peds_{RUN_TAG}_{datetime.now().strftime('%m%d%H%M')}"
LATEST_MARKER = THIS_DIR / "LATEST_RUN.txt"

STUDY_BOUNDS = {
    "b4c_r": (1.0, 6.0),
    "cr_frac": (0.0, 1.0),
    "fuel_r": (10.0, 40.0),
    "enrichment": (1.5, 10.0),
    "f_mod": (0.40, 0.80),
    "water_r": (40.0, 80.0),
}
PARAM_ORDER = ["b4c_r", "cr_frac", "fuel_r", "enrichment", "f_mod", "water_r"]
PARAM_RAW_IDX = {name: i for i, name in enumerate(PARAM_ORDER)}


def _stub_removed_plot_import() -> None:
    """Training snapshots import plot_functions.xs_heatmap at load time.

    That package is no longer in the tree. Screening never calls it.
    """
    if "plot_functions.xs_heatmap" in sys.modules:
        return
    plot_functions = types.ModuleType("plot_functions")
    xs_heatmap = types.ModuleType("plot_functions.xs_heatmap")
    xs_heatmap.plot_xs_subplots = lambda *args, **kwargs: None
    plot_functions.xs_heatmap = xs_heatmap
    sys.modules["plot_functions"] = plot_functions
    sys.modules["plot_functions.xs_heatmap"] = xs_heatmap


def compute_delta_rho_pcm(k_a: float, k_b: float) -> float:
    if not (np.isfinite(k_a) and np.isfinite(k_b) and k_a > 0 and k_b > 0):
        return float("nan")
    return abs(k_a - k_b) / (k_a * k_b) * 1e5


def geometry_is_valid(params_raw: np.ndarray) -> bool:
    return bool(
        params_raw[PARAM_RAW_IDX["b4c_r"]]
        < params_raw[PARAM_RAW_IDX["fuel_r"]]
        < params_raw[PARAM_RAW_IDX["water_r"]]
    )


def sample_lhs(n: int, lo: np.ndarray, hi: np.ndarray, seed: int) -> np.ndarray:
    sampler = LatinHypercube(d=6, seed=seed)
    unit = sampler.random(n=n)
    return qmc_scale(unit, lo, hi).astype(np.float32)


def resolve_run_dir(seed: int) -> str:
    primary = RUN_DIR_TEMPLATE.format(seed=seed)
    if os.path.isdir(primary):
        return primary
    fallback = PROJECT_ROOT / "example_applications" / "main_results" / "PEDS" / f"train_1000_seed_{seed}"
    if fallback.is_dir():
        return str(fallback)
    raise FileNotFoundError(f"Missing seed {seed} run dir. Tried:\n  {primary}\n  {fallback}")


def build_context_and_model(run_dir: str, train_size: int, seed: int):
    ckpt_path = os.path.join(run_dir, "checkpoints", "best_model.pkl")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")

    ctx = get_run_eval_context(run_dir)
    train_idx, _val_idx, _test_idx, split_source = _read_indices_from_split_log(run_dir, train_size, seed)
    print(f"  Split source: {split_source} ({len(train_idx)} train samples)")

    _geoms, _keffs, rawparams = load_dataset_arrays(run_dir)
    dataset_path = _detect_data_filepath(run_dir) or _DEFAULT_DATA_FILEPATH
    with np.load(dataset_path, allow_pickle=True) as d:
        bounds_lo = np.array(d["bounds_lo"], dtype=np.float32)
        bounds_hi = np.array(d["bounds_hi"], dtype=np.float32)

    logs_root = os.path.dirname(run_dir)
    norm_stats = get_or_build_norm_stats(
        LOGS_ROOT=logs_root,
        train_size=train_size,
        seed=seed,
        rawparams=rawparams,
        train_idx=train_idx,
        ctx=ctx,
    )

    state, meta = load_checkpoint(ckpt_path)
    model = build_model_from_metadata(ctx, meta, seed_for_init=seed)
    nnx.update(model, jax.tree_util.tree_map(jnp.asarray, state))
    print(f"  Loaded checkpoint: epoch={meta.get('epoch')}, val_mean_pcm={meta.get('val_mean_pcm')}")
    return ctx, model, norm_stats, bounds_lo, bounds_hi


def _phi_features_from_flux(ctx, params_raw: np.ndarray, phi_fwd_padded: np.ndarray) -> np.ndarray:
    geo_i = update_geo(ctx.GEO, params_raw)
    R = geo_i.boundaries[-1].radius
    I = int(R / geo_i.mesh_size)
    delta_r = geo_i.mesh_size
    G = geo_i.G
    region_radii = [b.radius for b in geo_i.boundaries]

    feats = []
    for gg in range(G):
        phi_g = np.asarray(phi_fwd_padded[gg * (I + 1): gg * (I + 1) + I], dtype=np.float64)
        r_prev = 0.0
        for r_reg in region_radii:
            centres = (np.arange(I) + 0.5) * delta_r
            mask = (centres >= r_prev) & (centres < r_reg)
            feats.append(float(np.mean(phi_g[mask])) if mask.any() else 0.0)
            r_prev = r_reg
    return np.asarray(feats, dtype=np.float32)


def _compute_diffusion_ppf(xs_tensor: np.ndarray, phi_fwd: np.ndarray, geo) -> float:
    xs = np.asarray(xs_tensor, dtype=np.float64)
    R = geo.boundaries[-1].radius
    I = int(R / geo.mesh_size)
    G = int(geo.G)
    dr = geo.mesh_size

    phi = np.asarray(phi_fwd, dtype=np.float64)
    if phi.ndim == 1:
        phi_by_group = np.zeros((G, I), dtype=np.float64)
        for gg in range(G):
            phi_by_group[gg] = phi[gg * (I + 1): gg * (I + 1) + I]
        phi = phi_by_group
    elif phi.shape != (G, I):
        return float("nan")

    edges = np.arange(I + 1, dtype=np.float64) * dr
    geom = geo.geometry.lower()
    if geom == "cylindrical":
        V = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    elif geom == "spherical":
        V = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    else:
        V = np.full(I, dr, dtype=np.float64)

    region_of_cell = np.zeros(I, dtype=int)
    boundaries = [b.radius for b in geo.boundaries]
    for i in range(I):
        rc = (i + 0.5) * dr
        for j, rb in enumerate(boundaries):
            if rc < rb:
                region_of_cell[i] = j
                break
        else:
            region_of_cell[i] = len(boundaries) - 1

    nuf_slice = xs_layout(G)["nuSigma_f"]
    P = np.zeros(I, dtype=np.float64)
    for i in range(I):
        nuf = xs[region_of_cell[i], nuf_slice]
        P[i] = V[i] * float(np.sum(phi[:, i] * nuf))

    pos = P[P > 0.0]
    if pos.size == 0:
        return float("nan")
    return float(pos.max() / pos.mean())


def effective_bounds_multi(bounds_lo_list: list[np.ndarray], bounds_hi_list: list[np.ndarray]):
    bounds_lo = np.max(np.stack(bounds_lo_list, axis=0), axis=0)
    bounds_hi = np.min(np.stack(bounds_hi_list, axis=0), axis=0)
    lo = np.zeros(6, dtype=np.float64)
    hi = np.zeros(6, dtype=np.float64)
    for i, name in enumerate(PARAM_ORDER):
        s_lo, s_hi = STUDY_BOUNDS[name]
        lo[i] = max(float(bounds_lo[i]), s_lo)
        hi[i] = min(float(bounds_hi[i]), s_hi)
        if lo[i] >= hi[i]:
            raise ValueError(f"Empty range for {name}: [{lo[i]}, {hi[i]}]")
        print(f"  {name:10s} study=[{s_lo},{s_hi}]  trained(intersection)=[{bounds_lo[i]:.4f},{bounds_hi[i]:.4f}]"
              f"  effective=[{lo[i]:.4f},{hi[i]:.4f}]")
    return lo, hi, bounds_lo, bounds_hi


def compute_baseline_and_phi(ctx, params_all: np.ndarray, valid_idx: np.ndarray):
    n = len(params_all)
    xs_baselines_all = np.zeros((n, *ctx.compute_batch_baselines(params_all[:1], ctx.GEO).shape[1:]), dtype=np.float32)
    phi_feats_all = np.zeros((n, ctx.GEO.G * 3), dtype=np.float32)

    t0 = time.perf_counter()
    for start in range(0, len(valid_idx), NN_BATCH):
        batch_ids = valid_idx[start:start + NN_BATCH]
        params_b = params_all[batch_ids]
        xs_b = np.asarray(ctx.compute_batch_baselines(params_b, ctx.GEO), dtype=np.float32)
        xs_baselines_all[batch_ids] = xs_b
        for j, sid in enumerate(batch_ids):
            _k, phi_fwd, _adj, _F = ctx._run_NT_solver(xs_b[j], params_b[j], np.array([int(sid)], dtype=np.int32))
            phi_feats_all[sid] = _phi_features_from_flux(ctx, params_b[j], phi_fwd)
        done = start + len(batch_ids)
        if done % (NN_BATCH * 4) == 0 or done == len(valid_idx):
            print(f"    [baseline+phi] {done:5d}/{len(valid_idx)}  {time.perf_counter()-t0:.1f}s elapsed", flush=True)
    print(f"  Baseline+phi (shared, physics-only): {time.perf_counter()-t0:.2f}s")
    return xs_baselines_all, phi_feats_all


def evaluate_with_model(ctx, model, norm_stats, bounds_lo, bounds_hi, params_all, xs_baselines_all, phi_feats_all, valid_idx):
    n = len(params_all)
    keffs = np.full(n, np.nan, dtype=np.float64)
    ppfs = np.full(n, np.nan, dtype=np.float64)
    wall_s = np.zeros(n, dtype=np.float64)

    span = (bounds_hi - bounds_lo).astype(np.float32)
    geoms_all = ((params_all - bounds_lo) / span).astype(np.float32)
    log_xs_mean_j = jnp.array(norm_stats["log_xs_mean"])
    log_xs_std_j = jnp.array(norm_stats["log_xs_std"])
    phi_mean = np.asarray(norm_stats["phi_mean"], dtype=np.float32)
    phi_std = np.asarray(norm_stats["phi_std"], dtype=np.float32)

    t_nn = t_solve = t_ppf = 0.0
    t0_all = time.perf_counter()
    for start in range(0, len(valid_idx), NN_BATCH):
        batch_ids = valid_idx[start:start + NN_BATCH]
        params_b = params_all[batch_ids]
        geoms_b = geoms_all[batch_ids]
        xs_baselines_b = xs_baselines_all[batch_ids]
        phi_norm_b = ((phi_feats_all[batch_ids] - phi_mean) / phi_std).astype(np.float32)

        t0 = time.perf_counter()
        xs_final = model.compute_xs(jnp.array(geoms_b), jnp.array(xs_baselines_b), jnp.array(phi_norm_b), log_xs_mean_j, log_xs_std_j)
        xs_final = np.asarray(xs_final)
        t_nn += time.perf_counter() - t0

        t0 = time.perf_counter()
        for j, sid in enumerate(batch_ids):
            sid_solver = int(sid) + 20_000_000
            t_one = time.perf_counter()
            k, phi_fwd, _adj, _F = ctx._run_NT_solver(xs_final[j], params_b[j], np.array([sid_solver], dtype=np.int32))
            keffs[sid] = float(k)
            geo_i = update_geo(ctx.GEO, params_b[j])
            t_ppf0 = time.perf_counter()
            ppfs[sid] = _compute_diffusion_ppf(xs_final[j], phi_fwd, geo_i)
            t_ppf += time.perf_counter() - t_ppf0
            wall_s[sid] = time.perf_counter() - t_one
        t_solve += time.perf_counter() - t0

        done = start + len(batch_ids)
        if done % (NN_BATCH * 4) == 0 or done == len(valid_idx):
            print(f"      [{done:5d}/{len(valid_idx)}]  {time.perf_counter()-t0_all:.1f}s elapsed", flush=True)

    total = time.perf_counter() - t0_all
    print(f"    NN compute_xs: {t_nn:.2f}s   corrected solve+PPF: {t_solve:.2f}s (of which PPF {t_ppf:.2f}s)   total: {total:.2f}s")
    return keffs, ppfs, wall_s, total


def _rank_key(row: dict):
    if RANK_MODE == "keff_first":
        return (row["delta_rho_pcm"], row["ppf_diffusion"], row["sample_id"])
    return (row["ppf_diffusion"], row["delta_rho_pcm"], row["sample_id"])


def main():
    _stub_removed_plot_import()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUT_DIR / "lhs_peds_ensemble_run_log.txt"

    class _Tee:
        def __init__(self, *files):
            self.files = files

        def write(self, data):
            for fh in self.files:
                fh.write(data)
                fh.flush()

        def flush(self):
            for fh in self.files:
                fh.flush()

    log_fh = open(log_path, "w", buffering=1)
    orig_out, orig_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(orig_out, log_fh)
    sys.stderr = _Tee(orig_err, log_fh)

    start_dt = datetime.now()
    start_perf = time.perf_counter()
    print(f"Started at   : {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Out dir      : {OUT_DIR}")
    print(f"Seeds        : {SEEDS}")
    print(f"N_LHS        : {N_LHS}  seed={LHS_SEED}")
    print(f"Target/tol   : keff={TARGET_KEFF}  |Δρ_ensemble_mean|≤{TOL_PCM} pcm")
    print(f"TOP_K        : {TOP_K}")
    print(f"RANK_MODE    : {RANK_MODE}")

    try:
        run_dirs = [resolve_run_dir(s) for s in SEEDS]
        ctxs, models, norm_stats_list = [], [], []
        bounds_lo_list, bounds_hi_list = [], []
        for seed, run_dir in zip(SEEDS, run_dirs):
            print(f"\n--- Loading ensemble member seed={seed} ---")
            ctx, model, norm_stats, blo, bhi = build_context_and_model(run_dir, TRAIN_SIZE, seed)
            ctxs.append(ctx)
            models.append(model)
            norm_stats_list.append(norm_stats)
            bounds_lo_list.append(blo)
            bounds_hi_list.append(bhi)

        lo, hi, bounds_lo, bounds_hi = effective_bounds_multi(bounds_lo_list, bounds_hi_list)

        print("\nSampling LHS ...")
        params_all = sample_lhs(N_LHS, lo, hi, LHS_SEED)
        valid_mask = np.array([geometry_is_valid(p) for p in params_all], dtype=bool)
        valid_idx = np.where(valid_mask)[0]
        print(f"  Geometry-valid: {len(valid_idx)} / {N_LHS}")

        print("\nComputing shared baseline XS + phi-features (model-independent) ...")
        xs_baselines_all, phi_feats_all = compute_baseline_and_phi(ctxs[0], params_all, valid_idx)

        n_models = len(SEEDS)
        keffs_per_model = np.full((n_models, N_LHS), np.nan, dtype=np.float64)
        ppfs_per_model = np.full((n_models, N_LHS), np.nan, dtype=np.float64)
        wall_per_model = np.zeros((n_models, N_LHS), dtype=np.float64)

        for m_idx, seed in enumerate(SEEDS):
            print(f"\n--- Evaluating ensemble member seed={seed} ({m_idx+1}/{n_models}) ---")
            k, p, wall, _total_t = evaluate_with_model(
                ctxs[m_idx], models[m_idx], norm_stats_list[m_idx],
                bounds_lo, bounds_hi, params_all, xs_baselines_all, phi_feats_all, valid_idx,
            )
            keffs_per_model[m_idx] = k
            ppfs_per_model[m_idx] = p
            wall_per_model[m_idx] = wall

        valid = valid_mask & np.all(np.isfinite(keffs_per_model), axis=0)
        keff_mean = np.nanmean(keffs_per_model, axis=0)
        keff_std = np.nanstd(keffs_per_model, axis=0)
        ppf_mean = np.nanmean(ppfs_per_model, axis=0)
        ppf_std = np.nanstd(ppfs_per_model, axis=0)
        wall_total = wall_per_model.sum(axis=0)

        rows = []
        for i in range(N_LHS):
            dpcm = compute_delta_rho_pcm(keff_mean[i], TARGET_KEFF) if valid[i] else float("nan")
            row = {
                "sample_id": i,
                **{name: float(params_all[i, j]) for j, name in enumerate(PARAM_ORDER)},
                "keff_peds": float(keff_mean[i]) if valid[i] else float("nan"),
                "keff_peds_std": float(keff_std[i]) if valid[i] else float("nan"),
                "ppf_diffusion": float(ppf_mean[i]) if valid[i] else float("nan"),
                "ppf_diffusion_std": float(ppf_std[i]) if valid[i] else float("nan"),
                "delta_rho_pcm": dpcm,
                "valid": int(valid[i]),
                "wall_time_peds_s": float(wall_total[i]),
            }
            for m_idx, seed in enumerate(SEEDS):
                row[f"keff_seed{seed}"] = float(keffs_per_model[m_idx, i]) if valid[i] else float("nan")
                row[f"ppf_seed{seed}"] = float(ppfs_per_model[m_idx, i]) if valid[i] else float("nan")
            rows.append(row)

        all_path = OUT_DIR / "lhs_peds_all_evals.csv"
        with open(all_path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)

        feasible = [
            r for r in rows
            if r["valid"]
            and np.isfinite(r["delta_rho_pcm"])
            and r["delta_rho_pcm"] <= TOL_PCM
            and np.isfinite(r["ppf_diffusion"])
            and r["ppf_diffusion"] > 0
        ]
        feasible.sort(key=_rank_key)
        for rank, r in enumerate(feasible, start=1):
            r["rank"] = rank

        ranked_path = OUT_DIR / "lhs_peds_feasible_ranked.csv"
        ranked_fields = ["rank", "sample_id"] + PARAM_ORDER + [
            "keff_peds", "keff_peds_std", "ppf_diffusion", "ppf_diffusion_std", "delta_rho_pcm", "wall_time_peds_s"
        ]
        with open(ranked_path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=ranked_fields)
            wr.writeheader()
            for r in feasible:
                wr.writerow({k: r[k] for k in ranked_fields})

        top_k = min(TOP_K, len(feasible))
        if top_k < TOP_K:
            print(f"  [warn] Only {len(feasible)} feasible candidates (< TOP_K={TOP_K}); writing all of them.")
        top = feasible[:top_k]

        top_path = OUT_DIR / "keff_scan_top_candidates.csv"
        top_fields = ["keff_target", "rank"] + PARAM_ORDER + ["keff_peds", "ppf_diffusion", "wall_time_peds_s"]
        with open(top_path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=top_fields)
            wr.writeheader()
            for r in top:
                wr.writerow({
                    "keff_target": TARGET_KEFF,
                    "rank": r["rank"],
                    **{name: r[name] for name in PARAM_ORDER},
                    "keff_peds": r["keff_peds"],
                    "ppf_diffusion": r["ppf_diffusion"],
                    "wall_time_peds_s": f"{r['wall_time_peds_s']:.4f}",
                })

        n_valid = int(valid.sum())
        n_feas = len(feasible)
        eval_wall = time.perf_counter() - start_perf
        mean_valid_s = float(np.mean(wall_total[valid])) if n_valid else float("nan")
        mean_ensemble_std_feasible = float(np.mean([r["keff_peds_std"] for r in feasible])) if feasible else float("nan")

        timing = {
            "started_at": start_dt.isoformat(timespec="seconds"),
            "n_lhs": N_LHS,
            "n_valid": n_valid,
            "n_feasible": n_feas,
            "top_k": len(top),
            "eval_wall_s": eval_wall,
            "mean_s_per_valid_all_seeds": mean_valid_s,
            "mean_keff_ensemble_std_on_feasible": mean_ensemble_std_feasible,
        }
        with open(OUT_DIR / "lhs_peds_timing.json", "w") as f:
            json.dump(timing, f, indent=2)

        with open(OUT_DIR / "config_snapshot.json", "w") as f:
            json.dump({
                "MODE": "ensemble",
                "SEEDS": SEEDS,
                "RUN_DIR_TEMPLATE": RUN_DIR_TEMPLATE,
                "RUN_DIRS": run_dirs,
                "TRAIN_SIZE": TRAIN_SIZE,
                "N_LHS": N_LHS,
                "LHS_SEED": LHS_SEED,
                "TARGET_KEFF": TARGET_KEFF,
                "TOL_PCM": TOL_PCM,
                "TOP_K": TOP_K,
                "NN_BATCH": NN_BATCH,
                "RANK_MODE": RANK_MODE,
                "STUDY_BOUNDS": STUDY_BOUNDS,
                "effective_lo": lo.tolist(),
                "effective_hi": hi.tolist(),
                "trained_lo_intersection": bounds_lo.tolist(),
                "trained_hi_intersection": bounds_hi.tolist(),
                "per_seed_trained_lo": [b.tolist() for b in bounds_lo_list],
                "per_seed_trained_hi": [b.tolist() for b in bounds_hi_list],
            }, f, indent=2)

        with open(LATEST_MARKER, "w") as f:
            f.write(str(OUT_DIR) + "\n")

        print("\n========== ENSEMBLE SCREEN SUMMARY ==========")
        print(f"Valid geometries (all seeds finite): {n_valid} / {N_LHS}")
        print(f"Feasible (|Δρ_mean|≤{TOL_PCM:.0f}): {n_feas} / {N_LHS} ({100*n_feas/N_LHS:.2f}%)")
        print(f"Ranking mode                    : {RANK_MODE}")
        print(f"Top-{len(top)} written               : {top_path}")
        print(f"Mean ensemble std(keff) on feasible: {mean_ensemble_std_feasible:.5f}")
        print(f"Eval wall time                  : {eval_wall:.1f}s ({eval_wall/60:.2f} min)")
        if top:
            b = top[0]
            print(
                f"Best (rank 1): keff_mean={b['keff_peds']:.5f} "
                f"dpcm={b['delta_rho_pcm']:.1f} ppf_mean={b['ppf_diffusion']:.4f} "
                f"keff_std={b['keff_peds_std']:.5f}"
            )
        print(f"All evals      : {all_path}")
        print(f"Ranked feasible: {ranked_path}")
        print(f"Next: python {THIS_DIR / 'openmc_verify_topk.py'}")

    finally:
        end_dt = datetime.now()
        elapsed = time.perf_counter() - start_perf
        with open(OUT_DIR / "lhs_peds_ensemble_timing.txt", "w") as tf:
            tf.write(f"Started at   : {start_dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
            tf.write(f"Finished at  : {end_dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
            tf.write(f"Elapsed time : {elapsed:.1f} s ({elapsed/60:.2f} min)\n")
        print(f"\nFinished at  : {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Elapsed time : {elapsed:.1f} s ({elapsed/60:.2f} min)")
        sys.stdout = orig_out
        sys.stderr = orig_err
        log_fh.close()


if __name__ == "__main__":
    main()
