"""
keff sweep — OpenMC-direct inverse design study (companion to keff_scan_design.py).

Same Bayesian-optimization framework (``skopt.gp_minimize``) and the same
6-parameter search space / physical-validity rules as the surrogate study
in ``keff_scan_design.py``, but every candidate geometry is evaluated with a
full OpenMC eigenvalue run (via ``solvers/HF_openMC``) instead of the
trained PEDS surrogate. This gives a direct, apples-to-apples comparison of
search efficiency and wall-clock cost: same BO algorithm/settings/search
space, different "black box" answering each query.

Because one OpenMC eval costs minutes (not milliseconds), the search budget
here is small and fixed by ``N_SEARCH`` (default 100 total OpenMC calls),
of which ``N_INITIAL_POINTS`` are LHS-sampled (``initial_point_generator=
"lhs"``) before the GP acquisition takes over — mirrors the random/LHS init
fraction used in the surrogate study.

Search-space parity with the surrogate study
----------------------------------------------
``BOUNDS_RUN_DIR`` below is only used to resolve the *same* bounds_lo/
bounds_hi (and the same 5% inward edge buffer) that ``keff_scan_design.py``
uses for its BO dimensions — no PEDS model is loaded here. This intentionally
reproduces ``keff_scan_design.py``'s own dataset-resolution logic (detect the
data file from the run's ``code_snapshot_*.py``, otherwise fall back to the
global PEDS default dataset) so both studies search literally the same box.

Timing
------
Wall time is tracked per-OpenMC-call (``wall_time_hf_s``) and cumulatively,
plus overall study wall time (``keff_scan_openmc_timing.txt``) — directly
comparable to ``keff_scan_timing.txt`` from the surrogate study.

Usage
-----
    conda activate mc-env
    export OPENMC_CROSS_SECTIONS=/path/to/cross_sections.xml   # or edit CONFIG
    python keff_scan_design_openmc.py

(No arguments — edit the CONFIG section below to change anything about the
study, including the output directory naming.)
"""
from __future__ import annotations

import os
import sys
import csv
import glob
import json
import re
import shutil
import time
import numpy as np
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# Path setup (same convention as keff_scan_design.py / keff_scan_hf_verify.py)
# ─────────────────────────────────────────────────────────────────────────────
THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # .../inverse_design/keff_sweep
INVERSE_DESIGN_DIR = os.path.dirname(THIS_DIR)                  # .../inverse_design
MODULES_DIR = os.path.dirname(INVERSE_DESIGN_DIR)                # .../modules
PROJECT_ROOT = os.path.dirname(MODULES_DIR)
SOLVERS_DIR = os.path.join(PROJECT_ROOT, "solvers")

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SOLVERS_DIR)

from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args


# ═════════════════════════════════════════════════════════════════════════════
# CONFIG — every decision for this study lives here.
# ═════════════════════════════════════════════════════════════════════════════

# ---- Search-space parity source ------------------------------------------
# Only used to resolve bounds_lo/bounds_hi exactly as keff_scan_design.py
# does for this run (no surrogate model is loaded).
BOUNDS_RUN_DIR = os.path.join(MODULES_DIR, "RUNS", "precise_param_strat", "train_1000_seed_1")
# Fallback dataset if BOUNDS_RUN_DIR's code_snapshot can't be parsed — must
# match PEDS.py's ``_DATA_FILEPATH`` so the fallback is identical to the
# surrogate script's own behavior.
_DEFAULT_DATA_FILEPATH = os.path.join(PROJECT_ROOT, "data", "highfidelity", "17jul_0.8_1.2.npz")

# ---- Study identity / output location ----------------------------------------
STUDY_NAME = "openmc_direct_bo"
RESULTS_ROOT = os.path.join(THIS_DIR, "results")
RUN_TAG = datetime.now().strftime("%m%d%H")
OUT_DIR = os.path.join(RESULTS_ROOT, f"{STUDY_NAME}_{RUN_TAG}")
LATEST_RUN_MARKER = os.path.join(THIS_DIR, f"LATEST_RUN_{STUDY_NAME}.txt")

# ---- keff target (single target — OpenMC evals are far more expensive than
# the surrogate's, so the whole budget below goes to one target) -------------
KEFF_TARGET = 1.00

# ---- BO search budget -------------------------------------------------------
N_SEARCH = 100                      # total OpenMC evaluations (the "budget")
N_INITIAL_FRAC = 0.20                # fraction of N_SEARCH spent on LHS init
N_INITIAL_POINTS = int(N_SEARCH * N_INITIAL_FRAC)
TOP_K = 10                          # candidates kept (best |keff_HF - target|)
RANDOM_STATE = 0
ACQ_FUNC = "EI"

# ---- Free vs. fixed parameters ------------------------------------------------
# Same 6-vector layout as keff_scan_design.py: [b4c_r, cr_frac, fuel_r,
# enrichment, f_mod, water_r]. All free, matching the surrogate study.
PARAM_ORDER = ["b4c_r", "cr_frac", "fuel_r", "enrichment", "f_mod", "water_r"]
PARAM_SPEC = {
    "b4c_r":      "free",
    "cr_frac":    "free",
    "fuel_r":     "free",
    "enrichment": "free",
    "f_mod":      "free",
    "water_r":    "free",
}

# ---- Optional additional per-parameter range constraints ---------------------
# Intersected with bounds_lo/bounds_hi (never widened beyond them). Empty here
# to match the (unconstrained) surrogate study.
CUSTOM_BOUNDS: dict[str, tuple[float, float]] = {}

# ---- Inward edge buffer (same rationale/value as keff_scan_design.py) --------
BOUNDARY_BUFFER_FRAC = 0.05

# ---- Physical validity ---------------------------------------------------
ENFORCE_GEOMETRY_ORDERING = True
GEOMETRY_PENALTY = 50.0     # returned loss for an invalid (unevaluated) geometry

# ---- Selection / ranking of top-K -------------------------------------------
SELECTION_MODE = "abs_keff"   # "abs_keff" | "ppf_among_feasible"
KEFF_FEASIBILITY_PCM = 1000.0

# ---- Hybrid BO objective (drives search, independent of ranking above) -------
# Same semantics as keff_scan_design.py. Currently keff-only; flip on later to
# reproduce the Stage-2-style hybrid keff+PPF objective against OpenMC PPF.
USE_PPF_OBJECTIVE = False
COMPUTE_HF_PPF = False      # compute OpenMC PPF each eval (needed if the above
                             # is True, or SELECTION_MODE=="ppf_among_feasible")

INVALID_PENALTY = 1e3        # non-finite keff from OpenMC
INFEASIBLE_BASE = 100.0      # loss floor for outside-feasibility candidates

# ---- OpenMC fidelity (None = use HF_openMC.config_run.CFG_ROD's built-in
# defaults: 20,000 particles x 200 batches / 50 inactive, keff_trigger_std=5e-4
# early stop) — same fidelity keff_scan_hf_verify.py uses for HF verification.
PARTICLES: int | None = None
BATCHES: int | None = None
INACTIVE: int | None = None

XS_XML = os.environ.get(
    "OPENMC_CROSS_SECTIONS",
    "/global/scratch/users/caterinafrau/openmc_data/endfb-viii.0-hdf5/cross_sections.xml",
)

# ---- Per-iteration OpenMC artifacts ------------------------------------------
# Each OpenMC call writes XML/statepoint/convergence files under
# OUT_DIR/openmc_runs/iter_XXXX/. With N_SEARCH=100 calls this adds up fast;
# delete each iteration's work dir right after its scalar results are logged
# (the BO log CSV already has everything needed for analysis/plots).
CLEANUP_WORK_DIRS = True

# ═════════════════════════════════════════════════════════════════════════════
# End of CONFIG
# ═════════════════════════════════════════════════════════════════════════════

PARAM_RAW_IDX = {name: i for i, name in enumerate(PARAM_ORDER)}


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


def _detect_data_filepath(run_dir: str) -> str | None:
    """Mirrors ``evaluate_test_metrics._detect_data_filepath`` exactly (kept
    self-contained here so this script has no jax/flax dependency under
    mc-env). Reproducing this logic verbatim — including its fallback
    behavior — is what guarantees the same effective search-space bounds as
    keff_scan_design.py for this run."""
    snapshots = glob.glob(os.path.join(run_dir, "code_snapshot_*.py"))
    if not snapshots:
        return None
    with open(snapshots[0]) as f:
        content = f.read()
    for line in content.splitlines():
        if "_DATA_FILEPATH" in line and "=" in line:
            quoted = re.findall(r'["\']([^"\']+)["\']', line)
            if quoted:
                filename = quoted[-1]
                candidate = os.path.join(PROJECT_ROOT, "data", "highfidelity", filename)
                if os.path.exists(candidate):
                    return candidate
                candidate2 = os.path.join(PROJECT_ROOT, filename)
                if os.path.exists(candidate2):
                    return candidate2
    return None


def load_bounds(run_dir: str):
    """Resolve bounds_lo/bounds_hi exactly as keff_scan_design.py's
    ``build_context_and_model`` does: detect the dataset from run_dir's code
    snapshot, else fall back to the PEDS default dataset."""
    dataset_path = _detect_data_filepath(run_dir) or _DEFAULT_DATA_FILEPATH
    with np.load(dataset_path, allow_pickle=True) as d:
        bounds_lo = np.array(d["bounds_lo"], dtype=np.float32)
        bounds_hi = np.array(d["bounds_hi"], dtype=np.float32)
    print(f"  Bounds dataset      : {dataset_path}")
    return bounds_lo, bounds_hi


def build_free_dims(bounds_lo, bounds_hi):
    """Identical logic to keff_scan_design.py's build_free_dims."""
    free_names, fixed_values, dims = [], {}, []

    for name in PARAM_ORDER:
        idx = PARAM_RAW_IDX[name]
        trained_lo, trained_hi = float(bounds_lo[idx]), float(bounds_hi[idx])
        spec = PARAM_SPEC[name]

        if spec == "free":
            lo, hi = trained_lo, trained_hi
            if name in CUSTOM_BOUNDS:
                c_lo, c_hi = CUSTOM_BOUNDS[name]
                lo, hi = max(lo, c_lo), min(hi, c_hi)
                if lo >= hi:
                    raise ValueError(
                        f"CUSTOM_BOUNDS[{name!r}]=({c_lo},{c_hi}) does not "
                        f"overlap trained range [{trained_lo},{trained_hi}]"
                    )
            span = hi - lo
            buf = BOUNDARY_BUFFER_FRAC * span
            lo_eff, hi_eff = lo + buf, hi - buf
            free_names.append(name)
            dims.append(Real(lo_eff, hi_eff, name=name))
            print(f"  Free param bound   : {name:10s} trained=[{trained_lo:.4f},{trained_hi:.4f}]"
                  f"  effective(after buffer)=[{lo_eff:.4f},{hi_eff:.4f}]")
        else:
            value = float(spec)
            if not (trained_lo <= value <= trained_hi):
                raise ValueError(
                    f"Fixed value for {name!r}={value} is outside the trained "
                    f"bounds [{trained_lo},{trained_hi}]"
                )
            fixed_values[name] = value
            print(f"  Fixed param        : {name:10s} = {value:.4f} "
                  f"(trained range [{trained_lo:.4f},{trained_hi:.4f}])")

    return free_names, fixed_values, dims


def make_params_raw(free_values: dict, fixed_values: dict) -> np.ndarray:
    merged = {**fixed_values, **free_values}
    return np.array([merged[name] for name in PARAM_ORDER], dtype=np.float32)


def geometry_is_valid(params_raw: np.ndarray) -> bool:
    if not ENFORCE_GEOMETRY_ORDERING:
        return True
    b4c_r = params_raw[PARAM_RAW_IDX["b4c_r"]]
    fuel_r = params_raw[PARAM_RAW_IDX["fuel_r"]]
    water_r = params_raw[PARAM_RAW_IDX["water_r"]]
    return bool(b4c_r < fuel_r < water_r)


def compute_delta_rho_pcm(k_a: float, k_b: float) -> float:
    """Reactivity difference |Δρ| in pcm — same formula as PEDS metrics /
    keff_scan_design.py / keff_scan_hf_verify.py."""
    if not (np.isfinite(k_a) and np.isfinite(k_b) and k_a > 0 and k_b > 0):
        return float("nan")
    return abs(k_a - k_b) / (k_a * k_b) * 1e5


def is_keff_feasible(keff: float, target: float, tol_pcm: float = None) -> bool:
    tol = KEFF_FEASIBILITY_PCM if tol_pcm is None else tol_pcm
    dpcm = compute_delta_rho_pcm(keff, target)
    return np.isfinite(dpcm) and dpcm <= tol


def scan_loss(keff: float, target: float, ppf: float) -> float:
    """BO objective — identical semantics to keff_scan_design.py's scan_loss."""
    if not np.isfinite(keff):
        return INVALID_PENALTY
    if not USE_PPF_OBJECTIVE:
        return float((keff - target) ** 2)

    dpcm = compute_delta_rho_pcm(keff, target)
    if not np.isfinite(dpcm) or dpcm > KEFF_FEASIBILITY_PCM:
        scale = dpcm / max(KEFF_FEASIBILITY_PCM, 1.0) if np.isfinite(dpcm) else 1e3
        return float(INFEASIBLE_BASE + scale ** 2)
    if not np.isfinite(ppf) or ppf <= 0.0:
        return float(INFEASIBLE_BASE)
    return float(ppf)


def evaluate_candidate(params_raw: np.ndarray, target: float, work_dir: str) -> dict:
    """One BO query: geometry -> full OpenMC eigenvalue run -> keff_hf
    (+ optional OpenMC PPF). Returns a dict with keff, keff_std, ppf, loss,
    valid, converged, work_dir."""
    if not geometry_is_valid(params_raw):
        return dict(keff=float("nan"), keff_std=float("nan"), ppf=float("nan"),
                    loss=GEOMETRY_PENALTY, valid=False, converged=None, work_dir=None)

    from HF_openMC.design_api import verify_design

    result = verify_design(
        params_raw, work_dir=work_dir, xs_xml=XS_XML,
        particles=PARTICLES, batches=BATCHES, inactive=INACTIVE,
        verbose=False, compute_ppf=COMPUTE_HF_PPF,
    )
    keff = float(result["keff_hf"])
    keff_std = float(result["keff_hf_std"])
    ppf = float(result.get("ppf_hf", float("nan")))
    loss = scan_loss(keff, target, ppf)
    return dict(keff=keff, keff_std=keff_std, ppf=ppf, loss=loss, valid=True,
                converged=result.get("keff_std_converged"), work_dir=result["work_dir"])


def select_top_candidates(valid_results: list, target: float) -> list:
    """Apply SELECTION_MODE and return up to TOP_K candidates, best-first."""
    if SELECTION_MODE == "abs_keff":
        ranked = sorted(valid_results, key=lambda r: abs(r["keff_hf"] - target))
        return ranked[:TOP_K]

    if SELECTION_MODE == "ppf_among_feasible":
        feasible = [
            r for r in valid_results
            if is_keff_feasible(r["keff_hf"], target)
            and np.isfinite(r.get("ppf_hf", float("nan")))
            and r["ppf_hf"] > 0.0
        ]
        feasible.sort(key=lambda r: (r["ppf_hf"], abs(r["keff_hf"] - target)))
        return feasible[:TOP_K]

    raise ValueError(f"Unknown SELECTION_MODE={SELECTION_MODE!r}")


def write_run_markers(out_dir: str) -> None:
    with open(LATEST_RUN_MARKER, "w") as f:
        f.write(out_dir + "\n")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    openmc_runs_root = os.path.join(OUT_DIR, "openmc_runs")
    os.makedirs(openmc_runs_root, exist_ok=True)

    log_path = os.path.join(OUT_DIR, "keff_scan_openmc_run_log.txt")
    bo_log_path = os.path.join(OUT_DIR, "keff_scan_openmc_bo_log.csv")
    top_candidates_path = os.path.join(OUT_DIR, "keff_scan_openmc_top_candidates.csv")
    timing_path = os.path.join(OUT_DIR, "keff_scan_openmc_timing.txt")
    config_snapshot_path = os.path.join(OUT_DIR, "config_snapshot_openmc.json")

    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    log_fh = open(log_path, "w", buffering=1)
    sys.stdout = _Tee(orig_stdout, log_fh)
    sys.stderr = _Tee(orig_stderr, log_fh)

    start_dt = datetime.now()
    start_perf = time.perf_counter()
    print(f"Started at          : {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Study name          : {STUDY_NAME}")
    print(f"Out dir             : {OUT_DIR}")
    print(f"keff target         : {KEFF_TARGET}")
    print(f"N_SEARCH (budget)   : {N_SEARCH}  ({N_INITIAL_POINTS} LHS init)")
    print(f"TOP_K               : {TOP_K}")
    print(f"SELECTION_MODE      : {SELECTION_MODE}")
    print(f"KEFF_FEASIBILITY_PCM: {KEFF_FEASIBILITY_PCM}")
    print(f"USE_PPF_OBJECTIVE   : {USE_PPF_OBJECTIVE}")
    print(f"COMPUTE_HF_PPF      : {COMPUTE_HF_PPF}")
    print(f"Boundary buffer     : {BOUNDARY_BUFFER_FRAC * 100:.1f}% inward on each free param")
    print(f"OpenMC fidelity     : particles={PARTICLES} batches={BATCHES} inactive={INACTIVE} "
          f"(None = CFG_ROD defaults)")
    print(f"OPENMC_CROSS_SECTIONS = {XS_XML}")

    with open(config_snapshot_path, "w") as f:
        json.dump({
            "STUDY_NAME": STUDY_NAME,
            "BOUNDS_RUN_DIR": BOUNDS_RUN_DIR,
            "KEFF_TARGET": KEFF_TARGET, "N_SEARCH": N_SEARCH,
            "N_INITIAL_POINTS": N_INITIAL_POINTS, "TOP_K": TOP_K,
            "RANDOM_STATE": RANDOM_STATE, "ACQ_FUNC": ACQ_FUNC,
            "PARAM_SPEC": PARAM_SPEC, "CUSTOM_BOUNDS": {k: list(v) for k, v in CUSTOM_BOUNDS.items()},
            "BOUNDARY_BUFFER_FRAC": BOUNDARY_BUFFER_FRAC,
            "ENFORCE_GEOMETRY_ORDERING": ENFORCE_GEOMETRY_ORDERING,
            "SELECTION_MODE": SELECTION_MODE,
            "KEFF_FEASIBILITY_PCM": KEFF_FEASIBILITY_PCM,
            "USE_PPF_OBJECTIVE": USE_PPF_OBJECTIVE,
            "COMPUTE_HF_PPF": COMPUTE_HF_PPF,
            "PARTICLES": PARTICLES, "BATCHES": BATCHES, "INACTIVE": INACTIVE,
        }, f, indent=2)

    try:
        bounds_lo, bounds_hi = load_bounds(BOUNDS_RUN_DIR)
        free_names, fixed_values, dims = build_free_dims(bounds_lo, bounds_hi)
        if not free_names:
            raise ValueError("PARAM_SPEC leaves no free parameters — nothing to search.")

        bo_log_fh = open(bo_log_path, "w", newline="")
        bo_log_fieldnames = (
            ["keff_target", "iteration"] + PARAM_ORDER +
            ["keff_hf", "keff_hf_std", "ppf_hf", "loss", "valid", "keff_std_converged",
             "wall_time_hf_s", "cumulative_wall_time_s", "work_dir"]
        )
        bo_log_writer = csv.DictWriter(bo_log_fh, fieldnames=bo_log_fieldnames)
        bo_log_writer.writeheader()

        iteration_counter = {"n": 0}
        all_results = []

        @use_named_args(dimensions=dims)
        def objective(**free_values):
            it = iteration_counter["n"]
            iteration_counter["n"] += 1
            params_raw = make_params_raw(free_values, fixed_values)
            work_dir = os.path.join(openmc_runs_root, f"iter_{it:04d}")
            os.makedirs(work_dir, exist_ok=True)

            t0 = time.perf_counter()
            out = evaluate_candidate(params_raw, KEFF_TARGET, work_dir)
            elapsed = time.perf_counter() - t0
            cumulative = time.perf_counter() - start_perf

            row = {"keff_target": KEFF_TARGET, "iteration": it}
            row.update({name: float(params_raw[PARAM_RAW_IDX[name]]) for name in PARAM_ORDER})
            row.update({
                "keff_hf": out["keff"], "keff_hf_std": out["keff_std"], "ppf_hf": out["ppf"],
                "loss": out["loss"], "valid": out["valid"], "keff_std_converged": out["converged"],
                "wall_time_hf_s": f"{elapsed:.4f}", "cumulative_wall_time_s": f"{cumulative:.4f}",
                "work_dir": out["work_dir"] or "",
            })
            bo_log_writer.writerow(row)
            bo_log_fh.flush()

            rec = {name: float(params_raw[PARAM_RAW_IDX[name]]) for name in PARAM_ORDER}
            rec.update({"keff_target": KEFF_TARGET, "keff_hf": out["keff"], "keff_hf_std": out["keff_std"],
                        "ppf_hf": out["ppf"], "loss": out["loss"], "valid": out["valid"],
                        "wall_time_hf_s": elapsed})
            all_results.append(rec)

            feas = is_keff_feasible(out["keff"], KEFF_TARGET) if out["valid"] else False
            print(f"    [{it + 1:4d}/{N_SEARCH}] keff_hf={out['keff']:.5f}±{out['keff_std']:.5f}  "
                  f"feas={feas}  loss={out['loss']:.4e}  ({elapsed:.1f}s, cum={cumulative/60:.1f}min)")

            if CLEANUP_WORK_DIRS and out["work_dir"]:
                shutil.rmtree(out["work_dir"], ignore_errors=True)

            return out["loss"]

        print(f"\nRunning gp_minimize: {N_SEARCH} calls, {N_INITIAL_POINTS} LHS init, "
              f"seed={RANDOM_STATE}, target={KEFF_TARGET}")
        t_bo0 = time.perf_counter()
        gp_minimize(
            func=objective,
            dimensions=dims,
            n_calls=N_SEARCH,
            n_initial_points=N_INITIAL_POINTS,
            initial_point_generator="lhs",
            random_state=RANDOM_STATE,
            acq_func=ACQ_FUNC,
            verbose=False,
        )
        bo_elapsed = time.perf_counter() - t_bo0
        print(f"\nFinished BO in {bo_elapsed:.1f}s ({bo_elapsed / N_SEARCH:.2f}s/eval avg)")

        bo_log_fh.close()

        valid_results = [r for r in all_results if r["valid"]]
        n_feas = sum(1 for r in valid_results if is_keff_feasible(r["keff_hf"], KEFF_TARGET))
        print(f"Valid={len(valid_results)}  feasible(≤{KEFF_FEASIBILITY_PCM:.0f} pcm)={n_feas}")

        top = select_top_candidates(valid_results, KEFF_TARGET)
        top_fieldnames = (
            ["keff_target", "rank"] + PARAM_ORDER +
            ["keff_hf", "ppf_hf", "wall_time_hf_s"]
        )
        with open(top_candidates_path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=top_fieldnames)
            wr.writeheader()
            for rank, r in enumerate(top, start=1):
                r["rank"] = rank
                dpcm = compute_delta_rho_pcm(r["keff_hf"], KEFF_TARGET)
                print(f"  #{rank:2d}  keff_hf={r['keff_hf']:.5f}  dpcm_vs_tgt={dpcm:.1f}  "
                      f"ppf_hf={r['ppf_hf']:.4f}  b4c_r={r['b4c_r']:.3f} cr_frac={r['cr_frac']:.3f} "
                      f"fuel_r={r['fuel_r']:.3f} enrichment={r['enrichment']:.3f} "
                      f"f_mod={r['f_mod']:.3f} water_r={r['water_r']:.3f}")
                wr.writerow({k: r[k] for k in top_fieldnames})
        if len(top) < TOP_K:
            print(f"[warn] Only {len(top)} candidates selected (< TOP_K={TOP_K}). "
                  f"Consider loosening KEFF_FEASIBILITY_PCM.")

        write_run_markers(OUT_DIR)

        print(f"\nAll OpenMC evaluations -> {bo_log_path}")
        print(f"Top-{TOP_K} candidates      -> {top_candidates_path}")

    finally:
        end_dt = datetime.now()
        elapsed_sec = time.perf_counter() - start_perf
        with open(timing_path, "w") as tf:
            tf.write(f"Started at    : {start_dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
            tf.write(f"Finished at   : {end_dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
            tf.write(f"Elapsed time  : {elapsed_sec:.1f} s ({elapsed_sec/60:.2f} min)\n")
            tf.write(f"Study name    : {STUDY_NAME}\n")
            tf.write(f"keff target   : {KEFF_TARGET}\n")
            tf.write(f"N_SEARCH      : {N_SEARCH}\n")
            tf.write(f"Avg s/eval    : {elapsed_sec / max(N_SEARCH, 1):.2f}\n")
            tf.write(f"SELECTION     : {SELECTION_MODE}\n")
        print(f"\nFinished at  : {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Elapsed time : {elapsed_sec:.1f} s ({elapsed_sec/60:.2f} min)")
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
        log_fh.close()


if __name__ == "__main__":
    main()
