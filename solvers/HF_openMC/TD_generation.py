
import os
os.environ['OPENMC_CROSS_SECTIONS'] = (
    '/global/scratch/users/caterinafrau/openmc_data/'
    'endfb-viii.0-hdf5/cross_sections.xml'
)
import warnings
import time
import json
import numpy as np
import pandas as pd
from pyDOE3 import lhs

import openmc

try:
    from .MC_solver import run_mc, apply_sample, _replace, persist_mc_results
    from .config_def import RegionSpec, SolverSettings, MCConfig
    from .config_run import CFG_ROD
except ImportError:
    from MC_solver import run_mc, apply_sample, _replace, persist_mc_results
    from config_def import RegionSpec, SolverSettings, MCConfig
    from config_run import CFG_ROD

warnings.filterwarnings('ignore', category=openmc.IDWarning)

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ══════════════════════════════════════════════════════════════════════════════
# SWEEP SETTINGS  — only thing to edit when changing the campaign
# ══════════════════════════════════════════════════════════════════════════════

N_SAMPLES   = int(os.environ.get("N_SAMPLES", 100))
LHS_SEED    = int(os.environ.get("LHS_SEED", 0))
CHECKPOINT  = 10          # save a checkpoint every N runs

name = f"seed_{LHS_SEED}"
_DATA_SUBDIR    = os.path.join('training_data', '2_aug', name)
DATA_DIR        = os.path.join(_PROJECT_ROOT, _DATA_SUBDIR)
DATASET_FILE    = os.path.join(DATA_DIR, 'LHS_full_dataset.csv')
DIAGNOSTICS_FILE = os.path.join(DATA_DIR, 'diagnostics.csv')
CHECKPOINT_FILE = os.path.join(DATA_DIR, 'checkpoint.csv')
TIMINGS_FILE = os.path.join(DATA_DIR, 'timings.csv')
SUMMARY_FILE = os.path.join(DATA_DIR, 'study_scalar_summary.csv')
LOG_FILE = os.path.join(DATA_DIR, 'generation_log.txt')

# ── Parameter bounds ──────────────────────────────────────────────────────────
# Keys follow the r{i}_{region.name}_{field} convention used by MC_solver.
# They must match the regions in whichever MCConfig you pass to generate_data().
# Comment any line out to fix that parameter at its nominal value.
BOUNDS = {
    'r0_b4c_rod_outer_radius':      (1.0,  6.0),
    'r0_b4c_rod_cr_fraction':       (0.0,   1.0),
    'r1_fuel_annulus_outer_radius': (10.0, 40.0),
    'r1_fuel_annulus_enrichment':   (1,  10.0),
    'r1_fuel_annulus_f_mod':        (0.40,   0.8),
    'r2_water_outer_radius':        (40.0, 80.0),
}

# Uncertainty control
# OpenMC already extends a single run from `batches` up to `trigger_max_batches`
# when keff_trigger_std is set (no restart, same simulation).  The outer retry
# loop below re-runs from scratch with doubled settings — expensive; keep off for
# production LHS sweeps unless you explicitly need it.
ENABLE_OUTER_STD_RETRIES = False
MAX_UNCERTAINTY_RETRIES = 3          # only used if ENABLE_OUTER_STD_RETRIES
STD_RETRY_GROWTH_FACTOR = 2

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _lhs_batch(n: int, seed: int) -> tuple[np.ndarray, list]:
    """Draw `n` LHS points scaled to BOUNDS. Returns (X_scaled, labels)."""
    labels       = list(BOUNDS.keys())
    bounds_array = np.array(list(BOUNDS.values()))          # (D, 2)
    X_unit       = lhs(len(BOUNDS), samples=n,
                       criterion='maximin', seed=seed)
    X_scaled     = bounds_array[:, 0] + X_unit * (bounds_array[:, 1]
                                                   - bounds_array[:, 0])
    return X_scaled, labels


def _iter_lhs_candidates(batch_size: int | None = None):
    """
    Yield LHS sample dicts indefinitely.

    Kept for future campaigns where rejected candidates are replaced by
    drawing additional LHS batches with seeds LHS_SEED, LHS_SEED+1, ...
    """
    n = batch_size or max(N_SAMPLES, 32)
    batch = 0
    while True:
        X_scaled, labels = _lhs_batch(n, seed=LHS_SEED + batch)
        for row in X_scaled:
            yield dict(zip(labels, row))
        batch += 1


def _override_output_paths(cfg, run_idx: int, data_dir: str,
                            dataset_file: str, diagnostics_file: str):
    """
    Point xs_output_path and plot_output at the training-data directory
    so LHS runs don't write into the single-run output folders.
    OpenMC XML/statepoint artifacts go into a dedicated per-run folder.
    verbose is set to False to keep the terminal readable during a sweep.
    """
    run_dir = os.path.join(data_dir, 'runs', f'run_{run_idx:04d}')
    new_settings = _replace(cfg.settings,
        openmc_work_dir        = run_dir,
        convergence_output_dir = run_dir,
        xs_output_path = dataset_file,
        plot_output    = os.path.join(data_dir, 'flux_plots', f'flux_run_{run_idx:04d}.png'),
        diagnostics_output_path=diagnostics_file,
        copy_statepoint=True,
        verbose        = False,
    )
    return _replace(cfg, settings=new_settings)


def _run_with_uncertainty_retries(cfg_i, std_target: float | None = None):
    """
    Run OpenMC for one LHS case.

    By default (ENABLE_OUTER_STD_RETRIES=False) this is a single OpenMC call.
    OpenMC's own keff trigger then extends batches in-place from `batches` to
    `trigger_max_batches` without restarting.

    If ENABLE_OUTER_STD_RETRIES=True, a failed case is re-run from scratch with
    doubled batch limits (expensive; discards prior work).
    """
    if std_target is None:
        std_target = cfg_i.settings.keff_trigger_std or 5e-4

    cfg_run = cfg_i
    retries = 0
    settings_used = []

    while True:
        sets = cfg_run.settings
        settings_used.append({
            'batches': int(sets.batches),
            'trigger_max_batches': int(sets.trigger_max_batches or sets.batches),
            'particles': int(sets.particles),
            'inactive': int(sets.inactive),
            'keff_trigger_std': float(sets.keff_trigger_std) if sets.keff_trigger_std is not None else None,
        })

        keff, flux_data, r_centers, lib, cells, conv = run_mc(cfg_run, persist_results=False)

        if not ENABLE_OUTER_STD_RETRIES or keff.std_dev <= std_target:
            persist_mc_results(cfg_run, keff, lib, cells, conv)
            return keff, flux_data, r_centers, lib, cells, conv, retries, settings_used

        if retries >= MAX_UNCERTAINTY_RETRIES:
            persist_mc_results(cfg_run, keff, lib, cells, conv)
            return keff, flux_data, r_centers, lib, cells, conv, retries, settings_used

        retries += 1
        new_batches = max(sets.batches + 1, int(np.ceil(sets.batches * STD_RETRY_GROWTH_FACTOR)))
        new_trigger_max = max(
            sets.trigger_max_batches or sets.batches,
            int(np.ceil((sets.trigger_max_batches or sets.batches) * STD_RETRY_GROWTH_FACTOR)),
        )
        print(
            f"  [std-retry {retries}/{MAX_UNCERTAINTY_RETRIES}] "
            f"keff_std={keff.std_dev:.6e} > {std_target:.6e}; "
            f"increasing batches {sets.batches}→{new_batches}, "
            f"trigger_max_batches {(sets.trigger_max_batches or sets.batches)}→{new_trigger_max}"
        )
        cfg_run = _replace(
            cfg_run,
            settings=_replace(
                sets,
                batches=new_batches,
                trigger_max_batches=new_trigger_max,
                keff_trigger_std=std_target,
            ),
        )


def _build_scalar_summary_row(run_id: int, attempt: int, keff, elapsed_s: float, retries: int, conv: dict):
    """
    Build scalar-only metrics row for study summary (no flux vectors).
    """
    return {
        'run_id': run_id,
        'attempt': attempt,
        'keff': float(keff.nominal_value),
        'keff_std': float(keff.std_dev),
        'runtime_s': float(elapsed_s),
        'openmc_runtime_total_s': conv.get('openmc_runtime_total_s'),
        'openmc_runtime_transport_s': conv.get('openmc_runtime_transport_s'),
        'openmc_runtime_inactive_s': conv.get('openmc_runtime_inactive_s'),
        'openmc_runtime_active_s': conv.get('openmc_runtime_active_s'),
        'std_retries': int(retries),
        'radial_power_peaking_factor': conv.get('radial_power_peaking_factor'),
        'fuel_avg_power_density': conv.get('fuel_avg_power_density'),
        'fuel_max_power_density': conv.get('fuel_max_power_density'),
        'flux_peak_factor_full': conv.get('flux_peak_factor_full'),
        'fuel_bin_count': conv.get('fuel_bin_count'),
        'keff_std_converged_flag': conv.get('keff_std_converged_flag'),
        'converged_flag': conv.get('converged_flag'),
    }


def _write_generation_log_header(fh, base_cfg: MCConfig) -> None:
    fh.write(f'{"═"*72}\n')
    fh.write('  LHS DATA GENERATION LOG\n')
    fh.write(f'{"═"*72}\n')
    fh.write(f'  Campaign dir : {DATA_DIR}\n')
    fh.write(f'  LHS seed     : {LHS_SEED}\n')
    fh.write(f'  N_SAMPLES    : {N_SAMPLES}\n')
    std_target = base_cfg.settings.keff_trigger_std or 5e-4
    fh.write(f'  STD target   : {std_target:.6e}\n')
    fh.write(f'  Outer restarts: {ENABLE_OUTER_STD_RETRIES}\n')
    fh.write(f'  Geometry     : {base_cfg.geometry}\n')
    fh.write(f'  Regions      : {[r.name for r in base_cfg.regions]}\n')
    fh.write(f'  Free params  : {list(BOUNDS.keys())}\n')
    fh.write(f'{"═"*72}\n\n')


def _fmt_float(v, prec: int = 6) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return 'n/a'
    return f'{float(v):.{prec}f}'


def _append_run_log(run_id: int, attempt: int, sample: dict, keff, elapsed_s: float,
                    retries: int, conv: dict, settings_used: list) -> None:
    last_settings = settings_used[-1] if settings_used else {}
    with open(LOG_FILE, 'a', encoding='utf-8') as fh:
        fh.write(f'{"-"*72}\n')
        fh.write(f'Run {run_id:04d}  |  candidate {attempt}  |  status: SUCCESS\n')
        fh.write(f'{"-"*72}\n')
        fh.write('LHS sample:\n')
        for k, v in sample.items():
            fh.write(f'  {k:<36s} = {v:.6f}\n')
        fh.write('\nResults:\n')
        fh.write(f'  k_eff                         = {keff.nominal_value:.8f}\n')
        fh.write(f'  k_eff std                     = {keff.std_dev:.8f}  '
                 f'({keff.std_dev / 1e-5:.2f} pcm)\n')
        fh.write(f'  batches used (statepoint)     = {conv.get("n_batches", "n/a")}\n')
        fh.write(f'  inactive batches              = {conv.get("n_inactive", last_settings.get("inactive", "n/a"))}\n')
        fh.write(f'  particles / batch             = {last_settings.get("particles", "n/a")}\n')
        fh.write(f'  trigger_max_batches           = {last_settings.get("trigger_max_batches", "n/a")}\n')
        fh.write(f'  keff trigger std              = {last_settings.get("keff_trigger_std", "n/a")}\n')
        fh.write(f'  std retries                   = {retries}\n')
        fh.write(f'  radial PPF (fuel-only)        = {_fmt_float(conv.get("radial_power_peaking_factor"))}\n')
        fh.write(f'  flux peak factor (all bins)   = {_fmt_float(conv.get("flux_peak_factor_full"))}\n')
        fh.write('\nTiming:\n')
        fh.write(f'  elapsed wall time (s)         = {elapsed_s:.2f}\n')
        fh.write(f'  OpenMC total runtime (s)        = {_fmt_float(conv.get("openmc_runtime_total_s"), 2)}\n')
        fh.write(f'  OpenMC transport runtime (s)    = {_fmt_float(conv.get("openmc_runtime_transport_s"), 2)}\n')
        fh.write(f'  OpenMC inactive runtime (s)     = {_fmt_float(conv.get("openmc_runtime_inactive_s"), 2)}\n')
        fh.write(f'  OpenMC active runtime (s)       = {_fmt_float(conv.get("openmc_runtime_active_s"), 2)}\n')
        if len(settings_used) > 1:
            fh.write('\nMC settings history (retries):\n')
            for i, s in enumerate(settings_used, start=1):
                fh.write(f'  attempt {i}: batches={s.get("batches")}, '
                         f'trigger_max={s.get("trigger_max_batches")}, '
                         f'particles={s.get("particles")}, '
                         f'inactive={s.get("inactive")}, '
                         f'keff_trigger_std={s.get("keff_trigger_std")}\n')
        fh.write('\n')


def _append_failure_log(attempt: int, sample: dict, exc: Exception) -> None:
    with open(LOG_FILE, 'a', encoding='utf-8') as fh:
        fh.write(f'{"-"*72}\n')
        fh.write(f'Candidate {attempt}  |  status: FAILED\n')
        fh.write(f'{"-"*72}\n')
        fh.write('LHS sample:\n')
        for k, v in sample.items():
            fh.write(f'  {k:<36s} = {v:.6f}\n')
        fh.write(f'\nError: {exc}\n\n')


def _write_generation_log_footer(n_full: int, n_attempt: int, failed: list,
                                 timings_mem: list) -> None:
    with open(LOG_FILE, 'a', encoding='utf-8') as fh:
        fh.write(f'{"═"*72}\n')
        fh.write('  SWEEP COMPLETE\n')
        fh.write(f'{"═"*72}\n')
        fh.write(f'  Successful runs : {n_full}/{N_SAMPLES}\n')
        fh.write(f'  Candidates tried: {n_attempt}\n')
        if failed:
            fh.write(f'  Failed attempts : {failed}\n')
        if timings_mem:
            avg_elapsed = float(np.mean([t['elapsed_s'] for t in timings_mem]))
            avg_openmc = float(np.mean([
                t['openmc_runtime_total_s'] for t in timings_mem
                if t.get('openmc_runtime_total_s') is not None and not np.isnan(t['openmc_runtime_total_s'])
            ])) if any(
                t.get('openmc_runtime_total_s') is not None and not np.isnan(t['openmc_runtime_total_s'])
                for t in timings_mem
            ) else np.nan
            fh.write(f'  Avg elapsed wall time (s)   : {avg_elapsed:.2f}\n')
            if not np.isnan(avg_openmc):
                fh.write(f'  Avg OpenMC total runtime (s): {avg_openmc:.2f}\n')
        fh.write(f'{"═"*72}\n')

# ══════════════════════════════════════════════════════════════════════════════
# MAIN GENERATOR
# ══════════════════════════════════════════════════════════════════════════════
def generate_data(base_cfg=CFG_ROD) -> pd.DataFrame:
    """
    Collect N_SAMPLES full MC runs over a true LHS candidate set in BOUNDS.

    The function is config-agnostic: pass any MCConfig and update BOUNDS to
    match its regions.  apply_sample() handles all field injection generically.

    Steps per accepted run
    ----------------------
    1. Draw / take next LHS candidate  →  flat sample dict
    2. apply_sample(base_cfg, sample)  →  perturbed MCConfig
    3. run_mc(cfg_i)                   →  keff, flux, XS
       └─ save_results() inside run_mc →  appends row to DATASET_FILE
    4. Checkpoint lightweight summary every CHECKPOINT full runs.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(DATASET_FILE):
        os.remove(DATASET_FILE)
        print(f'[info] removed existing {DATASET_FILE} — starting fresh\n')
    for stale_file in (DIAGNOSTICS_FILE, CHECKPOINT_FILE, TIMINGS_FILE, SUMMARY_FILE, LOG_FILE):
        if os.path.exists(stale_file):
            os.remove(stale_file)
            print(f'[info] removed existing {stale_file} — starting fresh\n')

    labels = list(BOUNDS.keys())
    X_scaled, _ = _lhs_batch(N_SAMPLES, seed=LHS_SEED)
    samples = [dict(zip(labels, row)) for row in X_scaled]
    failed   = []
    rows_mem = []
    timings_mem = []
    summary_mem = []
    n_full = 0
    n_attempt = 0

    std_target = base_cfg.settings.keff_trigger_std or 5e-4

    print(f'\n{"═"*64}')
    print(f'  LHS sweep  |  target {N_SAMPLES} full runs  |  {len(labels)} free params')
    print(f'  Config     :  {base_cfg.geometry}  |  '
          f'{len(base_cfg.regions)} regions: '
          f'{[r.name for r in base_cfg.regions]}')
    print(f'  MC fidelity:  {base_cfg.settings.particles} particles, '
          f'{base_cfg.settings.batches} batches '
          f'(extends to {base_cfg.settings.trigger_max_batches or base_cfg.settings.batches} in-run)')
    print(f'  keff target:  σ < {std_target:.6e}  |  outer restarts: {ENABLE_OUTER_STD_RETRIES}')
    print(f'  Output     :  {DATASET_FILE}')
    print(f'  Log file   :  {LOG_FILE}')
    print(f'{"═"*64}\n')

    with open(LOG_FILE, 'w', encoding='utf-8') as log_fh:
        _write_generation_log_header(log_fh, base_cfg)

    for sample in samples:
        n_attempt += 1

        print(f'\n{"-"*64}')
        print(f'  Candidate {n_attempt}  |  full run {n_full + 1}/{N_SAMPLES}')
        for k, v in sample.items():
            print(f'    {k:<36s} = {v:.4f}')
        print(f'{"-"*64}')

        # 1. Inject the sample into a copy of the base config
        cfg_i = apply_sample(base_cfg, sample)

        # 2. Redirect output paths — index by accepted full-run slot
        run_idx = n_full + 1
        cfg_i = _override_output_paths(cfg_i, run_idx, DATA_DIR, DATASET_FILE, DIAGNOSTICS_FILE)

        try:
            t0 = time.time()
            keff, flux_data, r_centers, lib, cells, conv, retries, settings_used = _run_with_uncertainty_retries(
                cfg_i, std_target=std_target
            )
            elapsed_s = time.time() - t0
            n_full += 1
            rows_mem.append({
                'run_id': n_full,
                'attempt': n_attempt,
                **sample,
                'keff': keff.nominal_value,
                'keff_std': keff.std_dev,
                'entropy_last': conv.get('entropy_last'),
                'entropy_rel_range_last': conv.get('entropy_rel_range_last'),
                'k_active_mean': conv.get('k_active_mean'),
                'k_active_std_sample': conv.get('k_active_std_sample'),
                'keff_std_converged_flag': conv.get('keff_std_converged_flag'),
                'k_window_shift_over_batch_noise': conv.get('k_window_shift_over_batch_noise'),
                'converged_flag': conv.get('converged_flag'),
                'openmc_runtime_total_s': conv.get('openmc_runtime_total_s'),
                'openmc_runtime_transport_s': conv.get('openmc_runtime_transport_s'),
                'openmc_runtime_inactive_s': conv.get('openmc_runtime_inactive_s'),
                'openmc_runtime_active_s': conv.get('openmc_runtime_active_s'),
                'radial_power_peaking_factor': conv.get('radial_power_peaking_factor'),
                'fuel_avg_power_density': conv.get('fuel_avg_power_density'),
                'fuel_max_power_density': conv.get('fuel_max_power_density'),
                'flux_peak_factor_full': conv.get('flux_peak_factor_full'),
                'fuel_bin_count': conv.get('fuel_bin_count'),
                'std_retries': retries,
                'elapsed_s': elapsed_s,
            })
            timings_mem.append({
                'run_id': n_full,
                'attempt': n_attempt,
                'elapsed_s': elapsed_s,
                'openmc_runtime_total_s': conv.get('openmc_runtime_total_s'),
                'openmc_runtime_transport_s': conv.get('openmc_runtime_transport_s'),
                'openmc_runtime_inactive_s': conv.get('openmc_runtime_inactive_s'),
                'openmc_runtime_active_s': conv.get('openmc_runtime_active_s'),
                'radial_power_peaking_factor': conv.get('radial_power_peaking_factor'),
                'flux_peak_factor_full': conv.get('flux_peak_factor_full'),
                'keff': keff.nominal_value,
                'keff_std': keff.std_dev,
                'std_target': std_target,
                'std_retries': retries,
                'settings_history': json.dumps(settings_used),
            })
            summary_mem.append(
                _build_scalar_summary_row(
                    run_id=n_full,
                    attempt=n_attempt,
                    keff=keff,
                    elapsed_s=elapsed_s,
                    retries=retries,
                    conv=conv,
                )
            )
            print(f'  ✓  k_eff = {keff.nominal_value:.6f}'
                  f' ± {keff.std_dev:.6f}   [{n_full}/{N_SAMPLES} full]'
                  f'  |  time={elapsed_s:.1f}s  retries={retries}')
            _append_run_log(
                run_id=n_full,
                attempt=n_attempt,
                sample=sample,
                keff=keff,
                elapsed_s=elapsed_s,
                retries=retries,
                conv=conv,
                settings_used=settings_used,
            )

        except Exception as exc:
            print(f'  ✗  Candidate {n_attempt} FAILED: {exc}')
            _append_failure_log(attempt=n_attempt, sample=sample, exc=exc)
            failed.append(n_attempt)
            rows_mem.append({
                'run_id': None,
                'attempt': n_attempt,
                **sample,
                'keff': np.nan,
                'keff_std': np.nan,
                'entropy_last': np.nan,
                'entropy_rel_range_last': np.nan,
                'k_active_mean': np.nan,
                'k_active_std_sample': np.nan,
                'keff_std_converged_flag': np.nan,
                'k_window_shift_over_batch_noise': np.nan,
                'converged_flag': np.nan,
                'openmc_runtime_total_s': np.nan,
                'openmc_runtime_transport_s': np.nan,
                'openmc_runtime_inactive_s': np.nan,
                'openmc_runtime_active_s': np.nan,
                'radial_power_peaking_factor': np.nan,
                'fuel_avg_power_density': np.nan,
                'fuel_max_power_density': np.nan,
                'flux_peak_factor_full': np.nan,
                'fuel_bin_count': np.nan,
                'std_retries': np.nan,
                'elapsed_s': np.nan,
            })

        if n_full > 0 and n_full % CHECKPOINT == 0:
            pd.DataFrame(rows_mem).to_csv(CHECKPOINT_FILE, index=False)
            pd.DataFrame(timings_mem).to_csv(TIMINGS_FILE, index=False)
            pd.DataFrame(summary_mem).to_csv(SUMMARY_FILE, index=False)
            print(f'  [checkpoint] {n_full} full runs → {CHECKPOINT_FILE}')

    pd.DataFrame(rows_mem).to_csv(CHECKPOINT_FILE, index=False)
    if timings_mem:
        pd.DataFrame(timings_mem).to_csv(TIMINGS_FILE, index=False)
    if summary_mem:
        pd.DataFrame(summary_mem).to_csv(SUMMARY_FILE, index=False)

    _write_generation_log_footer(n_full, n_attempt, failed, timings_mem)

    print(f'\n{"═"*64}')
    print(f'  Done.  {n_full}/{N_SAMPLES} full runs collected '
          f'({n_attempt} candidates tried).')
    if failed:
        print(f'  Failed attempts  : {failed}')
    if timings_mem:
        avg_time = float(np.mean([t['elapsed_s'] for t in timings_mem]))
        print(f'  Average runtime per successful run : {avg_time:.2f} s')
    print(f'  Full XS dataset      → {DATASET_FILE}')
    print(f'  Lightweight summary  → {CHECKPOINT_FILE}')
    print(f'  Runtime log          → {TIMINGS_FILE}')
    print(f'  Scalar study summary → {SUMMARY_FILE}')
    print(f'  Generation log       → {LOG_FILE}')
    print(f'{"═"*64}\n')

    return pd.read_csv(DATASET_FILE)

# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    df = generate_data(CFG_ROD)
    print(f'Dataset shape: {df.shape}')
