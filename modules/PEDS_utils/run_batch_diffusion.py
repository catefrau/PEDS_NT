"""
run_batch_diffusion.py
──────────────────────
Loop over every row in full_results.csv, reconstruct a GeometryConfig
from that row's geometry parameters, call run_diffusion_solver (which
internally runs the polynomial regression to predict XS), and append:

  • solver_keff   – k_eff returned by the diffusion solver
  • delta_pcm     – reactivity difference in pcm:
                    Δρ = (1/k_ref − 1/k_solver) × 1e5  [pcm]
                    positive → solver is MORE reactive than MC reference

Usage
─────
  python run_batch_diffusion.py \
      --csv  reg_and_data/inputs/CR/full_results.csv \
      --out  reg_and_data/output/CR/full_results_with_solver.csv

All paths default to the values hard-coded in diffusion_solver.py so you
can also just run `python run_batch_diffusion.py` with no arguments.
"""

import argparse
import sys
import os
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ── Make sure project root is on the path ─────────────────────────────────────
# Adjust this if your project layout differs.
PROJECT_ROOT = Path(__file__).resolve().parent

CSV_DEFAULT = PROJECT_ROOT / 'FILES' / 'LHS_filtered_dataset.csv'
OUT_DEFAULT = PROJECT_ROOT / 'FILES' / 'LHS_filtered_dataset_with_solver.csv'

# Solver lives one level up from this script, under solver/
SOLVER_ROOT = PROJECT_ROOT.parent / 'solvers'
print("Looking for solver at:", SOLVER_ROOT)
print("Exists?", SOLVER_ROOT.exists())
print("Contents:", list(SOLVER_ROOT.iterdir()))
sys.path.insert(0, str(SOLVER_ROOT))

# ── Imports ───────────────────────────────────────────────────────────────────
from NTdiffusion.diffusion_solver import (
    get_xs_basedon_geo,
    run_diffusion_solver,
)
from NTcode_config_data.config_def import (
    GeometryConfig,
    MaterialSpec,
    BoundarySpec,
    BoundaryCondition,
    MatProperties,
)
# ══════════════════════════════════════════════════════════════════════════════
#  FIXED GEOMETRY TEMPLATE  (everything that is NOT swept in the CSV)
# ══════════════════════════════════════════════════════════════════════════════
# Region names and order must match the CSV column prefixes:
#   r0_b4c_rod_*  →  region_index=0
#   r1_fuel_annulus_*  →  region_index=1
#   r2_water_*  →  region_index=2

_FIXED_BC          = BoundaryCondition(bc_type='vacuum')
_FIXED_MESH_SIZE   = 1        # cm
_FIXED_G           = 2
_FIXED_GEOMETRY    = 'cylindrical'

_REGION_SPECS = (
    MaterialSpec('b4c_rod',      region_index=0),
    MaterialSpec('fuel_annulus', region_index=1),
    MaterialSpec('water',        region_index=2),
)

# Boundary name labels (cosmetic only, don't affect physics)
_BOUNDARY_NAMES = ('CR_outer', 'core_outer', 'moderator_outer')


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: build a GeometryConfig from one CSV row
# ══════════════════════════════════════════════════════════════════════════════

def geo_from_row(row: pd.Series) -> GeometryConfig:
    """
    Reconstruct a GeometryConfig from a single CSV row.

    Columns read (all present in full_results.csv):
      r0_b4c_rod_outer_radius       → boundaries[0].radius
      r0_b4c_rod_cr_fraction        → mat_properties.cr_fraction
      r1_fuel_annulus_outer_radius  → boundaries[1].radius
      r1_fuel_annulus_enrichment    → mat_properties.enrichment
      r1_fuel_annulus_f_mod         → mat_properties.f_mod
      r2_water_outer_radius         → boundaries[2].radius
    """
    r0 = float(row['r0_b4c_rod_outer_radius'])
    r1 = float(row['r1_fuel_annulus_outer_radius'])
    r2 = float(row['r2_water_outer_radius'])

    cr_fraction = float(row['r0_b4c_rod_cr_fraction'])
    enrichment  = float(row['r1_fuel_annulus_enrichment'])
    f_mod       = float(row['r1_fuel_annulus_f_mod'])

    boundaries = (
        BoundarySpec(name=_BOUNDARY_NAMES[0], radius=r0),
        BoundarySpec(name=_BOUNDARY_NAMES[1], radius=r1),
        BoundarySpec(name=_BOUNDARY_NAMES[2], radius=r2),
    )

    mat_props = MatProperties(
        cr_fraction = cr_fraction,
        enrichment  = enrichment,
        f_mod       = f_mod,
    )

    return GeometryConfig(
        G            = _FIXED_G,
        regions      = _REGION_SPECS,
        boundaries   = boundaries,
        geometry     = _FIXED_GEOMETRY,
        mat_properties = mat_props,
        bc           = _FIXED_BC,
        mesh_size    = _FIXED_MESH_SIZE,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  REACTIVITY DIFFERENCE  [pcm]
# ══════════════════════════════════════════════════════════════════════════════

def delta_pcm(k_ref: float, k_solver: float) -> float:
    """
    Δρ = ρ_solver − ρ_ref  in pcm,  where  ρ = (k−1)/k = 1 − 1/k

    Δρ [pcm] =(k_solver - k_ref) / (k_solver * k_ref) × 1e5

    Sign convention:
      positive  →  solver predicts a MORE reactive system than MC
      negative  →  solver predicts a LESS  reactive system than MC
    """
    if k_ref <= 0 or k_solver <= 0:
        return float('nan')
    return ((k_solver - k_ref) / (k_solver * k_ref)) * 1e5


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN BATCH LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_batch(csv_path: Path, out_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    n  = len(df)
    print(f"\n[batch] Loaded {n} rows from {csv_path}")
    print(f"[batch] Output → {out_path}\n")

    solver_keffs = np.full(n, np.nan)
    delta_pcms   = np.full(n, np.nan)
    statuses     = [''] * n

    t0 = time.time()

    for idx, row in tqdm(df.iterrows(), total=n, desc='Solving', unit='case'):
        try:
            geo       = geo_from_row(row)
            xs_tensor = get_xs_basedon_geo(geo)
            k_solver, _, _ = run_diffusion_solver(xs_tensor, geo)

            k_ref = float(row['keff'])

            solver_keffs[idx] = k_solver
            delta_pcms[idx]   = delta_pcm(k_ref, k_solver)
            statuses[idx]     = 'ok'

        except Exception as exc:
            statuses[idx] = f'ERROR: {exc}'
            tqdm.write(f"  [row {idx}] FAILED: {exc}")
            if os.environ.get('BATCH_TRACEBACK'):
                traceback.print_exc()

    elapsed = time.time() - t0

    # ── Append new columns ────────────────────────────────────────────────────
    df['solver_keff'] = solver_keffs
    df['delta_pcm']   = delta_pcms
    df['status']      = statuses

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    # ── Summary stats ─────────────────────────────────────────────────────────
    ok_mask  = df['status'] == 'ok'
    n_ok     = ok_mask.sum()
    n_failed = n - n_ok

    print(f"\n{'─'*60}")
    print(f"  Completed  : {n_ok}/{n}  ({elapsed:.1f} s total, "
          f"{elapsed/max(n_ok,1):.2f} s/case)")
    print(f"  Failed     : {n_failed}")

    if n_ok > 0:
        errs = df.loc[ok_mask, 'delta_pcm']
        print(f"\n  Δρ [pcm] statistics  (solver vs MC reference):")
        print(f"    mean  = {errs.mean():+.1f} pcm")
        print(f"    std   = {errs.std():.1f} pcm")
        print(f"    min   = {errs.min():+.1f} pcm")
        print(f"    max   = {errs.max():+.1f} pcm")
        print(f"    |max| = {errs.abs().max():.1f} pcm")

        abs_errs = (df.loc[ok_mask, 'solver_keff'] - df.loc[ok_mask, 'keff']).abs()
        print(f"\n  Δk_eff statistics  (solver vs MC reference):")
        print(f"    mean  = {abs_errs.mean():.5f}")
        print(f"    max   = {abs_errs.max():.5f}")

    print(f"{'─'*60}")
    print(f"  Saved → {out_path}\n")

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(
        description='Batch diffusion solver over all CSV rows.')
    p.add_argument(
        '--csv', type=Path,
        default=CSV_DEFAULT,
        help='Input CSV path (default: reg_and_data/inputs/CR/full_results.csv)')
    p.add_argument(
        '--out', type=Path,
        default=OUT_DEFAULT,
        help='Output CSV path')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    run_batch(args.csv, args.out)
