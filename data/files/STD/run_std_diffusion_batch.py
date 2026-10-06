#!/usr/bin/env python
"""
Batch baseline LF diffusion runs for STD subset CSV files.

Input CSV must contain:
  r0_b4c_rod_outer_radius, r0_b4c_rod_cr_fraction,
  r1_fuel_annulus_outer_radius, r1_fuel_annulus_enrichment,
  r1_fuel_annulus_f_mod, r2_water_outer_radius, keff
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "config_and_run"))

from solvers.NTdiffusion.diffusion_solver import get_xs_basedon_geo, run_diffusion_solver
from config_and_run.NTcode_config_data.config_def import (
    BoundaryCondition,
    BoundarySpec,
    GeometryConfig,
    MaterialSpec,
    MatProperties,
)


_FIXED_BC = BoundaryCondition(bc_type="vacuum")
_FIXED_MESH_SIZE = 1
_FIXED_G = 2
_FIXED_GEOMETRY = "cylindrical"
_REGION_SPECS = (
    MaterialSpec("b4c_rod", region_index=0),
    MaterialSpec("fuel_annulus", region_index=1),
    MaterialSpec("water", region_index=2),
)
_BOUNDARY_NAMES = ("CR_outer", "core_outer", "moderator_outer")


def geo_from_row(row: pd.Series) -> GeometryConfig:
    r0 = float(row["r0_b4c_rod_outer_radius"])
    r1 = float(row["r1_fuel_annulus_outer_radius"])
    r2 = float(row["r2_water_outer_radius"])

    boundaries = (
        BoundarySpec(name=_BOUNDARY_NAMES[0], radius=r0),
        BoundarySpec(name=_BOUNDARY_NAMES[1], radius=r1),
        BoundarySpec(name=_BOUNDARY_NAMES[2], radius=r2),
    )
    mat_props = MatProperties(
        cr_fraction=float(row["r0_b4c_rod_cr_fraction"]),
        enrichment=float(row["r1_fuel_annulus_enrichment"]),
        f_mod=float(row["r1_fuel_annulus_f_mod"]),
    )
    return GeometryConfig(
        G=_FIXED_G,
        regions=_REGION_SPECS,
        boundaries=boundaries,
        geometry=_FIXED_GEOMETRY,
        mat_properties=mat_props,
        bc=_FIXED_BC,
        mesh_size=_FIXED_MESH_SIZE,
    )


def delta_pcm(k_ref: float, k_solver: float) -> float:
    if k_ref <= 0 or k_solver <= 0:
        return float("nan")
    return ((k_solver - k_ref) / (k_solver * k_ref)) * 1e5


def run_batch(csv_path: Path, out_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    n = len(df)
    print(f"[batch] Loaded {n} rows from {csv_path}")

    solver_keff = np.full(n, np.nan)
    delta_vals = np.full(n, np.nan)
    statuses = [""] * n

    t0 = time.time()
    for idx, row in tqdm(df.iterrows(), total=n, desc="LF baseline", unit="case"):
        try:
            geo = geo_from_row(row)
            xs_tensor = get_xs_basedon_geo(geo)
            k_fwd, _, _ = run_diffusion_solver(xs_tensor, geo)
            k_ref = float(row["keff"])
            solver_keff[idx] = k_fwd
            delta_vals[idx] = delta_pcm(k_ref, k_fwd)
            statuses[idx] = "ok"
        except Exception as exc:  # keep run going on row failures
            statuses[idx] = f"ERROR: {exc}"

    elapsed = time.time() - t0
    df["solver_keff"] = solver_keff
    df["delta_pcm"] = delta_vals
    df["status"] = statuses

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    ok_mask = df["status"] == "ok"
    n_ok = int(ok_mask.sum())
    print(f"[batch] Completed {n_ok}/{n} in {elapsed:.1f}s ({elapsed/max(n_ok,1):.2f}s/case)")
    print(f"[batch] Saved -> {out_path}")
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline LF solver for one CSV dataset.")
    parser.add_argument("--csv", type=Path, required=True, help="Input CSV path.")
    parser.add_argument("--out", type=Path, required=True, help="Output CSV path.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_batch(args.csv, args.out)
