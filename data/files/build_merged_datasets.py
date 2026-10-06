"""Merge every per-run CSV under ``data/files`` into three canonical datasets.

A "case" is one reactor configuration, identified by its six geometry/material
knobs. The same case was re-exported many times over the project's history with
different column subsets; this script collapses all of those copies into:

``dataset_keff_xs.csv``
    Cases with the full two-group cross-section set: 6 parameters, ``keff``,
    ``keff_std`` and all 36 cross-section columns.
``dataset_keff.csv``
    Every case: 6 parameters, ``keff`` and ``keff_std``.
``dataset_incomplete.csv``
    Cases where at least one field could not be recovered from any source file,
    with a ``missing_fields`` column naming what is absent.
``dataset_extras.csv``
    Everything else that only some source files recorded: the diffusion-solver
    cross-check (``solver_keff``, ``delta_pcm``) and the OpenMC power/runtime
    metrics. Deliberately kept out of the two main datasets; blank where a case
    was never measured that way.

Dropped on purpose because they carry no information: source-file provenance
columns, the constant ``geometry``/``G``/``*_material`` columns, the constant
solver ``status`` column, and per-run bookkeeping ids.

Usage:
    python build_merged_datasets.py
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent

PARAMS = [
    "r0_b4c_rod_outer_radius",
    "r0_b4c_rod_cr_fraction",
    "r1_fuel_annulus_outer_radius",
    "r1_fuel_annulus_enrichment",
    "r1_fuel_annulus_f_mod",
    "r2_water_outer_radius",
]

MATERIALS = ["b4c_rod", "fuel_annulus", "water"]
REACTIONS = [
    ("diffusion-coefficient", 2),
    ("absorption", 2),
    ("nu-fission", 2),
    ("scatter matrix", 4),
    ("chi", 2),
]
XS = [
    f"{mat}_{rxn}_g{g}"
    for mat in MATERIALS
    for rxn, ngroups in REACTIONS
    for g in range(1, ngroups + 1)
]

VALUE_COLS = ["keff", "keff_std"] + XS

# Recorded for only part of the pool; collected separately in dataset_extras.csv.
EXTRA_COLS = [
    "solver_keff",
    "delta_pcm",
    "radial_power_peaking_factor",
    "fuel_avg_power_density",
    "fuel_max_power_density",
    "flux_peak_factor_full",
    "fuel_bin_count",
    "openmc_runtime_total_s",
    "openmc_runtime_transport_s",
    "openmc_runtime_inactive_s",
    "openmc_runtime_active_s",
    "elapsed_s",
    "std_retries",
]

ALL_NUMERIC = VALUE_COLS + EXTRA_COLS

# Parameters are stored at full float64 repr in every source file; the distinct
# cases are >1e-3 apart in every dimension, so rounding here only absorbs
# formatting noise and cannot merge genuinely different cases.
KEY_DECIMALS = 10

# Relative difference above which two source files are considered to disagree.
CONFLICT_RTOL = 1e-6

# This file's solver columns come from an earlier version of the diffusion
# solver: its ``keff`` matches the newer runs exactly, but ``solver_keff``
# differs on all 536 of its cases by ~0.5%. The STD runs supersede it.
SUPERSEDED = {"older_datasets/LHS_filt_with_diffusion.csv": {"solver_keff", "delta_pcm"}}

OUT_XS = ROOT / "dataset_keff_xs.csv"
OUT_KEFF = ROOT / "dataset_keff.csv"
OUT_INCOMPLETE = ROOT / "dataset_incomplete.csv"
OUT_EXTRAS = ROOT / "dataset_extras.csv"
OUTPUTS = {OUT_XS, OUT_KEFF, OUT_INCOMPLETE, OUT_EXTRAS}


def read_case_table(path: Path) -> pd.DataFrame | None:
    """Load ``path`` as a case table, or return None if it is not one."""
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:  # noqa: BLE001
        print(f"  skip (unreadable): {path.name}: {exc}")
        return None
    if not set(PARAMS).issubset(df.columns):
        return None
    # A few exports have stray repeated header rows inside the body.
    df = df[df[PARAMS[0]].astype(str).str.strip() != PARAMS[0]]
    for col in PARAMS + [c for c in ALL_NUMERIC if c in df.columns]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=PARAMS)


def main() -> None:
    values: dict[tuple, dict[str, float]] = {}
    # Rounding is only used to match rows; the parameters themselves are written
    # back out at the full precision of whichever file supplied them first.
    exact_params: dict[tuple, list[float]] = {}
    conflicts: list[tuple[tuple, str, float, float, str]] = []
    n_sources = 0

    for path in sorted(ROOT.rglob("*.csv")):
        if path in OUTPUTS:
            continue
        df = read_case_table(path)
        if df is None:
            continue
        n_sources += 1
        rel = str(path.relative_to(ROOT))
        ignored = SUPERSEDED.get(rel, frozenset())
        present = [c for c in ALL_NUMERIC if c in df.columns and c not in ignored]
        for _, row in df.iterrows():
            key = tuple(round(float(row[p]), KEY_DECIMALS) for p in PARAMS)
            case = values.setdefault(key, {})
            exact_params.setdefault(key, [float(row[p]) for p in PARAMS])
            for col in present:
                new = row[col]
                if pd.isna(new):
                    continue
                new = float(new)
                if col not in case:
                    case[col] = new
                    continue
                old = case[col]
                if abs(old - new) / max(abs(old), abs(new), 1e-30) > CONFLICT_RTOL:
                    conflicts.append((key, col, old, new, rel))

    print(f"read {n_sources} source files -> {len(values)} unique cases")
    if conflicts:
        print(f"WARNING: {len(conflicts)} conflicting values; first value kept")
        for key, col, old, new, src in conflicts[:10]:
            print(f"  {col} = {old:.10g} vs {new:.10g} ({src})")

    records = []
    for key, case in values.items():
        rec = dict(zip(PARAMS, exact_params[key]))
        rec.update({c: case.get(c) for c in ALL_NUMERIC})
        rec["missing_fields"] = ";".join(c for c in VALUE_COLS if c not in case)
        records.append(rec)

    full = pd.DataFrame.from_records(records, columns=PARAMS + ALL_NUMERIC + ["missing_fields"])
    full = full.sort_values(PARAMS).reset_index(drop=True)
    assert not full[PARAMS].duplicated().any(), "duplicate parameter rows in output"

    complete = full[full["missing_fields"] == ""]
    complete[PARAMS + VALUE_COLS].to_csv(OUT_XS, index=False)

    has_keff = full[full[["keff", "keff_std"]].notna().all(axis=1)]
    has_keff[PARAMS + ["keff", "keff_std"]].to_csv(OUT_KEFF, index=False)

    incomplete = full[full["missing_fields"] != ""]
    incomplete[PARAMS + VALUE_COLS + ["missing_fields"]].to_csv(OUT_INCOMPLETE, index=False)

    extras = full[full[EXTRA_COLS].notna().any(axis=1)]
    extras[PARAMS + ["keff", "keff_std"] + EXTRA_COLS].to_csv(OUT_EXTRAS, index=False)

    for path, df in (
        (OUT_XS, complete),
        (OUT_KEFF, has_keff),
        (OUT_INCOMPLETE, incomplete),
        (OUT_EXTRAS, extras),
    ):
        ncols = len(pd.read_csv(path, nrows=0).columns)
        print(f"  {path.name:28s} {len(df):5d} rows x {ncols:2d} cols")
    for col in EXTRA_COLS:
        print(f"      {col:32s} present for {int(extras[col].notna().sum()):5d} cases")


if __name__ == "__main__":
    main()
