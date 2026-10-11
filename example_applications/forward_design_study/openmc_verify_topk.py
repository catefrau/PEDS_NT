"""
OpenMC verification of shortlisted candidates from `peds_ensemble_screen.py`.

Inputs:
- run directory containing `keff_scan_top_candidates.csv`
  (from `KEFF_SCAN_OUT_DIR` or `LATEST_RUN.txt`).

Outputs:
- `keff_scan_hf_diagnostics.csv` (per-candidate detailed verification),
- `keff_scan_final_summary.csv` (compact final summary),
- `keff_scan_hf_timing.txt` (timing and run/skip counts).
"""
from __future__ import annotations

import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
LATEST_RUN_MARKER = THIS_DIR / "LATEST_RUN.txt"
OUT_DIR_OVERRIDE: str | None = None

XS_XML = os.environ.get(
    "OPENMC_CROSS_SECTIONS",
    "/global/scratch/users/caterinafrau/openmc_data/endfb-viii.0-hdf5/cross_sections.xml",
)
PARTICLES: int | None = None
BATCHES: int | None = None
INACTIVE: int | None = None
COMPUTE_HF_PPF = True

PARAM_ORDER = ["b4c_r", "cr_frac", "fuel_r", "enrichment", "f_mod", "water_r"]


def resolve_out_dir() -> Path:
    override = OUT_DIR_OVERRIDE or os.environ.get("KEFF_SCAN_OUT_DIR")
    if override:
        out_dir = Path(override.strip())
        if not out_dir.is_dir():
            raise FileNotFoundError(f"Override run dir does not exist: {out_dir}")
        return out_dir
    if not LATEST_RUN_MARKER.exists():
        raise FileNotFoundError(
            f"{LATEST_RUN_MARKER} not found. Run peds_ensemble_screen.py first, "
            "or set KEFF_SCAN_OUT_DIR."
        )
    out_dir = Path(LATEST_RUN_MARKER.read_text().strip())
    if not out_dir.is_dir():
        raise FileNotFoundError(f"Recorded run dir does not exist: {out_dir}")
    return out_dir


def load_top_candidates(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rec = {k: float(row[k]) for k in PARAM_ORDER}
            rec["keff_target"] = float(row["keff_target"])
            rec["rank"] = int(float(row["rank"]))
            rec["keff_peds"] = float(row["keff_peds"])
            rec["wall_time_peds_s"] = row.get("wall_time_peds_s", "")
            ppf = row.get("ppf_diffusion", "")
            rec["ppf_diffusion"] = float(ppf) if ppf not in ("", None) else float("nan")
            rows.append(rec)
    return rows


def candidate_key(rec: dict) -> tuple:
    return (round(rec["keff_target"], 6),) + tuple(round(rec[k], 9) for k in PARAM_ORDER)


def load_done_keys(out_csv: Path) -> set:
    if not out_csv.exists():
        return set()
    done = set()
    with open(out_csv, newline="") as f:
        for row in csv.DictReader(f):
            try:
                key = (round(float(row["keff_target"]), 6),) + tuple(round(float(row[k]), 9) for k in PARAM_ORDER)
                done.add(key)
            except (KeyError, ValueError):
                continue
    return done


def compute_delta_rho_pcm(k_a: float, k_b: float) -> float:
    if not (k_a > 0 and k_b > 0):
        return float("nan")
    return abs(k_a - k_b) / (k_a * k_b) * 1e5


def main() -> None:
    out_dir = resolve_out_dir()
    top_candidates_path = out_dir / "keff_scan_top_candidates.csv"
    diag_path = out_dir / "keff_scan_hf_diagnostics.csv"
    summary_path = out_dir / "keff_scan_final_summary.csv"
    timing_path = out_dir / "keff_scan_hf_timing.txt"
    hf_work_root = out_dir / "openmc_runs"
    hf_work_root.mkdir(parents=True, exist_ok=True)

    print(f"Out dir              : {out_dir}")
    print(f"Top candidates       : {top_candidates_path}")

    from solvers.HF_openMC.design_api import ensure_openmc_data, verify_design

    xs = ensure_openmc_data(XS_XML)
    print(f"OPENMC_CROSS_SECTIONS = {xs}")

    candidates = load_top_candidates(top_candidates_path)
    done = load_done_keys(diag_path)
    print(f"Candidates to verify : {len(candidates)}")
    print(f"Already done         : {len(done)}")

    diag_fieldnames = (
        ["keff_target", "rank"]
        + PARAM_ORDER
        + [
            "keff_peds",
            "ppf_diffusion",
            "wall_time_peds_s",
            "keff_hf",
            "keff_hf_std",
            "ppf_hf",
            "delta_pcm",
            "keff_std_converged",
            "wall_time_hf_s",
            "work_dir",
        ]
    )
    write_header = not diag_path.exists() or diag_path.stat().st_size == 0
    diag_fh = open(diag_path, "a", newline="")
    diag_writer = csv.DictWriter(diag_fh, fieldnames=diag_fieldnames)
    if write_header:
        diag_writer.writeheader()

    start_dt = datetime.now()
    start_perf = time.perf_counter()
    n_run = n_skip = 0

    for idx, rec in enumerate(candidates, start=1):
        key = candidate_key(rec)
        tag = f"target{rec['keff_target']:.3f}_rank{rec['rank']:02d}"
        if key in done:
            print(f"[{idx}/{len(candidates)}] SKIP {tag} (already verified)")
            n_skip += 1
            continue

        work_dir = hf_work_root / tag
        work_dir.mkdir(parents=True, exist_ok=True)
        params_raw = [rec[k] for k in PARAM_ORDER]
        print(f"\n[{idx}/{len(candidates)}] VERIFY {tag}  params={params_raw}")

        t0 = time.perf_counter()
        result = verify_design(
            params_raw,
            work_dir=str(work_dir),
            xs_xml=XS_XML,
            particles=PARTICLES,
            batches=BATCHES,
            inactive=INACTIVE,
            verbose=True,
            compute_ppf=COMPUTE_HF_PPF,
        )
        hf_elapsed = time.perf_counter() - t0

        keff_hf = result["keff_hf"]
        keff_peds = rec["keff_peds"]
        delta_pcm = compute_delta_rho_pcm(keff_peds, keff_hf)

        row = {
            "keff_target": rec["keff_target"],
            "rank": rec["rank"],
            **{k: rec[k] for k in PARAM_ORDER},
            "keff_peds": keff_peds,
            "ppf_diffusion": rec.get("ppf_diffusion", float("nan")),
            "wall_time_peds_s": rec.get("wall_time_peds_s", ""),
            "keff_hf": keff_hf,
            "keff_hf_std": result["keff_hf_std"],
            "ppf_hf": result.get("ppf_hf", float("nan")),
            "delta_pcm": delta_pcm,
            "keff_std_converged": result.get("keff_std_converged", ""),
            "wall_time_hf_s": f"{hf_elapsed:.2f}",
            "work_dir": result["work_dir"],
        }
        diag_writer.writerow(row)
        diag_fh.flush()
        done.add(key)
        n_run += 1
        print(
            f"  keff_HF={keff_hf:.6f}±{result['keff_hf_std']:.6f} "
            f"keff_PEDS={keff_peds:.6f} delta_pcm={delta_pcm:.1f} "
            f"ppf_HF={row['ppf_hf']:.4f} ({hf_elapsed:.1f}s)"
        )

    diag_fh.close()

    with open(diag_path, newline="") as f:
        all_rows = list(csv.DictReader(f))
    all_rows.sort(key=lambda r: (float(r["keff_target"]), int(r["rank"])))

    summary_fieldnames = [
        "keff_target",
        "rank",
        "keff_peds",
        "keff_hf",
        "delta_pcm_vs_target",
        "delta_pcm",
        "ppf_diffusion",
        "ppf_hf",
    ] + PARAM_ORDER
    with open(summary_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=summary_fieldnames)
        wr.writeheader()
        for r in all_rows:
            try:
                k_p = float(r.get("keff_peds", "nan"))
                k_t = float(r.get("keff_target", "nan"))
                r = dict(r)
                r["delta_pcm_vs_target"] = f"{compute_delta_rho_pcm(k_p, k_t):.6f}"
            except (TypeError, ValueError):
                r = dict(r)
                r.setdefault("delta_pcm_vs_target", "")
            wr.writerow({k: r.get(k, "") for k in summary_fieldnames})

    total_elapsed = time.perf_counter() - start_perf
    with open(timing_path, "w") as tf:
        tf.write(f"Started at   : {start_dt.strftime('%Y-%m-%d %H:%M:%S')}\n")
        tf.write(f"Finished at  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        tf.write(f"Elapsed time : {total_elapsed:.1f} s ({total_elapsed/60:.2f} min)\n")
        tf.write(f"Verified     : {n_run}\n")
        tf.write(f"Skipped      : {n_skip}\n")

    print(f"\nDone. ran={n_run} skipped={n_skip} elapsed={total_elapsed/60:.2f} min")
    print(f"Diagnostics    -> {diag_path}")
    print(f"Final summary  -> {summary_path}")


if __name__ == "__main__":
    main()
