"""
Diffusion-only recheck for a PEDS-feasible shortlist (no NN correction).

What this script does:
1) reads ranked feasible candidates from PEDS screen output,
2) re-evaluates each geometry with diffusion-only baseline (`predict_xs`),
3) compares diffusion-only vs PEDS (and vs HF when available),
4) writes full/slim CSV tables plus summary JSON/Markdown.

Inputs:
- `lhs_peds_feasible_ranked.csv` (or `--feasible-csv`),
- optional `lhs_peds_hf_comparison.csv` (or `--hf-comparison-csv`).

Outputs:
- `diffusion_only_recheck.csv` (full),
- `diffusion_only_recheck_slim.csv` (compact),
- `diffusion_only_summary.json`,
- `diffusion_only_summary.md`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from NTcode_config_data.config_run import GEO_CYL as GEO
from PEDS_subdivision.context import update_geo
from solvers.NTdiffusion.diffusion_solver import (
    precompute_geometry,
    predict_xs,
    run_diffusion_solver,
    xs_layout,
)

LATEST_RUN = THIS_DIR / "LATEST_RUN.txt"
PARAM_ORDER = ["b4c_r", "cr_frac", "fuel_r", "enrichment", "f_mod", "water_r"]


def compute_delta_rho_pcm(k_a, k_b):
    k_a = np.asarray(k_a, dtype=float)
    k_b = np.asarray(k_b, dtype=float)
    if k_b.ndim == 0:
        k_b = np.full(k_a.shape, float(k_b))
    out = np.full(k_a.shape, np.nan, dtype=float)
    mask = np.isfinite(k_a) & np.isfinite(k_b) & (k_a > 0) & (k_b > 0)
    out[mask] = np.abs(k_a[mask] - k_b[mask]) / (k_a[mask] * k_b[mask]) * 1e5
    return out


def cell_volumes(geo) -> np.ndarray:
    R = geo.boundaries[-1].radius
    I = int(R / geo.mesh_size)
    dr = geo.mesh_size
    edges = np.arange(I + 1, dtype=np.float64) * dr
    geom = geo.geometry.lower()
    if geom == "cylindrical":
        return np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    if geom == "spherical":
        return (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    return np.full(I, dr, dtype=np.float64)


def compute_ppf_from_phi(xs_tensor: np.ndarray, phi_fwd: np.ndarray, geo) -> float:
    xs = np.asarray(xs_tensor, dtype=np.float64)
    phi = np.asarray(phi_fwd, dtype=np.float64)
    gd = precompute_geometry(geo)
    roc = np.asarray(gd["region_of_cell"], dtype=int)
    I = int(gd["I"])
    G = int(geo.G)
    if phi.shape[0] != G or phi.shape[1] != I:
        return float("nan")

    nuf_slice = xs_layout(G)["nuSigma_f"]
    V = cell_volumes(geo)
    P = np.zeros(I, dtype=np.float64)
    for i in range(I):
        nuf = xs[roc[i], nuf_slice]
        P[i] = V[i] * float(np.sum(phi[:, i] * nuf))

    pos = P[P > 0.0]
    if pos.size == 0:
        return float("nan")
    return float(pos.max() / pos.mean())


def evaluate_diffusion_only(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(df)
    t0 = time.perf_counter()
    for i, (_, r) in enumerate(df.iterrows(), start=1):
        params = np.array([float(r[p]) for p in PARAM_ORDER], dtype=np.float64)
        geo_i = update_geo(GEO, params)
        t_case = time.perf_counter()
        xs = np.asarray(predict_xs(geo_i), dtype=np.float64)
        k_base, phi_fwd, _phi_adj = run_diffusion_solver(xs, geo_i)
        ppf_base = compute_ppf_from_phi(xs, phi_fwd, geo_i)
        wall = time.perf_counter() - t_case

        rows.append({
            "rank_peds": int(r["rank"]),
            "sample_id": int(r["sample_id"]),
            **{p: float(r[p]) for p in PARAM_ORDER},
            "keff_peds": float(r["keff_peds"]),
            "ppf_peds": float(r["ppf_diffusion"]),
            "delta_pcm_peds_vs_target": float(r["delta_rho_pcm"]),
            "keff_diffonly": float(k_base),
            "ppf_diffonly": float(ppf_base),
            "wall_time_diffonly_s": wall,
        })
        if i % 20 == 0 or i == n:
            elapsed = time.perf_counter() - t0
            print(f"[{i:4d}/{n}] elapsed={elapsed:.1f}s  ({elapsed/i:.3f}s/case)", flush=True)

    return pd.DataFrame(rows)


def merge_hf_metrics(diff_df: pd.DataFrame, hf_df: pd.DataFrame, target_keff: float) -> pd.DataFrame:
    hf = hf_df.copy()
    if "rank" not in hf.columns:
        return diff_df
    hf["rank"] = hf["rank"].astype(int)

    keep = ["rank"]
    for col in ["keff_hf", "keff_hf_std", "ppf_hf", "wall_time_hf_s", "work_dir"]:
        if col in hf.columns:
            keep.append(col)
    hf = hf[keep].rename(columns={"rank": "rank_peds"})

    merged = diff_df.merge(hf, how="left", on="rank_peds")
    merged["keff_target"] = float(target_keff)

    if "keff_hf" in merged.columns:
        merged["delta_pcm_peds_vs_hf"] = compute_delta_rho_pcm(merged["keff_peds"].to_numpy(), merged["keff_hf"].to_numpy())
        merged["delta_pcm_diffonly_vs_hf"] = compute_delta_rho_pcm(merged["keff_diffonly"].to_numpy(), merged["keff_hf"].to_numpy())
        merged["ppf_err_peds_vs_hf"] = merged["ppf_peds"].to_numpy() - merged["ppf_hf"].to_numpy()
        merged["ppf_err_diffonly_vs_hf"] = merged["ppf_diffonly"].to_numpy() - merged["ppf_hf"].to_numpy()
        merged["abs_ppf_err_peds_vs_hf"] = np.abs(merged["ppf_err_peds_vs_hf"])
        merged["abs_ppf_err_diffonly_vs_hf"] = np.abs(merged["ppf_err_diffonly_vs_hf"])
        merged["delta_pcm_hf_vs_target"] = compute_delta_rho_pcm(merged["keff_hf"].to_numpy(), target_keff)
    return merged


def write_slim_csv(merged: pd.DataFrame, out_path: Path, target_keff: float) -> pd.DataFrame:
    slim = pd.DataFrame({
        "sample_id": merged["sample_id"],
        "rank_peds": merged["rank_peds"],
        "keff_target": merged.get("keff_target", target_keff),
        "keff_peds": merged["keff_peds"],
        "ppf_peds": merged["ppf_peds"],
        "delta_rho_pcm_peds_vs_target": merged["delta_pcm_peds_vs_target"],
        "peds_status": merged.get("peds_status"),
        "keff_diffonly": merged["keff_diffonly"],
        "ppf_diffonly": merged["ppf_diffonly"],
        "delta_rho_pcm_diffonly_vs_target": merged.get("delta_rho_pcm_diffonly", merged.get("delta_pcm_diffonly_vs_target")),
        "diffusion_only_status": merged.get("diffusion_only_status"),
        "keff_hf": merged["keff_hf"] if "keff_hf" in merged.columns else np.nan,
        "ppf_hf": merged["ppf_hf"] if "ppf_hf" in merged.columns else np.nan,
        "delta_rho_pcm_hf_vs_target": merged["delta_pcm_hf_vs_target"] if "delta_pcm_hf_vs_target" in merged.columns else np.nan,
        "delta_rho_pcm_peds_vs_hf": merged["delta_pcm_peds_vs_hf"] if "delta_pcm_peds_vs_hf" in merged.columns else np.nan,
        "delta_rho_pcm_diffonly_vs_hf": merged["delta_pcm_diffonly_vs_hf"] if "delta_pcm_diffonly_vs_hf" in merged.columns else np.nan,
        "delta_rho_pcm_peds_vs_diffonly": compute_delta_rho_pcm(merged["keff_peds"].to_numpy(), merged["keff_diffonly"].to_numpy()),
        "ppf_err_peds_vs_hf": merged["ppf_err_peds_vs_hf"] if "ppf_err_peds_vs_hf" in merged.columns else np.nan,
        "ppf_err_diffonly_vs_hf": merged["ppf_err_diffonly_vs_hf"] if "ppf_err_diffonly_vs_hf" in merged.columns else np.nan,
        "ppf_err_peds_vs_diffonly": merged["ppf_peds"] - merged["ppf_diffonly"],
    })
    slim.to_csv(out_path, index=False)
    return slim


def build_summary(out_dir: Path, merged: pd.DataFrame, tol_pcm: float, target_keff: float, top_k: int, eval_wall_s: float):
    n = len(merged)
    merged = merged.copy()
    merged["feasible_peds"] = merged["delta_pcm_peds_vs_target"] <= tol_pcm
    merged["delta_pcm_diffonly_vs_target"] = compute_delta_rho_pcm(merged["keff_diffonly"].to_numpy(), target_keff)
    merged["feasible_diffonly"] = merged["delta_pcm_diffonly_vs_target"] <= tol_pcm
    merged["delta_rho_pcm_diffonly"] = merged["delta_pcm_diffonly_vs_target"]
    merged["diffusion_only_status"] = np.where(merged["feasible_diffonly"], "PASS", "FAIL")
    merged["peds_status"] = np.where(merged["feasible_peds"], "PASS", "FAIL")

    rank_df = merged.sort_values(by=["ppf_diffonly", "delta_pcm_diffonly_vs_target", "rank_peds"], ascending=[True, True, True]).reset_index(drop=True)
    rank_df["rank_diffonly"] = np.arange(1, len(rank_df) + 1, dtype=int)
    rank_df["selected_topk_diffonly"] = rank_df["rank_diffonly"] <= top_k
    rank_df["selected_topk_peds"] = rank_df["rank_peds"] <= top_k
    rank_df["selected_by_both"] = rank_df["selected_topk_diffonly"] & rank_df["selected_topk_peds"]

    rank_lookup = rank_df[["sample_id", "rank_diffonly", "selected_topk_diffonly", "selected_topk_peds", "selected_by_both"]]
    merged = merged.merge(rank_lookup, on="sample_id", how="left")

    n_peds_feas = int(np.sum(merged["feasible_peds"]))
    n_diff_feas = int(np.sum(merged["feasible_diffonly"]))
    n_peds_not_diff = int(np.sum(merged["feasible_peds"] & (~merged["feasible_diffonly"])))

    top_mask = merged["rank_peds"] <= top_k
    n_top = int(np.sum(top_mask))
    n_top_overlap = int(np.sum(merged["selected_by_both"]))
    n_top_only_peds = int(np.sum(merged["selected_topk_peds"] & (~merged["selected_topk_diffonly"])))

    hf_mask = merged["keff_hf"].notna() if "keff_hf" in merged.columns else np.zeros(len(merged), dtype=bool)
    n_hf = int(np.sum(hf_mask))
    mae_k_peds = mae_k_diff = mae_ppf_peds = mae_ppf_diff = np.nan
    if n_hf > 0:
        mae_k_peds = float(np.nanmean(merged.loc[hf_mask, "delta_pcm_peds_vs_hf"]))
        mae_k_diff = float(np.nanmean(merged.loc[hf_mask, "delta_pcm_diffonly_vs_hf"]))
        mae_ppf_peds = float(np.nanmean(merged.loc[hf_mask, "abs_ppf_err_peds_vs_hf"]))
        mae_ppf_diff = float(np.nanmean(merged.loc[hf_mask, "abs_ppf_err_diffonly_vs_hf"]))

    summary = {
        "target_keff": target_keff,
        "tol_pcm": tol_pcm,
        "n_cases_from_peds_feasible_file": n,
        "n_feasible_peds": n_peds_feas,
        "n_feasible_diffusion_only": n_diff_feas,
        "n_peds_feasible_but_not_diffusion_feasible": n_peds_not_diff,
        "pct_peds_feasible_not_diffusion_feasible": 100.0 * n_peds_not_diff / max(n_peds_feas, 1),
        "top_k": top_k,
        "n_peds_top_k": n_top,
        "n_overlap_peds_vs_diffusion_top_k": n_top_overlap,
        "n_peds_top_k_not_selected_by_diffusion_top_k": n_top_only_peds,
        "pct_peds_top_k_not_selected_by_diffusion_top_k": 100.0 * n_top_only_peds / max(n_top, 1),
        "diffusion_eval_wall_s": eval_wall_s,
        "diffusion_eval_mean_s_per_case": eval_wall_s / max(n, 1),
        "n_hf_rows_available_for_accuracy_comparison": n_hf,
        "mae_pcm_peds_vs_hf": mae_k_peds,
        "mae_pcm_diffusion_only_vs_hf": mae_k_diff,
        "mae_ppf_peds_vs_hf": mae_ppf_peds,
        "mae_ppf_diffusion_only_vs_hf": mae_ppf_diff,
    }

    (out_dir / "diffusion_only_summary.json").write_text(json.dumps(summary, indent=2))

    md = []
    md.append("# Diffusion-Only Recheck")
    md.append("")
    md.append(f"- Target/gate: `keff={target_keff}` with `|Δρ| <= {tol_pcm:.0f} pcm`")
    md.append(f"- Input set size: **{n}** PEDS-feasible geometries")
    md.append(f"- Diffusion-only runtime: **{eval_wall_s:.1f} s** ({eval_wall_s/max(n,1):.3f} s/case)")
    md.append("")
    md.append("## Selection Shift")
    md.append("")
    md.append(f"- PEDS-feasible in input: **{n_peds_feas}**")
    md.append(f"- Diffusion-only feasible (same gate): **{n_diff_feas}**")
    md.append(
        f"- PEDS-feasible but NOT diffusion-feasible: **{n_peds_not_diff}** "
        f"({100.0*n_peds_not_diff/max(n_peds_feas,1):.1f}%)"
    )
    md.append(
        f"- Top-{top_k} overlap (PEDS rank vs diffusion-only rank): "
        f"**{n_top_overlap}/{top_k}**"
    )
    md.append(
        f"- PEDS top-{top_k} NOT selected by diffusion-only top-{top_k}: "
        f"**{n_top_only_peds}/{top_k}** ({100.0*n_top_only_peds/max(top_k,1):.1f}%)"
    )
    md.append("")
    if n_hf > 0:
        md.append("## Accuracy vs HF")
        md.append("")
        md.append(f"- Rows with HF comparison: **{n_hf}**")
        md.append(f"- Mean `|Δρ(PEDS, HF)|`: **{mae_k_peds:.1f} pcm**")
        md.append(f"- Mean `|Δρ(diffusion-only, HF)|`: **{mae_k_diff:.1f} pcm**")
        md.append(f"- Mean `|PPF_PEDS - PPF_HF|`: **{mae_ppf_peds:.4f}**")
        md.append(f"- Mean `|PPF_diffonly - PPF_HF|`: **{mae_ppf_diff:.4f}**")
    else:
        md.append("## Accuracy vs HF")
        md.append("")
        md.append("- No HF comparison rows available.")
    md.append("")
    md.append("Generated files:")
    md.append("- `diffusion_only_recheck.csv` (full)")
    md.append("- `diffusion_only_recheck_slim.csv` (compact)")
    md.append("- `diffusion_only_summary.json`")
    md.append("- `diffusion_only_summary.md`")
    (out_dir / "diffusion_only_summary.md").write_text("\n".join(md) + "\n")

    merged = merged.sort_values("rank_peds").reset_index(drop=True)
    merged.to_csv(out_dir / "diffusion_only_recheck.csv", index=False)
    write_slim_csv(merged, out_dir / "diffusion_only_recheck_slim.csv", target_keff)


def parse_args():
    default_run_dir = Path(LATEST_RUN.read_text().strip()) if LATEST_RUN.exists() else THIS_DIR
    p = argparse.ArgumentParser(description="Diffusion-only recheck of PEDS-feasible geometries.")
    p.add_argument("--feasible-csv", type=Path, default=default_run_dir / "lhs_peds_feasible_ranked.csv")
    p.add_argument("--hf-comparison-csv", type=Path, default=default_run_dir / "lhs_peds_hf_comparison.csv")
    p.add_argument("--out-dir", type=Path, default=default_run_dir / "diffusion_only_check")
    p.add_argument("--target-keff", type=float, default=1.0)
    p.add_argument("--tol-pcm", type=float, default=1500.0)
    p.add_argument("--top-k", type=int, default=30)
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.feasible_csv.exists():
        raise FileNotFoundError(f"Missing feasible CSV: {args.feasible_csv}")

    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Feasible CSV: {args.feasible_csv}")
    print(f"Out dir: {args.out_dir}")

    df = pd.read_csv(args.feasible_csv)
    needed = {"rank", "sample_id", "keff_peds", "ppf_diffusion", "delta_rho_pcm", *PARAM_ORDER}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")

    t0 = time.perf_counter()
    diff_df = evaluate_diffusion_only(df)
    eval_wall = time.perf_counter() - t0

    hf_df = pd.DataFrame()
    if args.hf_comparison_csv.exists():
        hf_df = pd.read_csv(args.hf_comparison_csv)
        print(f"Using HF comparison file: {args.hf_comparison_csv} ({len(hf_df)} rows)")
    else:
        print(f"HF comparison CSV not found: {args.hf_comparison_csv} (skipping HF metrics)")

    merged = merge_hf_metrics(diff_df, hf_df, args.target_keff)
    build_summary(args.out_dir, merged, args.tol_pcm, args.target_keff, args.top_k, eval_wall)

    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Wrote: {args.out_dir / 'diffusion_only_recheck.csv'}")
    print(f"Wrote: {args.out_dir / 'diffusion_only_recheck_slim.csv'}")
    print(f"Wrote: {args.out_dir / 'diffusion_only_summary.json'}")
    print(f"Wrote: {args.out_dir / 'diffusion_only_summary.md'}")


if __name__ == "__main__":
    main()
