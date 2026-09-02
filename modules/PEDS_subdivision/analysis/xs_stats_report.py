#!/usr/bin/env python3
"""Build XS correction statistics + clipping summaries for one run."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODULES_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(MODULES_DIR) not in sys.path:
    sys.path.insert(0, str(MODULES_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from NTcode_config_data.config_run import GEO_CYL as GEO
from PEDS_subdivision.context import update_geo
from PEDS_subdivision.analysis.config import (
    STUDY_FOLDER,
    STUDY_PARENT_FOLDER,
    XS_SOURCE_RUN,
    xs_final_csv,
    xs_logratio_csv,
    xs_stats_outdir,
)
from solvers.NTdiffusion.diffusion_solver import predict_xs

XS_TYPES = ["D1", "D2", "Sa1", "Sa2", "nSf1", "nSf2", "Ss11", "Ss22", "Ss12", "Ss21", "chi1", "chi2"]
REGION_LABEL_MAP = {"CR": "absorber", "Core": "fuel", "Mod": "moderator", "Moderator": "moderator"}
REGION_ORDER = ["absorber", "fuel", "moderator"]
REGION_TO_BASELINE_IDX = {"CR": 0, "Core": 1, "Mod": 2, "Moderator": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-folder", default=STUDY_FOLDER)
    parser.add_argument("--study-parent-folder", default=STUDY_PARENT_FOLDER)
    parser.add_argument("--xs-source-run", default=XS_SOURCE_RUN)
    parser.add_argument("--final-xs-csv", type=Path, default=None)
    parser.add_argument("--logratio-csv", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, default=None)
    return parser.parse_args()


def _infer_param_cols(final_df: pd.DataFrame) -> List[str]:
    if "sample_idx" not in final_df.columns:
        raise ValueError("Expected column 'sample_idx' in final XS CSV.")
    stop_cols = {"keff_ref", "keff_pred", "delta_rho_pcm"}
    cols = list(final_df.columns)
    sample_i = cols.index("sample_idx")
    stop_i = min([cols.index(c) for c in stop_cols if c in cols], default=None)
    if stop_i is None or stop_i <= sample_i + 1:
        raise ValueError("Could not infer geometry parameter columns.")
    return cols[sample_i + 1 : stop_i]


def _available_xs_columns(final_df: pd.DataFrame) -> List[Tuple[str, str, str]]:
    triples = []
    for col in final_df.columns:
        for region in ["CR", "Core", "Mod", "Moderator"]:
            prefix = f"{region}_"
            if col.startswith(prefix):
                xs_type = col[len(prefix) :]
                if xs_type in XS_TYPES:
                    triples.append((col, region, xs_type))
                break
    if not triples:
        raise ValueError("No XS columns found in final XS CSV.")
    return triples


def compute_baseline_xs(params: np.ndarray) -> np.ndarray:
    baseline = np.zeros((params.shape[0], 3, len(XS_TYPES)), dtype=np.float64)
    for i in range(params.shape[0]):
        geo_i = update_geo(GEO, params[i])
        baseline[i] = np.asarray(predict_xs(geo_i), dtype=np.float64)
    return baseline


def build_correction_frame(final_df: pd.DataFrame) -> pd.DataFrame:
    param_cols = _infer_param_cols(final_df)
    xs_cols = _available_xs_columns(final_df)
    params = final_df[param_cols].to_numpy(dtype=np.float64)
    baseline = compute_baseline_xs(params)

    rows = []
    for i, row in final_df.iterrows():
        sid = int(row["sample_idx"])
        for col, region, xs_type in xs_cols:
            ridx = REGION_TO_BASELINE_IDX[region]
            xidx = XS_TYPES.index(xs_type)
            region_label = REGION_LABEL_MAP[region]
            base = float(baseline[i, ridx, xidx])
            final_val = float(row[col])
            delta = final_val - base
            if abs(base) > 1e-12:
                rel = delta / base
            elif abs(delta) <= 1e-12:
                rel = 0.0
            else:
                rel = np.nan
            rows.append(
                {
                    "sample_idx": sid,
                    "region": region_label,
                    "xs_type": xs_type,
                    "baseline_xs": base,
                    "final_xs": final_val,
                    "delta_xs": delta,
                    "abs_delta_xs": abs(delta),
                    "relative_delta": rel,
                    "abs_relative_delta": abs(rel) if np.isfinite(rel) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def summarize_corrections(corr_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    def quant(v: pd.Series, q: float) -> float:
        vv = v[np.isfinite(v)]
        return np.nan if vv.empty else float(np.quantile(vv, q))

    def agg_block(group: pd.DataFrame) -> pd.Series:
        rel = group["relative_delta"]
        return pd.Series(
            {
                "n": len(group),
                "n_rel_finite": int(np.isfinite(rel).sum()),
                "mean_delta_xs": group["delta_xs"].mean(),
                "mean_abs_delta_xs": group["abs_delta_xs"].mean(),
                "mean_relative_delta_pct": 100.0 * np.nanmean(rel),
                "mean_abs_relative_delta_pct": 100.0 * np.nanmean(np.abs(rel)),
                "q05_relative_delta_pct": 100.0 * quant(rel, 0.05),
                "q25_relative_delta_pct": 100.0 * quant(rel, 0.25),
                "q50_relative_delta_pct": 100.0 * quant(rel, 0.50),
                "q75_relative_delta_pct": 100.0 * quant(rel, 0.75),
                "q95_relative_delta_pct": 100.0 * quant(rel, 0.95),
            }
        )

    by_region_xs = corr_df.groupby(["region", "xs_type"], as_index=False).apply(agg_block).reset_index(drop=True)
    by_xs_type = corr_df.groupby(["xs_type"], as_index=False).apply(agg_block).reset_index(drop=True)
    return by_region_xs, by_xs_type


def summarize_clipping(logratio_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    work = logratio_df.copy()
    work["region"] = work["region"].map(REGION_LABEL_MAP).fillna(work["region"])
    work["clip_fraction"] = work["frac_at_lower_clip"] + work["frac_at_upper_clip"]
    work["clip_fraction"] = work["clip_fraction"].clip(lower=0.0, upper=1.0)

    epoch = (
        work.groupby("epoch", as_index=False)["clip_fraction"]
        .agg(mean_clip_fraction="mean", max_clip_fraction="max")
        .sort_values("epoch")
    )
    by_channel = (
        work.groupby(["region", "xs_idx"], as_index=False)["clip_fraction"]
        .agg(mean_clip_fraction="mean", max_clip_fraction="max")
        .sort_values("mean_clip_fraction", ascending=False)
    )
    overall = {
        "overall_mean_clip_fraction": float(work["clip_fraction"].mean()),
        "overall_max_clip_fraction": float(work["clip_fraction"].max()),
        "rows_with_any_clipping_frac": float((work["clip_fraction"] > 0).mean()),
    }
    return epoch, by_channel, overall


def _plot_correction_boxplot(corr_df: pd.DataFrame, outpath: Path) -> None:
    plt.rcParams.update({"font.size": 14, "axes.titlesize": 18, "axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13})
    plt.figure(figsize=(12, 5))
    data = []
    labels = []
    for xs in XS_TYPES:
        vals = corr_df.loc[corr_df["xs_type"] == xs, "relative_delta"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)] * 100.0
        data.append(vals)
        labels.append(xs)
    plt.boxplot(data, tick_labels=labels, showfliers=False)
    plt.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    plt.ylabel("Relative correction (%)")
    plt.title("XS relative correction distribution by XS type")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def _plot_correction_heatmap(by_region_xs: pd.DataFrame, outpath: Path) -> None:
    pivot = by_region_xs.pivot(index="region", columns="xs_type", values="mean_relative_delta_pct").reindex(index=REGION_ORDER)
    pivot = pivot[[x for x in XS_TYPES if x in pivot.columns]]
    arr = pivot.to_numpy(dtype=float)
    vmax = np.nanmax(np.abs(arr)) if np.isfinite(arr).any() else 1.0
    vmax = max(vmax, 1e-6)
    plt.rcParams.update({"font.size": 14, "axes.titlesize": 18, "axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 14})
    plt.figure(figsize=(12, 4))
    im = plt.imshow(arr, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, label="Mean relative correction (%)")
    plt.yticks(ticks=np.arange(len(pivot.index)), labels=list(pivot.index))
    plt.xticks(ticks=np.arange(len(pivot.columns)), labels=list(pivot.columns), rotation=45)
    plt.title("Mean relative correction by region and XS type")
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def _plot_clip_epoch(epoch_clip: pd.DataFrame, outpath: Path) -> None:
    plt.rcParams.update({"font.size": 14, "axes.titlesize": 18, "axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 13})
    plt.figure(figsize=(10, 4))
    plt.plot(epoch_clip["epoch"], 100.0 * epoch_clip["mean_clip_fraction"], marker="o", linewidth=1.2, label="mean clipped fraction")
    plt.plot(epoch_clip["epoch"], 100.0 * epoch_clip["max_clip_fraction"], marker="s", linewidth=1.0, alpha=0.8, label="max clipped fraction")
    plt.xlabel("Epoch")
    plt.ylabel("Clipped (%)")
    plt.title("Log-ratio clipping trend")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def _plot_clip_heatmap(clip_by_channel: pd.DataFrame, outpath: Path) -> None:
    ch = clip_by_channel.copy()
    ch["xs_type"] = ch["xs_idx"].astype(int).map({i: x for i, x in enumerate(XS_TYPES)})
    pivot = ch.pivot(index="region", columns="xs_type", values="mean_clip_fraction").reindex(index=REGION_ORDER)
    pivot = pivot[[x for x in XS_TYPES if x in pivot.columns]]
    arr = 100.0 * pivot.to_numpy(dtype=float)
    vmax = np.nanmax(arr) if np.isfinite(arr).any() else 1.0
    vmax = max(vmax, 1e-6)
    plt.rcParams.update({"font.size": 14, "axes.titlesize": 18, "axes.labelsize": 15, "xtick.labelsize": 13, "ytick.labelsize": 14})
    plt.figure(figsize=(12, 4))
    im = plt.imshow(arr, aspect="auto", cmap="magma", vmin=0.0, vmax=vmax)
    plt.colorbar(im, label="Mean clipped fraction (%)")
    plt.yticks(ticks=np.arange(len(pivot.index)), labels=list(pivot.index))
    plt.xticks(ticks=np.arange(len(pivot.columns)), labels=list(pivot.columns), rotation=45)
    plt.title("Average clipping by region and XS type")
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()


def _format_table(df: pd.DataFrame, n: int = 8) -> str:
    show = df.head(n).copy()
    cols = list(show.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in show.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                vals.append(f"{v:.4f}" if np.isfinite(v) else "nan")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_markdown_summary(outpath: Path, final_df: pd.DataFrame, corr_df: pd.DataFrame, by_region_xs: pd.DataFrame, by_xs_type: pd.DataFrame, clip_epoch: pd.DataFrame, clip_by_channel: pd.DataFrame, clip_overall: Dict[str, float]) -> None:
    top_xs = by_xs_type.sort_values("mean_abs_relative_delta_pct", ascending=False)
    top_region_xs = by_region_xs.sort_values("mean_abs_relative_delta_pct", ascending=False)
    top_clip = clip_by_channel.sort_values("mean_clip_fraction", ascending=False)
    lines = [
        "# XS correction statistics summary",
        "",
        "## Scope",
        f"- Configurations analyzed: **{len(final_df)}**",
        f"- XS entries compared: **{len(corr_df)}**",
        "- Baseline XS reconstructed with `predict_xs(update_geo(...))` from geometry parameters.",
        "- Correction definition: `delta = final_xs - baseline_xs`, `relative = delta / baseline_xs`.",
        "",
        "## Overall correction patterns",
        "- Highest mean absolute relative correction by XS type:",
        "",
        _format_table(top_xs[["xs_type", "mean_relative_delta_pct", "mean_abs_relative_delta_pct", "q50_relative_delta_pct", "q95_relative_delta_pct"]], n=10),
        "",
        "- Highest mean absolute relative correction by region and XS type:",
        "",
        _format_table(top_region_xs[["region", "xs_type", "mean_relative_delta_pct", "mean_abs_relative_delta_pct", "q50_relative_delta_pct", "q95_relative_delta_pct"]], n=12),
        "",
        "## Gradient clipping from logratio_saturation",
        f"- Mean clipped fraction across all epoch/region/xs rows: **{100.0 * clip_overall['overall_mean_clip_fraction']:.3f}%**",
        f"- Maximum observed clipped fraction in a single row: **{100.0 * clip_overall['overall_max_clip_fraction']:.3f}%**",
        f"- Rows with any clipping (`frac_at_lower_clip + frac_at_upper_clip > 0`): **{100.0 * clip_overall['rows_with_any_clipping_frac']:.2f}%**",
        "",
        "- Most clipped channels:",
        "",
        _format_table(top_clip.assign(mean_clip_pct=100.0 * top_clip["mean_clip_fraction"], max_clip_pct=100.0 * top_clip["max_clip_fraction"])[["region", "xs_idx", "mean_clip_pct", "max_clip_pct"]], n=12),
        "",
        "- Epoch-level clipping trend snapshot:",
        "",
        _format_table(clip_epoch.assign(mean_clip_pct=100.0 * clip_epoch["mean_clip_fraction"], max_clip_pct=100.0 * clip_epoch["max_clip_fraction"])[["epoch", "mean_clip_pct", "max_clip_pct"]], n=min(10, len(clip_epoch))),
        "",
    ]
    outpath.write_text("\n".join(lines))


def generate_xs_stats(final_xs_csv: Path, logratio_csv: Path, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    final_df = pd.read_csv(final_xs_csv)
    corr_df = build_correction_frame(final_df)
    by_region_xs, by_xs_type = summarize_corrections(corr_df)
    logratio_df = pd.read_csv(logratio_csv)
    clip_epoch, clip_by_channel, clip_overall = summarize_clipping(logratio_df)
    corr_df.to_csv(outdir / "xs_corrections_detailed.csv", index=False)
    by_region_xs.to_csv(outdir / "xs_correction_summary_by_region_xstype.csv", index=False)
    by_xs_type.to_csv(outdir / "xs_correction_summary_by_xstype.csv", index=False)
    clip_epoch.to_csv(outdir / "logratio_clip_summary_by_epoch.csv", index=False)
    clip_by_channel.to_csv(outdir / "logratio_clip_summary_by_region_xs.csv", index=False)
    _plot_correction_boxplot(corr_df, outdir / "correction_boxplot_by_xstype.png")
    _plot_correction_heatmap(by_region_xs, outdir / "correction_heatmap_mean_relative_pct.png")
    _plot_clip_epoch(clip_epoch, outdir / "logratio_clipping_epoch_trend.png")
    _plot_clip_heatmap(clip_by_channel, outdir / "logratio_clipping_heatmap.png")
    write_markdown_summary(
        outpath=outdir / "xs_statistics_summary.md",
        final_df=final_df,
        corr_df=corr_df,
        by_region_xs=by_region_xs,
        by_xs_type=by_xs_type,
        clip_epoch=clip_epoch,
        clip_by_channel=clip_by_channel,
        clip_overall=clip_overall,
    )
    print(f"Wrote XS stats artifacts to: {outdir}")


def main() -> None:
    args = parse_args()
    final_xs = args.final_xs_csv or xs_final_csv(args.study_folder, args.study_parent_folder, args.xs_source_run)
    logratio = args.logratio_csv or xs_logratio_csv(args.study_folder, args.study_parent_folder, args.xs_source_run)
    outdir = args.outdir or xs_stats_outdir(args.study_folder, args.study_parent_folder)
    generate_xs_stats(final_xs_csv=final_xs, logratio_csv=logratio, outdir=outdir)


if __name__ == "__main__":
    main()

