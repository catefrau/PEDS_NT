#!/usr/bin/env python
"""
Build an LHS-like final dataset by prioritizing parameter-space uniformity.

Primary objective (lower is better):
  - mean 1D L2 discrepancy (marginal uniformity)
  - mean 2D pairwise chi-square statistic + 2D L2 discrepancy

Hard constraint:
  - keff_std <= MAX_KEFF_STD
  - optional keff bounds (disabled by default)

Workflow
--------
1. Filter by std (and optional keff bounds).
2. Apply starting PARAM_BOUNDS.
3. Greedily trim margins only when uniformity improves significantly.
4. Select N_SAMPLES with space-filling (maximin) greedy search, shortlisting
   low-std candidates each step.
5. Optional local swap refinement on the uniformity score.
6. Save metrics, plots, and final_dataset.csv.

Edit config.py, then run:
    python build_final_dataset.py
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import config as cfg


PARAM_COLS = list(cfg.PARAM_BOUNDS.keys())
ALL_COLS = PARAM_COLS + ["keff", "keff_std", "file_name"]
PAIR_COLS = list(combinations(PARAM_COLS, 2))


def load_source() -> pd.DataFrame:
    df = pd.read_csv(cfg.INPUT_CSV)
    missing = set(ALL_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {cfg.INPUT_CSV}: {sorted(missing)}")
    return df


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(subset=cfg.DEDUP_COLS).reset_index(drop=True)


def apply_hard_filters(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["keff_std"] <= cfg.MAX_KEFF_STD
    if cfg.USE_KEFF_BOUNDS:
        keff_lo, keff_hi = cfg.KEFF_BOUNDS
        mask &= (df["keff"] >= keff_lo) & (df["keff"] <= keff_hi)
    return df.loc[mask].copy()


def apply_param_bounds(df: pd.DataFrame, bounds: dict[str, tuple[float, float]]) -> pd.DataFrame:
    mask = np.ones(len(df), dtype=bool)
    for col, (lo, hi) in bounds.items():
        mask &= (df[col] >= lo) & (df[col] <= hi)
    return df.loc[mask].copy()


def l2_discrepancy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total == 0:
        return float("nan")
    probs = counts / total
    uniform = np.full_like(probs, 1.0 / len(probs), dtype=float)
    return float(np.sqrt(np.mean((probs - uniform) ** 2)))


def marginal_l2(values: np.ndarray, lo: float, hi: float, n_bins: int) -> float:
    counts, _ = np.histogram(values, bins=np.linspace(lo, hi, n_bins + 1))
    return l2_discrepancy(counts)


def pairwise_2d_metrics(
    df: pd.DataFrame,
    bounds: dict[str, tuple[float, float]],
    n_bins: int,
) -> pd.DataFrame:
    rows = []
    for c1, c2 in PAIR_COLS:
        lo1, hi1 = bounds[c1]
        lo2, hi2 = bounds[c2]
        counts, _, _ = np.histogram2d(
            df[c1].to_numpy(),
            df[c2].to_numpy(),
            bins=[
                np.linspace(lo1, hi1, n_bins + 1),
                np.linspace(lo2, hi2, n_bins + 1),
            ],
        )
        flat = counts.ravel()
        chi2, p_value = stats.chisquare(flat)
        rows.append(
            {
                "param_x": c1,
                "param_y": c2,
                "chi2_stat": float(chi2),
                "p_value": float(p_value),
                "l2_discrepancy": l2_discrepancy(flat),
                "likely_non_uniform_p_lt_0_05": bool(p_value < 0.05),
            }
        )
    return pd.DataFrame(rows)


def uniformity_metrics_1d(
    df: pd.DataFrame,
    bounds: dict[str, tuple[float, float]],
    n_bins: int,
) -> pd.DataFrame:
    rows = []
    for col in PARAM_COLS:
        lo, hi = bounds[col]
        values = df[col].to_numpy()
        counts, _ = np.histogram(values, bins=np.linspace(lo, hi, n_bins + 1))
        chi2, p_value = stats.chisquare(counts)
        rows.append(
            {
                "parameter": col,
                "n_samples": len(values),
                "bounds_lo": lo,
                "bounds_hi": hi,
                "chi2_stat": float(chi2),
                "p_value": float(p_value),
                "l2_discrepancy": l2_discrepancy(counts),
                "likely_non_uniform_p_lt_0_05": bool(p_value < 0.05),
            }
        )
    return pd.DataFrame(rows)


def combined_uniformity_score(
    df: pd.DataFrame,
    bounds: dict[str, tuple[float, float]],
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    metrics_1d = uniformity_metrics_1d(df, bounds, cfg.N_BINS_1D)
    metrics_2d = pairwise_2d_metrics(df, bounds, cfg.N_BINS_2D)
    score = (
        cfg.WEIGHT_L2_1D * metrics_1d["l2_discrepancy"].mean()
        + cfg.WEIGHT_CHI2_2D * metrics_2d["chi2_stat"].mean()
        + cfg.WEIGHT_L2_2D * metrics_2d["l2_discrepancy"].mean()
    )
    return float(score), metrics_1d, metrics_2d


def normalize_array(df: pd.DataFrame, bounds: dict[str, tuple[float, float]]) -> np.ndarray:
    cols = []
    for col in PARAM_COLS:
        lo, hi = bounds[col]
        span = hi - lo
        cols.append((df[col].to_numpy() - lo) / span)
    return np.column_stack(cols)


def trim_margins_if_helpful(
    pool: pd.DataFrame,
    bounds: dict[str, tuple[float, float]],
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]], list[dict]]:
    bounds = {k: tuple(v) for k, v in bounds.items()}
    history: list[dict] = []
    base_score, _, _ = combined_uniformity_score(pool, bounds)

    improved = True
    while improved:
        improved = False
        best_candidate = None

        for col in PARAM_COLS:
            values = pool[col].to_numpy()
            lo_cur, hi_cur = bounds[col]
            for step in cfg.MARGIN_TRIM_PERCENTILE_STEPS:
                for edge, new_val in (
                    ("lo", float(np.quantile(values, step))),
                    ("hi", float(np.quantile(values, 1.0 - step))),
                ):
                    new_bounds = dict(bounds)
                    if edge == "lo":
                        if new_val <= lo_cur or new_val >= hi_cur:
                            continue
                        new_bounds[col] = (new_val, hi_cur)
                    else:
                        if new_val >= hi_cur or new_val <= lo_cur:
                            continue
                        new_bounds[col] = (lo_cur, new_val)

                    sub = apply_param_bounds(pool, new_bounds)
                    if len(sub) < cfg.N_SAMPLES:
                        continue

                    new_score, _, _ = combined_uniformity_score(sub, new_bounds)
                    rel_improve = (base_score - new_score) / base_score
                    if rel_improve < cfg.MARGIN_TRIM_MIN_REL_IMPROVEMENT:
                        continue

                    candidate = {
                        "parameter": col,
                        "edge": edge,
                        "percentile_step": step,
                        "new_bounds": new_bounds,
                        "pool_size": len(sub),
                        "score": new_score,
                        "rel_improvement": rel_improve,
                    }
                    if best_candidate is None or candidate["score"] < best_candidate["score"]:
                        best_candidate = candidate

        if best_candidate is not None:
            bounds = best_candidate["new_bounds"]
            pool = apply_param_bounds(pool, bounds)
            base_score = best_candidate["score"]
            history.append(best_candidate)
            improved = True

    return pool, bounds, history


def select_space_filling(
    pool: pd.DataFrame,
    bounds: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    if len(pool) < cfg.N_SAMPLES:
        raise ValueError(
            f"Pool has only {len(pool)} unique cases after filters; need {cfg.N_SAMPLES}."
        )

    pool = pool.sort_values("keff_std", kind="mergesort").reset_index(drop=True)
    x_norm = normalize_array(pool, bounds)

    selected: list[int] = [0]  # lowest std after sorting
    remaining = set(range(1, len(pool)))

    while len(selected) < cfg.N_SAMPLES:
        sel_x = x_norm[selected]
        shortlist = sorted(remaining, key=lambda i: pool.loc[i, "keff_std"])[: cfg.LHS_SHORTLIST_SIZE]

        best_idx = None
        best_key = None
        for idx in shortlist:
            dists = np.linalg.norm(sel_x - x_norm[idx], axis=1)
            min_dist = float(dists.min())
            key = (min_dist, -float(pool.loc[idx, "keff_std"]))
            if best_key is None or key > best_key:
                best_key = key
                best_idx = idx

        selected.append(int(best_idx))
        remaining.remove(int(best_idx))

    selected_df = pool.iloc[selected].copy()
    selected_df["selection_source"] = "space_filling_greedy"
    return selected_df.reset_index(drop=True)


def refine_by_swaps(
    selected: pd.DataFrame,
    pool: pd.DataFrame,
    bounds: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    selected = selected.copy().reset_index(drop=True)
    pool = pool.copy().reset_index(drop=True)

    selected_keys = {tuple(row[c] for c in cfg.DEDUP_COLS) for _, row in selected.iterrows()}
    unselected = pool[
        ~pool.apply(lambda r: tuple(r[c] for c in cfg.DEDUP_COLS) in selected_keys, axis=1)
    ].sort_values("keff_std").reset_index(drop=True)

    best_score, _, _ = combined_uniformity_score(selected, bounds)
    swaps = 0

    for _ in range(cfg.LHS_REFINEMENT_SWAPS):
        if unselected.empty:
            break
        i = np.random.randint(0, len(selected))
        trial = selected.drop(index=i)
        improved = False

        for _, candidate in unselected.head(80).iterrows():
            trial_df = pd.concat([trial, pd.DataFrame([candidate])], ignore_index=True)
            score, _, _ = combined_uniformity_score(trial_df, bounds)
            if score + 1e-12 < best_score:
                selected = trial_df.sort_values("keff_std").reset_index(drop=True)
                best_score = score
                swaps += 1
                improved = True
                break
        if not improved:
            continue

    selected["selection_source"] = "space_filling_refined"
    selected.attrs["swap_count"] = swaps
    return selected


def _plot_keff_std_hist(frames: dict[str, pd.DataFrame], out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for label, frame in frames.items():
        ax.hist(frame["keff_std"], bins=35, alpha=0.55, edgecolor="black", label=f"{label} (N={len(frame)})")
    ax.axvline(cfg.MAX_KEFF_STD, color="red", linestyle="--", linewidth=1.2, label=f"max std = {cfg.MAX_KEFF_STD}")
    ax.set_xlabel("keff_std")
    ax.set_ylabel("Count")
    ax.set_title("keff_std distributions across selection stages")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _plot_keff_hist(frames: dict[str, pd.DataFrame], out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for label, frame in frames.items():
        ax.hist(frame["keff"], bins=35, alpha=0.55, edgecolor="black", label=f"{label} (N={len(frame)})")
    ax.set_xlabel("keff")
    ax.set_ylabel("Count")
    ax.set_title("keff distributions across selection stages")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _plot_param_hists(
    frames: dict[str, pd.DataFrame],
    bounds: dict[str, tuple[float, float]],
    out_png: Path,
) -> None:
    labels = list(frames.keys())
    fig, axes = plt.subplots(len(labels), len(PARAM_COLS), figsize=(16, 3.8 * len(labels)))
    if len(labels) == 1:
        axes = np.array([axes])

    for row_idx, label in enumerate(labels):
        frame = frames[label]
        for col_idx, col in enumerate(PARAM_COLS):
            ax = axes[row_idx, col_idx]
            lo, hi = bounds[col]
            ax.hist(frame[col], bins=30, range=(lo, hi), edgecolor="black", alpha=0.82)
            ax.axvline(lo, color="red", linestyle="--", linewidth=1.0)
            ax.axvline(hi, color="red", linestyle="--", linewidth=1.0)
            if row_idx == 0:
                ax.set_title(col, fontsize=9)
            if col_idx == 0:
                ax.set_ylabel(f"{label}\nCount", fontsize=9)

    fig.suptitle("Parameter distributions by stage (red = active bounds)", y=1.01)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_uniformity_comparison(
    before_1d: pd.DataFrame,
    after_1d: pd.DataFrame,
    out_png: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(PARAM_COLS))
    axes[0].bar(x - 0.2, before_1d["l2_discrepancy"], width=0.4, label="before")
    axes[0].bar(x + 0.2, after_1d["l2_discrepancy"], width=0.4, label="after")
    axes[0].set_xticks(x, [c.replace("_", "\n") for c in PARAM_COLS], fontsize=7)
    axes[0].set_ylabel("L2 discrepancy")
    axes[0].set_title("Marginal L2 discrepancy")
    axes[0].legend()

    axes[1].bar(x - 0.2, before_1d["p_value"], width=0.4, label="before")
    axes[1].bar(x + 0.2, after_1d["p_value"], width=0.4, label="after")
    axes[1].axhline(0.05, color="black", linestyle=":", linewidth=1.2)
    axes[1].set_xticks(x, [c.replace("_", "\n") for c in PARAM_COLS], fontsize=7)
    axes[1].set_ylabel("Chi-square p-value")
    axes[1].set_title("Marginal 1D uniformity p-values")
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _plot_pairwise_heatmaps(
    metrics_2d: pd.DataFrame,
    value_col: str,
    out_png: Path,
    title: str,
) -> None:
    n = len(PARAM_COLS)
    mat = np.full((n, n), np.nan)
    short = [c.replace("r0_b4c_rod_", "r0_").replace("r1_fuel_annulus_", "r1_").replace("r2_water_", "r2_") for c in PARAM_COLS]

    idx = {c: i for i, c in enumerate(PARAM_COLS)}
    for _, row in metrics_2d.iterrows():
        i, j = idx[row["param_x"]], idx[row["param_y"]]
        mat[i, j] = row[value_col]
        mat[j, i] = row[value_col]

    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(mat, cmap="viridis_r")
    ax.set_xticks(range(n), short, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n), short, fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=value_col)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _plot_pairwise_scatter_grid(df: pd.DataFrame, bounds: dict[str, tuple[float, float]], out_png: Path) -> None:
    n_pairs = len(PAIR_COLS)
    ncols = 3
    nrows = int(np.ceil(n_pairs / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 4.0 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax_idx, (c1, c2) in enumerate(PAIR_COLS):
        ax = axes[ax_idx]
        lo1, hi1 = bounds[c1]
        lo2, hi2 = bounds[c2]
        ax.hexbin(
            df[c1], df[c2],
            gridsize=18,
            extent=(lo1, hi1, lo2, hi2),
            mincnt=1,
            cmap="cividis",
        )
        ax.set_xlabel(c1.split("_")[-1], fontsize=8)
        ax.set_ylabel(c2.split("_")[-1], fontsize=8)

    for ax in axes[n_pairs:]:
        ax.axis("off")

    fig.suptitle("Pairwise 2D projections (final selected)", y=1.01)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    np.random.seed(0)
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plots_dir = cfg.OUTPUT_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    source = load_source()
    hard_pool = deduplicate(apply_hard_filters(source))
    in_bounds = deduplicate(apply_param_bounds(hard_pool, cfg.PARAM_BOUNDS))

    score_start, metrics_1d_start, metrics_2d_start = combined_uniformity_score(in_bounds, cfg.PARAM_BOUNDS)
    trimmed_pool, final_bounds, trim_history = trim_margins_if_helpful(in_bounds, cfg.PARAM_BOUNDS)
    score_trimmed, metrics_1d_trimmed, metrics_2d_trimmed = combined_uniformity_score(trimmed_pool, final_bounds)

    selected = select_space_filling(trimmed_pool, final_bounds)
    selected = refine_by_swaps(selected, trimmed_pool, final_bounds)
    score_final, metrics_1d_final, metrics_2d_final = combined_uniformity_score(selected, final_bounds)

    selection_report = pd.DataFrame(
        {
            "stage": [
                "source_total",
                "after_std_filter",
                "after_starting_bounds",
                "after_margin_trim",
                "final_selected",
            ],
            "count": [
                len(source),
                len(hard_pool),
                len(in_bounds),
                len(trimmed_pool),
                len(selected),
            ],
        }
    )

    score_report = pd.DataFrame(
        [
            {"stage": "after_starting_bounds", "uniformity_score": score_start},
            {"stage": "after_margin_trim", "uniformity_score": score_trimmed},
            {"stage": "final_selected", "uniformity_score": score_final},
        ]
    )

    hard_pool.to_csv(cfg.OUTPUT_DIR / "pool_after_std_filter.csv", index=False)
    in_bounds.to_csv(cfg.OUTPUT_DIR / "pool_after_starting_bounds.csv", index=False)
    trimmed_pool.to_csv(cfg.OUTPUT_DIR / "pool_after_margin_trim.csv", index=False)
    selected.to_csv(cfg.OUTPUT_DIR / "final_dataset.csv", index=False)

    metrics_1d_start.to_csv(cfg.OUTPUT_DIR / "uniformity_1d_after_starting_bounds.csv", index=False)
    metrics_1d_trimmed.to_csv(cfg.OUTPUT_DIR / "uniformity_1d_after_margin_trim.csv", index=False)
    metrics_1d_final.to_csv(cfg.OUTPUT_DIR / "uniformity_1d_final_selected.csv", index=False)
    metrics_2d_start.to_csv(cfg.OUTPUT_DIR / "uniformity_2d_after_starting_bounds.csv", index=False)
    metrics_2d_trimmed.to_csv(cfg.OUTPUT_DIR / "uniformity_2d_after_margin_trim.csv", index=False)
    metrics_2d_final.to_csv(cfg.OUTPUT_DIR / "uniformity_2d_final_selected.csv", index=False)
    selection_report.to_csv(cfg.OUTPUT_DIR / "selection_report.csv", index=False)
    score_report.to_csv(cfg.OUTPUT_DIR / "uniformity_score_by_stage.csv", index=False)

    with open(cfg.OUTPUT_DIR / "final_bounds.json", "w") as fh:
        json.dump({k: list(v) for k, v in final_bounds.items()}, fh, indent=2)
    with open(cfg.OUTPUT_DIR / "margin_trim_history.json", "w") as fh:
        json.dump(trim_history, fh, indent=2, default=str)

    with open(cfg.OUTPUT_DIR / "final_bounds.py", "w") as fh:
        fh.write("# Active bounds after optional margin trimming\n")
        fh.write("FINAL_PARAM_BOUNDS = {\n")
        for col, (lo, hi) in final_bounds.items():
            fh.write(f'    "{col}": ({lo:.6f}, {hi:.6f}),\n')
        fh.write("}\n")

    stage_frames = {
        "std_filter": hard_pool,
        "starting_bounds": in_bounds,
        "margin_trim": trimmed_pool,
        "final_lhs": selected,
    }
    _plot_keff_hist(stage_frames, plots_dir / "keff_by_stage.png")
    _plot_keff_std_hist(stage_frames, plots_dir / "keff_std_by_stage.png")
    _plot_param_hists(stage_frames, final_bounds, plots_dir / "params_by_stage.png")
    _plot_uniformity_comparison(
        metrics_1d_start,
        metrics_1d_final,
        plots_dir / "uniformity_1d_before_vs_final.png",
        "Marginal uniformity: starting bounds vs final LHS selection",
    )
    _plot_pairwise_heatmaps(
        metrics_2d_start,
        "chi2_stat",
        plots_dir / "pairwise_chi2_before.png",
        "Pairwise 2D chi-square (starting bounds pool)",
    )
    _plot_pairwise_heatmaps(
        metrics_2d_final,
        "chi2_stat",
        plots_dir / "pairwise_chi2_final.png",
        "Pairwise 2D chi-square (final selected)",
    )
    _plot_pairwise_heatmaps(
        metrics_2d_final,
        "l2_discrepancy",
        plots_dir / "pairwise_l2_final.png",
        "Pairwise 2D L2 discrepancy (final selected)",
    )
    _plot_pairwise_scatter_grid(selected, final_bounds, plots_dir / "pairwise_scatter_final.png")

    print(f"Saved outputs under {cfg.OUTPUT_DIR}")
    print(selection_report.to_string(index=False))
    print("\nUniformity score (lower is better):")
    print(score_report.to_string(index=False))
    print("\nFinal bounds:")
    for col, (lo, hi) in final_bounds.items():
        print(f"  {col}: [{lo:.4f}, {hi:.4f}]")
    print(f"\nMargin trims applied: {len(trim_history)}")
    print(
        "Mean 1D L2: "
        f"start={metrics_1d_start['l2_discrepancy'].mean():.5f} -> "
        f"final={metrics_1d_final['l2_discrepancy'].mean():.5f}"
    )
    print(
        "Mean 2D chi2: "
        f"start={metrics_2d_start['chi2_stat'].mean():.1f} -> "
        f"final={metrics_2d_final['chi2_stat'].mean():.1f}"
    )
    print(
        "Mean 2D L2: "
        f"start={metrics_2d_start['l2_discrepancy'].mean():.5f} -> "
        f"final={metrics_2d_final['l2_discrepancy'].mean():.5f}"
    )


if __name__ == "__main__":
    main()
