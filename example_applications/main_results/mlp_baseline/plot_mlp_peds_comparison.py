#!/usr/bin/env python3
"""Generate MLP-vs-PEDS comparison plots and refresh COMPARISON_WITH_PEDS.md."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STUDY_ROOT = Path(__file__).resolve().parent
COMPLETE_STRAT = STUDY_ROOT.parent
SEEDS = [0, 1, 2, 3, 4]
PLOT_DIR = STUDY_ROOT / "analysis" / "comparison_plots"
COMPARISON_DIR = STUDY_ROOT / "comparison"

# Publication-scale fonts (aligned with pub_keff_plots.py, +2 for half-page figures)
FS_LABEL = 19
FS_TICK = 17
FS_LEGEND = 16

PEDS_GENERATOR_PARAMS = 73_524
MLP_PARAMS = 71_497

RUN_DIRS = {
    "geom": STUDY_ROOT / "mlp_geom_only",
    "phi_xs": STUDY_ROOT / "mlp_with_phi_xs",
    "peds": COMPLETE_STRAT / "PEDS",
}

# Source columns for mean |Δk|; converted to MAE (keff units) via / 1e5.
DELTA_K_COLS = {
    "geom": ("train_mean_delta_k_pcm", "val_mean_delta_k_pcm"),
    "phi_xs": ("train_mean_delta_k_pcm", "val_mean_delta_k_pcm"),
    "peds": ("train_mean_pcm", "val_mean_pcm"),
}

MODEL_LABELS = {
    "geom": "MLP geom-features",
    "phi_xs": "MLP all-features",
    "peds": "PEDS",
}


# One hue per model; train = solid, val = dashed + lighter shade
MODEL_COLORS = {
    "geom": "#d62728",
    "phi_xs": "#2ca02c",
    "peds": "#1f77b4",
}


def load_epoch_curves(run_dir: Path, seeds: list[int], method: str) -> pd.DataFrame:
    frames = []
    for seed in seeds:
        path = run_dir / f"train_1000_seed_{seed}" / "epoch_metrics.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing epoch metrics: {path}")
        df = pd.read_csv(path)
        df["seed"] = seed
        df["method"] = method
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def aggregate_epoch_stats(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    rows = []
    for epoch, grp in df.groupby("epoch"):
        row = {"epoch": int(epoch)}
        for col in value_cols:
            if col not in grp.columns:
                continue
            row[f"{col}_mean"] = grp[col].mean()
            row[f"{col}_std"] = grp[col].std(ddof=0)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("epoch")


def plot_train_val_delta_k(aggs: dict[str, pd.DataFrame], out: Path) -> None:
    """Single axes: mean |Δk| (MAE on keff) vs epoch for 3 models × train/val."""
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    pcm_to_mae = 1e-5

    for key in ("geom", "phi_xs", "peds"):
        agg = aggs[key]
        train_col, val_col = DELTA_K_COLS[key]
        color = MODEL_COLORS[key]
        label = MODEL_LABELS[key]

        ax.plot(
            agg["epoch"],
            agg[f"{train_col}_mean"] * pcm_to_mae,
            color=color,
            linestyle="-",
            linewidth=2.4,
            label=f"{label} (train)",
        )
        ax.plot(
            agg["epoch"],
            agg[f"{val_col}_mean"] * pcm_to_mae,
            color=color,
            linestyle="--",
            linewidth=2.4,
            alpha=0.7,
            label=f"{label} (val)",
        )

    ax.set_yscale("log")
    ax.set_xlabel("Epoch", fontsize=FS_LABEL)
    ax.set_ylabel(r"MAE ($|\Delta k|$)", fontsize=FS_LABEL)
    ax.tick_params(axis="both", labelsize=FS_TICK)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=FS_LEGEND, frameon=False, loc="best")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_train_val_mse(mlp_agg: pd.DataFrame, peds_agg: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)

    for ax, split in zip(axes, ("train", "val")):
        col = f"{split}_mse_k"
        ax.plot(mlp_agg["epoch"], mlp_agg[f"{col}_mean"], "-", color="#d62728", label="MLP baseline")
        ax.fill_between(
            mlp_agg["epoch"],
            mlp_agg[f"{col}_mean"] - mlp_agg[f"{col}_std"],
            mlp_agg[f"{col}_mean"] + mlp_agg[f"{col}_std"],
            color="#d62728",
            alpha=0.2,
        )
        ax.plot(peds_agg["epoch"], peds_agg[f"{col}_mean"], "-", color="#1f77b4", label="PEDS")
        ax.fill_between(
            peds_agg["epoch"],
            peds_agg[f"{col}_mean"] - peds_agg[f"{col}_std"],
            peds_agg[f"{col}_mean"] + peds_agg[f"{col}_std"],
            color="#1f77b4",
            alpha=0.2,
        )
        ax.set_yscale("log")
        ax.set_xlabel("Epoch", fontsize=FS_LABEL)
        ax.set_ylabel(f"{split.capitalize()} MSE (k)", fontsize=FS_LABEL)
        ax.tick_params(labelsize=FS_TICK)
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=FS_LEGEND)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_val_error_metrics(mlp_agg: pd.DataFrame, peds_agg: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)

    ax = axes[0]
    ax.plot(mlp_agg["epoch"], mlp_agg["val_mean_delta_k_pcm_mean"], "o-", color="#d62728", label="MLP |Δk|")
    ax.fill_between(
        mlp_agg["epoch"],
        mlp_agg["val_mean_delta_k_pcm_mean"] - mlp_agg["val_mean_delta_k_pcm_std"],
        mlp_agg["val_mean_delta_k_pcm_mean"] + mlp_agg["val_mean_delta_k_pcm_std"],
        color="#d62728",
        alpha=0.2,
    )
    ax.plot(peds_agg["epoch"], peds_agg["val_mean_pcm_mean"], "o-", color="#1f77b4", label="PEDS |Δρ|")
    ax.fill_between(
        peds_agg["epoch"],
        peds_agg["val_mean_pcm_mean"] - peds_agg["val_mean_pcm_std"],
        peds_agg["val_mean_pcm_mean"] + peds_agg["val_mean_pcm_std"],
        color="#1f77b4",
        alpha=0.2,
    )
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation error (pcm)")
    ax.set_title("Validation physics error vs epoch")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()

    ax = axes[1]
    ax.plot(mlp_agg["epoch"], mlp_agg["val_mean_frac_error_mean"], "o-", color="#d62728", label="MLP")
    ax.fill_between(
        mlp_agg["epoch"],
        mlp_agg["val_mean_frac_error_mean"] - mlp_agg["val_mean_frac_error_std"],
        mlp_agg["val_mean_frac_error_mean"] + mlp_agg["val_mean_frac_error_std"],
        color="#d62728",
        alpha=0.2,
    )
    peds_frac = peds_agg["val_mae_k_mean"] / 1.0  # proxy not ideal - use mae_k as scale
    # PEDS epoch log has val_mae_k; fractional error not logged -> derive approx from mae/keff~1
    if "val_mae_k_mean" in peds_agg.columns:
        ax.plot(peds_agg["epoch"], peds_agg["val_mae_k_mean"], "o-", color="#1f77b4", label="PEDS MAE(k)")
        ax.fill_between(
            peds_agg["epoch"],
            peds_agg["val_mae_k_mean"] - peds_agg["val_mae_k_std"],
            peds_agg["val_mae_k_mean"] + peds_agg["val_mae_k_std"],
            color="#1f77b4",
            alpha=0.2,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Error scale")
    ax.set_title("Validation MAE(k) / fractional error proxy")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_test_metric_bars(mlp_test: pd.DataFrame, peds_dk: pd.DataFrame, out: Path) -> None:
    metrics = [
        ("test_mse_k", "Test MSE (k)", True),
        ("test_mean_delta_k_pcm", "Mean |Δk| (pcm)", False),
        ("test_mean_frac_error", "Mean |fractional error|", False),
        ("test_frac_below_650_delta_k", "Frac |Δk| < 650 pcm", False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    x = np.arange(len(SEEDS))
    width = 0.36

    for ax, (mlp_col, title, logy) in zip(axes.ravel(), metrics):
        peds_col = {
            "test_mse_k": "test_mse_k",
            "test_mean_delta_k_pcm": "test_mean_pcm",
            "test_mean_frac_error": None,
            "test_frac_below_650_delta_k": "test_frac_below_650",
        }[mlp_col]

        mlp_vals = [mlp_test.loc[mlp_test.seed == s, mlp_col].iloc[0] for s in SEEDS]
        if peds_col is not None:
            peds_vals = [peds_dk.loc[peds_dk.seed == s, peds_col].iloc[0] for s in SEEDS]
            ax.bar(x - width / 2, mlp_vals, width, label="MLP", color="#d62728", alpha=0.85)
            ax.bar(x + width / 2, peds_vals, width, label="PEDS", color="#1f77b4", alpha=0.85)
        else:
            ax.bar(x, mlp_vals, width, label="MLP", color="#d62728", alpha=0.85)

        if logy:
            ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([f"seed {s}" for s in SEEDS])
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()

    fig.suptitle("Held-out test metrics by seed (train_size=1000)", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_test_box_summary(mlp_test: pd.DataFrame, peds_dk: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    data = [
        mlp_test["test_mean_delta_k_pcm"].values,
        peds_dk["test_mean_pcm"].values,
    ]
    bp = ax.boxplot(data, tick_labels=["MLP |Δk|", "PEDS |Δk|"], patch_artist=True)
    colors = ["#d62728", "#1f77b4"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    ax.set_ylabel("Per-run mean |Δk| on test (pcm)")
    ax.set_title("Test-set error distribution across 5 seeds")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def write_summary_markdown(
    mlp_test: pd.DataFrame,
    peds_dk: pd.DataFrame,
    peds_rho: pd.DataFrame,
    mlp_agg: pd.DataFrame,
    peds_agg: pd.DataFrame,
    out: Path,
) -> None:
    b = mlp_test.mean(numeric_only=True)
    p_dk = peds_dk.mean(numeric_only=True)
    p_rho = peds_rho.mean(numeric_only=True)

    final_mlp_val_mse = float(mlp_agg.iloc[-1]["val_mse_k_mean"])
    final_peds_val_mse = float(peds_agg.iloc[-1]["val_mse_k_mean"])
    ratio_mse = final_mlp_val_mse / max(final_peds_val_mse, 1e-12)

    lines = [
        "# MLP baseline vs PEDS (`complete_strat`)",
        "",
        "## What was done",
        "- Trained a vanilla NN-only baseline ensemble (seeds 0–4, train size 1000).",
        "- Reused the exact PEDS `split_log.csv` for each seed (same train/val/test indices).",
        "- Inputs: only the 6 normalized geometry features from `params`.",
        "- Target: direct `keff` prediction (no XS prediction, no diffusion solver).",
        "- Generated comparison plots in `analysis/comparison_plots/`.",
        "",
        "## Architecture comparison",
        "",
        "### Shared design choices",
        "- Both models use the same **trunk hidden widths**: `128 → 256 → 128` with ReLU.",
        "- Both are trained with Adam, cosine LR schedule, MSE on `keff`, 70 epochs, batch size 32.",
        "- Parameter budgets are intentionally matched (~71.5k vs ~73.5k).",
        "",
        "### MLP baseline (`6 → 128 → 256 → 128 → 36 → 1`)",
        "| Block | Shape | Role |",
        "|---|---|---|",
        "| Input | 6 | geometry only (`b4c_r`, `cr_frac`, `fuel_r`, `enrichment`, `f_mod`, `water_r`) |",
        "| Trunk | 6→128→256→128 | feature extraction |",
        "| Latent | 128→36 | mirrors PEDS XS-correction dimensionality |",
        "| Output head | 36→1 | **learned replacement** for low-fidelity solver mapping to `keff` |",
        "| Parameters | **71,497** | |",
        "",
        "### PEDS generator + solver",
        "| Block | Shape | Role |",
        "|---|---|---|",
        "| Input | 12 (=6 geom + 6 φ features) | geometry + low-fidelity flux features |",
        "| Trunk | 12→128→256→128 | feature extraction |",
        "| XS head | (128+36)→36 | predicts log-ratio XS corrections per region/group |",
        "| Physics | diffusion eigenvalue solver | maps corrected XS + geometry → `keff` |",
        "| Parameters (generator only) | **73,524** | solver has no trainable NN params |",
        "",
        "### Key differences",
        "1. **Inputs**: MLP uses geometry only; PEDS also uses φ features and baseline XS in the head.",
        "2. **Output space**: MLP predicts scalar `keff`; PEDS predicts 36 XS corrections then solves physics.",
        "3. **Inductive bias**: PEDS enforces a physics pathway (XS → diffusion); MLP must learn the map end-to-end.",
        "4. **Final layer**: MLP has an explicit `36→1` FC layer; PEDS replaces this with the NT diffusion solver.",
        "5. **Same trunk width, not identical graph**: input dimension and head wiring differ even though parameter counts are close.",
        "",
        "## Training evolution (validation)",
        f"- Final-epoch mean val MSE: MLP `{final_mlp_val_mse:.4g}` vs PEDS `{final_peds_val_mse:.4g}` "
        f"(~`{ratio_mse:.0f}×` higher for MLP).",
        f"- Final-epoch mean val |Δk|: MLP `{mlp_agg.iloc[-1]['val_mean_delta_k_pcm_mean']:.0f}` pcm vs "
        f"PEDS val |Δρ| `{peds_agg.iloc[-1]['val_mean_pcm_mean']:.0f}` pcm "
        "(metrics differ slightly; see test table below).",
        "- MLP validation error decreases slowly and plateaus around ~10–12k pcm |Δk|.",
        "- PEDS validation error decreases to ~600 pcm |Δρ| and keeps improving through epoch 70.",
        "",
        "Plots:",
        "- `analysis/comparison_plots/train_val_mse_vs_epoch.png`",
        "- `analysis/comparison_plots/val_error_vs_epoch.png`",
        "- `analysis/comparison_plots/test_metrics_by_seed.png`",
        "- `analysis/comparison_plots/test_delta_k_boxplot.png`",
        "",
        "## Held-out test comparison (mean over 5 seeds)",
        "",
        "| Metric | MLP baseline | PEDS | Ratio (MLP/PEDS) |",
        "|---|---:|---:|---:|",
        f"| MSE(k) | {b['test_mse_k']:.4g} | {p_dk['test_mse_k']:.4g} | {b['test_mse_k']/p_dk['test_mse_k']:.0f}× |",
        f"| Mean |Δk| (pcm) | {b['test_mean_delta_k_pcm']:.1f} | {p_dk['test_mean_pcm']:.1f} | "
        f"{b['test_mean_delta_k_pcm']/p_dk['test_mean_pcm']:.1f}× |",
        f"| Median |Δk| (pcm) | {b['test_median_delta_k_pcm']:.1f} | {p_dk['test_median_pcm']:.1f} | "
        f"{b['test_median_delta_k_pcm']/p_dk['test_median_pcm']:.1f}× |",
        f"| Mean |Δρ| (pcm) | {b['test_mean_delta_rho_pcm']:.1f} | {p_rho['test_mean_pcm']:.1f} | "
        f"{b['test_mean_delta_rho_pcm']/p_rho['test_mean_pcm']:.1f}× |",
        f"| Mean fractional error | {b['test_mean_frac_error']:.4f} | — | — |",
        f"| Frac below 650 pcm (|Δk|) | {b['test_frac_below_650_delta_k']:.3f} | {p_dk['test_frac_below_650']:.3f} | "
        f"{b['test_frac_below_650_delta_k']/max(p_dk['test_frac_below_650'],1e-9):.2f}× |",
        "",
        "## Per-seed test |Δk| (pcm)",
        "",
        "| Seed | MLP | PEDS |",
        "|---|---:|---:|",
    ]
    for seed in SEEDS:
        m = float(mlp_test.loc[mlp_test.seed == seed, "test_mean_delta_k_pcm"].iloc[0])
        p = float(peds_dk.loc[peds_dk.seed == seed, "test_mean_pcm"].iloc[0])
        lines.append(f"| {seed} | {m:.1f} | {p:.1f} |")

    lines.extend(
        [
            "",
            "## Interpretation: is 1000 points enough for NN-only accuracy?",
            "- **No, not at PEDS-level accuracy.** With the same splits and similar parameter count, the MLP baseline remains ~20× worse in test |Δk| and captures only ~4% of test points within 650 pcm vs ~70% for PEDS.",
            "- The MLP does learn a coarse trend (val fractional error drops from ~0.94 to ~0.11), but it cannot match the fine keff accuracy achieved when physics structure is embedded.",
            "- This supports the conclusion that **data volume alone is insufficient** here: the solver-informed PEDS pathway provides strong inductive bias that a geometry-only MLP cannot recover with 1000 training samples.",
            "",
            "## Files",
            "- Baseline metrics: `analysis/testset_results/test_metrics_all_runs.csv`",
            "- PEDS metrics: `../PEDS/analysis/testset_results_dk/test_metrics_all_runs.csv`",
            "- Per-seed epoch logs: `train_1000_seed_<seed>/epoch_metrics.csv`",
        ]
    )
    out.write_text("\n".join(lines), encoding="utf-8")


def write_comparison_metrics_csv(out: Path) -> Path:
    """Mean±std across seeds for geom-features / all-features / PEDS test metrics."""
    geom = pd.read_csv(RUN_DIRS["geom"] / "analysis" / "testset_results" / "test_metrics_all_runs.csv")
    allf = pd.read_csv(RUN_DIRS["phi_xs"] / "analysis" / "testset_results" / "test_metrics_all_runs.csv")
    peds = pd.read_csv(COMPLETE_STRAT / "PEDS" / "analysis" / "testset_results_dk" / "test_metrics_all_runs.csv")

    # (display_name, geom_col, allf_col, peds_col) — None if unavailable for that model
    specs = [
        ("test_mse_k", "test_mse_k", "test_mse_k", "test_mse_k"),
        ("test_mae_k", "test_mae_k", "test_mae_k", "test_mae_k"),
        ("test_mean_delta_k_pcm", "test_mean_delta_k_pcm", "test_mean_delta_k_pcm", "test_mean_pcm"),
        ("test_median_delta_k_pcm", "test_median_delta_k_pcm", "test_median_delta_k_pcm", "test_median_pcm"),
        ("test_mean_frac_error", "test_mean_frac_error", "test_mean_frac_error", None),
        ("test_frac_below_650_delta_k", "test_frac_below_650_delta_k", "test_frac_below_650_delta_k", "test_frac_below_650"),
        ("test_mean_delta_rho_pcm", "test_mean_delta_rho_pcm", "test_mean_delta_rho_pcm", None),
        ("test_frac_below_650_delta_rho", "test_frac_below_650_delta_rho", "test_frac_below_650_delta_rho", None),
    ]

    def mean_std(df: pd.DataFrame, col: str | None) -> tuple[float, float]:
        if col is None or col not in df.columns:
            return (float("nan"), float("nan"))
        return (float(df[col].mean()), float(df[col].std(ddof=0)))

    rows = []
    for name, gcol, acol, pcol in specs:
        g_m, g_s = mean_std(geom, gcol)
        a_m, a_s = mean_std(allf, acol)
        p_m, p_s = mean_std(peds, pcol)
        rows.append(
            {
                "metric": name,
                "mlp_geom_features_mean": g_m,
                "mlp_geom_features_std": g_s,
                "mlp_all_features_mean": a_m,
                "mlp_all_features_std": a_s,
                "peds_mean": p_m,
                "peds_std": p_s,
            }
        )

    out_df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    return out


def main() -> None:
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    aggs: dict[str, pd.DataFrame] = {}
    for key, run_dir in RUN_DIRS.items():
        train_col, val_col = DELTA_K_COLS[key]
        epochs = load_epoch_curves(run_dir, SEEDS, method=key)
        aggs[key] = aggregate_epoch_stats(epochs, [train_col, val_col])

    out_delta = COMPARISON_DIR / "train_val_delta_k_vs_epoch.png"
    plot_train_val_delta_k(aggs, out_delta)
    print(f"Wrote {out_delta}")

    out_csv = write_comparison_metrics_csv(COMPARISON_DIR / "test_metrics_summary_mean_std.csv")
    print(f"Wrote {out_csv}")

    # Optional legacy plots (geom-only MLP vs PEDS), if metrics CSVs exist.
    geom_test_path = RUN_DIRS["geom"] / "analysis" / "testset_results" / "test_metrics_all_runs.csv"
    peds_dk_path = COMPLETE_STRAT / "PEDS" / "analysis" / "testset_results_dk" / "test_metrics_all_runs.csv"
    peds_rho_path = COMPLETE_STRAT / "PEDS" / "analysis" / "testset_results" / "test_metrics_all_runs.csv"
    if geom_test_path.exists() and peds_dk_path.exists() and peds_rho_path.exists():
        geom_epochs = load_epoch_curves(RUN_DIRS["geom"], SEEDS, method="geom")
        peds_epochs = load_epoch_curves(RUN_DIRS["peds"], SEEDS, method="peds")
        mlp_cols = ["train_mse_k", "val_mse_k", "val_mean_delta_k_pcm", "val_mean_frac_error"]
        peds_cols = ["train_mse_k", "val_mse_k", "val_mean_pcm", "val_mae_k"]
        mlp_agg = aggregate_epoch_stats(geom_epochs, mlp_cols)
        peds_agg = aggregate_epoch_stats(peds_epochs, peds_cols)
        mlp_test = pd.read_csv(geom_test_path)
        peds_dk = pd.read_csv(peds_dk_path)
        peds_rho = pd.read_csv(peds_rho_path)
        plot_train_val_mse(mlp_agg, peds_agg, PLOT_DIR / "train_val_mse_vs_epoch.png")
        plot_val_error_metrics(mlp_agg, peds_agg, PLOT_DIR / "val_error_vs_epoch.png")
        plot_test_metric_bars(mlp_test, peds_dk, PLOT_DIR / "test_metrics_by_seed.png")
        plot_test_box_summary(mlp_test, peds_dk, PLOT_DIR / "test_delta_k_boxplot.png")
        write_summary_markdown(
            mlp_test,
            peds_dk,
            peds_rho,
            mlp_agg,
            peds_agg,
            STUDY_ROOT / "COMPARISON_WITH_PEDS.md",
        )
        print(f"Wrote legacy plots to {PLOT_DIR}")


if __name__ == "__main__":
    main()
