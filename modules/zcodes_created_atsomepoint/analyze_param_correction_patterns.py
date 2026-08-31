"""
analyze_param_correction_patterns.py
=========================================================================
Ad-hoc study: which regions of the 6D input-parameter space is PEDS bad
at correcting?

For every sample in every split (train/val/test) and every seed run under
modules/RUNS/precise_param_strat/, this script pairs up the reactivity
discrepancy |delta_rho| BEFORE any NN correction (epoch 0, i.e. baseline
solver XS) and AFTER training (epoch 70 / best epoch), and asks:

    Does PEDS' relative improvement depend on where a sample sits in
    *parameter* space (b4c_r, cr_frac, fuel_r, enrichment, f_mod, water_r),
    rather than simply on how big the baseline error was (keff space)?

Everything is read from CSVs already written during training / by
evaluate_test_metrics.py — no training or physics-solver re-run needed:

  train_1000_seed_{s}/keff_epoch_log_train.csv   (epoch 0 & 70 delta_rho_pcm)
  train_1000_seed_{s}/XS/final_xs_train.csv      (raw geometry params, join key = sample_idx)
  train_1000_seed_{s}/keff_epoch_log_val.csv     (epoch 0 & 70 delta_rho_pcm)
  train_1000_seed_{s}/XS/final_xs_val.csv        (raw geometry params, join key = sample_idx)
  testset_results/run_train1000_seed{s}_keff_comparison.csv
                                                  (epoch 0 & 70 delta_rho_pcm + raw params, already joined)

Outputs (written to <RUN_ROOT>/param_correction_analysis/):
  per_sample_improvement.csv           - the combined long-format table used for everything below
  param_bin_summary.csv                - mean/median rel. improvement per parameter quantile-bin, per split
  poorly_corrected_samples.csv         - the worst-improvement tail, for manual inspection
  correlation_summary.csv              - Spearman corr(param, rel_improvement) per split & seed

  figs/binned_improvement_vs_<param>.png   - improvement vs parameter, binned, split as color, seed as error band
  figs/poor_vs_good_param_distributions.png - histogram overlay: bottom-tail vs rest, one panel per parameter
  figs/before_after_scatter_colored_by_f_mod.png - delta_before vs delta_after, log-log, colored by f_mod
  figs/correlation_heatmap.png             - param x split heatmap of mean Spearman corr across seeds
  figs/f_mod_focus.png                     - dedicated multi-seed panel zooming in on the f_mod hypothesis

USAGE
-----
    python analyze_param_correction_patterns.py
    python analyze_param_correction_patterns.py --run-root /path/to/precise_param_strat --epoch-final 70
=========================================================================
"""
import os
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RUN_ROOT = os.path.join(THIS_DIR, "RUNS", "precise_param_strat")

# Short canonical parameter names used throughout this script.
PARAM_NAMES = ["b4c_r", "cr_frac", "fuel_r", "enrichment", "f_mod", "water_r"]
PRETTY_NAMES = {
    "b4c_r": "B4C rod radius",
    "cr_frac": "Control-rod B4C fraction",
    "fuel_r": "Fuel annulus outer radius",
    "enrichment": "Fuel enrichment",
    "f_mod": "Fuel moderation fraction (f_mod)",
    "water_r": "Water outer radius",
}
# test comparison csv uses these long names, in the SAME order as PARAM_NAMES.
TEST_PARAM_COLS = [
    "r0_b4c_rod_outer_radius",
    "r0_b4c_rod_cr_fraction",
    "r1_fuel_annulus_outer_radius",
    "r1_fuel_annulus_enrichment",
    "r1_fuel_annulus_f_mod",
    "r2_water_outer_radius",
]

SPLIT_COLORS = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"}
N_BINS = 8
# fraction of samples (by rel. improvement, ascending) considered "poorly corrected"
POOR_TAIL_FRAC = 0.15
EPS_PCM = 1.0  # floor for delta_before when computing ratios, avoids /0


def find_seed_dirs(run_root):
    dirs = sorted(glob.glob(os.path.join(run_root, "train_*_seed_*")))
    seeds = []
    for d in dirs:
        base = os.path.basename(d)
        try:
            seed = int(base.rsplit("_", 1)[-1])
        except ValueError:
            continue
        seeds.append((seed, d))
    return sorted(seeds)


def load_train_or_val_split(run_dir, split, epoch_final):
    """Build [sample_idx, PARAM_NAMES..., delta_before, delta_after] for one split/run."""
    log_path = os.path.join(run_dir, f"keff_epoch_log_{split}.csv")
    xs_path = os.path.join(run_dir, "XS", f"final_xs_{split}.csv")
    if not (os.path.exists(log_path) and os.path.exists(xs_path)):
        return None

    log_df = pd.read_csv(log_path)
    d0 = log_df[log_df["epoch"] == 0][["sample_idx", "delta_rho_pcm"]].rename(
        columns={"delta_rho_pcm": "delta_before"})
    dF = log_df[log_df["epoch"] == epoch_final][["sample_idx", "delta_rho_pcm"]].rename(
        columns={"delta_rho_pcm": "delta_after"})
    if d0.empty or dF.empty:
        return None
    paired = d0.merge(dF, on="sample_idx", how="inner")

    xs_df = pd.read_csv(xs_path)[["sample_idx"] + PARAM_NAMES]
    merged = paired.merge(xs_df, on="sample_idx", how="inner")
    return merged


def load_test_split(run_root, run_dir, seed, epoch_final):
    """Test set already has raw params + epoch-0/epoch-final delta_rho pre-joined
    by evaluate_test_metrics.py — find whichever train size subfolder it wrote for."""
    base = os.path.basename(run_dir)  # train_<N>_seed_<S>
    try:
        train_size = int(base.split("_")[1])
    except (IndexError, ValueError):
        train_size = None
    candidates = []
    if train_size is not None:
        candidates.append(os.path.join(run_root, "testset_results",
                                        f"run_train{train_size}_seed{seed}_keff_comparison.csv"))
    candidates += sorted(glob.glob(os.path.join(run_root, "testset_results",
                                                 f"run_train*_seed{seed}_keff_comparison.csv")))
    path = next((c for c in candidates if os.path.exists(c)), None)
    if path is None:
        return None

    df = pd.read_csv(path)
    d0 = df[df["epoch"] == 0][["sample_idx", "delta_rho_pcm"] + TEST_PARAM_COLS].rename(
        columns={"delta_rho_pcm": "delta_before"})
    epochs_present = sorted(df["epoch"].unique())
    final_epoch = epoch_final if epoch_final in epochs_present else epochs_present[-1]
    dF = df[df["epoch"] == final_epoch][["sample_idx", "delta_rho_pcm"]].rename(
        columns={"delta_rho_pcm": "delta_after"})
    merged = d0.merge(dF, on="sample_idx", how="inner")
    merged = merged.rename(columns=dict(zip(TEST_PARAM_COLS, PARAM_NAMES)))
    return merged[["sample_idx", "delta_before", "delta_after"] + PARAM_NAMES]


def build_combined_table(run_root, epoch_final):
    rows = []
    for seed, run_dir in find_seed_dirs(run_root):
        for split in ("train", "val"):
            df = load_train_or_val_split(run_dir, split, epoch_final)
            if df is None or df.empty:
                print(f"  [warn] seed={seed} split={split}: no data, skipping")
                continue
            df = df.copy()
            df["seed"] = seed
            df["split"] = split
            rows.append(df)

        df_test = load_test_split(run_root, run_dir, seed, epoch_final)
        if df_test is None or df_test.empty:
            print(f"  [warn] seed={seed} split=test: no data, skipping")
        else:
            df_test = df_test.copy()
            df_test["seed"] = seed
            df_test["split"] = "test"
            rows.append(df_test)

    if not rows:
        raise RuntimeError(f"No usable data found under {run_root}")

    combined = pd.concat(rows, ignore_index=True)

    # ── improvement metrics ────────────────────────────────────────────
    safe_before = np.maximum(combined["delta_before"].to_numpy(), EPS_PCM)
    combined["abs_improvement"] = combined["delta_before"] - combined["delta_after"]
    combined["rel_improvement"] = combined["abs_improvement"] / safe_before
    # ratio close to 1 → no improvement; close to 0 → fully corrected; >1 → PEDS made it worse
    combined["residual_ratio"] = combined["delta_after"] / safe_before
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# Summaries
# ─────────────────────────────────────────────────────────────────────────────
def build_param_bin_summary(combined, n_bins=N_BINS):
    rows = []
    for split, sub_split in combined.groupby("split"):
        for param in PARAM_NAMES:
            vals = sub_split[param].to_numpy()
            try:
                bin_idx, edges = pd.qcut(vals, n_bins, labels=False, retbins=True, duplicates="drop")
            except ValueError:
                continue
            for b in range(len(edges) - 1):
                mask = bin_idx == b
                if mask.sum() == 0:
                    continue
                sub = sub_split[mask]
                rows.append(dict(
                    split=split, param=param, bin=b,
                    lo=edges[b], hi=edges[b + 1],
                    n=int(mask.sum()),
                    mean_delta_before=float(sub["delta_before"].mean()),
                    mean_delta_after=float(sub["delta_after"].mean()),
                    mean_rel_improvement=float(sub["rel_improvement"].mean()),
                    median_rel_improvement=float(sub["rel_improvement"].median()),
                    std_rel_improvement=float(sub["rel_improvement"].std()),
                    frac_poor=float((sub["rel_improvement"] < sub_split["rel_improvement"].quantile(POOR_TAIL_FRAC)).mean()),
                ))
    return pd.DataFrame(rows)


def build_correlation_summary(combined):
    rows = []
    for (split, seed), sub in combined.groupby(["split", "seed"]):
        for param in PARAM_NAMES:
            rho, p = stats.spearmanr(sub[param], sub["rel_improvement"])
            rows.append(dict(split=split, seed=seed, param=param,
                              spearman_rho=rho, p_value=p, n=len(sub)))
    return pd.DataFrame(rows)


def identify_poor_samples(combined, tail_frac=POOR_TAIL_FRAC):
    out = []
    for split, sub in combined.groupby("split"):
        thresh = sub["rel_improvement"].quantile(tail_frac)
        poor = sub[sub["rel_improvement"] <= thresh].copy()
        poor["poor_threshold_rel_improvement"] = thresh
        out.append(poor)
    return pd.concat(out, ignore_index=True).sort_values(["split", "rel_improvement"])


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────
def plot_binned_improvement_vs_param(combined, param, out_path, n_bins=N_BINS):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax_imp, ax_delta = axes

    for split in ("train", "val", "test"):
        sub_split = combined[combined["split"] == split]
        if sub_split.empty:
            continue
        # bin edges computed once per split across all seeds pooled, for a stable x-axis
        try:
            bin_idx, edges = pd.qcut(sub_split[param], n_bins, labels=False, retbins=True, duplicates="drop")
        except ValueError:
            continue
        centers = 0.5 * (edges[:-1] + edges[1:])

        # per-seed means within each bin -> mean & std across seeds (shows consistency)
        per_seed_means_imp = []
        per_seed_means_before = []
        per_seed_means_after = []
        for seed, sub_seed in sub_split.groupby("seed"):
            seed_bin_idx = np.digitize(sub_seed[param], edges[1:-1])
            m_imp = np.full(len(edges) - 1, np.nan)
            m_before = np.full(len(edges) - 1, np.nan)
            m_after = np.full(len(edges) - 1, np.nan)
            for b in range(len(edges) - 1):
                mask = seed_bin_idx == b
                if mask.sum() > 0:
                    m_imp[b] = sub_seed["rel_improvement"].to_numpy()[mask].mean()
                    m_before[b] = sub_seed["delta_before"].to_numpy()[mask].mean()
                    m_after[b] = sub_seed["delta_after"].to_numpy()[mask].mean()
            per_seed_means_imp.append(m_imp)
            per_seed_means_before.append(m_before)
            per_seed_means_after.append(m_after)

        arr_imp = np.vstack(per_seed_means_imp)
        arr_before = np.vstack(per_seed_means_before)
        arr_after = np.vstack(per_seed_means_after)
        mean_imp, std_imp = np.nanmean(arr_imp, axis=0), np.nanstd(arr_imp, axis=0)
        mean_before = np.nanmean(arr_before, axis=0)
        mean_after = np.nanmean(arr_after, axis=0)

        c = SPLIT_COLORS[split]
        ax_imp.plot(centers, mean_imp, "-o", color=c, label=split)
        ax_imp.fill_between(centers, mean_imp - std_imp, mean_imp + std_imp, color=c, alpha=0.2)

        ax_delta.plot(centers, mean_before, "--o", color=c, alpha=0.6, label=f"{split} before")
        ax_delta.plot(centers, mean_after, "-s", color=c, label=f"{split} after")

    ax_imp.axhline(0, color="grey", ls=":", alpha=0.7)
    ax_imp.set_xlabel(PRETTY_NAMES[param])
    ax_imp.set_ylabel("Relative improvement\n(Δρ_before − Δρ_after) / Δρ_before")
    ax_imp.set_title(f"Correction strength vs {PRETTY_NAMES[param]}\n(mean ± std across seeds)")
    ax_imp.legend(fontsize=9)
    ax_imp.grid(alpha=0.3)

    ax_delta.set_yscale("log")
    ax_delta.set_xlabel(PRETTY_NAMES[param])
    ax_delta.set_ylabel("mean |Δρ| (pcm, log scale)")
    ax_delta.set_title("Before vs after, binned")
    ax_delta.legend(fontsize=8, ncol=2)
    ax_delta.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_poor_vs_good_distributions(combined, out_path, tail_frac=POOR_TAIL_FRAC):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.ravel()

    for i, param in enumerate(PARAM_NAMES):
        ax = axes[i]
        for split, c, ls in (("train", SPLIT_COLORS["train"], "-"),
                             ("val", SPLIT_COLORS["val"], "-"),
                             ("test", SPLIT_COLORS["test"], "-")):
            sub = combined[combined["split"] == split]
            if sub.empty:
                continue
            thresh = sub["rel_improvement"].quantile(tail_frac)
            poor = sub[sub["rel_improvement"] <= thresh]
            good = sub[sub["rel_improvement"] > thresh]
            bins = np.linspace(sub[param].min(), sub[param].max(), 25)
            ax.hist(good[param], bins=bins, density=True, alpha=0.25, color=c,
                    label=f"{split} rest" if i == 0 else None)
            ax.hist(poor[param], bins=bins, density=True, histtype="step", lw=2.2, color=c,
                    label=f"{split} worst {int(tail_frac*100)}%" if i == 0 else None)
        ax.set_xlabel(PRETTY_NAMES[param], fontsize=10)
        ax.set_ylabel("density")
        ax.grid(alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle(f"Parameter distributions: worst {int(tail_frac*100)}% relative-improvement samples "
                 "(step outline) vs the rest (filled)", y=1.09, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_before_after_scatter(combined, color_param, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), sharex=True, sharey=True)
    splits = ("train", "val", "test")
    vmin = combined[color_param].quantile(0.02)
    vmax = combined[color_param].quantile(0.98)

    sc = None
    for ax, split in zip(axes, splits):
        sub = combined[combined["split"] == split]
        if sub.empty:
            continue
        sc = ax.scatter(sub["delta_before"], sub["delta_after"], c=sub[color_param],
                         cmap="viridis", vmin=vmin, vmax=vmax, s=14, alpha=0.7, edgecolors="none")
        lims = [1.0, max(sub["delta_before"].max(), sub["delta_after"].max()) * 1.2]
        ax.plot(lims, lims, "--", color="grey", alpha=0.6, label="y = x (no improvement)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Δρ before (epoch 0, pcm)")
        ax.set_title(split)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_ylabel("Δρ after (final epoch, pcm)")
    if sc is not None:
        cbar = fig.colorbar(sc, ax=axes, shrink=0.85)
        cbar.set_label(PRETTY_NAMES.get(color_param, color_param))
    fig.suptitle("PEDS correction strength: points far above the diagonal are hardest to correct", fontsize=13)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_correlation_heatmap(corr_df, out_path):
    agg = corr_df.groupby(["split", "param"])["spearman_rho"].agg(["mean", "std"]).reset_index()
    splits = ["train", "val", "test"]
    mat = np.full((len(PARAM_NAMES), len(splits)), np.nan)
    for i, param in enumerate(PARAM_NAMES):
        for j, split in enumerate(splits):
            row = agg[(agg["param"] == param) & (agg["split"] == split)]
            if not row.empty:
                mat[i, j] = row["mean"].values[0]

    fig, ax = plt.subplots(figsize=(6.5, 6))
    vmax = np.nanmax(np.abs(mat))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(splits)))
    ax.set_xticklabels(splits)
    ax.set_yticks(range(len(PARAM_NAMES)))
    ax.set_yticklabels([PRETTY_NAMES[p] for p in PARAM_NAMES])
    for i in range(len(PARAM_NAMES)):
        for j in range(len(splits)):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        color="black", fontsize=10)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("mean Spearman ρ(param, rel. improvement) across seeds")
    ax.set_title("Does this parameter predict poor PEDS correction?\n(negative = higher param → worse relative improvement)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


def plot_f_mod_focus(combined, out_path, tail_frac=POOR_TAIL_FRAC):
    """Dedicated diagnostic panel for the f_mod hypothesis, split by seed to
    show the effect replicates rather than being a single-seed artifact."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    splits = ("train", "val", "test")
    seeds = sorted(combined["seed"].unique())
    cmap = plt.get_cmap("plasma", max(len(seeds), 2))

    for ax, split in zip(axes, splits):
        sub_split = combined[combined["split"] == split]
        if sub_split.empty:
            continue
        for si, seed in enumerate(seeds):
            sub = sub_split[sub_split["seed"] == seed]
            if sub.empty:
                continue
            try:
                bin_idx, edges = pd.qcut(sub["f_mod"], N_BINS, labels=False, retbins=True, duplicates="drop")
            except ValueError:
                continue
            centers = 0.5 * (edges[:-1] + edges[1:])
            means = [sub["rel_improvement"].to_numpy()[bin_idx == b].mean() for b in range(len(edges) - 1)]
            ax.plot(centers, means, "-o", color=cmap(si), alpha=0.85, label=f"seed {seed}")
        ax.axhline(0, color="grey", ls=":", alpha=0.7)
        ax.set_xlabel(PRETTY_NAMES["f_mod"])
        ax.set_title(split)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Relative improvement")
    axes[-1].legend(fontsize=8, loc="best")
    fig.suptitle("Relative improvement vs f_mod, per seed — checking the effect is not seed-specific", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default=DEFAULT_RUN_ROOT,
                         help="Folder containing train_<N>_seed_<S>/ subfolders "
                              "and testset_results/ (default: modules/RUNS/precise_param_strat)")
    parser.add_argument("--epoch-final", type=int, default=70,
                         help="Final epoch to compare against epoch 0 (default 70)")
    parser.add_argument("--tail-frac", type=float, default=POOR_TAIL_FRAC,
                         help="Fraction of samples (by rel. improvement) flagged as 'poorly corrected'")
    args = parser.parse_args()

    out_dir = os.path.join(args.run_root, "param_correction_analysis")
    fig_dir = os.path.join(out_dir, "figs")
    os.makedirs(fig_dir, exist_ok=True)

    print(f"Scanning {args.run_root} for seed runs …")
    combined = build_combined_table(args.run_root, args.epoch_final)
    print(f"\nCombined table: {len(combined)} rows "
          f"(splits: {combined['split'].value_counts().to_dict()})")
    combined.to_csv(os.path.join(out_dir, "per_sample_improvement.csv"), index=False)

    print("\n=== Relative improvement summary by split ===")
    print(combined.groupby("split")["rel_improvement"].describe().to_string())

    bin_summary = build_param_bin_summary(combined, n_bins=N_BINS)
    bin_summary.to_csv(os.path.join(out_dir, "param_bin_summary.csv"), index=False)

    corr_df = build_correlation_summary(combined)
    corr_df.to_csv(os.path.join(out_dir, "correlation_summary.csv"), index=False)
    print("\n=== Mean Spearman corr(param, rel_improvement) across seeds, by split ===")
    print(corr_df.groupby(["split", "param"])["spearman_rho"].mean()
          .unstack("split").round(3).to_string())

    poor_df = identify_poor_samples(combined, tail_frac=args.tail_frac)
    poor_df.to_csv(os.path.join(out_dir, "poorly_corrected_samples.csv"), index=False)
    print(f"\nFlagged {len(poor_df)} poorly-corrected samples "
          f"(worst {int(args.tail_frac*100)}% per split) → poorly_corrected_samples.csv")
    print("\nMean parameter values: poor-tail vs overall (by split):")
    for split, sub in combined.groupby("split"):
        poor_sub = poor_df[poor_df["split"] == split]
        comp = pd.DataFrame({
            "overall_mean": sub[PARAM_NAMES].mean(),
            "poor_tail_mean": poor_sub[PARAM_NAMES].mean(),
        })
        comp["pct_shift"] = 100 * (comp["poor_tail_mean"] - comp["overall_mean"]) / comp["overall_mean"].abs()
        print(f"\n  -- {split} --")
        print(comp.round(3).to_string())

    print("\nGenerating plots …")
    for param in PARAM_NAMES:
        plot_binned_improvement_vs_param(
            combined, param, os.path.join(fig_dir, f"binned_improvement_vs_{param}.png"))

    plot_poor_vs_good_distributions(
        combined, os.path.join(fig_dir, "poor_vs_good_param_distributions.png"),
        tail_frac=args.tail_frac)

    plot_before_after_scatter(
        combined, "f_mod", os.path.join(fig_dir, "before_after_scatter_colored_by_f_mod.png"))

    plot_correlation_heatmap(corr_df, os.path.join(fig_dir, "correlation_heatmap.png"))

    plot_f_mod_focus(combined, os.path.join(fig_dir, "f_mod_focus.png"), tail_frac=args.tail_frac)

    print(f"\nDone. All outputs under: {out_dir}")


if __name__ == "__main__":
    main()
