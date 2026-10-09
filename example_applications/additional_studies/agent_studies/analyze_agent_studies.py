"""analyze_agent_studies.py — Aggregate and compare agent study results.

Reads per-seed epoch_metrics.csv from each study directory under
RUNS/agent_studies/ and produces:
  - A comparison table (mean ± std across seeds) for headline metrics
  - A bin-by-bin breakdown (uses keff_epoch_log_val.csv) comparable
    to the reference precise_param_strat/testset_results/keff_dist_bins_by_split.csv
  - Saves results to RUNS/agent_studies/aggregate_results.csv

Usage (on login node, no compute):
    cd /global/home/users/caterinafrau/PEDS_NT/modules
    python RUNS/agent_studies/analyze_agent_studies.py
"""

import os
import sys
import glob
import numpy as np
import pandas as pd

THIS_DIR  = os.path.dirname(os.path.abspath(__file__))   # RUNS/agent_studies/
MOD_DIR   = os.path.dirname(os.path.dirname(THIS_DIR))   # modules/

# ── Ordered list of studies (same order as STUDY_CONFIGS in PEDS_agent.py) ──
STUDY_ORDER = [
    "s01_baseline",
    "s02_weighted_loss",
    "s03_dropout",
    "s04_adamw",
    "s05_keff_input",
    "s06_residual",
    "s07_elu",
    "s08_wider",
    "s09_noise_aug",
    "s10_balanced",
]

STUDY_DESCRIPTIONS = {
    "s01_baseline":       "Baseline [128,256,128] ReLU MSE",
    "s02_weighted_loss":  "PCM-weighted loss (w=1/k^4)",
    "s03_dropout":        "Dropout 0.15",
    "s04_adamw":          "AdamW wd=1e-4",
    "s05_keff_input":     "keff_baseline as 7th input",
    "s06_residual":       "Residual trunk [128,128,128]",
    "s07_elu":            "ELU activation",
    "s08_wider":          "Wider [256,512,256]",
    "s09_noise_aug":      "Input noise σ=0.01",
    "s10_balanced":       "Balanced sampling (low-keff 2×)",
}

SEEDS = [0, 1, 2]
TRAIN_SIZE = 500


def find_run_dir(study, seed):
    exp_name = f"train_{TRAIN_SIZE}_seed_{seed}"
    path = os.path.join(THIS_DIR, study, exp_name)
    return path if os.path.isdir(path) else None


def load_epoch_metrics(run_dir):
    """Load epoch_metrics.csv, return the row with the lowest val_mean_pcm."""
    p = os.path.join(run_dir, "epoch_metrics.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    if df.empty:
        return None
    best_row = df.loc[df["val_mean_pcm"].idxmin()]
    return best_row


def load_per_sample_val(run_dir, best_epoch):
    """Load keff_epoch_log_val.csv for the best epoch."""
    p = os.path.join(run_dir, "keff_epoch_log_val.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    if df.empty:
        return None
    df_ep = df[df["epoch"] == best_epoch].copy()
    return df_ep


def compute_bin_metrics(df_ep, df_ep0, n_bins=10):
    """Compute keff-bin breakdown similar to keff_dist_bins_by_split.csv.

    df_ep  : per-sample predictions at best epoch (columns: keff_openmc, keff_peds, delta_rho_pcm)
    df_ep0 : per-sample predictions at epoch 0 (baseline, before correction)
    """
    if df_ep is None or df_ep0 is None:
        return None

    # Merge on sample_idx to get both before/after
    df = df_ep[["sample_idx", "keff_openmc", "keff_peds", "delta_rho_pcm"]].copy()
    df = df.rename(columns={"delta_rho_pcm": "delta_rho_after"})
    df0 = df_ep0[["sample_idx", "delta_rho_pcm"]].rename(columns={"delta_rho_pcm": "delta_rho_before"})
    df  = df.merge(df0, on="sample_idx", how="inner")
    if df.empty:
        return None

    # Cut into bins by keff_openmc
    bins = np.percentile(df["keff_openmc"], np.linspace(0, 100, n_bins + 1))
    bins[0]  -= 1e-6
    bins[-1] += 1e-6
    df["bin"] = pd.cut(df["keff_openmc"], bins=bins, labels=False)

    rows = []
    for b in range(n_bins):
        sub = df[df["bin"] == b]
        if len(sub) == 0:
            continue
        rows.append(dict(
            bin                  = b,
            n_samples            = len(sub),
            avg_keff             = sub["keff_openmc"].mean(),
            avg_delta_rho_before = sub["delta_rho_before"].mean(),
            avg_delta_rho_after  = sub["delta_rho_after"].mean(),
            frac_below_650_after = (sub["delta_rho_after"] < 650).mean(),
            pct_improvement      = (1 - sub["delta_rho_after"].mean() / sub["delta_rho_before"].mean()) * 100,
        ))
    return pd.DataFrame(rows)


def aggregate_study(study):
    """Collect metrics across all seeds for one study."""
    seed_metrics = []
    bin_dfs      = []

    for seed in SEEDS:
        rdir = find_run_dir(study, seed)
        if rdir is None:
            continue

        best_row = load_epoch_metrics(rdir)
        if best_row is None:
            continue
        best_epoch = int(best_row["epoch"])

        seed_metrics.append({
            "seed":               seed,
            "best_epoch":         best_epoch,
            "val_mean_pcm":       float(best_row["val_mean_pcm"]),
            "val_frac_below_650": float(best_row.get("val_frac_below_650", np.nan)),
            "train_mean_pcm":     float(best_row.get("train_mean_pcm", np.nan)),
        })

        # Per-sample val predictions for bin analysis
        df_ep  = load_per_sample_val(rdir, best_epoch)
        df_ep0 = load_per_sample_val(rdir, 0)
        bdf    = compute_bin_metrics(df_ep, df_ep0)
        if bdf is not None:
            bdf["seed"] = seed
            bin_dfs.append(bdf)

    if not seed_metrics:
        return None, None

    m = pd.DataFrame(seed_metrics)
    summary = {
        "study":                study,
        "description":          STUDY_DESCRIPTIONS.get(study, ""),
        "n_seeds":              len(m),
        "val_mean_pcm_mean":    m["val_mean_pcm"].mean(),
        "val_mean_pcm_std":     m["val_mean_pcm"].std(),
        "val_frac_below_650_mean": m["val_frac_below_650"].mean(),
        "val_frac_below_650_std":  m["val_frac_below_650"].std(),
        "train_mean_pcm_mean":  m["train_mean_pcm"].mean(),
        "best_epoch_mean":      m["best_epoch"].mean(),
    }

    bin_summary = None
    if bin_dfs:
        all_bins = pd.concat(bin_dfs)
        bin_summary = (
            all_bins.groupby("bin")
            .agg(
                avg_keff_mean            = ("avg_keff", "mean"),
                avg_delta_rho_before_mean= ("avg_delta_rho_before", "mean"),
                avg_delta_rho_after_mean = ("avg_delta_rho_after", "mean"),
                frac_below_650_mean      = ("frac_below_650_after", "mean"),
                pct_improvement_mean     = ("pct_improvement", "mean"),
                pct_improvement_std      = ("pct_improvement", "std"),
            )
            .reset_index()
        )
        bin_summary["study"] = study

    return summary, bin_summary


def main():
    all_summaries = []
    all_bin_summaries = []

    for study in STUDY_ORDER:
        print(f"Processing {study} …", end=" ", flush=True)
        summary, bin_summary = aggregate_study(study)
        if summary is None:
            print("no data yet")
            continue
        print(f"n_seeds={summary['n_seeds']}  "
              f"val_mean={summary['val_mean_pcm_mean']:.1f}±{summary['val_mean_pcm_std']:.1f} pcm  "
              f"frac<650={summary['val_frac_below_650_mean']*100:.1f}%")
        all_summaries.append(summary)
        if bin_summary is not None:
            all_bin_summaries.append(bin_summary)

    if not all_summaries:
        print("\nNo results found yet. Check that runs have completed.")
        return

    # ── Main comparison table ────────────────────────────────────────────────
    df_sum = pd.DataFrame(all_summaries)
    print("\n" + "="*90)
    print("STUDY COMPARISON — headline metrics (mean ± std across seeds)")
    print("="*90)
    ref_val  = df_sum.loc[df_sum["study"] == "s01_baseline", "val_mean_pcm_mean"].values
    ref_frac = df_sum.loc[df_sum["study"] == "s01_baseline", "val_frac_below_650_mean"].values

    for _, row in df_sum.iterrows():
        delta_pcm  = row["val_mean_pcm_mean"] - ref_val[0] if len(ref_val) else 0.0
        delta_frac = (row["val_frac_below_650_mean"] - ref_frac[0]) * 100 if len(ref_frac) else 0.0
        sign_pcm   = "↓" if delta_pcm < -5 else ("↑" if delta_pcm > 5 else "~")
        sign_frac  = "↑" if delta_frac > 1 else ("↓" if delta_frac < -1 else "~")
        print(
            f"  {row['study']:<22s}  "
            f"val_pcm={row['val_mean_pcm_mean']:7.1f}±{row['val_mean_pcm_std']:5.1f}  "
            f"({sign_pcm}{abs(delta_pcm):5.1f})  "
            f"frac<650={row['val_frac_below_650_mean']*100:5.1f}%  "
            f"({sign_frac}{abs(delta_frac):.1f}pp)  "
            f"trn_pcm={row['train_mean_pcm_mean']:7.1f}  "
            f"| {row['description']}"
        )

    out_csv = os.path.join(THIS_DIR, "aggregate_results.csv")
    df_sum.to_csv(out_csv, index=False)
    print(f"\nSaved aggregate results → {out_csv}")

    # ── Bin-by-bin comparison ────────────────────────────────────────────────
    if all_bin_summaries:
        df_bins = pd.concat(all_bin_summaries, ignore_index=True)
        bin_out = os.path.join(THIS_DIR, "bin_comparison.csv")
        df_bins.to_csv(bin_out, index=False)
        print(f"Saved bin comparison     → {bin_out}")

        print("\n" + "="*90)
        print("BIN-BY-BIN IMPROVEMENT (pct_improvement_mean across seeds, by study)")
        print("="*90)
        pivot = df_bins.pivot_table(
            index="study", columns="bin",
            values="pct_improvement_mean", aggfunc="mean"
        ).round(1)
        pivot.columns = [f"bin{c}" for c in pivot.columns]
        print(pivot.to_string())


if __name__ == "__main__":
    main()
