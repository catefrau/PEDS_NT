"""
param_error_analysis.py
------------------------
Answers: "which parameter ranges, or combinations of parameters, cause the
biggest discrepancy (delta_rho_pcm) between the NN-corrected keff and the
OpenMC reference?" -- for the TRAIN, VAL and TEST sets, each written to its
own subfolder, and for BOTH epoch 0 (before the NN does anything -- i.e. the
bias already present in the physics-based PEDS surrogate) and the last epoch
(after training), so you can see what the NN fixed and what it didn't.

INPUT DATA
----------
* Test set: one or more "keff_comparison" csvs that already contain the 6
  geometry parameters as columns alongside keff_openmc / keff_peds /
  delta_rho_pcm, for epoch 0 and the final epoch (see TEST_CSV_GLOB).

* Train / val sets: for every run directory matching RUN_DIR_GLOB (one per
  seed), this script reads keff_epoch_log_train.csv / keff_epoch_log_val.csv
  (columns: epoch, sample_idx, keff_openmc, keff_peds, delta_rho_pcm,
  train_loss, val_loss -- NO parameters). Note that `sample_idx` in these
  files is a local, per-split, per-epoch enumeration index, NOT the sample's
  index in the original npz -- it can't be used to join anything. Instead we
  recover the 6 geometry parameters straight from `keff_openmc` (rounded to
  6 decimals in these logs) by nearest-match against NPZ_PATH's `keffs` /
  `params_raw` arrays (empirically accurate to ~5e-7, far below the spacing
  between distinct samples' keffs).

Only epoch 0 and the last logged epoch are kept for every run (as requested
-- intermediate epochs aren't interesting for this analysis).

Aggregation across seeds: for a given dataset (train/val/test) and epoch tag
(epoch0/final), rows from every matched seed/run are grouped by their
(rounded) parameter vector and delta_rho_pcm is averaged across seeds -- this
reduces training-noise in the error estimate. Samples that only appear in one
seed's split just don't get averaged with anything.

OUTPUTS (per dataset, under OUTDIR/<train|val|test>/)
-------------------------------------------------------
1. importance_table_epoch0.csv / importance_table_final.csv: for each of the
   6 parameters, how much it explains the error, by three measures (Pearson
   correlation, Spearman correlation, Random-Forest permutation importance).
2. importance_comparison.png: single grouped bar chart, RF importance per
   parameter, epoch0 vs final side by side.
3. binned_error_grid.png: single figure, one subplot per parameter (grid),
   each showing mean delta_rho_pcm per bin for epoch0 vs final side by side.
   A bin that stands out above the overall mean is a "hard" range for that
   parameter alone (marginal effect).
4. interactions_grid.png: single figure, one row per top parameter pair (by
   final-epoch importance), 4 columns (observed epoch0, residual epoch0,
   observed final, residual final). The "residual vs additive" panel is the
   observed 2D cell mean minus what you'd expect if the two parameters acted
   independently -- a high residual means the *combination* of ranges matters
   more than either parameter's marginal effect alone. Interaction scores are
   printed and saved to interaction_scores.csv. Only 2D cells with at least
   MIN_CELL_N_FOR_INTERACTION samples contribute; pairs whose sparsest
   occupied cell is below that floor are marked unreliable (score = NaN).
5. suggested_bin_weights.csv: for each parameter, a suggested relative
   sampling weight per bin (higher where final-epoch error is higher/more
   variable), which you can feed into your existing bounds-cutting function
   to build a stratified LHS. This is a *marginal* (per-parameter)
   allocation -- check interactions_grid.png before trusting this blindly.
6. abs_error_scatter_epoch0.png / abs_error_scatter_final.png: 2x3 scatter
   of each geometry parameter vs |delta_rho_pcm|, with Pearson r annotated
   and the worst 5% of samples highlighted in red (same samples across
   all panels).
7. samples_keff_error_params.csv: one row per unique geometry for epoch0 and
   final, with keff_openmc / keff_peds / delta_rho_pcm and the 6 parameters
   (train/val params recovered via match_keff_to_params against the npz;
   test params come from the comparison csvs). Sorted by descending error
   within each epoch so the worst geometries are at the top.

OUTPUTS (cross-set, under OUTDIR/)
------------------------------------
8. metrics_summary_by_set.csv: same metrics as evaluate_test_metrics.py's
   new_test_metrics_summary_by_train_size.csv (mean/median/p95/std pcm,
   frac below 650/100, mse_k, mae_k), but one row per (dataset, nn_correction)
   with mean +/- std across seeds. nn_correction="no" is epoch 0 (physics
   baseline), "yes" is the final epoch (NN-corrected).
9. importance_summary_by_set.csv: pearson / spearman / RF importance for
   every parameter, one row per (parameter, dataset, epoch_tag), so you can
   check whether correlations hold across train / val / test.
10. binned_error_by_set_epoch0.png / binned_error_by_set_final.png: for each
    parameter, mean delta_rho_pcm per bin with shared bin edges across
    splits; train / val / test as side-by-side bars (different colors) to
    confirm the same hard ranges. Companion CSV: binned_error_by_set.csv.
11. param_distributions_by_set.png: one subplot per parameter, with the
    train / val / test distributions overlaid (density histograms of unique
    samples), so you can spot under-represented ranges in any split.
12. correlation_by_keff_bin_{epoch0,final}.csv (+ heatmap PNG): Pearson /
    Spearman correlation of each geometry parameter vs delta_rho_pcm, computed
    separately inside each of N_KEFF_BINS shared keff ranges — so you can see
    whether a correlation only exists in a particular keff band. Heatmap cells
    annotate Pearson r and the bin's % of samples (paper-sized fonts/DPI).
13. correlation_by_keff_bin_by_set_{epoch0,final}.png and
    correlation_by_keff_bin_by_set_lines_{epoch0,final}.png: cross-split
    Pearson summary (train/val/test; reliable bins only).

WHAT TO EDIT
------------
Only the "SPECS" block right below the imports.
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

# ============================== SPECS (edit) ==============================
# Test set: glob matching every per-sample "keff_comparison" csv you want to
# include. Each file must already contain the 6 parameter columns plus
# epoch/sample_idx/keff_openmc/keff_peds/delta_rho_pcm.
study_folder = "decay_study/decay_70_EPOCHS_100_different_holdout"

TEST_CSV_GLOB = f"RUNS/{study_folder}/testset_results/rep_train*_keff_comparison.csv"

# NPZ file used to train/derive the LHS samples. Must contain "keffs"
# (unique per sample) and "params_raw" (the 6 physical parameter values) plus
# "param_names". Used to recover the 6 parameters for train/val rows, which
# aren't stored directly in keff_epoch_log_*.csv.
NPZ_PATH = "../data/highfidelity/13jul_merged_bigR.npz"

# Train/val sets: glob matching every run directory (one per seed) that
# contains keff_epoch_log_train.csv, keff_epoch_log_val.csv and
# split_log.csv.
RUN_DIR_GLOB = f"RUNS/{study_folder}/train_1000_seed_*"

OUTDIR = f"error_attribution_report/{study_folder}"


# Max allowed |keff_a - keff_b| when matching a logged keff to an npz keff.
# train/val logs round keff_openmc to 6 decimals, so matches are typically
# accurate to ~5e-7; keep this tight so close-but-distinct npz samples
# (some pairs are only ~1e-6 apart) don't get confused with each other.
KEFF_MATCH_TOL = 2e-5

# Which column is the error to explain.
ERROR_COL = "delta_rho_pcm"

# Non-parameter columns to ignore when auto-detecting the 6 parameters in the
# test csvs (auto-detection isn't needed for train/val -- params come from
# the npz join and are known by name: param_names in the npz).
# signed_delta_rho_pcm is an error diagnostic column, NOT a geometry param.
NON_PARAM_COLS = ["epoch", "sample_idx", "keff_openmc", "keff_peds", ERROR_COL,
                   "signed_delta_rho_pcm", "keff", "train_loss", "val_loss",
                   "split", "seed_dir", "epoch_tag"]

SPLIT_COLORS = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"}

N_BINS = 8                 # bins per parameter for the marginal error plots
N_KEFF_BINS = 10           # keff ranges for correlation-by-keff-bin analysis
MIN_SAMPLES_PER_KEFF_BIN = 20  # skip corr in a bin with fewer unique samples
TOP_K_FOR_INTERACTIONS = 6  # look at pairwise interactions among the top-K
                             # most important parameters (by final-epoch
                             # importance)
MIN_CELL_N_FOR_INTERACTION = 15  # min samples in a 2D bin for that cell to
                                 # enter the interaction score; if any occupied
                                 # cell is below this, the pair is unreliable
WORST_FRAC = 0.05          # fraction of highest-|error| samples highlighted
                             # red on the abs-error scatter plots
# ===========================================================================


# --------------------------- npz parameter lookup ---------------------------

def load_npz_lookup(npz_path):
    d = np.load(npz_path)
    keffs = np.asarray(d["keffs"], dtype=np.float64)
    params_raw = np.asarray(d["params_raw"], dtype=np.float64)
    param_names = [str(x) for x in d["param_names"]]
    order = np.argsort(keffs)
    return keffs[order], params_raw[order], param_names


def match_keff_to_params(keff_values, sorted_keffs, sorted_params, tol=KEFF_MATCH_TOL):
    """Nearest-neighbor match of each keff value against the npz's sorted
    keffs array. Returns (matched_params [N,6], ok_mask [N])."""
    keff_values = np.asarray(keff_values, dtype=np.float64)
    idx = np.searchsorted(sorted_keffs, keff_values)
    idx = np.clip(idx, 1, len(sorted_keffs) - 1)
    left, right = idx - 1, idx
    dist_l = np.abs(sorted_keffs[left] - keff_values)
    dist_r = np.abs(sorted_keffs[right] - keff_values)
    use_r = dist_r < dist_l
    best = np.where(use_r, right, left)
    dist = np.where(use_r, dist_r, dist_l)
    ok = dist <= tol
    n_bad = int((~ok).sum())
    if n_bad:
        print(f"    WARNING: {n_bad}/{len(keff_values)} keff values had no npz match "
              f"within tol={tol} (max dist={dist.max():.2e}) -- dropped.")
    return sorted_params[best], ok


# ------------------------------- data loading -------------------------------

def load_train_or_val_raw(split_name, npz_lookup):
    """Loads keff_epoch_log_<split_name>.csv (epoch 0 + last epoch only) for
    every run dir matching RUN_DIR_GLOB, recovers the 6 parameters straight
    from keff_openmc via the npz lookup, and tags each row 'epoch0' or
    'final'. Returns a single concatenated DataFrame with columns: epoch,
    epoch_tag, sample_idx, keff_openmc, keff_peds, delta_rho_pcm,
    <param_names>, seed_dir."""
    sorted_keffs, sorted_params, param_names = npz_lookup
    run_dirs = sorted(glob.glob(RUN_DIR_GLOB))
    if not run_dirs:
        raise FileNotFoundError(f"No run dirs matched RUN_DIR_GLOB={RUN_DIR_GLOB!r}")

    frames = []
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        log_path = run_dir / f"keff_epoch_log_{split_name}.csv"
        if not log_path.exists():
            print(f"  skipping {run_dir} (missing {log_path.name})")
            continue

        log_df = pd.read_csv(log_path)
        if log_df.empty:
            continue

        max_epoch = log_df["epoch"].max()
        log_df["epoch_tag"] = np.where(log_df["epoch"] == 0, "epoch0",
                                 np.where(log_df["epoch"] == max_epoch, "final", None))
        log_df = log_df[log_df["epoch_tag"].notna()].copy()

        params, ok = match_keff_to_params(log_df["keff_openmc"].values, sorted_keffs, sorted_params)
        log_df = log_df.loc[ok].copy()
        params = params[ok]
        for i, name in enumerate(param_names):
            log_df[name] = params[:, i]

        log_df["seed_dir"] = run_dir.name
        frames.append(log_df)

    if not frames:
        raise FileNotFoundError(f"No usable {split_name} data found under RUN_DIR_GLOB={RUN_DIR_GLOB!r}")
    full = pd.concat(frames, ignore_index=True)
    print(f"  loaded {split_name}: {len(full)} rows from {len(frames)} run dir(s) "
          f"({(full['epoch_tag'] == 'epoch0').sum()} epoch0 + {(full['epoch_tag'] == 'final').sum()} final)")
    return full, param_names


def load_test_raw():
    """Loads every csv matching TEST_CSV_GLOB (params already present as
    columns) and tags each row 'epoch0' or 'final' (per-file max epoch)."""
    files = sorted(glob.glob(TEST_CSV_GLOB, recursive=True))
    if not files:
        raise FileNotFoundError(f"No files matched TEST_CSV_GLOB={TEST_CSV_GLOB!r}")
    print(f"  found {len(files)} test file(s): {files}")

    frames = []
    param_cols = None
    for f in files:
        df = pd.read_csv(f)
        if param_cols is None:
            param_cols = [c for c in df.columns if c not in NON_PARAM_COLS]
        max_epoch = df["epoch"].max()
        df["epoch_tag"] = np.where(df["epoch"] == 0, "epoch0",
                             np.where(df["epoch"] == max_epoch, "final", None))
        df = df[df["epoch_tag"].notna()]
        df["seed_dir"] = Path(f).stem
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    print(f"  loaded test: {len(full)} rows from {len(frames)} file(s) "
          f"({(full['epoch_tag'] == 'epoch0').sum()} epoch0 + {(full['epoch_tag'] == 'final').sum()} final)")
    return full, param_cols


def aggregate_by_param(full, param_cols, keep_keff=False):
    """Groups rows by (rounded) parameter vector, averaging ERROR_COL across
    seeds/runs. `full` should already be filtered to a single epoch_tag.
    If keep_keff=True, also keeps keff_openmc (first) and keff_peds (mean)."""
    round_cols = [f"_r_{c}" for c in param_cols]
    full = full.copy()
    for c, rc in zip(param_cols, round_cols):
        full[rc] = full[c].round(6)

    agg_spec = {**{c: "first" for c in param_cols}, ERROR_COL: ["mean", "std", "count"]}
    if keep_keff:
        agg_spec["keff_openmc"] = "first"
        agg_spec["keff_peds"] = "mean"

    agg = full.groupby(round_cols, as_index=False).agg(agg_spec)
    new_cols = []
    for c in agg.columns:
        if isinstance(c, tuple):
            base, func = c
            new_cols.append(base if func in ("first", "") else f"{base}_{func}")
        else:
            new_cols.append(c)
    agg.columns = new_cols
    agg = agg.drop(columns=round_cols)
    rename = {f"{ERROR_COL}_mean": ERROR_COL,
              f"{ERROR_COL}_std": f"{ERROR_COL}_std_across_seeds",
              f"{ERROR_COL}_count": "n_seeds"}
    if keep_keff and f"keff_peds_mean" in agg.columns:
        rename["keff_peds_mean"] = "keff_peds"
    agg = agg.rename(columns=rename)
    return agg


def save_samples_keff_error_params(df0_raw, dfF_raw, param_cols, outdir, dataset_label):
    """One CSV per split: unique geometries at epoch0 + final with keff, error,
    and the 6 parameters. Sorted by descending error within each epoch."""
    frames = []
    for tag, raw in [("epoch0", df0_raw), ("final", dfF_raw)]:
        if raw.empty:
            continue
        agg = aggregate_by_param(raw, param_cols, keep_keff=True)
        agg.insert(0, "epoch_tag", tag)
        agg = agg.sort_values(ERROR_COL, ascending=False).reset_index(drop=True)
        frames.append(agg)

    if not frames:
        print(f"  [{dataset_label}] no samples to write for samples_keff_error_params.csv")
        return None

    out = pd.concat(frames, ignore_index=True)
    # Stable, readable column order
    lead = ["epoch_tag", "keff_openmc", "keff_peds", ERROR_COL,
            f"{ERROR_COL}_std_across_seeds", "n_seeds"]
    cols = [c for c in lead if c in out.columns] + [c for c in param_cols if c in out.columns]
    out = out[cols]
    out_path = Path(outdir) / "samples_keff_error_params.csv"
    out.to_csv(out_path, index=False)
    n0 = int((out["epoch_tag"] == "epoch0").sum())
    nF = int((out["epoch_tag"] == "final").sum())
    print(f"  samples keff/error/params saved → {out_path} "
          f"(epoch0={n0}, final={nF}; sorted by descending {ERROR_COL})")
    return out


# ------------------------------- analysis core -------------------------------

def importance_table(df, param_cols):
    X = df[param_cols].values
    y = df[ERROR_COL].values

    rows = []
    for i, name in enumerate(param_cols):
        pear = np.corrcoef(X[:, i], y)[0, 1]
        spear = pd.Series(X[:, i]).corr(pd.Series(y), method="spearman")
        rows.append({"parameter": name, "pearson_corr": pear, "spearman_corr": spear})
    table = pd.DataFrame(rows)

    rf = RandomForestRegressor(n_estimators=500, max_depth=6, min_samples_leaf=3, random_state=0)
    rf.fit(X, y)
    perm = permutation_importance(rf, X, y, n_repeats=30, random_state=0)
    table["rf_permutation_importance"] = perm.importances_mean
    table = table.sort_values("rf_permutation_importance", ascending=False).reset_index(drop=True)
    return table, rf


def _safe_corr(x, y, method="pearson"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    if method == "pearson":
        return float(np.corrcoef(x, y)[0, 1])
    return float(pd.Series(x).corr(pd.Series(y), method="spearman"))


def correlation_by_keff_bin(df, param_cols, edges=None, n_bins=N_KEFF_BINS,
                            min_n=MIN_SAMPLES_PER_KEFF_BIN):
    """
    Split samples into shared keff bins and compute Pearson / Spearman
    correlation of each parameter vs ERROR_COL inside each bin.

    Returns (table DataFrame, edges ndarray). `df` must contain keff_openmc.
    """
    if "keff_openmc" not in df.columns or df.empty:
        return pd.DataFrame(), edges

    k = df["keff_openmc"].astype(float).to_numpy()
    if edges is None:
        lo, hi = float(np.min(k)), float(np.max(k))
        if hi <= lo:
            hi = lo + 1e-6
        edges = np.linspace(lo, hi, n_bins + 1)
    else:
        n_bins = len(edges) - 1

    bin_idx = np.digitize(k, edges[1:-1], right=False)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    y = df[ERROR_COL].astype(float).to_numpy()

    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        n = int(mask.sum())
        avg_keff = float(np.mean(k[mask])) if n else float("nan")
        avg_err = float(np.mean(y[mask])) if n else float("nan")
        for name in param_cols:
            x = df[name].astype(float).to_numpy()[mask]
            if n < min_n:
                pear = spear = float("nan")
            else:
                pear = _safe_corr(x, y[mask], method="pearson")
                spear = _safe_corr(x, y[mask], method="spearman")
            rows.append({
                "bin": b,
                "keff_lo": float(edges[b]),
                "keff_hi": float(edges[b + 1]),
                "n_samples": n,
                "avg_keff": avg_keff,
                "avg_delta_rho_pcm": avg_err,
                "parameter": name,
                "pearson_corr": pear,
                "spearman_corr": spear,
                "reliable": bool(n >= min_n),
            })
    return pd.DataFrame(rows), edges


def _short_param_label(name):
    """Shorter y-tick labels for paper figures."""
    return (name
            .replace("r0_b4c_rod_", "r0 ")
            .replace("r1_fuel_annulus_", "r1 ")
            .replace("r2_water_", "r2 ")
            .replace("_", " "))


def _keff_bin_ticklabels(table, param_cols, bins):
    # One row per bin (edges are shared); drop duplicates when `table` spans
    # multiple datasets / parameters.
    edge_src = (table.drop_duplicates(subset=["bin"])
                     .set_index("bin")
                     .reindex(bins))
    labels = [f"[{lo:.3f}, {hi:.3f})" for lo, hi in zip(edge_src["keff_lo"], edge_src["keff_hi"])]
    if labels:
        labels[-1] = labels[-1][:-1] + "]"
    return labels


def _bin_pct_lookup(table):
    """Map bin index → percent of total samples (same for every parameter)."""
    per_bin = (table.drop_duplicates(subset=["bin"])
                    .set_index("bin")["n_samples"]
                    .astype(float))
    total = float(per_bin.sum()) if len(per_bin) else 0.0
    if total <= 0:
        return {int(b): 0.0 for b in per_bin.index}
    return {int(b): 100.0 * float(n) / total for b, n in per_bin.items()}


def plot_correlation_by_keff_bin(table, param_cols, outdir, dataset_label, epoch_label,
                                 corr_col="pearson_corr"):
    """Heatmap: parameters (rows) × keff bins (cols), colored by correlation.

    Cell annotation: Pearson value + sample % of this split (not raw n).
    Sized for paper figures (large fonts, high DPI).
    """
    if table is None or table.empty:
        return
    bins = sorted(table["bin"].unique())
    pct_by_bin = _bin_pct_lookup(table)
    mat = np.full((len(param_cols), len(bins)), np.nan)
    annot_pct = np.full((len(param_cols), len(bins)), np.nan)
    for i, name in enumerate(param_cols):
        sub = table[table["parameter"] == name].set_index("bin")
        for j, b in enumerate(bins):
            if b in sub.index:
                mat[i, j] = sub.loc[b, corr_col]
                annot_pct[i, j] = pct_by_bin.get(int(b), np.nan)

    fig_w = max(18.0, 1.7 * len(bins) + 6.0)
    fig_h = max(9.5, 1.35 * len(param_cols) + 4.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    vmax = np.nanmax(np.abs(mat)) if np.isfinite(mat).any() else 1.0
    vmax = max(float(vmax), 0.05)
    im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_yticks(np.arange(len(param_cols)))
    ax.set_yticklabels([_short_param_label(p) for p in param_cols], fontsize=22)
    labels = _keff_bin_ticklabels(table, param_cols, bins)
    ax.set_xticks(np.arange(len(bins)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=18)
    for i in range(len(param_cols)):
        for j in range(len(bins)):
            val = mat[i, j]
            pct = annot_pct[i, j]
            pct_txt = f"{pct:.0f}%" if np.isfinite(pct) else ""
            if np.isnan(val):
                ax.text(j, i, pct_txt if pct_txt else "—", ha="center", va="center",
                        fontsize=18, color="#555555", fontweight="medium")
            else:
                # white text on strong |r|, black otherwise
                txt_color = "white" if abs(val) > 0.55 * vmax else "black"
                ax.text(j, i, f"{val:.2f}\n{pct_txt}", ha="center", va="center",
                        fontsize=18, color=txt_color, fontweight="bold", linespacing=1.35)
    ax.set_xlabel(r"$k_{\mathrm{eff}}$ bin", fontsize=24)
    ax.set_ylabel("geometry parameter", fontsize=24)
    ax.set_title(
        f"{dataset_label} / {epoch_label}: Pearson "
        rf"$r$(param, $|\Delta\rho|$) by $k_{{\mathrm{{eff}}}}$ bin",
        fontsize=24, pad=14,
    )
    ax.tick_params(axis="both", which="both", length=0)
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Pearson $r$", fontsize=22)
    cbar.ax.tick_params(labelsize=18)
    fig.tight_layout()
    out_path = Path(outdir) / f"correlation_by_keff_bin_{epoch_label}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  correlation-by-keff-bin heatmap saved → {out_path}")


def plot_correlation_by_keff_bin_by_set(summary_df, outdir, epoch_tag="final"):
    """
    Paper figure for the cross-split Pearson summary (reliable cells only).

    Two companion views written for `epoch_tag`:
      1. Side-by-side heatmaps (train | val | test), unreliable masked.
      2. Multi-panel lines: Pearson vs keff-bin center, one panel per
         parameter, train/val/test overlaid (markers only where reliable).
    """
    if summary_df is None or summary_df.empty:
        return
    sub = summary_df[summary_df["epoch_tag"] == epoch_tag].copy()
    if sub.empty:
        return

    # normalize reliability to bool (CSV may store strings)
    if sub["reliable"].dtype != bool:
        sub["reliable"] = sub["reliable"].astype(str).str.lower().isin(["true", "1", "yes"])

    splits = [s for s in ("train", "val", "test") if s in set(sub["dataset"])]
    param_cols = list(dict.fromkeys(sub["parameter"].tolist()))
    bins = sorted(sub["bin"].unique())
    labels = _keff_bin_ticklabels(sub, param_cols, bins)
    centers = (
        sub.drop_duplicates("bin")
           .set_index("bin")
           .reindex(bins)
    )
    keff_centers = (0.5 * (centers["keff_lo"] + centers["keff_hi"])).to_numpy()

    # Shared color scale across splits (reliable Pearson only)
    rel_vals = sub.loc[sub["reliable"], "pearson_corr"].astype(float)
    vmax = float(np.nanmax(np.abs(rel_vals))) if len(rel_vals) and np.isfinite(rel_vals).any() else 1.0
    vmax = max(vmax, 0.05)

    # ── 1) side-by-side heatmaps ──────────────────────────────────────────
    fig, axes = plt.subplots(
        1, len(splits),
        figsize=(7.8 * len(splits) + 2.0, 1.4 * len(param_cols) + 5.5),
        sharey=True, constrained_layout=True,
    )
    if len(splits) == 1:
        axes = [axes]
    last_im = None
    for ax, split in zip(axes, splits):
        sdf = sub[sub["dataset"] == split]
        pct_by_bin = _bin_pct_lookup(sdf)
        mat = np.full((len(param_cols), len(bins)), np.nan)
        pct_mat = np.full((len(param_cols), len(bins)), np.nan)
        for i, name in enumerate(param_cols):
            psub = sdf[sdf["parameter"] == name].set_index("bin")
            for j, b in enumerate(bins):
                if b not in psub.index:
                    continue
                row = psub.loc[b]
                pct_mat[i, j] = pct_by_bin.get(int(b), np.nan)
                if bool(row["reliable"]) and np.isfinite(row["pearson_corr"]):
                    mat[i, j] = float(row["pearson_corr"])
        last_im = ax.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_xticks(np.arange(len(bins)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=16)
        ax.set_yticks(np.arange(len(param_cols)))
        if ax is axes[0]:
            ax.set_yticklabels([_short_param_label(p) for p in param_cols], fontsize=20)
        ax.set_title(split, fontsize=24, pad=10)
        ax.set_xlabel(r"$k_{\mathrm{eff}}$ bin", fontsize=20)
        ax.tick_params(axis="both", which="both", length=0)
        for i in range(len(param_cols)):
            for j in range(len(bins)):
                val = mat[i, j]
                pct = pct_mat[i, j]
                pct_txt = f"{pct:.0f}%" if np.isfinite(pct) else ""
                if np.isnan(val):
                    ax.text(j, i, pct_txt if pct_txt else "—", ha="center", va="center",
                            fontsize=15, color="#666666", fontweight="medium")
                else:
                    txt_color = "white" if abs(val) > 0.55 * vmax else "black"
                    ax.text(j, i, f"{val:.2f}\n{pct_txt}", ha="center", va="center",
                            fontsize=15, color=txt_color, fontweight="bold",
                            linespacing=1.3)
    axes[0].set_ylabel("geometry parameter", fontsize=22)
    fig.suptitle(
        f"Pearson $r$(param, $|\\Delta\\rho|$) by $k_{{\\mathrm{{eff}}}}$ bin "
        f"— reliable cells only ({epoch_tag})",
        fontsize=24, y=1.02,
    )
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes, fraction=0.02, pad=0.02)
        cbar.set_label("Pearson $r$", fontsize=20)
        cbar.ax.tick_params(labelsize=16)

    out_hm = Path(outdir) / f"correlation_by_keff_bin_by_set_{epoch_tag}.png"
    fig.savefig(out_hm, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  cross-set Pearson heatmap saved → {out_hm}")

    # ── 2) multi-panel lines (reliable only) ──────────────────────────────
    ncols = 3
    nrows = int(np.ceil(len(param_cols) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(6.5 * ncols, 4.8 * nrows),
        sharex=True, sharey=True, squeeze=False,
        constrained_layout=True,
    )
    markers = {"train": "o", "val": "s", "test": "D"}
    for i, name in enumerate(param_cols):
        ax = axes[i // ncols][i % ncols]
        ax.axhline(0.0, color="0.55", lw=1.2, zorder=0)
        for split in splits:
            sdf = sub[(sub["dataset"] == split) & (sub["parameter"] == name)
                      & (sub["reliable"])].copy()
            if sdf.empty:
                continue
            sdf = sdf.set_index("bin").reindex(bins)
            y = sdf["pearson_corr"].astype(float).to_numpy()
            ax.plot(
                keff_centers, y,
                color=SPLIT_COLORS.get(split, "C0"),
                marker=markers.get(split, "o"),
                markersize=11, linewidth=2.8,
                label=split, zorder=2,
            )
        ax.set_title(_short_param_label(name), fontsize=20, pad=8)
        ax.set_ylim(-1.05, 1.05)
        ax.tick_params(labelsize=16)
        ax.grid(True, alpha=0.3)
        if i // ncols == nrows - 1:
            ax.set_xlabel(r"$k_{\mathrm{eff}}$ (bin center)", fontsize=18)
        if i % ncols == 0:
            ax.set_ylabel("Pearson $r$", fontsize=18)
        if i == 0:
            ax.legend(fontsize=16, frameon=True, loc="best")

    for j in range(len(param_cols), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(
        f"Pearson $r$ vs $k_{{\\mathrm{{eff}}}}$ bin — reliable only ({epoch_tag})",
        fontsize=24,
    )
    out_ln = Path(outdir) / f"correlation_by_keff_bin_by_set_lines_{epoch_tag}.png"
    fig.savefig(out_ln, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  cross-set Pearson lines saved → {out_ln}")


def analyze_correlation_by_keff_range(df0, dfF, param_cols, outdir, dataset_label,
                                      shared_edges=None):
    """Per-dataset keff-binned correlations for epoch0 and final."""
    outdir = Path(outdir)
    # Build shared edges from whichever df has keff (prefer union of both).
    if shared_edges is None and "keff_openmc" in df0.columns and "keff_openmc" in dfF.columns:
        all_k = np.concatenate([
            df0["keff_openmc"].astype(float).to_numpy(),
            dfF["keff_openmc"].astype(float).to_numpy(),
        ])
        lo, hi = float(np.min(all_k)), float(np.max(all_k))
        if hi <= lo:
            hi = lo + 1e-6
        shared_edges = np.linspace(lo, hi, N_KEFF_BINS + 1)

    table0, edges = correlation_by_keff_bin(df0, param_cols, edges=shared_edges)
    tableF, _ = correlation_by_keff_bin(dfF, param_cols, edges=edges)

    if not table0.empty:
        table0.to_csv(outdir / "correlation_by_keff_bin_epoch0.csv", index=False)
        plot_correlation_by_keff_bin(table0, param_cols, outdir, dataset_label, "epoch0")
    if not tableF.empty:
        tableF.to_csv(outdir / "correlation_by_keff_bin_final.csv", index=False)
        plot_correlation_by_keff_bin(tableF, param_cols, outdir, dataset_label, "final")

        # Compact print: strongest |pearson| bin per parameter (final only)
        print(f"\n=== [{dataset_label}] strongest |pearson| param↔error by keff bin (final) ===")
        reliable = tableF[tableF["reliable"]].copy()
        if reliable.empty:
            print("  (no keff bin met MIN_SAMPLES_PER_KEFF_BIN)")
        else:
            reliable["abs_pearson"] = reliable["pearson_corr"].abs()
            top = (reliable.sort_values("abs_pearson", ascending=False)
                           .groupby("parameter", as_index=False)
                           .first())
            show = top[["parameter", "bin", "keff_lo", "keff_hi", "n_samples",
                        "pearson_corr", "spearman_corr"]]
            print(show.to_string(index=False))

    return {"table0": table0, "tableF": tableF, "edges": edges}


def summarize_correlation_by_keff_bin_across_sets(corr_results, outdir):
    """
    corr_results: list of (dataset_label, table0, tableF).
    Writes correlation_by_keff_bin_by_set.csv and paper-ready Pearson figures
    comparing train / val / test (reliable cells only).
    """
    rows = []
    for label, table0, tableF in corr_results:
        for epoch_tag, table in [("epoch0", table0), ("final", tableF)]:
            if table is None or table.empty:
                continue
            t = table.copy()
            t.insert(0, "dataset", label)
            t.insert(1, "epoch_tag", epoch_tag)
            rows.append(t)
    if not rows:
        return None
    out = pd.concat(rows, ignore_index=True)
    out_path = Path(outdir) / "correlation_by_keff_bin_by_set.csv"
    out.to_csv(out_path, index=False)
    print(f"\n=== correlation by keff bin across sets (saved → {out_path}) ===")
    for epoch_tag in ("epoch0", "final"):
        plot_correlation_by_keff_bin_by_set(out, outdir, epoch_tag=epoch_tag)
    return out


def marginal_bins(df, name):
    vals = df[name].values
    edges = np.linspace(vals.min(), vals.max(), N_BINS + 1)
    bin_idx = np.clip(np.digitize(vals, edges[1:-1]), 0, N_BINS - 1)
    means, counts = [], []
    for b in range(N_BINS):
        mask = bin_idx == b
        counts.append(int(mask.sum()))
        means.append(df[ERROR_COL].values[mask].mean() if mask.sum() > 0 else np.nan)
    return edges, np.array(means), np.array(counts)


def plot_binned_grid(df0, dfF, param_cols, outdir, dataset_label):
    """Single figure, one subplot per parameter, epoch0 vs final bars."""
    n = len(param_cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows), squeeze=False)

    for i, name in enumerate(param_cols):
        ax = axes[i // ncols][i % ncols]
        edges0, means0, counts0 = marginal_bins(df0, name)
        edgesF, meansF, countsF = marginal_bins(dfF, name)
        centers = 0.5 * (edgesF[:-1] + edgesF[1:])
        width = np.diff(edgesF) * 0.4
        ax.bar(centers - width / 2, means0, width=width, color="#8C8C8C", edgecolor="white", label="epoch 0")
        ax.bar(centers + width / 2, meansF, width=width, color="#55A868", edgecolor="white", label="final")
        ax.axhline(df0[ERROR_COL].mean(), color="#8C8C8C", ls=":", lw=1)
        ax.axhline(dfF[ERROR_COL].mean(), color="#C44E52", ls="--", lw=1, label="overall mean (final)")
        for c, m, n_ in zip(centers + width / 2, meansF, countsF):
            if not np.isnan(m):
                ax.text(c, m, f"{n_}", ha="center", va="bottom", fontsize=6)
        ax.set_title(name, fontsize=9)
        ax.set_ylabel(ERROR_COL, fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle(f"{dataset_label}: mean {ERROR_COL} per bin, epoch 0 vs final", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(Path(outdir) / "binned_error_grid.png", dpi=130)
    plt.close(fig)


def compute_interaction(df, pA, pB, min_cell_n=MIN_CELL_N_FOR_INTERACTION):
    """2D bin the pair (pA, pB), compute observed cell means and residuals vs
    an additive (row+col effect) model. Returns dict of matrices/edges plus a
    normalized interaction score = std(residual) / std(observed cell means).

    Only cells with >= min_cell_n samples contribute to cell means / score.
    If any occupied cell falls below that floor, reliable=False and score=NaN.
    """
    n_bins_2d = max(3, N_BINS // 2)
    xa, xb, y = df[pA].values, df[pB].values, df[ERROR_COL].values

    edges_a = np.linspace(xa.min(), xa.max(), n_bins_2d + 1)
    edges_b = np.linspace(xb.min(), xb.max(), n_bins_2d + 1)
    bin_a = np.clip(np.digitize(xa, edges_a[1:-1]), 0, n_bins_2d - 1)
    bin_b = np.clip(np.digitize(xb, edges_b[1:-1]), 0, n_bins_2d - 1)

    cell_mean = np.full((n_bins_2d, n_bins_2d), np.nan)
    cell_count = np.zeros((n_bins_2d, n_bins_2d), dtype=int)
    for i in range(n_bins_2d):
        for j in range(n_bins_2d):
            mask = (bin_a == i) & (bin_b == j)
            cell_count[i, j] = mask.sum()
            if mask.sum() >= min_cell_n:
                cell_mean[i, j] = y[mask].mean()

    occupied = cell_count[cell_count > 0]
    min_occupied = int(occupied.min()) if occupied.size else 0
    reliable = bool(occupied.size > 0 and min_occupied >= min_cell_n)

    grand = np.nanmean(cell_mean)
    row_eff = np.nanmean(cell_mean, axis=1) - grand
    col_eff = np.nanmean(cell_mean, axis=0) - grand
    additive = grand + row_eff[:, None] + col_eff[None, :]
    residual = cell_mean - additive

    valid = ~np.isnan(cell_mean)
    if reliable and valid.sum() > 0:
        obs_std = np.nanstd(cell_mean[valid])
        resid_std = np.nanstd(residual[valid])
        score = resid_std / obs_std if obs_std > 0 else np.nan
    else:
        score = np.nan

    return {"cell_mean": cell_mean, "residual": residual, "cell_count": cell_count,
            "score": score, "min_cell_n": min_occupied, "reliable": reliable,
            "edges_a": edges_a, "edges_b": edges_b}


def plot_interactions_grid(df0, dfF, pairs, outdir, dataset_label):
    """Single figure: one row per pair, 4 columns (obs0, resid0, obsF, residF)."""
    if not pairs:
        return pd.DataFrame()

    nrows = len(pairs)
    fig, axes = plt.subplots(nrows, 4, figsize=(16, 3.6 * nrows), squeeze=False)
    score_rows = []

    def _score_label(res):
        if not res["reliable"] or not np.isfinite(res["score"]):
            return (f"unreliable (min_n={res['min_cell_n']}"
                    f"<{MIN_CELL_N_FOR_INTERACTION})")
        return f"score={res['score']:.2f}"

    for r, (pA, pB) in enumerate(pairs):
        res0 = compute_interaction(df0, pA, pB)
        resF = compute_interaction(dfF, pA, pB)
        score_rows.append({"param_A": pA, "param_B": pB,
                            "interaction_score_epoch0": res0["score"],
                            "interaction_score_final": resF["score"],
                            "min_cell_n_epoch0": res0["min_cell_n"],
                            "min_cell_n_final": resF["min_cell_n"],
                            "reliable_epoch0": res0["reliable"],
                            "reliable_final": resF["reliable"]})

        panels = [("observed epoch0", res0["cell_mean"], "viridis"),
                  (f"residual epoch0 ({_score_label(res0)})", res0["residual"], "coolwarm"),
                  ("observed final", resF["cell_mean"], "viridis"),
                  (f"residual final ({_score_label(resF)})", resF["residual"], "coolwarm")]
        for c, (title, mat, cmap) in enumerate(panels):
            ax = axes[r][c]
            im = ax.imshow(mat, origin="lower", aspect="auto", cmap=cmap)
            ax.set_title(title, fontsize=8)
            ax.tick_params(labelsize=6)
            if c == 0:
                ax.set_ylabel(f"{pA}\n({pB} on x)", fontsize=7)
            plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(f"{dataset_label}: pairwise interactions, epoch 0 vs final", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(Path(outdir) / "interactions_grid.png", dpi=130)
    plt.close(fig)

    score_table = pd.DataFrame(score_rows)
    score_table.to_csv(Path(outdir) / "interaction_scores.csv", index=False)
    return score_table


def plot_importance_comparison(table0, tableF, outdir, dataset_label):
    order = tableF["parameter"].tolist()
    t0 = table0.set_index("parameter").reindex(order)
    tF = tableF.set_index("parameter").reindex(order)

    y_pos = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(order) + 1.5))
    height = 0.38
    ax.barh(y_pos + height / 2, t0["rf_permutation_importance"], height=height,
            color="#8C8C8C", label="epoch 0")
    ax.barh(y_pos - height / 2, tF["rf_permutation_importance"], height=height,
            color="#55A868", label="final")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(order, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("RF permutation importance")
    ax.set_title(f"{dataset_label}: parameter importance for {ERROR_COL}, epoch 0 vs final")
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(outdir) / "importance_comparison.png", dpi=130)
    plt.close(fig)


def plot_abs_error_scatter(df, param_cols, outdir, dataset_label, epoch_label):
    """2x3 scatter of each parameter vs |delta_rho_pcm|; worst WORST_FRAC in red."""
    y = np.abs(df[ERROR_COL].astype(float).to_numpy())
    thresh = np.quantile(y, 1.0 - WORST_FRAC)
    worst = y >= thresh

    n = len(param_cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows), squeeze=False)

    for i, name in enumerate(param_cols):
        ax = axes[i // ncols][i % ncols]
        x = df[name].astype(float).to_numpy()
        ax.scatter(x[~worst], y[~worst], s=14, c="#B8B8B8", alpha=0.45,
                   edgecolors="none", zorder=1)
        ax.scatter(x[worst], y[worst], s=42, c="#C44E52", alpha=0.9,
                   edgecolors="k", linewidths=0.45, zorder=2)
        if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0:
            r = float(np.corrcoef(x, y)[0, 1])
        else:
            r = float("nan")
        ax.text(0.5, 0.98, f"r = {r:.2f}", transform=ax.transAxes,
                ha="center", va="top", fontsize=12)
        ax.set_xlabel(name, fontsize=12)
        ax.set_ylabel(r"$|\Delta\rho|$ (pcm)", fontsize=12)
        ax.tick_params(labelsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    pct = int(round(WORST_FRAC * 100))
    fig.suptitle(
        f"{dataset_label}: Absolute reactivity error vs. geometry parameters — "
        f"{epoch_label} (red = worst {pct}% of samples)",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = Path(outdir) / f"abs_error_scatter_{epoch_label}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  abs-error scatter saved → {out_path} "
          f"(n={len(y)}, worst {pct}% threshold={thresh:.1f} pcm, n_worst={int(worst.sum())})")


def suggested_bin_weights(df, param_cols, outdir):
    """Computed from the final-epoch data -- the ranges that are still
    problematic after training are what matter for future resampling."""
    rows = []
    for name in param_cols:
        edges, means, counts = marginal_bins(df, name)
        means_clipped = np.clip(means, 0.0, None)
        means_clipped = np.nan_to_num(means_clipped)
        w = np.sqrt(means_clipped)
        w = w / w.sum() if w.sum() > 0 else np.ones(N_BINS) / N_BINS
        for b in range(N_BINS):
            rows.append({
                "parameter": name, "bin": b,
                "lo": edges[b], "hi": edges[b + 1],
                "mean_error_in_bin": means[b],
                "n_in_bin": int(counts[b]),
                "suggested_weight": w[b],
            })
    table = pd.DataFrame(rows)
    table.to_csv(Path(outdir) / "suggested_bin_weights.csv", index=False)
    return table


# -------------------- cross-set metrics + param distributions --------------------

METRIC_NAMES = [
    "mean_pcm", "median_pcm", "frac_below_650", "frac_below_100",
    "mse_k", "mae_k", "p95_pcm", "std_pcm",
]


def _metrics_from_rows(df):
    """Same headline metrics as PEDS.compute_metrics / evaluate_test_metrics."""
    k_ref = df["keff_openmc"].astype(float).to_numpy()
    k_pred = df["keff_peds"].astype(float).to_numpy()
    dr = df[ERROR_COL].astype(float).to_numpy()
    return {
        "mean_pcm": float(np.mean(dr)),
        "median_pcm": float(np.median(dr)),
        "frac_below_650": float(np.mean(dr < 650.0)),
        "frac_below_100": float(np.mean(dr < 100.0)),
        "mse_k": float(np.mean((k_pred - k_ref) ** 2)),
        "mae_k": float(np.mean(np.abs(k_pred - k_ref))),
        "p95_pcm": float(np.percentile(dr, 95.0)),
        "std_pcm": float(np.std(dr)),
    }


def summarize_metrics_by_set(datasets, outdir):
    """
    datasets: list of (label, full_df) where full_df has epoch_tag + seed_dir
              and keff_openmc / keff_peds / delta_rho_pcm.
    Writes metrics_summary_by_set.csv with one row per (dataset, nn_correction),
    mean +/- std of each metric across seeds (same shape as
    evaluate_test_metrics' new_test_metrics_summary_by_train_size.csv).
    """
    rows = []
    for label, full in datasets:
        for epoch_tag, nn_corr in [("epoch0", "no"), ("final", "yes")]:
            sub = full[full["epoch_tag"] == epoch_tag]
            if sub.empty:
                continue
            per_seed = []
            for _, seed_df in sub.groupby("seed_dir"):
                per_seed.append(_metrics_from_rows(seed_df))
            seed_table = pd.DataFrame(per_seed)
            row = {
                "dataset": label,
                "n_seeds": int(len(seed_table)),
                "nn_correction": nn_corr,
                "n_samples_per_seed_mean": float(sub.groupby("seed_dir").size().mean()),
            }
            for m in METRIC_NAMES:
                row[f"{m}_mean"] = float(seed_table[m].mean())
                row[f"{m}_std"] = float(seed_table[m].std(ddof=0)) if len(seed_table) > 1 else 0.0
            rows.append(row)

    summary = pd.DataFrame(rows)
    # stable column order matching evaluate_test_metrics priority
    ordered = ["dataset", "n_seeds", "nn_correction", "n_samples_per_seed_mean"]
    priority = ["mean_pcm", "median_pcm", "frac_below_650", "frac_below_100",
                "mse_k", "mae_k", "p95_pcm", "std_pcm"]
    for m in priority:
        ordered += [f"{m}_mean", f"{m}_std"]
    summary = summary[ordered]
    summary = summary.sort_values(["dataset", "nn_correction"]).reset_index(drop=True)

    out_path = Path(outdir) / "metrics_summary_by_set.csv"
    summary.to_csv(out_path, index=False)
    print(f"\n=== metrics summary across sets (saved → {out_path}) ===")
    show = ["dataset", "nn_correction", "n_seeds", "mean_pcm_mean", "mean_pcm_std",
            "median_pcm_mean", "frac_below_650_mean", "frac_below_100_mean"]
    print(summary[show].to_string(index=False))
    return summary


def plot_param_distributions_by_set(datasets, param_cols, outdir):
    """
    datasets: list of (label, unique_param_df) -- one row per unique sample
              (params only; epoch doesn't matter since params are fixed).
    Overlays train/val/test density histograms for each of the 6 parameters
    so under-represented ranges jump out visually.
    """
    n = len(param_cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows), squeeze=False)

    for i, name in enumerate(param_cols):
        ax = axes[i // ncols][i % ncols]
        # shared bin edges across sets so the overlay is fair
        all_vals = np.concatenate([df[name].values for _, df in datasets if name in df.columns])
        edges = np.linspace(all_vals.min(), all_vals.max(), N_BINS + 1)
        for label, df in datasets:
            vals = df[name].values
            ax.hist(vals, bins=edges, density=True, alpha=0.45,
                    color=SPLIT_COLORS.get(label, None), label=f"{label} (n={len(df)})",
                    edgecolor="white", linewidth=0.5)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel(name, fontsize=8)
        ax.set_ylabel("density", fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle("Parameter distributions by set (unique samples, density)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = Path(outdir) / "param_distributions_by_set.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  param distributions saved → {out_path}")


def summarize_importance_by_set(importance_results, outdir):
    """
    importance_results: list of (dataset_label, table0, tableF).
    Writes importance_summary_by_set.csv with one row per
    (parameter, dataset, epoch_tag) so correlations can be compared
    across train / val / test.
    """
    rows = []
    metric_cols = ["pearson_corr", "spearman_corr", "rf_permutation_importance"]
    for label, table0, tableF in importance_results:
        for epoch_tag, table in [("epoch0", table0), ("final", tableF)]:
            for _, row in table.iterrows():
                out = {"parameter": row["parameter"], "dataset": label, "epoch_tag": epoch_tag}
                for m in metric_cols:
                    out[m] = float(row[m])
                rows.append(out)

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(
        ["epoch_tag", "parameter", "dataset"]
    ).reset_index(drop=True)

    out_path = Path(outdir) / "importance_summary_by_set.csv"
    summary.to_csv(out_path, index=False)
    print(f"\n=== importance summary across sets (saved → {out_path}) ===")
    # Compact print: pearson only, final epoch, wide by dataset
    final = summary[summary["epoch_tag"] == "final"]
    if not final.empty:
        wide = final.pivot(index="parameter", columns="dataset", values="pearson_corr")
        # Stable column order when present
        for col in ["train", "val", "test"]:
            if col not in wide.columns:
                wide[col] = np.nan
        wide = wide[["train", "val", "test"]]
        print("Pearson corr (final epoch) by dataset:")
        print(wide.to_string())
    return summary


def plot_binned_error_by_set(datasets_by_epoch, param_cols, outdir):
    """
    datasets_by_epoch: dict epoch_tag -> list of (label, unique_sample_df)
    For each epoch, one 2x3 figure: mean ERROR_COL per bin with shared edges
    across splits; train/val/test as grouped bars.
    Also writes binned_error_by_set.csv with the underlying numbers.
    """
    csv_rows = []
    for epoch_tag, datasets in datasets_by_epoch.items():
        if not datasets:
            continue

        n = len(param_cols)
        ncols = 3
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.6 * nrows), squeeze=False)
        n_sets = len(datasets)

        for i, name in enumerate(param_cols):
            ax = axes[i // ncols][i % ncols]
            all_vals = np.concatenate(
                [df[name].values for _, df in datasets if name in df.columns]
            )
            edges = np.linspace(all_vals.min(), all_vals.max(), N_BINS + 1)
            centers = 0.5 * (edges[:-1] + edges[1:])
            # Equal-width bins from linspace -> use a scalar bar width so the
            # per-split offset vector stays shape (n_sets,) not (n_sets, n_bins).
            bin_width = float(edges[1] - edges[0])
            bar_w = bin_width * (0.8 / max(n_sets, 1))
            offsets = (np.arange(n_sets) - (n_sets - 1) / 2.0) * bar_w

            for s_i, (label, df) in enumerate(datasets):
                vals = df[name].values
                y = df[ERROR_COL].values
                bin_idx = np.clip(np.digitize(vals, edges[1:-1]), 0, N_BINS - 1)
                means, counts = [], []
                for b in range(N_BINS):
                    mask = bin_idx == b
                    counts.append(int(mask.sum()))
                    means.append(float(y[mask].mean()) if mask.sum() > 0 else np.nan)
                    csv_rows.append({
                        "epoch_tag": epoch_tag,
                        "parameter": name,
                        "dataset": label,
                        "bin": b,
                        "lo": edges[b],
                        "hi": edges[b + 1],
                        "mean_error_in_bin": means[-1],
                        "n_in_bin": counts[-1],
                    })
                means = np.asarray(means, dtype=float)
                x = centers + offsets[s_i]
                ax.bar(x, means, width=bar_w,
                       color=SPLIT_COLORS.get(label, f"C{s_i}"),
                       edgecolor="white", linewidth=0.5,
                       label=label if i == 0 else None)
                for c, m, n_ in zip(x, means, counts):
                    if not np.isnan(m) and n_ > 0:
                        ax.text(c, m, f"{n_}", ha="center", va="bottom", fontsize=5)

            ax.set_title(name, fontsize=9)
            ax.set_ylabel(ERROR_COL, fontsize=8)
            ax.tick_params(labelsize=7)
            if i == 0:
                ax.legend(fontsize=7)

        for j in range(n, nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")

        fig.suptitle(
            f"Mean {ERROR_COL} per bin by split — {epoch_tag} "
            f"(shared bin edges; colors = train/val/test)",
            fontsize=11,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out_path = Path(outdir) / f"binned_error_by_set_{epoch_tag}.png"
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        print(f"  binned error by set saved → {out_path}")

    if csv_rows:
        csv_path = Path(outdir) / "binned_error_by_set.csv"
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
        print(f"  binned error by set table saved → {csv_path}")


# --------------------------------- driver -----------------------------------

def analyze_dataset(full, param_cols, outdir, dataset_label, keff_edges=None):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Never treat error / bookkeeping columns as geometry parameters.
    param_cols = [c for c in param_cols if c not in NON_PARAM_COLS and c in full.columns]

    df0_raw = full[full["epoch_tag"] == "epoch0"]
    dfF_raw = full[full["epoch_tag"] == "final"]
    # keep keff so correlation-by-keff-bin can condition on ranges
    df0 = aggregate_by_param(df0_raw, param_cols, keep_keff=True)
    dfF = aggregate_by_param(dfF_raw, param_cols, keep_keff=True)
    print(f"  {dataset_label}: epoch0 -> {len(df0)} unique samples, final -> {len(dfF)} unique samples")

    save_samples_keff_error_params(df0_raw, dfF_raw, param_cols, outdir, dataset_label)

    table0, _ = importance_table(df0, param_cols)
    tableF, _ = importance_table(dfF, param_cols)
    table0.to_csv(outdir / "importance_table_epoch0.csv", index=False)
    tableF.to_csv(outdir / "importance_table_final.csv", index=False)
    print(f"\n=== [{dataset_label}] importance (final epoch) for {ERROR_COL} ===")
    print(tableF.to_string(index=False))

    plot_importance_comparison(table0, tableF, outdir, dataset_label)
    plot_binned_grid(df0, dfF, param_cols, outdir, dataset_label)
    plot_abs_error_scatter(df0, param_cols, outdir, dataset_label, "epoch0")
    plot_abs_error_scatter(dfF, param_cols, outdir, dataset_label, "final")

    corr_keff = analyze_correlation_by_keff_range(
        df0, dfF, param_cols, outdir, dataset_label, shared_edges=keff_edges,
    )

    top_params = tableF["parameter"].tolist()[:TOP_K_FOR_INTERACTIONS]
    pairs = [(top_params[i], top_params[j])
             for i in range(len(top_params)) for j in range(i + 1, len(top_params))]
    score_table = plot_interactions_grid(df0, dfF, pairs, outdir, dataset_label)
    if not score_table.empty:
        print(f"\n=== [{dataset_label}] interaction scores (top {TOP_K_FOR_INTERACTIONS}: {top_params}) ===")
        print(score_table.to_string(index=False))

    suggested_bin_weights(dfF, param_cols, outdir)

    print(f"\n[{dataset_label}] outputs saved to {outdir}/")
    print("  samples_keff_error_params.csv (epoch0 + final, sorted by descending error)")
    print("  importance_table_epoch0.csv, importance_table_final.csv")
    print("  importance_comparison.png")
    print("  binned_error_grid.png")
    print("  abs_error_scatter_epoch0.png, abs_error_scatter_final.png")
    print("  correlation_by_keff_bin_{epoch0,final}.csv/.png")
    print("  interactions_grid.png, interaction_scores.csv")
    print("  suggested_bin_weights.csv (based on final epoch)")
    return {
        "df0": df0,
        "dfF": dfF,
        "table0": table0,
        "tableF": tableF,
        "corr_keff": corr_keff,
    }


def main():
    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading npz parameter lookup...")
    npz_lookup = load_npz_lookup(NPZ_PATH)

    # Load all three sets first so keff-bin edges are shared across splits.
    print("\n--- Loading raw splits ---")
    test_full, test_params = load_test_raw()
    val_full, val_params = load_train_or_val_raw("val", npz_lookup)
    train_full, train_params = load_train_or_val_raw("train", npz_lookup)

    all_keffs = np.concatenate([
        train_full["keff_openmc"].astype(float).to_numpy(),
        val_full["keff_openmc"].astype(float).to_numpy(),
        test_full["keff_openmc"].astype(float).to_numpy(),
    ])
    lo, hi = float(np.min(all_keffs)), float(np.max(all_keffs))
    if hi <= lo:
        hi = lo + 1e-6
    shared_keff_edges = np.linspace(lo, hi, N_KEFF_BINS + 1)
    print(f"  shared keff bin edges ({N_KEFF_BINS} bins): "
          f"[{shared_keff_edges[0]:.4f}, {shared_keff_edges[-1]:.4f}]")

    print("\n--- TEST set ---")
    test_res = analyze_dataset(
        test_full, test_params, outdir / "test", "test", keff_edges=shared_keff_edges,
    )

    print("\n--- VAL set ---")
    val_res = analyze_dataset(
        val_full, val_params, outdir / "val", "val", keff_edges=shared_keff_edges,
    )

    print("\n--- TRAIN set ---")
    train_res = analyze_dataset(
        train_full, train_params, outdir / "train", "train", keff_edges=shared_keff_edges,
    )

    # Prefer the train-set param column order (comes from the npz); fall back
    # to whatever the test csv detected if train somehow had none.
    param_cols = train_params or val_params or test_params
    # Drop any leftover non-param columns (e.g. signed_delta_rho_pcm) if a
    # test csv listed them before NON_PARAM_COLS was updated.
    param_cols = [c for c in param_cols if c not in NON_PARAM_COLS]

    print("\n--- Cross-set summaries ---")
    summarize_metrics_by_set(
        [("train", train_full), ("val", val_full), ("test", test_full)],
        outdir,
    )
    summarize_importance_by_set(
        [
            ("train", train_res["table0"], train_res["tableF"]),
            ("val", val_res["table0"], val_res["tableF"]),
            ("test", test_res["table0"], test_res["tableF"]),
        ],
        outdir,
    )
    summarize_correlation_by_keff_bin_across_sets(
        [
            ("train", train_res["corr_keff"]["table0"], train_res["corr_keff"]["tableF"]),
            ("val", val_res["corr_keff"]["table0"], val_res["corr_keff"]["tableF"]),
            ("test", test_res["corr_keff"]["table0"], test_res["corr_keff"]["tableF"]),
        ],
        outdir,
    )
    plot_binned_error_by_set(
        {
            "epoch0": [
                ("train", train_res["df0"]),
                ("val", val_res["df0"]),
                ("test", test_res["df0"]),
            ],
            "final": [
                ("train", train_res["dfF"]),
                ("val", val_res["dfF"]),
                ("test", test_res["dfF"]),
            ],
        },
        param_cols,
        outdir,
    )
    plot_param_distributions_by_set(
        [
            ("train", train_res["dfF"]),
            ("val", val_res["dfF"]),
            ("test", test_res["dfF"]),
        ],
        param_cols,
        outdir,
    )

    print(f"\nAll done. See {outdir}/{{test,val,train}}/ plus")
    print(f"  {outdir}/metrics_summary_by_set.csv")
    print(f"  {outdir}/importance_summary_by_set.csv")
    print(f"  {outdir}/correlation_by_keff_bin_by_set.csv")
    print(f"  {outdir}/binned_error_by_set_{{epoch0,final}}.png")
    print(f"  {outdir}/binned_error_by_set.csv")
    print(f"  {outdir}/param_distributions_by_set.png")


if __name__ == "__main__":
    main()
