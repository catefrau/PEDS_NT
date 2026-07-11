"""
param_error_analysis.py
------------------------
Answers: "which parameter ranges, or combinations of parameters, cause the
biggest discrepancy (delta_rho_pcm) between the NN-corrected keff and the
OpenMC reference?"

It expects one or more CSV files (one per training seed/run), each already
containing the parameter values alongside the error for every test sample,
e.g.:

    epoch,sample_idx,keff_openmc,keff_peds,delta_rho_pcm,
    r0_b4c_rod_outer_radius,r0_b4c_rod_cr_fraction,
    r1_fuel_annulus_outer_radius,r1_fuel_annulus_enrichment,
    r1_fuel_annulus_f_mod,r2_water_outer_radius

No join against the training npz is needed anymore -- the parameters are
already on each row.

WHAT TO EDIT
------------
Only the "SPECS" block right below the imports. Everything else can be left
alone unless you want to change the method.

WHAT THE OUTPUTS MEAN
----------------------
1. importance_table (csv + printed): for each of the 6 parameters, how much
   it explains the error, by three different measures (Pearson correlation,
   Spearman correlation, and Random-Forest permutation importance). They
   agree on the ranking most of the time; when they don't, it usually means
   a nonlinear or interaction effect (RF importance picks it up, correlation
   doesn't).

2. binned_error_<param>.png: for each parameter, its range is cut into
   N_BINS equal-width bins, and the mean (+ std, + sample count) of
   delta_rho_pcm is plotted per bin. A bin that stands out well above the
   overall mean is a "hard" range for that parameter alone (marginal
   effect). Low sample count in a bin means low confidence in that bin's
   estimate -- check the printed counts, don't just trust the bar height.

3. interaction_<paramA>_<paramB>.png: 2D heatmap of mean delta_rho_pcm
   over bins of the top parameter pairs (by importance). Alongside it,
   an "interaction score" is printed: this compares the observed 2D cell
   means against what you'd expect if the two parameters' effects were
   purely additive (i.e. each acting independently). A high interaction
   score means the *combination* of ranges matters more than either
   parameter's marginal effect alone -- this is precisely the "combination
   of parameters" effect you asked about. A low score means the marginal
   (1D) plots already tell the whole story for that pair.

4. suggested_bin_weights.csv: for each parameter, a suggested relative
   sampling weight per bin (higher where error is higher/more variable),
   which you can feed into your existing bounds-cutting function to build
   a stratified LHS. This is a *marginal* (per-parameter) allocation -- if
   step 3 shows strong interactions for some pair, a purely marginal
   scheme may still under-sample the specific bad combination, so check
   the interaction plots before trusting this table blindly.

Aggregation across seeds: if TEST_CSV_GLOB matches multiple files (multiple
seeds of the same experiment, same test set), rows are grouped by their
parameter vector (rounded) and delta_rho_pcm is averaged across seeds --
this reduces training-noise in the error estimate. If your seeds actually
use different test sets, that's still fine: rows with unique parameter
vectors just won't be averaged with anything.
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
# Glob pattern matching every test-set csv you want to include (one seed
# each, or just one file). Use ** for recursive matching.
TEST_CSV_GLOB = "RUNS/different_ranges/bounded_poly/testset_results/representative_train1000_seed1_keff_comparison.csv"

# Which column is the error to explain.
ERROR_COL = "delta_rho_pcm"

# Non-parameter columns to ignore when auto-detecting the 6 parameters.
# Everything else in the csv is treated as a parameter.
NON_PARAM_COLS = ["epoch", "sample_idx", "keff_openmc", "keff_peds", ERROR_COL]

# If a csv has multiple epochs per sample_idx (e.g. it's actually a
# per-epoch log, not a single final test evaluation), keep only the row
# with the max epoch per sample_idx. Set to False if every row is already
# a single evaluation and you want to keep them all as-is.
KEEP_ONLY_LAST_EPOCH_PER_SAMPLE = True
KEEP_ONLY_FIRST_EPOCH_PER_SAMPLE = False

N_BINS = 8                 # bins per parameter for the marginal error plots
TOP_K_FOR_INTERACTIONS = 3  # look at pairwise interactions among the top-K
                             # most important parameters
OUTDIR = "error_attribution_report"
# ===========================================================================


def load_and_aggregate():
    files = sorted(glob.glob(TEST_CSV_GLOB, recursive=True))
    if not files:
        raise FileNotFoundError(f"No files matched TEST_CSV_GLOB={TEST_CSV_GLOB!r}")
    print(f"Found {len(files)} file(s):")
    for f in files:
        print(f"  {f}")

    frames = []
    for f in files:
        df = pd.read_csv(f)
        if KEEP_ONLY_LAST_EPOCH_PER_SAMPLE and "epoch" in df.columns:
            df = df.loc[df.groupby("sample_idx")["epoch"].idxmax()]
        elif KEEP_ONLY_FIRST_EPOCH_PER_SAMPLE and "epoch" in df.columns:
            df = df.loc[df.groupby("sample_idx")["epoch"].idxmin()]
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)

    param_cols = [c for c in full.columns if c not in NON_PARAM_COLS]
    print(f"\nDetected {len(param_cols)} parameter columns: {param_cols}")

    # group by (rounded) parameter vector to average delta_rho across seeds
    round_cols = [f"_r_{c}" for c in param_cols]
    for c, rc in zip(param_cols, round_cols):
        full[rc] = full[c].round(6)

    agg = (
        full.groupby(round_cols, as_index=False)
        .agg({**{c: "first" for c in param_cols}, ERROR_COL: ["mean", "std", "count"]})
    )
    # flatten MultiIndex columns: keep plain param names as-is (agg func "first"
    # collapses to a single column per param), only the ERROR_COL gets suffixed
    new_cols = []
    for c in agg.columns:
        if isinstance(c, tuple):
            base, func = c
            new_cols.append(base if func in ("first", "") else f"{base}_{func}")
        else:
            new_cols.append(c)
    agg.columns = new_cols
    agg = agg.drop(columns=round_cols)
    agg = agg.rename(columns={f"{ERROR_COL}_mean": ERROR_COL, f"{ERROR_COL}_std": f"{ERROR_COL}_std_across_seeds",
                               f"{ERROR_COL}_count": "n_seeds"})
    print(f"\n{len(full)} raw rows -> {len(agg)} unique samples after averaging across seeds.")
    return agg, param_cols


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


def plot_marginal_bins(df, param_cols, outdir):
    grand_mean = df[ERROR_COL].mean()
    for name in param_cols:
        vals = df[name].values
        edges = np.linspace(vals.min(), vals.max(), N_BINS + 1)
        bin_idx = np.clip(np.digitize(vals, edges[1:-1]), 0, N_BINS - 1)

        means, counts = [], []
        for b in range(N_BINS):
            mask = bin_idx == b
            counts.append(mask.sum())
            means.append(df[ERROR_COL].values[mask].mean() if mask.sum() > 0 else np.nan)

        centers = 0.5 * (edges[:-1] + edges[1:])
        fig, ax = plt.subplots(figsize=(6, 3.5))
        bars = ax.bar(centers, means, width=np.diff(edges) * 0.9, color="#55A868", edgecolor="white")
        ax.axhline(grand_mean, color="#C44E52", ls="--", label="overall mean error")
        for c, m, n in zip(centers, means, counts):
            if not np.isnan(m):
                ax.text(c, m, f"n={n}", ha="center", va="bottom", fontsize=7)
        ax.set_title(f"mean {ERROR_COL} by bin -- {name}")
        ax.set_xlabel(name)
        ax.set_ylabel(ERROR_COL)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(Path(outdir) / f"binned_error_{name}.png", dpi=130)
        plt.close(fig)


def interaction_score_and_plot(df, pA, pB, outdir):
    """
    2D bin the pair (pA, pB), compute observed cell means, compare to the
    additive prediction (row_effect + col_effect - grand_mean). Returns a
    normalized interaction score = std(residual) / std(observed cell means)
    and saves a heatmap.
    """
    n_bins_2d = max(3, N_BINS // 2)  # coarser bins in 2D to keep cell counts reasonable
    xa = df[pA].values
    xb = df[pB].values
    y = df[ERROR_COL].values

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
            if mask.sum() > 0:
                cell_mean[i, j] = y[mask].mean()

    grand = np.nanmean(cell_mean)
    row_eff = np.nanmean(cell_mean, axis=1) - grand
    col_eff = np.nanmean(cell_mean, axis=0) - grand
    additive = grand + row_eff[:, None] + col_eff[None, :]
    residual = cell_mean - additive

    valid = ~np.isnan(cell_mean)
    obs_std = np.nanstd(cell_mean[valid])
    resid_std = np.nanstd(residual[valid])
    score = resid_std / obs_std if obs_std > 0 else np.nan

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].imshow(cell_mean, origin="lower", aspect="auto", cmap="viridis")
    axes[0].set_title(f"observed mean {ERROR_COL}")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(residual, origin="lower", aspect="auto", cmap="coolwarm")
    axes[1].set_title(f"residual vs additive model\n(interaction score={score:.2f})")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    for ax, edges, axis_name, other_edges, other_name in [
        (axes[0], edges_a, pA, edges_b, pB), (axes[1], edges_a, pA, edges_b, pB)
    ]:
        ax.set_xlabel(pB)
        ax.set_ylabel(pA)
    fig.suptitle(f"{pA}  x  {pB}")
    fig.tight_layout()
    fig.savefig(Path(outdir) / f"interaction_{pA}_{pB}.png", dpi=130)
    plt.close(fig)

    return score, cell_count


def suggested_bin_weights(df, param_cols, outdir):
    rows = []
    for name in param_cols:
        vals = df[name].values
        edges = np.linspace(vals.min(), vals.max(), N_BINS + 1)
        bin_idx = np.clip(np.digitize(vals, edges[1:-1]), 0, N_BINS - 1)
        means = []
        for b in range(N_BINS):
            mask = bin_idx == b
            m = df[ERROR_COL].values[mask].mean() if mask.sum() > 0 else 0.0
            means.append(max(m, 0.0))
        means = np.array(means)
        # Neyman-like allocation: weight proportional to sqrt(mean error);
        # this is a common stratified-sampling heuristic (allocate more to
        # strata with higher variance/magnitude of the quantity of interest).
        w = np.sqrt(means)
        w = w / w.sum() if w.sum() > 0 else np.ones(N_BINS) / N_BINS
        for b in range(N_BINS):
            rows.append({
                "parameter": name, "bin": b,
                "lo": edges[b], "hi": edges[b + 1],
                "mean_error_in_bin": means[b],
                "suggested_weight": w[b],
            })
    table = pd.DataFrame(rows)
    table.to_csv(Path(outdir) / "suggested_bin_weights.csv", index=False)
    return table


def main():
    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)

    df, param_cols = load_and_aggregate()

    table, rf = importance_table(df, param_cols)
    print("\n=== Parameter importance for explaining", ERROR_COL, "===")
    print(table.to_string(index=False))
    table.to_csv(outdir / "importance_table.csv", index=False)

    plot_marginal_bins(df, param_cols, outdir)

    top_params = table["parameter"].tolist()[:TOP_K_FOR_INTERACTIONS]
    print(f"\n=== Pairwise interaction scores among top {TOP_K_FOR_INTERACTIONS}: {top_params} ===")
    for i in range(len(top_params)):
        for j in range(i + 1, len(top_params)):
            pA, pB = top_params[i], top_params[j]
            score, counts = interaction_score_and_plot(df, pA, pB, outdir)
            print(f"  {pA:35s} x {pB:35s}  interaction_score={score:.2f}  "
                  f"(min cell n={counts.min()}, watch out if this is small)")

    weights = suggested_bin_weights(df, param_cols, outdir)

    print(f"\nAll outputs saved to {outdir}/")
    print("  importance_table.csv")
    print("  binned_error_<param>.png  (one per parameter)")
    print("  interaction_<A>_<B>.png   (one per top pair)")
    print("  suggested_bin_weights.csv")


if __name__ == "__main__":
    main()
