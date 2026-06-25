"""
plot_peds_results.py
====================
Visualise PEDS training/validation results from the keff epoch log,
with 6-parameter backtrace from the full LHS dataset.

Usage
-----
    python plot_peds_results.py

Expects in the same folder (or adjust paths below):
    - keff_epoch_log.csv        (or keff_epoch_log_val.csv / _train.csv)
    - LHS_full_dataset.csv

Outputs (saved next to the script):
    1. peds_loss_curve.png          — train/val MSE loss over epochs
    2. peds_keff_scatter_final.png  — keff_peds vs keff_openmc at last epoch
    3. peds_delta_rho_evolution.png — per-sample Δρ (pcm) over epochs
    4. peds_parallel_coords.png     — parallel coordinates of 6 params, coloured by Δρ at last epoch
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# 0.  CONFIGURATION  — adjust paths here
# ──────────────────────────────────────────────

project_name = "v10_wloss"
LOG_PATH = f"../LOGS/{project_name}/keff_epoch_log_val.csv"

FULL_PATH = "../FILES/1000_clean.csv"
#PARALLEL_PATH = f"../LOGS/{project_name}/metrics_plot/peds_parallel_coords.png"
#SCATTER_PATH = f"../LOGS/{project_name}/metrics_plot/peds_keff_scatter_final.png"
MATCH_TOL = 1e-5                        # keff tolerance for backtrace matching

PARAMS = [
    "r0_b4c_rod_outer_radius",
    "r0_b4c_rod_cr_fraction",
    "r1_fuel_annulus_outer_radius",
    "r1_fuel_annulus_enrichment",
    "r1_fuel_annulus_f_mod",
    "r2_water_outer_radius",
]
PARAM_LABELS = [
    "B4C outer\nradius",
    "CR\nfraction",
    "Fuel annulus\nouter radius",
    "Fuel\nenrichment",
    "Fuel f_mod",
    "Water outer\nradius",
]

# ──────────────────────────────────────────────
# 1.  LOAD & BACKTRACE
# ──────────────────────────────────────────────

def load_and_merge(log_path, full_path, tol=MATCH_TOL):
    df_log  = pd.read_csv(log_path)
    df_full = pd.read_csv(full_path)

    # Make sure numeric types are correct
    df_log["keff_openmc"]    = df_log["keff_openmc"].astype(float)
    df_log["keff_peds"]      = df_log["keff_peds"].astype(float)
    df_log["delta_rho_pcm"]  = df_log["delta_rho_pcm"].astype(float)

    # ── If param columns are already in the log, skip matching ──────────
    if PARAMS[0] in df_log.columns:
        print("Param columns already present in log — skipping backtrace.")
        return df_log

    # ── Backtrace via keff matching ──────────────────────────────────────
    log_unique = df_log.drop_duplicates("sample_idx")[["sample_idx", "keff_openmc"]]
    matched, unmatched = [], []

    for _, row in log_unique.iterrows():
        diffs    = abs(df_full["keff"] - row["keff_openmc"])
        best_idx = diffs.idxmin()
        best_diff = float(diffs.min())
        if best_diff > tol:
            unmatched.append((int(row["sample_idx"]), row["keff_openmc"], best_diff))
            continue
        params = df_full.loc[best_idx, PARAMS].to_dict()
        matched.append({
            "sample_idx": int(row["sample_idx"]),
            **params,
        })

    if unmatched:
        print(f"\n  ⚠  {len(unmatched)} sample(s) could not be matched "
              f"(keff not found within tol={tol}):")
        for sidx, kval, diff in unmatched:
            print(f"     sample_idx={sidx}  keff_openmc={kval:.6f}  "
                  f"closest diff={diff:.2e}")
        print("  These rows will have NaN for the 6 parameters.\n")

    df_lookup  = pd.DataFrame(matched) if matched else pd.DataFrame(columns=["sample_idx"] + PARAMS)
    df_enriched = df_log.merge(df_lookup, on="sample_idx", how="left")

    n_ok = df_enriched.drop_duplicates("sample_idx")[PARAMS[0]].notna().sum()
    n_tot = df_enriched["sample_idx"].nunique()
    print(f"  Matched {n_ok}/{n_tot} samples to LHS parameters.")
    return df_enriched


# ──────────────────────────────────────────────
# 2.  PLOT 1 — Loss curves
# ──────────────────────────────────────────────

def plot_loss_curves(df, save_path="peds_loss_curve.png"):
    # Aggregate per epoch (one loss value per epoch)
    ep = df.groupby("epoch").agg(
        train_loss=("epoch_train_loss", "first"),
        val_loss  =("epoch_val_loss",   "first"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ep["epoch"], ep["train_loss"], "o-", color="#2563EB", lw=1.8,
            ms=4, label="Train MSE")
    ax.plot(ep["epoch"], ep["val_loss"],   "s--", color="#DC2626", lw=1.8,
            ms=4, label="Val MSE")

    ax.set_yscale("log")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("MSE loss (log scale)", fontsize=11)
    ax.set_title("PEDS — Training & Validation Loss", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────
# 3.  PLOT 2 — keff scatter at final epoch
# ──────────────────────────────────────────────

def plot_keff_scatter(df, epoch = None, save_path=None, vmin=None, vmax=None):
    if epoch is None:
        last_epoch = df["epoch"].max()
    df_fin = df[df["epoch"] == epoch].copy()

    if save_path is None:
        save_path = f"../LOGS/{project_name}/metrics_plot/scatter_epoch{epoch}.png"

    fig, ax = plt.subplots(figsize=(6, 6))

    sc = ax.scatter(
        df_fin["keff_openmc"], df_fin["keff_peds"],
        c=df_fin["delta_rho_pcm"], cmap="RdYlGn_r",
        s=50, edgecolors="k", linewidths=0.4, zorder=3,
        vmin=vmin, vmax=vmax,    
    )
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Δρ (pcm)", fontsize=10)

    # Perfect prediction line
    lims = [
        min(df_fin["keff_openmc"].min(), df_fin["keff_peds"].min()) * 0.995,
        max(df_fin["keff_openmc"].max(), df_fin["keff_peds"].max()) * 1.005,
    ]
    ax.plot(lims, lims, "k--", lw=1, label="y = x")
    ax.set_xlim(lims); ax.set_ylim(lims)

    # Annotate each point with sample index
    """ for _, row in df_fin.iterrows():
        ax.annotate(
            f"s{int(row['sample_idx'])}",
            (row["keff_openmc"], row["keff_peds"]),
            textcoords="offset points", xytext=(5, 4),
            fontsize=7, color="#374151",
        ) """

    ax.set_xlabel("k$_{eff}$ — OpenMC", fontsize=11)
    ax.set_ylabel("k$_{eff}$ — PEDS",   fontsize=11)
    ax.set_title(f"keff Prediction at Epoch {epoch}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────
# 4.  PLOT 3 — Δρ evolution per sample over epochs
# ──────────────────────────────────────────────

def plot_delta_rho_evolution(df, save_path="peds_delta_rho_evolution.png"):
    sample_ids = sorted(df["sample_idx"].unique())
    cmap_s = cm.get_cmap("tab10", len(sample_ids))

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, sid in enumerate(sample_ids):
        sub = df[df["sample_idx"] == sid].sort_values("epoch")
        ax.plot(sub["epoch"], sub["delta_rho_pcm"],
                "o-", color=cmap_s(i), lw=1.6, ms=4, label=f"Sample {int(sid)}")

    # Target line
    ax.axhline(650, color="gray", linestyle="--", lw=1, label="650 pcm target")

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Δρ (pcm)", fontsize=11)
    ax.set_title("Per-sample Δρ Evolution over Training", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────
# 5.  PLOT 4 — Parallel coordinates, coloured by Δρ at final epoch
# ──────────────────────────────────────────────

def plot_parallel_coords(df, epoch= None, save_path=None):
    if epoch is None:    
        last_epoch = df["epoch"].max()
    df_fin = df[df["epoch"] == epoch].copy()

    if save_path is None:
        save_path = f"../LOGS/{project_name}/metrics_plot/parallel_coords_epoch{epoch}.png"

    # Only rows where all 6 params are known
    df_plot = df_fin.dropna(subset=PARAMS).copy()
    if df_plot.empty:
        print("  ⚠  No matched samples for parallel coordinates plot — skipping.")
        return

    # Normalise each parameter to [0,1]
    norm_params = df_plot[PARAMS].copy()
    for col in PARAMS:
        mn, mx = norm_params[col].min(), norm_params[col].max()
        norm_params[col] = (norm_params[col] - mn) / (mx - mn + 1e-30)

    delta_rho_vals = df_plot["delta_rho_pcm"].values
    norm_dr        = (delta_rho_vals - delta_rho_vals.min()) / \
                     (delta_rho_vals.max() - delta_rho_vals.min() + 1e-30)
    base_colors = cm.RdYlGn_r(norm_dr)   # red = high error, green = low error

    fig, ax = plt.subplots(figsize=(11, 5))
    x_pos = np.arange(len(PARAMS))

    # threshold = upper half of Δρ range
    high_mask =  norm_dr > 0.3
    low_mask  = ~high_mask

    # 1) draw low-error lines first, faint
    for i, (_, row) in enumerate(norm_params.iterrows()):
        if low_mask[i]:
            y_vals = row[PARAMS].values.astype(float)
            color = base_colors[i].copy()
            color[-1] = 0.4   # more transparent
            ax.plot(x_pos, y_vals, color=color, linewidth=1.5, zorder=1)

    # 2) draw high-error lines second, bold
    for i, (_, row) in enumerate(norm_params.iterrows()):
        if high_mask[i]:
            y_vals = row[PARAMS].values.astype(float)
            color = base_colors[i].copy()
            color[-1] = 0.9   # almost opaque
            ax.plot(x_pos, y_vals, color=color, linewidth=2.5, zorder=3)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(PARAM_LABELS, fontsize=9)
    ax.set_ylabel("Normalised parameter value", fontsize=10)
    ax.set_title(
        f"Parallel Coordinates — 6 parameters coloured by Δρ at epoch {epoch}",
        fontsize=12, fontweight="bold",
    )
    ax.set_xlim(-0.1, len(PARAMS) - 0.9)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    # Colorbar
    sm   = cm.ScalarMappable(cmap="RdYlGn_r",
                              norm=mcolors.Normalize(vmin=delta_rho_vals.min(),
                                                     vmax=delta_rho_vals.max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Δρ (pcm)", fontsize=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────
# 6.  MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading and merging data …")
    df = load_and_merge(LOG_PATH, FULL_PATH)

    print("\nGenerating plots …")
    #plot_loss_curves(df)
    #plot_delta_rho_evolution(df)
    first_epoch = df["epoch"].min()   # typically 1
    last_epoch  = df["epoch"].max()

    # ✅ Compute shared colorbar range across BOTH epochs
    both_epochs = df[df["epoch"].isin([first_epoch, last_epoch])]
    global_vmin = both_epochs["delta_rho_pcm"].min()
    global_vmax = both_epochs["delta_rho_pcm"].max()

    plot_keff_scatter(df, epoch=first_epoch, vmin=global_vmin, vmax=global_vmax)
    plot_keff_scatter(df, epoch=last_epoch,  vmin=global_vmin, vmax=global_vmax)

    plot_parallel_coords(df, epoch=first_epoch)   # 🆕 epoch 1
    plot_parallel_coords(df, epoch=last_epoch) 

    print("\nDone. All plots saved.")
