import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path


project_name = "v1_baseline"
csv_path_val = f"../LOGS/{project_name}/keff_epoch_log_val.csv"
rhodiff_path_val = f"../LOGS/{project_name}/metrics_plot/keff_pcm_evolution_val.png"
hist_path_val = f"../LOGS/{project_name}/metrics_plot/loss_histogram_val.png"
scatt_path_val = f"../LOGS/{project_name}/metrics_plot/keff_scatter_val.png"

csv_path_train = f"../LOGS/{project_name}/keff_epoch_log_train.csv"
rhodiff_path_train = f"../LOGS/{project_name}/metrics_plot/keff_pcm_evolution_train.png"
hist_path_train = f"../LOGS/{project_name}/metrics_plot/loss_histogram_train.png"
scatt_path_train = f"../LOGS/{project_name}/metrics_plot/keff_scatter_train.png"

def plot_keff_pcm(log_path, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(log_path)
    epoch_stats = df.groupby("epoch")["delta_rho_pcm"].agg(["mean","min","max"]).reset_index()
    samples = sorted(df["sample_idx"].unique())
    kref_map = {s: df[df["sample_idx"]==s]["keff_openmc"].iloc[0] for s in samples}
    print(f"Samples number: {len(samples)}")
    plt.style.use("default")
    colors = ["#4C9EFF","#FF6B6B","#6BCB77","#FFD166","#C77DFF"]

    n_legend_rows = int(np.ceil(len(samples) / 10))   # 10 columns in legend
    legend_row_height = 0.22                           # inches per legend row
    legend_height = n_legend_rows * legend_row_height + 0.4  # total legend area in inches

    plot_w   = 18.0    # fixed plot width  (inches)
    plot_h   = 7.0     # fixed plot height (inches) — NEVER changes
    fig_h    = plot_h + legend_height   # figure grows downward

    fig = plt.figure(figsize=(plot_w, fig_h))

    # Place the axes at the top, with a fixed height in figure-fraction units
    top_margin   = 0.08               # fraction of fig height for title
    plot_frac    = plot_h / fig_h     # fraction the plot occupies

    ax = fig.add_axes([0.06,                         # left
                    legend_height / fig_h + 0.05, # bottom — above the legend zone
                    0.91,                          # width
                    plot_frac - top_margin])       # height

    cmap = plt.cm.get_cmap("tab10", len(samples))  # or "hsv", "Set1", "rainbow"
    for i, s in enumerate(samples):
        color = cmap(i)
        sub = df[df["sample_idx"]==s].sort_values("epoch")
        ax.plot(sub["epoch"], sub["delta_rho_pcm"], color=color, linewidth=1.8,
                marker="o", markersize=4, alpha=0.75, label=f"S{s}  k={kref_map[s]:.3f}")

    """ ax.fill_between(epoch_stats["epoch"], epoch_stats["min"], epoch_stats["max"],
                    color="white", alpha=0.06, label="Min-Max band") """
    final_mean_rho = epoch_stats["mean"].iloc[-1]   # value at the last epoch
    ax.plot(epoch_stats["epoch"], epoch_stats["mean"], color="black", linewidth=2.8,
            linestyle="--", marker="D", markersize=7, label=f"Mean (final: {final_mean_rho:.1f} pcm)")
    # ── β_eff reference line ───────────────────────────────────────────────
    beta_eff_pcm = 650   # pcm — typical U-235 LWR value; replace with your own!
    ax.axhline(y=beta_eff_pcm, color="#00008B", linewidth=1.8,
            linestyle=(0, (5, 3)),   # long-dash pattern
            label=f"β_eff = {beta_eff_pcm} pcm")
    ax.text(x=ax.get_xlim()[1], y=beta_eff_pcm + 60,
            s=f"β_eff = {beta_eff_pcm} pcm",
            color="#00008B", fontsize=9, ha="right")
    ax.set_xlabel("Epoch", fontsize=16)
    ax.set_ylabel("Delta-rho (pcm)", fontsize=16)
    ax.set_title(f"Reactivity Error (pcm) — PEDS vs. OpenMC - {project_name}", fontsize=18)
    all_epochs = sorted(df["epoch"].unique())
    ax.xaxis.set_major_locator(MaxNLocator(nbins=12, integer=True))
    ax.grid(True, alpha=0.15)
    #ax.set_yscale("log")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),  # centered, below the x-axis
        borderaxespad=0,
        fontsize=7,
        framealpha=0.3,
        ncol=10                        # 10 columns so it stays compact horizontally
    )
    plt.tight_layout()  # make sure this is called AFTER the legend    
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {save_path}")

def plot_loss_histogram(
    log_path,
    save_path="../LOGS/loss_histogram_final_epoch.png",
    n_bins=20
):
    df = pd.read_csv(log_path)

    # Snapshot at the last epoch
    final_epoch = df["epoch"].max()
    df_final = df[df["epoch"] == final_epoch].copy()

    # Use absolute value of delta_rho_pcm so bins go left (small error) to right (large)
    df_final["abs_delta_rho"] = df_final["delta_rho_pcm"].abs()

    # Color map: blue=low keff, red=high keff
    keff_vals = df_final["keff_openmc"].values
    norm = plt.Normalize(vmin=keff_vals.min(), vmax=keff_vals.max())
    cmap = plt.cm.get_cmap("coolwarm")

    # Build bins manually so we can color each sample individually
    bin_edges = np.linspace(df_final["abs_delta_rho"].min(),
                            df_final["abs_delta_rho"].max(), n_bins + 1)
    df_final["bin"] = pd.cut(df_final["abs_delta_rho"], bins=bin_edges, include_lowest=True)

    fig, ax = plt.subplots(figsize=(11, 5.5))

    # For each bin, stack one rectangle per sample inside it
    for bin_interval, group in df_final.groupby("bin", observed=True):
        # Sort by keff so colors are ordered nicely within the stack
        group = group.sort_values("keff_openmc")
        x_center = (bin_interval.left + bin_interval.right) / 2
        bar_width = (bin_edges[1] - bin_edges[0]) * 0.85

        for stack_pos, (_, row) in enumerate(group.iterrows()):
            color = cmap(norm(row["keff_openmc"]))
            ax.bar(x_center, 1, bottom=stack_pos,
                   width=bar_width, color=color,
                   edgecolor="white", linewidth=0.4, alpha=0.88)
            # Label sample index inside the bar if bar is tall enough
            ax.text(x_center, stack_pos + 0.5, f"S{int(row['sample_idx'])}",
                    ha="center", va="center", fontsize=5.5, color="white", fontweight="bold")

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label("k_eff (OpenMC)", fontsize=11)

    # β_eff reference line (convert to pcm on x axis)
    beta_eff_pcm = 650
    ax.axvline(x=beta_eff_pcm, color="#00008B", linewidth=1.8,
               linestyle=(0, (5, 3)), label=f"β_eff = {beta_eff_pcm} pcm")

    ax.set_xlabel("|Δρ| at Final Epoch (pcm)", fontsize=12)
    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.set_title(f"Distribution of Reactivity Error at Final Epoch ({final_epoch})", fontsize=13)
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.15, axis="x")
    ax.legend(fontsize=10, framealpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Histogram saved to {save_path}")


def plot_keff_scatter(
    log_path,
    save_path="../LOGS/keff_scatter_final_epoch.png",
    beta_eff_pcm=650
):
    df = pd.read_csv(log_path)

    # Snapshot at the last epoch
    final_epoch = df["epoch"].max()
    df_final = df[df["epoch"] == final_epoch].copy()
    df_final["abs_delta_rho"] = df_final["delta_rho_pcm"].abs()

    # Color: blue = below beta_eff threshold, red = above
    colors = ["#4C9EFF" if v <= beta_eff_pcm else "#FF4444"
              for v in df_final["abs_delta_rho"]]

    fig, ax = plt.subplots(figsize=(7, 5.5))

    ax.scatter(
        df_final["keff_openmc"],
        df_final["abs_delta_rho"],
        c=colors,
        s=80,           # dot size
        alpha=0.80,
        edgecolors="white",
        linewidths=0.5
    )

    # β_eff reference line
    ax.axhline(y=beta_eff_pcm, color="#CC0000", linewidth=1.8,
               linestyle="--", label=f"β_eff = {beta_eff_pcm} pcm")

    # Optional: label outliers with their sample index
    threshold = df_final["abs_delta_rho"].quantile(0.90)  # top 10% get labeled
    for _, row in df_final[df_final["abs_delta_rho"] >= threshold].iterrows():
        ax.annotate(f"S{int(row['sample_idx'])}",
                    xy=(row["keff_openmc"], row["abs_delta_rho"]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=7.5, color="#333333")

    ax.set_xlabel("k_eff (OpenMC)", fontsize=12)
    ax.set_ylabel("|Δρ| (pcm)", fontsize=12)
    ax.set_title(f"Final Epoch Error vs. k_eff  [epoch {final_epoch}]", fontsize=13)
    ax.grid(True, alpha=0.15)
    ax.legend(fontsize=10, framealpha=0.3)

    # Custom legend patches for the color meaning
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4C9EFF", edgecolor="white", label=f"|Δρ| ≤ β_eff ({beta_eff_pcm} pcm)"),
        Patch(facecolor="#FF4444", edgecolor="white", label=f"|Δρ| > β_eff ({beta_eff_pcm} pcm)"),
        plt.Line2D([0], [0], color="#CC0000", linewidth=1.8,
                   linestyle="--", label=f"β_eff = {beta_eff_pcm} pcm")
    ]
    ax.legend(handles=legend_elements, fontsize=9.5, framealpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Scatter plot saved to {save_path}")

def plot_loss_from_csv(
    log_path ,
    save_path = "../LOGS/loss_evolution.png"
):

    df = pd.read_csv(log_path)
    df["sq_error"] = (df["keff_peds"] - df["keff_openmc"]) ** 2

    loss_stats = df.groupby("epoch")["sq_error"].agg(
        sum_mse  = "sum",
        mean_mse = "mean"
    ).reset_index()

    # ── Compact but publication-readable figure ────────────────────────
    fig, ax = plt.subplots(figsize=(5.5, 3.8))           # small canvas

    ax.plot(loss_stats["epoch"], loss_stats["sum_mse"],
            color="#4C9EFF", linewidth=2.2, marker="o",
            markersize=7, label="Sum MSE (val)")
    ax.plot(loss_stats["epoch"], loss_stats["mean_mse"],
            color="#FF6B6B", linewidth=2.2, marker="D",
            markersize=7, linestyle="--", label="Mean MSE (val)")

    # ── Axes labels — big for publications ────────────────────────────
    ax.set_xlabel("Epoch",    fontsize=15, labelpad=6)
    ax.set_ylabel("MSE Loss", fontsize=15, labelpad=6)
    ax.set_title("Validation Loss — PEDS vs. OpenMC",
                 fontsize=14, pad=8)

    # ── Ticks — one per epoch, large font ─────────────────────────────
    ax.set_xticks(loss_stats["epoch"])                   # exactly 1–N, no gaps
    ax.tick_params(axis="both", labelsize=14)

    ax.set_yscale("log")
    ax.grid(True, alpha=0.15)

    ax.legend(fontsize=12, framealpha=0.3,
              loc="upper right")

    plt.tight_layout(pad=0.8)                            # minimal outer padding
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Loss plot saved to {save_path}")

#==========================================
#                MAIN 
#==========================================
if __name__ == "__main__":
    print("Generating keff evolution plot...")
    plot_keff_pcm(csv_path_val, rhodiff_path_val)
    #plot_loss_from_csv(log_path)
    plot_loss_histogram(csv_path_val, hist_path_val)
    plot_keff_scatter(csv_path_val, scatt_path_val)

    plot_keff_pcm(csv_path_train, rhodiff_path_train)
    #plot_loss_from_csv(log_path)
    plot_loss_histogram(csv_path_train, hist_path_train)
    plot_keff_scatter(csv_path_train, scatt_path_train)
