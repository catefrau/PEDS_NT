import csv
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt

log_path="../LOGS/good_keff_epoch_log.csv"

def plot_keff_pcm(log_path, save_path="../LOGS/keff_pcm_evolution.png"):
    import pandas as pd
    df = pd.read_csv(log_path)
    epoch_stats = df.groupby("epoch")["delta_rho_pcm"].agg(["mean","min","max"]).reset_index()
    samples = sorted(df["sample_idx"].unique())
    kref_map = {s: df[df["sample_idx"]==s]["keff_openmc"].iloc[0] for s in samples}

    plt.style.use("default")
    colors = ["#4C9EFF","#FF6B6B","#6BCB77","#FFD166","#C77DFF"]
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for s, c in zip(samples, colors):
        sub = df[df["sample_idx"]==s].sort_values("epoch")
        ax.plot(sub["epoch"], sub["delta_rho_pcm"], color=c, linewidth=1.8,
                marker="o", markersize=4, alpha=0.75, label=f"S{s}  k={kref_map[s]:.3f}")

    """ ax.fill_between(epoch_stats["epoch"], epoch_stats["min"], epoch_stats["max"],
                    color="white", alpha=0.06, label="Min-Max band") """
    ax.plot(epoch_stats["epoch"], epoch_stats["mean"], color="black",
            linewidth=2.8, linestyle="--", marker="D", markersize=7, label="Mean")
    # ── β_eff reference line ───────────────────────────────────────────────
    beta_eff_pcm = 650   # pcm — typical U-235 LWR value; replace with your own!
    ax.axhline(y=beta_eff_pcm, color="#FF4444", linewidth=1.8,
            linestyle=(0, (5, 3)),   # long-dash pattern
            label=f"β_eff = {beta_eff_pcm} pcm")
    ax.text(x=ax.get_xlim()[1], y=beta_eff_pcm + 60,
            s=f"β_eff = {beta_eff_pcm} pcm",
            color="#FF4444", fontsize=9, ha="right")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Delta-rho (pcm)", fontsize=12)
    ax.set_title("Reactivity Error (pcm) — PEDS vs. OpenMC", fontsize=13)
    ax.set_xticks(range(1, len(epoch_stats) + 1))
    ax.grid(True, alpha=0.15)
    ax.legend(loc="upper right", fontsize=9.5, framealpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {save_path}")


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
    ax.tick_params(axis="both", labelsize=13)

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

print("Generating keff evolution plot...")
plot_keff_pcm(log_path)
plot_loss_from_csv(log_path)
