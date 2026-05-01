import csv
import matplotlib.pyplot as plt

def plot_keff_pcm(log_path="./LOGS/keff_epoch_log.csv", save_path="./LOGS/keff_pcm_evolution.png"):
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

print("Generating keff evolution plot...")
plot_keff_pcm()