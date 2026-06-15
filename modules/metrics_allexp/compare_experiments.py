import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# ─────────────────────────────────────────────
#  CONFIGURATION  –  edit these as needed
# ─────────────────────────────────────────────
LOGS_DIR        = "../LOGS"                  # folder that contains all experiment sub-folders
METRIC          = "train_mae_k"             # column to extract and compare
SUMMARY_FILE    = "epoch_metrics.csv"     # filename inside each experiment folder
OUTPUT_CSV      = "combined_metrics_train.csv"  # output combined CSV
OUTPUT_PLOT     = "comparison_plot_train.png"   # output plot image
# ─────────────────────────────────────────────

def collect_metrics(logs_dir, metric, summary_file):
    """Walk every sub-folder in logs_dir and collect the chosen metric column."""
    collected = {}

    subfolders = sorted([
        d for d in os.listdir(logs_dir)
        if os.path.isdir(os.path.join(logs_dir, d))
    ])

    if not subfolders:
        raise FileNotFoundError(f"No sub-folders found in '{logs_dir}'.")

    for folder in subfolders:
        csv_path = os.path.join(logs_dir, folder, summary_file)
        if not os.path.isfile(csv_path):
            print(f"  [SKIP] '{folder}' – '{summary_file}' not found.")
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  [SKIP] '{folder}' – could not read CSV: {e}")
            continue

        if metric not in df.columns:
            print(f"  [SKIP] '{folder}' – column '{metric}' not found.")
            continue

        collected[folder] = df[metric].reset_index(drop=True)
        print(f"  [OK]   '{folder}' – {len(collected[folder])} epochs loaded.")

    return collected


def save_combined_csv(collected, output_csv):
    """Combine all series into one DataFrame (padded with NaN) and save."""
    combined = pd.DataFrame(collected)   # pandas aligns by index, pads with NaN automatically
    combined.index.name = "epoch"
    combined.index = combined.index + 1  # 1-based epoch numbering
    combined.to_csv(output_csv)
    print(f"\nCombined CSV saved → {output_csv}")
    return combined


def plot_metrics(combined, metric, output_plot):
    """Plot all experiments on a log-scale y-axis and save as PNG."""
    n = len(combined.columns)
    colors = cm.tab20(np.linspace(0, 1, max(n, 1)))

    fig, ax = plt.subplots(figsize=(12, 6))

    for (col, color) in zip(combined.columns, colors):
        series = combined[col].dropna()
        ax.plot(series.index, series.values, label=col, color=color, linewidth=1.5)

    ax.set_yscale("log")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(metric + "  [log scale]", fontsize=12)
    ax.set_title(f"Experiment comparison – {metric}", fontsize=14)
    ax.legend(
        loc="upper right",
        fontsize=7,
        ncol=max(1, n // 20 + 1),
        framealpha=0.7,
    )
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_plot, dpi=150)
    plt.close()
    print(f"Plot saved → {output_plot}")


# ── Main ──────────────────────────────────────
if __name__ == "__main__":
    print(f"Scanning '{LOGS_DIR}' for '{SUMMARY_FILE}'  |  metric: '{METRIC}'\n")

    collected = collect_metrics(LOGS_DIR, METRIC, SUMMARY_FILE)

    if not collected:
        print("\nNo valid experiments found. Nothing to do.")
    else:
        combined = save_combined_csv(collected, OUTPUT_CSV)
        plot_metrics(combined, METRIC, OUTPUT_PLOT)
        print(f"\nDone! {len(collected)} experiment(s) processed.")
