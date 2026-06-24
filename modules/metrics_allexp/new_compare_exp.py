import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


# ─────────────────────────────────────────────
# CONFIGURATION – edit these as needed
# ─────────────────────────────────────────────
LOGS_DIR = "../LOGS"
METRIC = "train_mae_k"
SUMMARY_FILE = "epoch_metrics.csv"

OUTPUT_CSV = "combined_mae_train.csv"

VERSION_PLOTS_DIR = "version_plots_train"   # folder for v1, v2, ..., v10 figures
COMPARE_PLOT = "compare_train.png"
LAST_EPOCH_SUMMARY_CSV = "last_epoch_summary.csv"

# Put exact experiment folder names here when you want a custom comparison plot
SELECTED_EXPERIMENTS = [
    "v4_phifeat",
    "v5_biggestDS",
    "v6_ste",
    "v7_121",
    "v8_phifixed",
    "v9_huberloss",
    "v10_oldloss",
]
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
    combined = pd.DataFrame(collected)
    combined.index.name = "epoch"
    combined.index = combined.index + 1
    combined.to_csv(output_csv)
    print(f"\nCombined CSV saved → {output_csv}")
    return combined


def extract_version(experiment_name):
    """
    Extract version prefix like v1, v2, ..., v10 from experiment folder name.
    Assumes experiment names start with v<number>.
    """
    match = re.match(r"^(v\d+)", experiment_name)
    return match.group(1) if match else None


def plot_single_group(df_group, metric, title, output_path):
    """Plot one dataframe group and save."""
    n = len(df_group.columns)
    colors = cm.tab20(np.linspace(0, 1, max(n, 1)))

    fig, ax = plt.subplots(figsize=(12, 6))

    for col, color in zip(df_group.columns, colors):
        series = df_group[col].dropna()
        ax.plot(series.index, series.values, label=col, color=color, linewidth=1.5)

    ax.set_yscale("log")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(f"{metric} [log scale]", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(
        loc="upper right",
        fontsize=7,
        ncol=max(1, n // 20 + 1),
        framealpha=0.7,
    )
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.6)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved → {output_path}")


def plot_by_version(combined, metric, output_dir):
    """
    Create one figure per version prefix (v1, v2, ..., v10).
    """
    os.makedirs(output_dir, exist_ok=True)

    version_groups = {}

    for col in combined.columns:
        version = extract_version(col)
        if version is None:
            print(f"  [WARN] Could not extract version from '{col}', skipping group plot.")
            continue
        version_groups.setdefault(version, []).append(col)

    # Sort versions numerically: v1, v2, ..., v10
    sorted_versions = sorted(
        version_groups.keys(),
        key=lambda x: int(x[1:])
    )

    for version in sorted_versions:
        cols = version_groups[version]
        df_group = combined[cols]
        output_path = os.path.join(output_dir, f"{version}_{metric}.png")
        title = f"Experiments in {version} – {metric}"
        plot_single_group(df_group, metric, title, output_path)


def plot_selected_experiments(combined, selected_experiments, metric, output_plot):
    """
    Plot only the manually selected experiments for direct comparison.
    """
    if not selected_experiments:
        print("\nNo selected experiments provided. Skipping comparison plot.")
        return

    missing = [name for name in selected_experiments if name not in combined.columns]
    if missing:
        print("\n[WARNING] These selected experiments were not found:")
        for name in missing:
            print(f"  - {name}")

    valid = [name for name in selected_experiments if name in combined.columns]
    if not valid:
        print("\nNo valid selected experiments found. Skipping comparison plot.")
        return

    df_selected = combined[valid]
    title = f"Selected experiment comparison – {metric}"
    plot_single_group(df_selected, metric, title, output_plot)

def save_last_epoch_metrics(logs_dir, summary_file, output_csv):
    """
    Save one row per experiment with the metrics at the last available epoch.
    Handles experiments with different numbers of epochs.
    """
    rows = []

    subfolders = sorted([
        d for d in os.listdir(logs_dir)
        if os.path.isdir(os.path.join(logs_dir, d))
    ])

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

        needed_cols = ["val_mae_k", "train_mae_k", "val_mean_pcm"]
        available_cols = [c for c in needed_cols if c in df.columns]

        if not available_cols:
            print(f"  [SKIP] '{folder}' – none of the target columns found.")
            continue

        # Use only the columns that exist, then find the last row with at least one value
        temp = df[available_cols].copy()
        temp["last_epoch"] = df.index + 1

        valid_rows = temp.dropna(how="all", subset=available_cols)
        if valid_rows.empty:
            print(f"  [SKIP] '{folder}' – no valid metric values found.")
            continue

        last_row = valid_rows.iloc[-1]

        row = {
            "experiment": folder,
            "last_epoch": int(last_row["last_epoch"]),
            "val_mae_k": last_row["val_mae_k"] if "val_mae_k" in df.columns else np.nan,
            "train_mae_k": last_row["train_mae_k"] if "train_mae_k" in df.columns else np.nan,
            "val_mean_pcm": last_row["val_mean_pcm"] if "val_mean_pcm" in df.columns else np.nan,
        }
        rows.append(row)

        print(f"  [OK]   '{folder}' – last epoch = {row['last_epoch']}")

    if not rows:
        print("\nNo valid last-epoch data found. Nothing saved.")
        return None

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values("experiment").reset_index(drop=True)
    out_df.to_csv(output_csv, index=False)

    print(f"\nLast-epoch summary saved → {output_csv}")
    return out_df

# ── Main ──────────────────────────────────────
if __name__ == "__main__":
    print(f"Scanning '{LOGS_DIR}' for '{SUMMARY_FILE}' | metric: '{METRIC}'\n")

    collected = collect_metrics(LOGS_DIR, METRIC, SUMMARY_FILE)

    if not collected:
        print("\nNo valid experiments found. Nothing to do.")
    else:
        combined = save_combined_csv(collected, OUTPUT_CSV)

        # 0) Save last-epoch values for each experiment
        last_epoch_df = save_last_epoch_metrics(LOGS_DIR, SUMMARY_FILE, LAST_EPOCH_SUMMARY_CSV)

        # 1) Save one plot per version group
        #plot_by_version(combined, METRIC, VERSION_PLOTS_DIR)

        # 2) Save one plot for your manually selected experiments
        #plot_selected_experiments(combined, SELECTED_EXPERIMENTS, METRIC, COMPARE_PLOT)

        print(f"\nDone! {len(collected)} experiment(s) processed.")