import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

ROOT_DIR = "../LOGS/train_500_seeds"
CSV_NAME = "split_log.csv"

OUT_PNG = os.path.join(ROOT_DIR, "keff_split_kde_overlay.png")
OUT_PDF = os.path.join(ROOT_DIR, "keff_split_kde_overlay.pdf")

COL_SAMPLE = "sample_idx"
COL_SPLIT = "split"
COL_KEFF = "keff"

EXPECTED_SPLITS = {"train", "val", "test"}
STRICT_SHARED_VAL_TEST = True
TOL = 1e-12

VAL_COLOR = "royalblue"
TEST_COLOR = "red"

TRAIN_ALPHA_LINE = 0.45
TRAIN_ALPHA_FILL = 0.06
VAL_ALPHA_FILL = 0.20
TEST_ALPHA_FILL = 0.20

LINEWIDTH_TRAIN = 1.6
LINEWIDTH_MAIN = 2.2

GRID_SIZE = 500
BW_METHOD = None
BW_ADJUST = 0.9  # 0.7 gives sharper curve, while 1.1 gives smoother curve

def find_split_logs(root_dir, csv_name):
    pattern = os.path.join(root_dir, "**", csv_name)
    files = sorted(glob.glob(pattern, recursive=True))
    return [f for f in files if os.path.isfile(f)]

def validate_columns(df, path):
    required = {COL_SAMPLE, COL_SPLIT, COL_KEFF}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing columns {sorted(missing)}; found {list(df.columns)}"
        )

def validate_splits(df, path):
    found = set(df[COL_SPLIT].astype(str).str.strip().unique())
    missing = EXPECTED_SPLITS - found
    if missing:
        raise ValueError(
            f"{path} is missing split labels {sorted(missing)}; found {sorted(found)}"
        )

def canonical_split(df, split_name):
    sub = df[df[COL_SPLIT] == split_name][[COL_SAMPLE, COL_KEFF]].copy()
    sub = sub.sort_values(by=[COL_SAMPLE, COL_KEFF]).reset_index(drop=True)
    return sub

def compare_shared_split(ref_df, cur_df, split_name, ref_path, cur_path):
    ref = canonical_split(ref_df, split_name)
    cur = canonical_split(cur_df, split_name)

    if len(ref) != len(cur):
        return False, f"{split_name}: size mismatch between {ref_path} and {cur_path}"

    same_idx = np.array_equal(
        ref[COL_SAMPLE].to_numpy(),
        cur[COL_SAMPLE].to_numpy()
    )
    same_keff = np.allclose(
        ref[COL_KEFF].to_numpy(),
        cur[COL_KEFF].to_numpy(),
        rtol=0.0,
        atol=TOL
    )

    if not same_idx or not same_keff:
        return False, f"{split_name}: content mismatch between {ref_path} and {cur_path}"
    return True, None

def kde_curve(values, xgrid, bw_method=None, bw_adjust=1.0):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        raise ValueError("Need at least 2 values for KDE.")
    kde = gaussian_kde(values, bw_method=bw_method)
    kde.set_bandwidth(kde.factor * bw_adjust)
    return kde(xgrid)

csv_files = find_split_logs(ROOT_DIR, CSV_NAME)
if not csv_files:
    raise FileNotFoundError(f"No {CSV_NAME} files found under {ROOT_DIR}")

experiment_data = []
for path in csv_files:
    df = pd.read_csv(path)
    validate_columns(df, path)
    df[COL_SPLIT] = df[COL_SPLIT].astype(str).str.strip()
    validate_splits(df, path)
    df[COL_SAMPLE] = pd.to_numeric(df[COL_SAMPLE], errors="raise").astype(int)
    df[COL_KEFF] = pd.to_numeric(df[COL_KEFF], errors="raise")

    exp_name = os.path.basename(os.path.dirname(path))
    experiment_data.append((exp_name, path, df))

print(f"Found {len(experiment_data)} experiments.")

ref_name, ref_path, ref_df = experiment_data[0]
mismatches = []
for exp_name, path, df in experiment_data[1:]:
    ok_val, msg_val = compare_shared_split(ref_df, df, "val", ref_path, path)
    ok_test, msg_test = compare_shared_split(ref_df, df, "test", ref_path, path)
    if not ok_val:
        mismatches.append(msg_val)
    if not ok_test:
        mismatches.append(msg_test)

if mismatches:
    print("\nShared validation/test consistency check FAILED:\n")
    for msg in mismatches:
        print(msg)
    if STRICT_SHARED_VAL_TEST:
        raise ValueError(
            "Validation/test splits differ across experiments. "
            "Fix the split generation or disable STRICT_SHARED_VAL_TEST."
        )
else:
    print("Validation and test splits are identical across experiments.")

all_keff = np.concatenate([df[COL_KEFF].to_numpy() for _, _, df in experiment_data])
xmin, xmax = all_keff.min(), all_keff.max()
pad = 0.03 * (xmax - xmin)
xgrid = np.linspace(xmin - pad, xmax + pad, GRID_SIZE)

fig, ax = plt.subplots(figsize=(10, 6))

train_colors = plt.cm.tab10(np.linspace(0, 1, max(len(experiment_data), 1)))

for i, (exp_name, path, df) in enumerate(experiment_data):
    train_vals = df.loc[df[COL_SPLIT] == "train", COL_KEFF].to_numpy()
    y = kde_curve(train_vals, xgrid, bw_method=BW_METHOD, bw_adjust=BW_ADJUST)

    ax.plot(
        xgrid, y,
        color=train_colors[i],
        lw=LINEWIDTH_TRAIN,
        alpha=TRAIN_ALPHA_LINE,
        label=f"train - {exp_name}"
    )
    ax.fill_between(
        xgrid, 0, y,
        color=train_colors[i],
        alpha=TRAIN_ALPHA_FILL
    )

val_vals = ref_df.loc[ref_df[COL_SPLIT] == "val", COL_KEFF].to_numpy()
test_vals = ref_df.loc[ref_df[COL_SPLIT] == "test", COL_KEFF].to_numpy()

y_val = kde_curve(val_vals, xgrid, bw_method=BW_METHOD, bw_adjust=BW_ADJUST)
y_test = kde_curve(test_vals, xgrid, bw_method=BW_METHOD, bw_adjust=BW_ADJUST)

ax.plot(xgrid, y_val, color=VAL_COLOR, lw=LINEWIDTH_MAIN, label="val (shared)")
ax.fill_between(xgrid, 0, y_val, color=VAL_COLOR, alpha=VAL_ALPHA_FILL)

ax.plot(xgrid, y_test, color=TEST_COLOR, lw=LINEWIDTH_MAIN, label="test (shared)")
ax.fill_between(xgrid, 0, y_test, color=TEST_COLOR, alpha=TEST_ALPHA_FILL)

ax.set_title("keff distribution by split across experiments")
ax.set_xlabel("keff")
ax.set_ylabel("Density")
ax.grid(True, alpha=0.25)
ax.legend(fontsize=9, ncol=2)
fig.tight_layout()

fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved: {OUT_PNG}")
