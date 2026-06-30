import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# USER SETTINGS
# =========================
ROOT_DIR = "../LOGS/train_500_seeds"
CSV_NAME = "split_log.csv"

OUT_PNG = os.path.join(ROOT_DIR, "keff_split_density_overlay.png")
OUT_PDF = os.path.join(ROOT_DIR, "keff_split_density_overlay.pdf")

NBINS = 35
STRICT_SHARED_VAL_TEST = True
TOL = 1e-12

# exact expected column names
COL_SAMPLE = "sample_idx"
COL_SPLIT = "split"
COL_KEFF = "keff"

EXPECTED_SPLITS = {"train", "val", "test"}

# colors
VAL_COLOR = "blue"
TEST_COLOR = "red"

# =========================
# HELPERS
# =========================
def find_split_logs(root_dir, csv_name):
    pattern = os.path.join(root_dir, "**", csv_name)
    files = sorted(glob.glob(pattern, recursive=True))
    return [f for f in files if os.path.isfile(f)]

def validate_columns(df, path):
    required = {COL_SAMPLE, COL_SPLIT, COL_KEFF}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing columns: {sorted(missing)}. "
            f"Found columns: {list(df.columns)}"
        )

def validate_splits(df, path):
    splits_in_file = set(df[COL_SPLIT].unique())
    missing = EXPECTED_SPLITS - splits_in_file
    if missing:
        raise ValueError(
            f"{path} is missing split labels: {sorted(missing)}. "
            f"Found: {sorted(splits_in_file)}"
        )

def canonical_split(df, split_name):
    sub = df[df[COL_SPLIT] == split_name][[COL_SAMPLE, COL_KEFF]].copy()
    sub = sub.sort_values(by=[COL_SAMPLE, COL_KEFF]).reset_index(drop=True)
    return sub

def compare_shared_split(ref_df, cur_df, split_name, ref_path, cur_path):
    ref = canonical_split(ref_df, split_name)
    cur = canonical_split(cur_df, split_name)

    if len(ref) != len(cur):
        return (
            False,
            f"{split_name}: size mismatch -> {ref_path} has {len(ref)}, "
            f"{cur_path} has {len(cur)}"
        )

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
        return (
            False,
            f"{split_name}: content mismatch between\n"
            f"  REF: {ref_path}\n"
            f"  CUR: {cur_path}"
        )

    return True, None

def density_hist_curve(values, bin_edges):
    hist, _ = np.histogram(values, bins=bin_edges, density=True)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return centers, hist

# =========================
# LOAD FILES
# =========================
csv_files = find_split_logs(ROOT_DIR, CSV_NAME)

if not csv_files:
    raise FileNotFoundError(f"No '{CSV_NAME}' files found under: {ROOT_DIR}")

experiment_data = []

for path in csv_files:
    df = pd.read_csv(path)
    validate_columns(df, path)
    validate_splits(df, path)

    df[COL_SPLIT] = df[COL_SPLIT].astype(str).str.strip()
    df[COL_KEFF] = pd.to_numeric(df[COL_KEFF], errors="raise")
    df[COL_SAMPLE] = pd.to_numeric(df[COL_SAMPLE], errors="raise").astype(int)

    exp_name = os.path.basename(os.path.dirname(path))
    experiment_data.append((exp_name, path, df))

print(f"Found {len(experiment_data)} experiment split logs.")
for exp_name, path, _ in experiment_data:
    print(f"  - {exp_name}: {path}")

# =========================
# CHECK SHARED VAL/TEST
# =========================
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
            "Validation/test splits are not identical across experiments. "
            "Fix the split generation or set STRICT_SHARED_VAL_TEST = False."
        )
else:
    print("\nValidation and test splits are identical across experiments.")

# =========================
# FIXED GLOBAL BINS
# =========================
all_keff = []
for _, _, df in experiment_data:
    all_keff.append(df[COL_KEFF].to_numpy())

all_keff = np.concatenate(all_keff)
xmin = all_keff.min()
xmax = all_keff.max()

if np.isclose(xmin, xmax):
    raise ValueError("All keff values are identical; cannot build histogram bins.")

bin_edges = np.linspace(xmin, xmax, NBINS + 1)

# =========================
# PLOT
# =========================
fig, ax = plt.subplots(figsize=(10, 6))

# training curve for each experiment
for exp_name, path, df in experiment_data:
    train_vals = df.loc[df[COL_SPLIT] == "train", COL_KEFF].to_numpy()
    x, y = density_hist_curve(train_vals, bin_edges)
    ax.step(x, y, where="mid", linewidth=1.8, alpha=0.9, label=f"train - {exp_name}")

# one shared val/test curve from the reference file
val_vals = ref_df.loc[ref_df[COL_SPLIT] == "val", COL_KEFF].to_numpy()
test_vals = ref_df.loc[ref_df[COL_SPLIT] == "test", COL_KEFF].to_numpy()

xv, yv = density_hist_curve(val_vals, bin_edges)
xt, yt = density_hist_curve(test_vals, bin_edges)

ax.step(xv, yv, where="mid", linewidth=2.8, color=VAL_COLOR, label="val (shared)")
ax.step(xt, yt, where="mid", linewidth=2.8, color=TEST_COLOR, label="test (shared)")

ax.set_xlabel("keff")
ax.set_ylabel("Density")
ax.set_title("keff distribution by split across experiments")
ax.grid(True, alpha=0.25)
ax.legend(fontsize=9, ncol=2)
fig.tight_layout()

fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nSaved figure:")
print(f"  {OUT_PNG}")
