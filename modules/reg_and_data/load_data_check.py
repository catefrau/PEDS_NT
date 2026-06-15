import os
import numpy as np
import matplotlib.pyplot as plt


# =========================
# User settings
# =========================
data_path = "../../data/highfidelity/MCruns_big.npz"
train_size = 800
test_size = 200
seed = 42
n_bins = 20

save_dir = "split_diagnostics"
os.makedirs(save_dir, exist_ok=True)

hist_path = os.path.join(save_dir, "keff_hist_train_test.png")
binrange_path = os.path.join(save_dir, "keff_range_per_bin.png")


# =========================
# Same split logic as loaddata()
# =========================
data = np.load(data_path, allow_pickle=True)
keffs = np.array(data["keffs"], dtype=np.float32)

rng = np.random.default_rng(seed)

sorted_idx = np.argsort(keffs)
split_bins = np.array_split(sorted_idx, n_bins)

test_per_bin = max(1, test_size // n_bins)

test_idx = []
train_idx = []
bin_stats = []

for b, bin_idx in enumerate(split_bins):
    bin_idx = np.array(bin_idx, copy=True)
    rng.shuffle(bin_idx)

    n_test_here = min(test_per_bin, len(bin_idx) - 1)

    test_take = bin_idx[:n_test_here]
    train_take = bin_idx[n_test_here:]

    test_idx.extend(test_take.tolist())
    train_idx.extend(train_take.tolist())

    kvals = keffs[bin_idx]
    bin_stats.append({
        "bin": b,
        "count_total": len(bin_idx),
        "count_test": len(test_take),
        "count_train": len(train_take),
        "k_min": float(kvals.min()),
        "k_max": float(kvals.max()),
        "k_mean": float(kvals.mean()),
    })

test_idx = np.array(test_idx[:test_size], dtype=int)
train_idx = np.array(train_idx, dtype=int)

rng.shuffle(train_idx)
train_idx = train_idx[:train_size]

train_keff = keffs[train_idx]
test_keff = keffs[test_idx]


# =========================
# Print diagnostics
# =========================
print(f"{'bin':>3} | {'k min':>8} | {'k max':>8} | {'total':>5} | {'train':>5} | {'test':>4}")
print("-" * 50)
for s in bin_stats:
    print(
        f"{s['bin']:>3} | "
        f"{s['k_min']:>8.5f} | "
        f"{s['k_max']:>8.5f} | "
        f"{s['count_total']:>5} | "
        f"{s['count_train']:>5} | "
        f"{s['count_test']:>4}"
    )

print()
print(f"Train size = {len(train_keff)}")
print(f"Test size  = {len(test_keff)}")
print(f"Train keff range = [{train_keff.min():.5f}, {train_keff.max():.5f}]")
print(f"Test  keff range = [{test_keff.min():.5f}, {test_keff.max():.5f}]")


# =========================
# Plot 1: histogram distribution
# =========================
plt.figure(figsize=(9, 5.5))

bins_hist = 30
plt.hist(train_keff, bins=bins_hist, alpha=0.60, color="tab:blue", label=f"Train ({len(train_keff)})")
plt.hist(test_keff, bins=bins_hist, alpha=0.60, color="tab:red", label=f"Test ({len(test_keff)})")

plt.xlabel("k_eff")
plt.ylabel("Number of samples")
plt.title("k_eff distribution — train vs test")
plt.grid(True, alpha=0.20)
plt.legend()
plt.tight_layout()
plt.savefig(hist_path, dpi=180, bbox_inches="tight")
plt.close()

print(f"Saved histogram to: {hist_path}")


# =========================
# Plot 2: keff range per bin
# =========================
bin_id = [s["bin"] for s in bin_stats]
k_min = [s["k_min"] for s in bin_stats]
k_max = [s["k_max"] for s in bin_stats]
k_mean = [s["k_mean"] for s in bin_stats]

plt.figure(figsize=(9, 5.5))

for i in range(len(bin_id)):
    plt.vlines(x=bin_id[i], ymin=k_min[i], ymax=k_max[i], color="tab:green", linewidth=4, alpha=0.75)
    plt.scatter(bin_id[i], k_mean[i], color="black", s=45, zorder=3)

plt.xticks(bin_id)
plt.xlabel("Bin index")
plt.ylabel("k_eff")
plt.title("k_eff range per bin")
plt.grid(True, alpha=0.20)

for i in range(len(bin_id)):
    plt.text(
        bin_id[i] + 0.06,
        k_max[i],
        f"[{k_min[i]:.3f}, {k_max[i]:.3f}]",
        fontsize=8,
        va="bottom"
    )

plt.tight_layout()
plt.savefig(binrange_path, dpi=180, bbox_inches="tight")
plt.close()

print(f"Saved bin-range plot to: {binrange_path}")