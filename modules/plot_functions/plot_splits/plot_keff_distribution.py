import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("../FILES/lhs_merged_newbounds.csv")

keff = df["keff"].dropna()

FIGSIZE = (7.2, 3.6)  # slightly wider and less tall
FS_LABEL = 17         # match parity-plot label scale

fig, ax = plt.subplots(figsize=FIGSIZE)

# weights make each bar height = fraction of total samples (probability)
weights = [1 / len(keff)] * len(keff)

ax.hist(
    keff,
    bins=40,
    weights=weights,
    color="steelblue",
    edgecolor="white",
    linewidth=0.5
)

ax.set_xlabel(r"$k_{eff}$", fontsize=FS_LABEL)
ax.set_ylabel("Probability", fontsize=FS_LABEL)
ax.tick_params(axis="both", labelsize=FS_LABEL)
ax.grid(True, alpha=0.25)

fig.tight_layout()
fig.savefig("keff_histogram_probability.png", dpi=300)
plt.show()

print("Saved: keff_histogram_probability.png")