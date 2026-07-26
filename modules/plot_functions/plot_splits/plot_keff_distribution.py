import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("../FILES/lhs_merged_newbounds.csv")

keff = df["keff"].dropna()

fig, ax = plt.subplots(figsize=(6, 4))

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

ax.set_title(f"Distribution of keff values ({len(keff)} samples)")
ax.set_xlabel("keff")
ax.set_ylabel("Probability")
ax.grid(True, alpha=0.25)

fig.tight_layout()
fig.savefig("keff_histogram_probability.png", dpi=300)
plt.show()

print("Saved: keff_histogram_probability.png")