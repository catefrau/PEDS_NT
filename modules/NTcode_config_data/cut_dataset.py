import pandas as pd

# --- Load the dataset ---
df = pd.read_csv("FILES/LHS_full_dataset.csv")

print(f"Original dataset size: {len(df)} samples")

# --- Define your filtering conditions ---
# We keep rows that do NOT match the unwanted characteristics
condition_remove = (
    (df["r0_b4c_rod_outer_radius"] > 8.0) |       # remove if outer radius > 8
    (df["r1_fuel_annulus_f_mod"] < 0.4)         # remove if f_mod < 0.4
)

df_filtered = df[~condition_remove]  # ~ means "NOT" — keep everything else

print(f"Filtered dataset size: {len(df_filtered)} samples")
print(f"Samples removed: {len(df) - len(df_filtered)}")

# --- Save the filtered dataset ---
df_filtered.to_csv("LHS_filtered_dataset.csv", index=False)
print("Saved to LHS_filtered_dataset.csv ✅")