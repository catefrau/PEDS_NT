import pandas as pd
from pathlib import Path

PARAMS = [
    "r0_b4c_rod_outer_radius",
    "r0_b4c_rod_cr_fraction",
    "r1_fuel_annulus_outer_radius",
    "r1_fuel_annulus_enrichment",
    "r1_fuel_annulus_f_mod",
    "r2_water_outer_radius",
]
KEFF_COL = "keff"
OUT_COLS = PARAMS + [KEFF_COL, "source_file"]

files_dir = Path("../FILES")
lhs_path = files_dir / "lhs_0.8_bounds.csv"
jul_path = files_dir / "13jul.csv"
out_path = files_dir / "17jul_boundsand13jul.csv"

# --- load master ---
lhs = pd.read_csv(lhs_path)

# --- normalize 7jul to the same schema ---
jul = pd.read_csv(jul_path)
jul_norm = jul[PARAMS + [KEFF_COL]].copy()
jul_norm["source_file"] = jul_path.name

# --- stack ---
merged = pd.concat([lhs[OUT_COLS], jul_norm[OUT_COLS]], ignore_index=True)

# --- optional: drop exact duplicates (same 6 params + keff) ---
before = len(merged)
merged = merged.drop_duplicates(subset=PARAMS + [KEFF_COL], keep="last")
print(f"rows: {before} -> {len(merged)} after dedup")

merged.to_csv(out_path, index=False)
print(f"saved {out_path}")