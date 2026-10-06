"""Edit these settings, then run: python build_final_dataset.py"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
INPUT_CSV = HERE.parent / "std_all_cases.csv"
OUTPUT_DIR = HERE / "outputs"

# ── Hard constraints ─────────────────────────────────────────────────────────
MAX_KEFF_STD = 0.00060
USE_KEFF_BOUNDS = False
KEFF_BOUNDS = (0.7, 1.3)  # ignored when USE_KEFF_BOUNDS is False

# ── Starting parameter bounds (margins trimmed only if uniformity improves) ─────
PARAM_BOUNDS = {
    "r0_b4c_rod_outer_radius": (1.0, 6.0),
    "r0_b4c_rod_cr_fraction": (0.0, 1.0),
    "r1_fuel_annulus_outer_radius": (10.0, 35.0),
    "r1_fuel_annulus_enrichment": (1.5, 8.0),
    "r1_fuel_annulus_f_mod": (0.4, 0.8),
    "r2_water_outer_radius": (40.0, 80.0),
}

# ── Selection ─────────────────────────────────────────────────────────────────
N_SAMPLES = 1500
DEDUP_COLS = [
    "r0_b4c_rod_outer_radius",
    "r0_b4c_rod_cr_fraction",
    "r1_fuel_annulus_outer_radius",
    "r1_fuel_annulus_enrichment",
    "r1_fuel_annulus_f_mod",
    "r2_water_outer_radius",
    "keff",
    "keff_std",
]

# ── Uniformity objective ──────────────────────────────────────────────────────
N_BINS_1D = 10
N_BINS_2D = 8
WEIGHT_L2_1D = 1.0
WEIGHT_CHI2_2D = 0.35  # scales mean 2D chi-square statistic (lower is better)
WEIGHT_L2_2D = 0.65

# Margin trimming: only accept a cut if the combined score drops by this fraction
MARGIN_TRIM_MIN_REL_IMPROVEMENT = 0.03
MARGIN_TRIM_PERCENTILE_STEPS = (0.01, 0.02, 0.03, 0.05, 0.07, 0.10)

# LHS-style subset selection from the pool
LHS_SHORTLIST_SIZE = 250   # lowest-std candidates considered each greedy step
LHS_REFINEMENT_SWAPS = 400  # attempted single-point swap improvements at the end
