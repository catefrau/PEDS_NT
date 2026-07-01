import numpy as np
import pandas as pd


def csv_to_npz(
    csv_path: str,
    param_cols: list[str],
    keff_col: str = "keff",
    output_path: str = "../FILES/MCruns.npz",
) -> None:
    """
    Convert a CSV simulation dataset into a structured .npz file.

    Parameters
    ----------
    csv_path    : Path to the input CSV file.
    param_cols  : List of column names to use as input parameters.
                  All other columns (cross-sections, materials, etc.) are discarded.
    keff_col    : Name of the keff column in the CSV. Default: "keff".
    output_path : Name/path of the output .npz file. Default: "MCruns.npz".

    Saved arrays
    ------------
    params       : (N, P) float32 — inputs normalized to [0, 1]
    params_raw   : (N, P) float32 — original physical values
    keffs        : (N,)   float32 — effective multiplication factor
    param_names  : (P,)   str     — names of the P input parameters
    bounds_lo    : (P,)   float32 — per-column minimum (used for normalization)
    bounds_hi    : (P,)   float32 — per-column maximum (used for normalization)
    """

    # ── 1. Load CSV ──────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path)

    # ── 2. Extract raw parameter matrix ──────────────────────────────────────
    missing = [c for c in param_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in CSV: {missing}")

    params_raw = df[param_cols].to_numpy(dtype=np.float32)   # shape (N, P)

    # ── 3. Compute bounds and normalize to [0, 1] ────────────────────────────
    bounds_lo = params_raw.min(axis=0)   # shape (P,)
    bounds_hi = params_raw.max(axis=0)   # shape (P,)

    # Avoid division by zero for constant columns (edge case)
    span = bounds_hi - bounds_lo
    span[span == 0] = 1.0

    params = (params_raw - bounds_lo) / span             # shape (N, P), float32

    # ── 4. Extract keff vector ───────────────────────────────────────────────
    if keff_col not in df.columns:
        raise ValueError(f"keff column '{keff_col}' not found in CSV.")

    keffs = df[keff_col].to_numpy(dtype=np.float32)      # shape (N,)

    # ── 5. Save .npz ─────────────────────────────────────────────────────────
    np.savez(
        output_path,
        params=params,
        params_raw=params_raw,
        keffs=keffs,
        param_names=np.array(param_cols),
        bounds_lo=bounds_lo.astype(np.float32),
        bounds_hi=bounds_hi.astype(np.float32),
    )
    print(f"Saved '{output_path}' with {len(keffs)} samples and {len(param_cols)} parameters.")



# --- Load the dataset ---
def cut_dataset(csv_path: str, csv_path_filtered: str) -> None:
    df = pd.read_csv(csv_path)
    print(f"Original dataset size: {len(df)} samples")

    # --- Define your filtering conditions ---
    condition_remove = (
        (df["keff"] < 0.8) |       # remove if outer radius > 8
        (df["keff"] > 1.2)         # remove if f_mod < 0.4
    )

    df_filtered = df[~condition_remove]  # ~ means "NOT" — keep everything else
    print(f"Filtered dataset size: {len(df_filtered)} samples")
    print(f"Samples removed: {len(df) - len(df_filtered)}")
    # --- Save the filtered dataset ---
    df_filtered.to_csv(csv_path_filtered, index=False)
    print(f"Saved to {csv_path_filtered} ✅")

# ── Example usage ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    PARAM_COLUMNS = [
        "r0_b4c_rod_outer_radius",
        "r0_b4c_rod_cr_fraction",
        "r1_fuel_annulus_outer_radius",
        "r1_fuel_annulus_enrichment",
        "r1_fuel_annulus_f_mod",
        "r2_water_outer_radius",
    ]

    """ cut_dataset(
        csv_path="../FILES/30june_piece.csv",
        csv_path_filtered="../FILES/30june_piece_filtered.csv",
    )
 """
    csv_to_npz(
        csv_path="../FILES/30_june_full.csv",
        param_cols=PARAM_COLUMNS,
        keff_col="keff",
        output_path="../FILES/30_june_full.npz",
    )
