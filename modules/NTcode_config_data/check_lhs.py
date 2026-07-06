import numpy as np
import pandas as pd
from scipy.stats import chisquare, pearsonr

# ══════════════════════════════════════════════════════════════
# EDIT THIS
CSV_PATH = '../FILES/2jul_full.csv'
KEFF_COL = 'keff'
PARAM_COLS = [   # list your 6 parameter column names as they appear in the CSV
    'r0_b4c_rod_outer_radius',
    'r0_b4c_rod_cr_fraction',
    'r1_fuel_annulus_outer_radius',
    'r1_fuel_annulus_enrichment',
    'r1_fuel_annulus_f_mod',
    'r2_water_outer_radius',
]
N_BINS = 10
EDGE_FRAC = 0.1
BOUNDS = {
    'r0_b4c_rod_outer_radius':      (1.0,  6.0),
    'r0_b4c_rod_cr_fraction':       (0.0,   1.0),
    'r1_fuel_annulus_outer_radius': (12.5, 40.0),
    'r1_fuel_annulus_enrichment':   (2,    10.0),
    'r1_fuel_annulus_f_mod':        (0.40, 0.8),
    'r2_water_outer_radius':        (40.0, 60.0),
}
# ══════════════════════════════════════════════════════════════

def infer_bounds_from_data(df, param_cols, round_to='auto'):
    rows, inferred_bounds = [], {}
    for p in param_cols:
        x = df[p].dropna().values
        obs_min, obs_max = x.min(), x.max()
        if round_to == 'auto':
            decimals_seen = [len(f"{v:.10f}".rstrip('0').split('.')[1])
                              for v in x if '.' in f"{v:.10f}".rstrip('0')]
            n_dec = min(int(np.median(decimals_seen)) if decimals_seen else 0, 3)
            factor = 10 ** n_dec
            lo_guess = np.floor(obs_min * factor) / factor
            hi_guess = np.ceil(obs_max * factor) / factor
        else:
            lo_guess, hi_guess = obs_min, obs_max
        inferred_bounds[p] = (lo_guess, hi_guess)
        rows.append({'parameter': p, 'observed_min': round(obs_min, 4),
                      'observed_max': round(obs_max, 4),
                      'inferred_lower_bound': lo_guess,
                      'inferred_upper_bound': hi_guess})
    return inferred_bounds, pd.DataFrame(rows)

def check_lhs_coverage_bounded_only(df, bounds, keff_col='keff', n_bins=10, edge_frac=0.1):
    n = len(df)
    rows = []
    for p, (lo, hi) in bounds.items():
        x = df[p].dropna().values
        bin_edges = np.linspace(lo, hi, n_bins + 1)
        counts, _ = np.histogram(x, bins=bin_edges)
        expected = np.full(n_bins, len(x) / n_bins)
        chi2_stat, chi2_p = chisquare(counts, expected)
        empty_bins = int(np.sum(counts == 0))
        edge_width = edge_frac * (hi - lo)
        low_edge_count  = np.sum(x <= lo + edge_width)
        high_edge_count = np.sum(x >= hi - edge_width)
        expected_edge_count = n * edge_frac
        low_edge_ratio  = low_edge_count  / expected_edge_count
        high_edge_ratio = high_edge_count / expected_edge_count
        r, r_p = pearsonr(df[p], df[keff_col])
        rows.append({'parameter': p, 'n_samples': len(x),
                      'empty_bins_of_10': empty_bins,
                      'chi2_p_value': round(chi2_p, 4),
                      'uniform_flag': 'OK' if chi2_p > 0.05 else 'NON-UNIFORM',
                      'low_edge_density_ratio': round(low_edge_ratio, 2),
                      'high_edge_density_ratio': round(high_edge_ratio, 2),
                      'corr_with_keff': round(r, 3),
                      'corr_p_value': round(r_p, 4)})
    return pd.DataFrame(rows)

df = pd.read_csv(CSV_PATH)
#inferred_bounds, bounds_report = infer_bounds_from_data(df, PARAM_COLS)
#print(bounds_report.to_string(index=False))

summary = check_lhs_coverage_bounded_only(df, BOUNDS, KEFF_COL, N_BINS, EDGE_FRAC)
print(summary.to_string(index=False))