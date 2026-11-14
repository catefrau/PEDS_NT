# High-fidelity dataset comparison

Parameters: r0_b4c_rod_outer_radius, r0_b4c_rod_cr_fraction, r1_fuel_annulus_outer_radius, r1_fuel_annulus_enrichment, r1_fuel_annulus_f_mod, r2_water_outer_radius

## 1. Samples

| file | N | NaN keff | duplicate rows | unique rows |
|---|---|---|---|---|
| 17jul_0.8_1.2.npz | 2648 | 0 | 0 | 2648 |
| LHS_0.8_newbounds.npz | 1904 | 0 | 50 | 1854 |
| MC_1315.npz | 1315 | 0 | 0 | 1315 |
| MC_2200.npz | 2203 | 0 | 0 | 2203 |
| complete_LHS_2900.npz | 2931 | 0 | 63 | 2868 |
| merged_lhs_0.8_1.2.npz | 2179 | 0 | 53 | 2126 |

## 2. Bounds

Stored bounds (bounds_lo / bounds_hi) vs. observed min/max of `params_raw`.

### 17jul_0.8_1.2.npz

| param | stored lo | stored hi | obs min | obs max | span used |
|---|---|---|---|---|---|
| r0_b4c_rod_outer_radius | 1.004 | 6 | 1.004 | 6 | 1.000 |
| r0_b4c_rod_cr_fraction | 0.0005025 | 0.9999 | 0.0005025 | 0.9999 | 1.000 |
| r1_fuel_annulus_outer_radius | 10.11 | 44.92 | 10.11 | 44.92 | 1.000 |
| r1_fuel_annulus_enrichment | 1.013 | 9.992 | 1.013 | 9.992 | 1.000 |
| r1_fuel_annulus_f_mod | 0.4005 | 0.7996 | 0.4005 | 0.7996 | 1.000 |
| r2_water_outer_radius | 40 | 79.97 | 40 | 79.97 | 1.000 |

keff: min 0.8003, max 1.1999, mean 1.0247, std 0.1089, p1/p99 0.8062/1.1977

`params` (normalised) range: [0.000, 1.000]

### LHS_0.8_newbounds.npz

| param | stored lo | stored hi | obs min | obs max | span used |
|---|---|---|---|---|---|
| r0_b4c_rod_outer_radius | 1.004 | 6 | 1.004 | 6 | 1.000 |
| r0_b4c_rod_cr_fraction | 0.0009685 | 0.9999 | 0.0009685 | 0.9999 | 1.000 |
| r1_fuel_annulus_outer_radius | 10.11 | 44.92 | 10.11 | 44.92 | 1.000 |
| r1_fuel_annulus_enrichment | 1.502 | 9.992 | 1.502 | 9.992 | 1.000 |
| r1_fuel_annulus_f_mod | 0.4005 | 0.7996 | 0.4005 | 0.7996 | 1.000 |
| r2_water_outer_radius | 40 | 79.95 | 40 | 79.95 | 1.000 |

keff: min 0.8003, max 1.1998, mean 1.0390, std 0.1059, p1/p99 0.8087/1.1977

`params` (normalised) range: [0.000, 1.000]

### MC_1315.npz

| param | stored lo | stored hi | obs min | obs max | span used |
|---|---|---|---|---|---|
| r0_b4c_rod_outer_radius | 1.004 | 7.979 | 1.004 | 7.979 | 1.000 |
| r0_b4c_rod_cr_fraction | 0.0009685 | 0.9996 | 0.0009685 | 0.9996 | 1.000 |
| r1_fuel_annulus_outer_radius | 10.11 | 44.92 | 10.11 | 44.92 | 1.000 |
| r1_fuel_annulus_enrichment | 1.502 | 9.995 | 1.502 | 9.995 | 1.000 |
| r1_fuel_annulus_f_mod | 0.3011 | 0.7996 | 0.3011 | 0.7996 | 1.000 |
| r2_water_outer_radius | 40 | 79.95 | 40 | 79.95 | 1.000 |

keff: min 0.8003, max 1.2683, mean 1.0326, std 0.1075, p1/p99 0.8068/1.1977

`params` (normalised) range: [0.000, 1.000]

### MC_2200.npz

| param | stored lo | stored hi | obs min | obs max | span used |
|---|---|---|---|---|---|
| r0_b4c_rod_outer_radius | 1.004 | 7.979 | 1.004 | 7.979 | 1.000 |
| r0_b4c_rod_cr_fraction | 0.0009685 | 0.9999 | 0.0009685 | 0.9999 | 1.000 |
| r1_fuel_annulus_outer_radius | 10.11 | 44.92 | 10.11 | 44.92 | 1.000 |
| r1_fuel_annulus_enrichment | 1.502 | 9.995 | 1.502 | 9.995 | 1.000 |
| r1_fuel_annulus_f_mod | 0.3011 | 0.7996 | 0.3011 | 0.7996 | 1.000 |
| r2_water_outer_radius | 40 | 79.95 | 40 | 79.95 | 1.000 |

keff: min 0.8003, max 1.1998, mean 1.0378, std 0.1049, p1/p99 0.8097/1.1977

`params` (normalised) range: [0.000, 1.000]

### complete_LHS_2900.npz

| param | stored lo | stored hi | obs min | obs max | span used |
|---|---|---|---|---|---|
| r0_b4c_rod_outer_radius | 1.004 | 11.98 | 1.004 | 11.98 | 1.000 |
| r0_b4c_rod_cr_fraction | 0.0009685 | 0.9999 | 0.0009685 | 0.9999 | 1.000 |
| r1_fuel_annulus_outer_radius | 10.02 | 44.99 | 10.02 | 44.99 | 1.000 |
| r1_fuel_annulus_enrichment | 1.502 | 9.998 | 1.502 | 9.998 | 1.000 |
| r1_fuel_annulus_f_mod | 0.3011 | 0.7996 | 0.3011 | 0.7996 | 1.000 |
| r2_water_outer_radius | 40 | 79.95 | 40 | 79.95 | 1.000 |

keff: min 0.5543, max 1.4268, mean 1.0714, std 0.1518, p1/p99 0.6912/1.3661

`params` (normalised) range: [0.000, 1.000]

### merged_lhs_0.8_1.2.npz

| param | stored lo | stored hi | obs min | obs max | span used |
|---|---|---|---|---|---|
| r0_b4c_rod_outer_radius | 1.004 | 11.98 | 1.004 | 11.98 | 1.000 |
| r0_b4c_rod_cr_fraction | 0.0009685 | 0.9999 | 0.0009685 | 0.9999 | 1.000 |
| r1_fuel_annulus_outer_radius | 10.11 | 44.92 | 10.11 | 44.92 | 1.000 |
| r1_fuel_annulus_enrichment | 1.502 | 9.992 | 1.502 | 9.992 | 1.000 |
| r1_fuel_annulus_f_mod | 0.3011 | 0.7996 | 0.3011 | 0.7996 | 1.000 |
| r2_water_outer_radius | 40 | 79.95 | 40 | 79.95 | 1.000 |

keff: min 0.8001, max 1.1998, mean 1.0363, std 0.1052, p1/p99 0.8093/1.1964

`params` (normalised) range: [0.000, 1.000]

## 3. LHS diagnostics

Using `params_raw` mapped to [0,1] with the stored bounds (u). For a true LHS of N points, each of N equal bins per dimension has exactly one point (occupancy=1.0, max/bin=1). Pure random gives occupancy ≈0.632. Also: KS vs uniform, max |Spearman| between params, and occupancy at coarse stratification (N/10 bins), which merged/filtered LHS sets may retain.

| file | N | occ(N bins) mean over dims | occ(N/10 bins) | max count/bin (N/10) | expected random occ(N/10) | KS p (min over dims) | mean KS D | max abs Spearman | chi2 p (N/10 bins, min) |
|---|---|---|---|---|---|---|---|---|---|
| 17jul_0.8_1.2.npz | 2648 | 0.613 | 0.990 | 27 (mean 10) | 1.000 | 2.6e-255 | 0.1254 | 0.339 | 2.1e-145 |
| LHS_0.8_newbounds.npz | 1904 | 0.609 | 0.944 | 28 (mean 10) | 1.000 | 0 | 0.1566 | 0.286 | 8.7e-267 |
| MC_1315.npz | 1315 | 0.620 | 0.981 | 27 (mean 10) | 1.000 | 3.4e-191 | 0.1968 | 0.324 | 3.3e-121 |
| MC_2200.npz | 2203 | 0.596 | 0.955 | 26 (mean 10) | 1.000 | 0 | 0.2025 | 0.289 | 3.4e-258 |
| complete_LHS_2900.npz | 2931 | 0.591 | 0.986 | 27 (mean 10) | 1.000 | 0 | 0.1958 | 0.269 | 0 |
| merged_lhs_0.8_1.2.npz | 2179 | 0.579 | 0.991 | 28 (mean 10) | 1.000 | 0 | 0.2130 | 0.325 | 6.7e-235 |

Per-dimension chi2 / bin-count spread (N/10 bins; std of counts vs sqrt(mean) expected for random, 0 for LHS-like):

| file | r0_b4c_rod_outer_radius | r0_b4c_rod_cr_fraction | r1_fuel_annulus_outer_radius | r1_fuel_annulus_enrichment | r1_fuel_annulus_f_mod | r2_water_outer_radius |
|---|---|---|---|---|---|---|
| 17jul_0.8_1.2.npz | 2.68 (rand 3.17) | 2.61 (rand 3.17) | 6.22 (rand 3.17) | 3.82 (rand 3.17) | 3.24 (rand 3.17) | 7.16 (rand 3.17) |
| LHS_0.8_newbounds.npz | 2.51 (rand 3.17) | 2.42 (rand 3.17) | 6.24 (rand 3.17) | 3.50 (rand 3.17) | 3.32 (rand 3.17) | 9.83 (rand 3.17) |
| MC_1315.npz | 5.54 (rand 3.17) | 2.51 (rand 3.17) | 5.70 (rand 3.17) | 3.98 (rand 3.17) | 4.66 (rand 3.17) | 8.45 (rand 3.17) |
| MC_2200.npz | 5.99 (rand 3.16) | 2.28 (rand 3.16) | 6.14 (rand 3.16) | 3.41 (rand 3.16) | 5.55 (rand 3.16) | 9.21 (rand 3.16) |
| complete_LHS_2900.npz | 9.34 (rand 3.16) | 1.92 (rand 3.16) | 3.89 (rand 3.16) | 2.38 (rand 3.16) | 4.98 (rand 3.16) | 8.38 (rand 3.16) |
| merged_lhs_0.8_1.2.npz | 8.96 (rand 3.17) | 2.47 (rand 3.17) | 5.73 (rand 3.17) | 3.38 (rand 3.17) | 5.28 (rand 3.17) | 8.14 (rand 3.17) |

## 4. Overlap between files (identical param rows)

| | 17jul_0.8_1.2.npz | LHS_0.8_newbounds.npz | MC_1315.npz | MC_2200.npz | complete_LHS_2900.npz | merged_lhs_0.8_1.2.npz |
|---|---|---|---|---|---|---|
| 17jul_0.8_1.2.npz | 2648 | 1854 | 1061 | 1854 | 1854 | 1854 |
| LHS_0.8_newbounds.npz | 1854 | 1854 | 1061 | 1854 | 1854 | 1854 |
| MC_1315.npz | 1061 | 1061 | 1315 | 1313 | 1164 | 1162 |
| MC_2200.npz | 1854 | 1854 | 1313 | 2203 | 1955 | 1955 |
| complete_LHS_2900.npz | 1854 | 1854 | 1164 | 1955 | 2868 | 2126 |
| merged_lhs_0.8_1.2.npz | 1854 | 1854 | 1162 | 1955 | 2126 | 2126 |