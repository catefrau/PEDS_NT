# `dataset_keff.csv`: analysis and relation to the `data/highfidelity` files

## 1. Duplicate check

- Rows: 4599. Parameter columns: r0_b4c_rod_outer_radius, r0_b4c_rod_cr_fraction, r1_fuel_annulus_outer_radius, r1_fuel_annulus_enrichment, r1_fuel_annulus_f_mod, r2_water_outer_radius.
- Exact duplicate 6-parameter vectors (float64): **0**; after float32 casting: **0**.
- Near-duplicate pairs (Euclidean distance < 1e-06 in range-scaled space): **0**.
- Near-duplicate pairs (Euclidean distance < 0.0001 in range-scaled space): **0**.
- Therefore no rows had to be removed and `dataset_keff.csv` was used as is (no new file written).

- NaN values: 0.

## 2. Samples, bounds, keff

N = **4599** unique cases.

| param | min | max | mean |
|---|---|---|---|
| r0_b4c_rod_outer_radius | 1.001 | 11.98 | 3.85 |
| r0_b4c_rod_cr_fraction | 0.0005025 | 0.9999 | 0.5088 |
| r1_fuel_annulus_outer_radius | 10 | 44.99 | 25.01 |
| r1_fuel_annulus_enrichment | 1 | 9.998 | 5.48 |
| r1_fuel_annulus_f_mod | 0.3011 | 0.7996 | 0.5833 |
| r2_water_outer_radius | 40 | 79.97 | 54.04 |

keff: min 0.4283, max 1.4268, mean 1.0523, std 0.1548; share in [0.8,1.2]: 76.1%. Mean MC keff_std 5.20e-04 (max 9.09e-04).

## 3. Relation to the high-fidelity files

Rows are matched on the 6 parameters after casting to float32 (the precision of the npz files), and keff is compared on matches.

| file | N unique | found in dataset_keff | % | keff max abs diff on matches | in file but missing from csv |
|---|---|---|---|---|---|
| 17jul_0.8_1.2 | 2648 | 2648 | 100.0% | 6e-08 | 0 |
| LHS_0.8_newbounds | 1854 | 1854 | 100.0% | 6e-08 | 0 |
| MC_1315 | 1315 | 1315 | 100.0% | 6e-08 | 0 |
| MC_2200 | 2203 | 2203 | 100.0% | 6e-08 | 0 |
| complete_LHS_2900 | 2868 | 2868 | 100.0% | 6e-08 | 0 |
| merged_lhs_0.8_1.2 | 2126 | 2126 | 100.0% | 6e-08 | 0 |

Union of matched rows: **3910** of 4599 csv rows (85.0%). **689** rows (15.0%) of the csv are not in any of the 6 npz files.

### Rows only in the csv (not in any npz) vs rows shared

| | N | keff min | keff max | keff mean | share in [0.8,1.2] |
|---|---|---|---|---|---|
| shared with npz | 3910 | 0.554 | 1.427 | 1.053 | 81.0% |
| csv only | 689 | 0.428 | 1.393 | 1.048 | 48.5% |

Parameter extent of csv-only rows:

| param | min | max |
|---|---|---|
| r0_b4c_rod_outer_radius | 1.001 | 9.488 |
| r0_b4c_rod_cr_fraction | 0.0005396 | 0.9988 |
| r1_fuel_annulus_outer_radius | 10 | 42.85 |
| r1_fuel_annulus_enrichment | 1 | 9.982 |
| r1_fuel_annulus_f_mod | 0.4007 | 0.7994 |
| r2_water_outer_radius | 40.07 | 79.94 |

### Inclusion between files

- Of the csv rows, 3502 have keff in [0.8,1.2]; of those, 3168 appear in some npz.
- `17jul_0.8_1.2` is fully contained in the csv (2648/2648).
- `LHS_0.8_newbounds` is fully contained in the csv (1854/1854).
- `MC_1315` is fully contained in the csv (1315/1315).
- `MC_2200` is fully contained in the csv (2203/2203).
- `complete_LHS_2900` is fully contained in the csv (2868/2868).
- `merged_lhs_0.8_1.2` is fully contained in the csv (2126/2126).

## 4. Is dataset_keff.csv LHS?

Mapped to [0,1] with its own min/max. A true LHS has one point per N-bin in each dimension (occupancy 1.0, random 0.63).

| bins m | occupancy (mean over 6 dims) | random expectation | Cr fraction + enrichment only |
|---|---|---|---|
| N/10 | 0.965 | 1.000 | 1.000 |
| N/4 | 0.889 | 0.982 | 0.986 |
| N/2 | 0.785 | 0.865 | 0.882 |
| N/1 | 0.589 | 0.632 | 0.639 |

max |Spearman| between parameters: 0.150. KS vs uniform p-values per dim: 0.0e+00, 7.7e-02, 5.8e-67, 1.5e-07, 6.9e-126, 0.0e+00.

10-bin counts per parameter:

| param | counts |
|---|---|
| r0_b4c_rod_outer_radius | [918, 937, 979, 980, 537, 45, 51, 47, 51, 54] |
| r0_b4c_rod_cr_fraction | [453, 459, 439, 433, 444, 455, 468, 469, 490, 489] |
| r1_fuel_annulus_outer_radius | [407, 591, 609, 628, 602, 508, 454, 410, 288, 102] |
| r1_fuel_annulus_enrichment | [271, 519, 566, 523, 486, 483, 475, 462, 417, 397] |
| r1_fuel_annulus_f_mod | [53, 60, 578, 590, 629, 621, 594, 594, 527, 353] |
| r2_water_outer_radius | [748, 763, 760, 708, 630, 201, 194, 190, 205, 200] |

## 5. Summary

- **Duplicates:** none (exact or near), so nothing was removed and no new csv was written.
- **Size:** 4599 cases, versus 1315-2931 rows (1854-2868 unique) in the npz files.
- **Superset:** every unique row of all 6 npz files is in the csv, with identical keff to float32 precision. The csv is the union of them (3910 rows) plus 689 extra rows.
- **Extra rows:** they have a lower share in the 0.8-1.2 window (48.5% vs 81%) and reach keff down to 0.43, below the npz minimum of 0.554. They look like later or unfiltered additions, not LHS-filtered points. Their r0 radius reaches 9.5, r1 radius 42.9 and f_mod starts at 0.40.
- **Bounds:** these are the widest of all the files: r0 radius 1-12, enrichment 1-10 and r2 radius 40-80. Enrichment goes down to 1.0, like `17jul_0.8_1.2` but unlike the other npz files (1.5). keff is 0.43-1.43, wider than the unfiltered `complete_LHS_2900` (0.55-1.43).
- **Distribution:** not LHS. Occupancy at N bins is 0.59, which is at or below the random level of 0.63. Marginals are strongly non-uniform, and the dense-core plus sparse-tail structure of `complete_LHS_2900` carries over (e.g. r0 radius ~920-980 per bin below 6 and ~50 per bin above). Cr fraction is the only parameter near uniform (KS p=0.08). Parameter correlations are smaller than in the npz files (max |Spearman| 0.15), because the keff filter is not applied to the whole csv.
