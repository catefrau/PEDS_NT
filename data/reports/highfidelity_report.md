# High-fidelity datasets: comparison report

Scope: the 6 `.npz` files directly in `data/highfidelity/` (subfolders ignored).
Each file holds `params`/`params_raw` (N×6), `keffs` (N), `param_names`, `bounds_lo`, `bounds_hi`.
`params` is `params_raw` rescaled to [0,1] with the stored bounds (checked: max difference 6e-8).
Reproduce with `reports/analyse_highfidelity.py` (full tables in `reports/highfidelity_stats.md`).

Parameters (physical units): r0 B4C rod outer radius, r0 B4C Cr fraction, r1 fuel annulus outer radius, r1 fuel enrichment, r1 fuel f_mod, r2 water outer radius.

## 1. Number of samples

| file | N | duplicate rows | unique | keff window |
|---|---|---|---|---|
| `LHS_0.8_newbounds` | 1904 | 50 | 1854 | 0.80–1.20 |
| `17jul_0.8_1.2` | 2648 | 0 | 2648 | 0.80–1.20 |
| `merged_lhs_0.8_1.2` | 2179 | 53 | 2126 | 0.80–1.20 |
| `MC_1315` | 1315 | 0 | 1315 | 0.80–1.27 |
| `MC_2200` | 2203 | 0 | 2203 | 0.80–1.20 |
| `complete_LHS_2900` | 2931 | 63 | 2868 | 0.55–1.43 (unfiltered) |

No NaN keff anywhere. The names do not match the real N (`MC_2200` has 2203, `complete_LHS_2900` has 2931), and the three LHS-named files contain exact duplicate parameter rows. A true LHS would have none.

## 2. Parameter bounds and keff range

The stored bounds equal the observed min/max in every file, so they describe the data extent, not necessarily the original sampling box.

| parameter | LHS_0.8_newbounds | 17jul_0.8_1.2 | merged_lhs | MC_1315 | MC_2200 | complete_LHS_2900 |
|---|---|---|---|---|---|---|
| r0 rod outer radius | 1.00–6.00 | 1.00–6.00 | 1.00–**11.98** | 1.00–**7.98** | 1.00–**7.98** | 1.00–**11.98** |
| r0 Cr fraction | 0.001–1.0 | 0.0005–1.0 | 0.001–1.0 | 0.001–1.0 | 0.001–1.0 | 0.001–1.0 |
| r1 fuel outer radius | 10.11–44.92 | 10.11–44.92 | 10.11–44.92 | 10.11–44.92 | 10.11–44.92 | **10.02–44.99** |
| r1 enrichment | **1.50**–9.99 | **1.01**–9.99 | 1.50–9.99 | 1.50–10.0 | 1.50–10.0 | 1.50–10.0 |
| r1 f_mod | 0.40–0.80 | 0.40–0.80 | **0.30**–0.80 | **0.30**–0.80 | **0.30**–0.80 | **0.30**–0.80 |
| r2 water outer radius | 40–79.95 | 40–79.97 | 40–79.95 | 40–79.95 | 40–79.95 | 40–79.95 |
| **keff min / max** | 0.8003 / 1.1998 | 0.8003 / 1.1999 | 0.8001 / 1.1998 | 0.8003 / **1.2683** | 0.8003 / 1.1998 | **0.554 / 1.427** |
| keff mean / std | 1.039 / 0.106 | 1.025 / 0.109 | 1.036 / 0.105 | 1.033 / 0.108 | 1.038 / 0.105 | 1.071 / 0.152 |

Observations:
- `LHS_0.8_newbounds` and `17jul_0.8_1.2` share the narrowest box: r0 radius ≤ 6 and f_mod ≥ 0.4. `17jul` also extends enrichment down to 1.0.
- The MC files widen r0 radius to ~8 and f_mod down to 0.3.
- `complete_LHS_2900` and `merged_lhs_0.8_1.2` have the widest r0 radius (~12) and f_mod down to 0.3.
- `merged_lhs_0.8_1.2` is `complete_LHS_2900` restricted to keff in [0.8, 1.2]. All 2126 of its unique rows are in `complete_LHS_2900`. Its bounds are the same except r1 outer radius and enrichment.
- Only `complete_LHS_2900` is unfiltered in keff. The others are cut to about [0.8, 1.2], and `MC_1315` has a few points up to 1.27.

## 3. Is the distribution LHS?

Short answer: no file shows LHS stratification as stored.

Evidence:
1. **Stratification test.** For a real LHS of N points, each of N equal bins per dimension holds exactly one point, so occupancy is 1.0. Random sampling gives 1−1/e ≈ 0.63. I used the Cr fraction and enrichment, which are the dimensions least affected by filtering, on unique rows:

   | file | occupancy at m=N | random expectation |
   |---|---|---|
   | LHS_0.8_newbounds | 0.653 | 0.632 |
   | 17jul_0.8_1.2 | 0.630 | 0.632 |
   | merged_lhs | 0.646 | 0.632 |
   | MC_1315 | 0.656 | 0.632 |
   | MC_2200 | 0.651 | 0.632 |
   | complete_LHS_2900 | 0.660 | 0.632 |

   The same holds at m=N/2 (≈0.87–0.90 vs 0.865). All files are statistically close to random, and none is LHS-like. `complete_LHS_2900` is the closest to stratified (0.998 at m=N/4 vs 0.982 random), but not enough to count as LHS.
2. **Marginals are not flat.** With the filter and merging, the 1-D marginals fail uniformity (KS p < 1e-100 everywhere). The 10-bin counts for `complete_LHS_2900` show why:
   - r0 radius: ~540–610 per bin for 1–6, then ~50 per bin for 6–12.
   - r2 radius: the same pattern, with a dense part below ~60 and ~50 per bin above.
   - f_mod: ~55 per bin for 0.3–0.4, then ~370 for 0.4–0.8.
   - So it is a union of nested boxes: a dense core box plus sparse extensions to wider bounds. It is not one LHS over the stated bounds.
3. **Geometry-driven shapes.** r1 outer radius is hump-shaped (about 200 in the edge bins vs 400 in the middle). Constraints between radii (r0 < r1 < r2) and the keff filter reshape the distribution. Filtering at 0.8–1.2 pushes `17jul`, `LHS_0.8_newbounds` and the MC files further from uniform. Their bin-count std is 2–3× the random level for r1 and r2 radii.
4. **Duplicates and nesting.** The files overlap heavily:
   - The 1854 unique rows of `LHS_0.8_newbounds` are contained in `17jul`, `MC_2200`, `complete_LHS_2900` and `merged_lhs`.
   - `MC_2200` is a superset of `LHS_0.8_newbounds`, plus about 350 rows.
   - Duplicated rows (50–63) are the sign of merging or concatenation.
5. **Correlations.** The max |Spearman| between parameters is 0.27–0.34. That is much larger than the ~0.03 expected from a clean LHS of this size, and it comes from the geometric constraints and the filtering.

Interpretation of the labels:
- `LHS_*`, `merged_lhs_*` and `complete_LHS_*` were probably generated as LHS in parts. Rejecting invalid geometries and keff outliers, then merging batches, removed the stratification.
- The `MC_*` files are consistent with random sampling and with the LHS rows being added to.
- The "LHS" label is therefore best read as "LHS-seeded, then filtered and merged", not as a strict LHS design.

## 4. Practical implications
- The datasets are not independent: for example, `merged_lhs` ⊂ `complete_LHS_2900`, and `LHS_0.8_newbounds` ⊂ all the others. Avoid combining them in train/test splits without deduplication.
- The input domains differ (r0 radius max 6, 8 or 12; f_mod min 0.3 or 0.4), so a model trained on one and tested on another is partly an extrapolation test.
- Dedupe `params_raw` before training or evaluation (50–63 duplicates in the LHS files).
- If strict space-filling is needed, resample or add points with a fresh LHS or Sobol design over a fixed box, filling the sparse regions (r0 > 6, r2 > 60, f_mod < 0.4), rather than relying on the stored sets.
