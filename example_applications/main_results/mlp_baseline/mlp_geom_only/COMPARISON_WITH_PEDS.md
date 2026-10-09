# MLP baseline vs PEDS (complete_strat)

## What was done
- Trained a vanilla NN-only baseline ensemble with 5 runs (seeds 0..4).
- For each seed, used the exact split from `train_1000_seed_<seed>/split_log.csv`.
- Inputs are only the 6 geometry features (`params`); target is direct `keff`.
- Baseline architecture: `6 -> 128 -> 256 -> 128 -> 1` (Boltzmann-app analogue: final hidden->XS layer swapped for hidden->1).
- Capacity check: PEDS generator estimate `73524` params vs baseline `66945` params.

## Test comparison (mean over seeds)
- Mean absolute fractional error (baseline): `0.11018`.
- Mean |Δk| (pcm): baseline `11132.0` vs PEDS `574.9` (+1836.3%).
- Mean |Δρ| (pcm): baseline `11118.8` vs PEDS `611.1` (+1719.5%).
- Fraction below 650 pcm in |Δk|: baseline `0.041` vs PEDS `0.698`.
- Fraction below 650 pcm in |Δρ|: baseline `0.045` vs PEDS `0.690`.

## Interpretation on data sufficiency
- If baseline and PEDS metrics are close, 1000 training points are likely enough for a pure NN surrogate at this geometry-only setting.
- If baseline remains noticeably worse, the PEDS solver-informed structure is extracting useful inductive bias not recovered by data alone.
- Use `<output_dir>/analysis/testset_results/test_metrics_all_runs.csv` for per-seed inspection.