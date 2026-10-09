# MLP baseline vs PEDS (`complete_strat`)

## What was done
- Trained a vanilla NN-only baseline ensemble (seeds 0–4, train size 1000).
- Reused the exact PEDS `split_log.csv` for each seed (same train/val/test indices).
- Inputs: only the 6 normalized geometry features from `params`.
- Target: direct `keff` prediction (no XS prediction, no diffusion solver).
- Generated comparison plots in `analysis/comparison_plots/`.

## Architecture comparison

### Shared design choices
- Both models use the same **trunk hidden widths**: `128 → 256 → 128` with ReLU.
- Both are trained with Adam, cosine LR schedule, MSE on `keff`, 70 epochs, batch size 32.
- Parameter budgets are intentionally matched (~71.5k vs ~73.5k).

### MLP baseline (`6 → 128 → 256 → 128 → 36 → 1`)
| Block | Shape | Role |
|---|---|---|
| Input | 6 | geometry only (`b4c_r`, `cr_frac`, `fuel_r`, `enrichment`, `f_mod`, `water_r`) |
| Trunk | 6→128→256→128 | feature extraction |
| Latent | 128→36 | mirrors PEDS XS-correction dimensionality |
| Output head | 36→1 | **learned replacement** for low-fidelity solver mapping to `keff` |
| Parameters | **71,497** | |

### PEDS generator + solver
| Block | Shape | Role |
|---|---|---|
| Input | 12 (=6 geom + 6 φ features) | geometry + low-fidelity flux features |
| Trunk | 12→128→256→128 | feature extraction |
| XS head | (128+36)→36 | predicts log-ratio XS corrections per region/group |
| Physics | diffusion eigenvalue solver | maps corrected XS + geometry → `keff` |
| Parameters (generator only) | **73,524** | solver has no trainable NN params |

### Key differences
1. **Inputs**: MLP uses geometry only; PEDS also uses φ features and baseline XS in the head.
2. **Output space**: MLP predicts scalar `keff`; PEDS predicts 36 XS corrections then solves physics.
3. **Inductive bias**: PEDS enforces a physics pathway (XS → diffusion); MLP must learn the map end-to-end.
4. **Final layer**: MLP has an explicit `36→1` FC layer; PEDS replaces this with the NT diffusion solver.
5. **Same trunk width, not identical graph**: input dimension and head wiring differ even though parameter counts are close.

## Training evolution (validation)
- Final-epoch mean val MSE: MLP `0.02195` vs PEDS `6.199e-05` (~`354×` higher for MLP).
- Final-epoch mean val |Δk|: MLP `11625` pcm vs PEDS val |Δρ| `597` pcm (metrics differ slightly; see test table below).
- MLP validation error decreases slowly and plateaus around ~10–12k pcm |Δk|.
- PEDS validation error decreases to ~600 pcm |Δρ| and keeps improving through epoch 70.

Plots:
- `analysis/comparison_plots/train_val_mse_vs_epoch.png`
- `analysis/comparison_plots/val_error_vs_epoch.png`
- `analysis/comparison_plots/test_metrics_by_seed.png`
- `analysis/comparison_plots/test_delta_k_boxplot.png`

## Held-out test comparison (mean over 5 seeds)

| Metric | MLP baseline | PEDS | Ratio (MLP/PEDS) |
|---|---:|---:|---:|
| MSE(k) | 0.02033 | 6.396e-05 | 318× |
| Mean |Δk| (pcm) | 11132.0 | 574.9 | 19.4× |
| Median |Δk| (pcm) | 9240.2 | 405.5 | 22.8× |
| Mean |Δρ| (pcm) | 11118.8 | 611.1 | 18.2× |
| Mean fractional error | 0.1102 | — | — |
| Frac below 650 pcm (|Δk|) | 0.041 | 0.698 | 0.06× |

## Per-seed test |Δk| (pcm)

| Seed | MLP | PEDS |
|---|---:|---:|
| 0 | 10892.3 | 615.5 |
| 1 | 10857.7 | 532.0 |
| 2 | 9832.7 | 576.5 |
| 3 | 12378.7 | 548.8 |
| 4 | 11698.4 | 601.8 |

## Interpretation: is 1000 points enough for NN-only accuracy?
- **No, not at PEDS-level accuracy.** With the same splits and similar parameter count, the MLP baseline remains ~20× worse in test |Δk| and captures only ~4% of test points within 650 pcm vs ~70% for PEDS.
- The MLP does learn a coarse trend (val fractional error drops from ~0.94 to ~0.11), but it cannot match the fine keff accuracy achieved when physics structure is embedded.
- This supports the conclusion that **data volume alone is insufficient** here: the solver-informed PEDS pathway provides strong inductive bias that a geometry-only MLP cannot recover with 1000 training samples.

## Files
- Baseline metrics: `analysis/testset_results/test_metrics_all_runs.csv`
- PEDS metrics: `../PEDS/analysis/testset_results_dk/test_metrics_all_runs.csv`
- Per-seed epoch logs: `train_1000_seed_<seed>/epoch_metrics.csv`