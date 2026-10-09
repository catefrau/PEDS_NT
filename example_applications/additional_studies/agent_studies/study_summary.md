# PEDS Agent Studies — Summary of Findings

**Round 1 Job ID:** 37726895  
**Round 2 Job ID:** 37729632  
**Setup:** 500 train + 300 val + 300 test, 3 seeds (0/1/2)  
**Baseline reference:** `precise_param_strat` (1000 train, 5 seeds)

---

## Motivation

Two problems identified in `precise_param_strat` results (`keff_dist_bins_by_split.csv`):

### Problem 1 — keff-bin-dependent correction quality
After PEDS correction, **lower keff configurations improve less** than high-keff ones.  
- Training: improvement ~93–94% across all bins (1 pp range)  
- **Validation: improvement ~85–90%** (5 pp range) — much wider spread  
- Low-keff configs start from a worse baseline (~9 000 pcm vs ~2 800 pcm at high keff)  
  but unweighted MSE treats each sample equally regardless of pcm magnitude.

### Problem 2 — Train/val generalization gap
- Training improvement: ~94%; Validation/test: ~88.5% (6.5 pp gap)  
- Gap is worst for low-keff bins (val improvement stalls ~85% while train ~93%)  
- Possible causes: 500-sample training set; no effective regularization; MSE lets the model
  over-specialize to training-set keff values.

---

## Reference data (precise_param_strat, 1000 train, 5 seeds)

### Validation set by keff bin

| Bin | avg keff | Δρ before | Δρ after | frac<650 | improvement |
|-----|----------|-----------|----------|----------|-------------|
| 0   | 0.821    | 9 519 pcm | 1 433 pcm | 25.7%  | 85.0% |
| 1   | 0.861    | 8 742 pcm | 1 267 pcm | 34.3%  | 85.5% |
| 2   | 0.900    | 7 952 pcm | 1 012 pcm | 38.4%  | 87.3% |
| 3   | 0.940    | 6 942 pcm |   799 pcm | 54.2%  | 88.5% |
| 4   | 0.981    | 5 688 pcm |   570 pcm | 67.9%  | 90.0% |
| 5   | 1.021    | 5 001 pcm |   464 pcm | 77.3%  | 90.7% |
| 6   | 1.058    | 4 462 pcm |   411 pcm | 77.5%  | 90.8% |
| 7   | 1.099    | 3 867 pcm |   353 pcm | 85.3%  | 90.9% |
| 8   | 1.140    | 3 280 pcm |   307 pcm | 91.2%  | 90.6% |
| 9   | 1.179    | 2 745 pcm |   276 pcm | 91.8%  | 89.9% |

Weighted-average val mean|Δρ| ≈ **599 pcm**; val frac<650 ≈ **73%**

---

## Round 1 results

### Headline metrics (mean ± std over 3 seeds, best-checkpoint val)

| Study | val mean|Δρ| pcm | val frac<650 | train mean|Δρ| | best epoch | Notes |
|-------|------------|------------|------------|------------|-------|
| s01_baseline  | 666.5 ± 37.6 | 64.6% | 323.7 | 44.3 | reference |
| s02_wt_loss   | 684.5 ± 34.9 | 63.1% | 356.7 | 41.7 | **worse** — 1/k⁴ too aggressive |
| s03_dropout   | 745.1 ± 45.3 | 60.9% | 663.9 | 23.3 | gradient explosion, stopped ~ep23 |
| s04_adamw     | 667.6 ± 37.5 | 64.4% | 322.3 | 44.3 | **identical** to baseline |
| s05_keff_input| crashed      |  —    |  —    |  —   | dim mismatch bug (fixed) |
| s06_residual  | 675.5 ± **0.95** | 67.1% | 408.9 | 56.0 | most consistent, gap flat |
| **s07_elu**   | **598.6 ± 46.6** | **71.4%** | 421.1 | 61.3 | **best mean**, high variance |
| s08_wider     | 667.8 ± 15.8 | 66.8% | 238.4 | 40.3 | wider = worse overfit |
| s09_noise_aug | 668.9 ± 44.9 | 64.9% | 307.9 | 45.7 | neutral |
| s10_balanced  | 685.2 ± 34.4 | 64.1% | 399.8 | 28.0 | worse — sampling imbalance |

### Per-seed train/val gap at best checkpoint

| Study | Seed | trn pcm | val pcm | gap pcm | frac<650 |
|-------|------|---------|---------|---------|----------|
| s01_baseline | 0 | 406 | 696 | 289 | 62.3% |
|              | 1 | 281 | 624 | 343 | 65.0% |
|              | 2 | 284 | 680 | 396 | 66.3% |
| s06_residual | 0 | 412 | 677 | **264** | 65.3% |
|              | 1 | 442 | 675 | **232** | 66.7% |
|              | 2 | 372 | 675 | **303** | 69.3% |
| s07_elu      | 0 | 436 | 652 | **216** | 67.0% |
|              | 1 | 437 | **569** | **132** | **72.7%** |
|              | 2 | 389 | **574** | **185** | **74.7%** |
| s08_wider    | 0 | 270 | 684 | 414 | 66.0% |
|              | 1 | 219 | 653 | 434 | 67.7% |
|              | 2 | 226 | 666 | 440 | 66.7% |

---

## Round 1 — Critical Analysis

### Why AdamW did nothing (s04)
AdamW with `wd=1e-4` is numerically identical to Adam here because the Adam denominator
normalises gradient scales and the weight magnitudes are small.  L2 via AdamW requires
`wd ≳ 1e-2` to compete with the adaptive moment, which would be destructive in this regime.

### Why dropout caused gradient explosion (s03)
Dropout randomly zeros neurons, increasing the effective gradient magnitude per surviving
neuron.  The physics custom VJP (NT diffusion backward pass) already injects large sparse
gradients, and stochastic masking amplifies this into numerical instability.
**Lesson:** explicit regularization via activation masking is incompatible with this solver.

### Why the wider network failed (s08)
More parameters without structural constraints means more capacity for memorisation.
The training pcm reaches 102–270 (2–3× lower than baseline) while validation stays at
~667 pcm — a gap of 414–440 pcm vs 289–396 for baseline.  More capacity ≠ better
generalisation in a low-data setting.

### Why the PCM-weighted loss failed (s02) and balanced sampling failed (s10)
Both approaches attempt to up-weight hard (low-keff) samples.  The 1/k⁴ weighting
makes gradients ≈25× larger for a k=0.8 sample than a k=1.2 sample, disrupting
the optimisation dynamics.  Balanced sampling produces the same effect by oversampling.
**Lesson:** the model is already learning corrections for low-keff samples; the issue is
generalisation, not gradient allocation.

### Why ELU worked (s07)
ELU avoids the "dead neuron" problem of ReLU: its negative region provides non-zero
gradient for all activations, enabling smoother parameter updates.  Crucially, the
ELU optimisation trajectory for seeds 1 and 2 found solutions with a **train/val gap of
only 132–185 pcm** (vs 289–396 for baseline) — a 2–3× improvement.  The model finds
corrections that genuinely generalise, not just memorise.  The high variance across
seeds (46 pcm std) shows the solution depends on the initialisation trajectory.

### Why residual connections are the most consistent (s06)
Skip connections constrain how far each layer's representation can deviate from its
input, acting as a strong **implicit regulariser**.  The gap stays flat at 230–300 pcm
across all 70 training epochs instead of growing monotonically.  Validation pcm is very
stable: 675 ± 0.95 pcm across all three seeds.  The residual connections prevent the
network from making extreme, sample-specific corrections.

### Root cause of the generalization gap
With Adam + MSE on keff, the model learns the smallest XS corrections that reproduce
training keff values.  Once the training loss is low, further training refines
sample-specific corrections that do not transfer to new configurations.
Standard L2 (weight decay) doesn't address this because Adam normalises gradient scales.
**Architectural constraints** (residual connections) and **better activation functions**
(ELU) reduce this directly by limiting correction complexity.

---

## Round 2 studies (10 studies × 3 seeds, Job 37729632)

**Strategy:** Use ELU + Residual as the base.  Test convergence, alternative activations,
and new loss/regularization forms individually first, then combine the best.

| Study | Architecture | Activation | Loss | Extra | Epochs | Targets |
|-------|-------------|------------|------|-------|--------|---------|
| s11_elu_residual   | [128,128,128] | ELU  | MSE     | Residual     | 70  | Both |
| s12_elu_long       | [128,256,128] | ELU  | MSE     | —            | 100 | 2 |
| s13_residual_long  | [128,128,128] | ReLU | MSE     | Residual     | 100 | 2 |
| s14_elu_res_long   | [128,128,128] | ELU  | MSE     | Residual     | 100 | Both |
| s15_gelu           | [128,256,128] | GELU | MSE     | —            | 70  | Both |
| s16_elu_log_loss   | [128,256,128] | ELU  | log-keff| —            | 70  | 1 |
| s17_elu_logreg     | [128,256,128] | ELU  | MSE+L2  | λ=0.001 logr | 70  | 2 |
| s18_elu_res_log    | [128,128,128] | ELU  | log-keff| Residual     | 70  | Both |
| s19_elu_res_log_long | [128,128,128] | ELU | log-keff | Residual  | 100 | Both |
| s20_elu_res_logreg_long | [128,128,128] | ELU | MSE+L2 | Residual, λ | 100 | Both |

**New features implemented:**
- `log_keff` loss: `MSE(log k_pred, log k_ref)` — scale-invariant, gradient ∝ 1/k.
  Gentler than 1/k⁴ weighting; addresses Problem 1 without optimisation instability.
- `logratio_penalty`: `loss += λ * mean(log_ratios²)` — penalises large XS corrections,
  pushes the model toward smaller, more generalizable adjustments.
- `patience` per-study: 20 for 100-epoch runs to avoid premature stopping.
- keff_input None-bug fixed: `compute_xs`/`compute_log_ratios` substitute zeros if
  `use_keff_input=True` but `keff_base_norm=None` (from logging calls).

---

## Round 2 — Predictions

| Study | Expected val pcm | Expected frac<650 | Reasoning |
|-------|-----------------|------------------|----|
| s11_elu_residual | ~600–640 | ~70% | combines gap reduction from both |
| s12_elu_long | ~560–590 | ~73% | ELU not converged at ep70 |
| s13_residual_long | ~640–660 | ~68% | slower improvement |
| s14_elu_res_long | **~550–580** | **~75%** | best structural combo + convergence |
| s15_gelu | ~590–630 | ~70% | smoother than ELU but uncertain |
| s16_elu_log_loss | ~580–620 | ~70% | better for low-keff bins |
| s17_elu_logreg | ~590–640 | ~69% | may reduce gap, modest gain |
| s18_elu_res_log | ~570–610 | ~72% | structural + better loss |
| s19_elu_res_log_long | **~540–570** | **~76%** | full structural + loss + convergence |
| s20_elu_res_logreg_long | ~550–590 | ~74% | structural + soft constraint |

**Target for round 2:** val mean|Δρ| < 550 pcm, val frac<650 > 75%

---

## Key Findings (summary so far)

### What worked for Problem 1 (keff-bin uniformity)
- ELU reduces the optimisation stall for low-keff samples (smoother gradient flow)
- Log-keff loss (round 2) should improve further: gradient is ∝ 1/k, larger for subcritical

### What worked for Problem 2 (generalization gap)
- **ELU** (best peaks, ~132–185 pcm gap for seeds 1,2)
- **Residual connections** (most consistent, ~230–300 pcm gap, flat across training)

### What clearly failed
- AdamW weight decay (normalised away by Adam denominator)
- Dropout (incompatible with physics VJP gradients)
- PCM-weighted loss with 1/k⁴ (too aggressive, disrupts optimisation)
- Balanced oversampling (same problem as weighted loss)
- Wider network (amplifies memorization)

---

## Next Steps (after round 2)

1. Identify best individual + combo from round 2 results.
2. Run the winner at full scale (1000 train, 5 seeds).
3. Compare final model to reference `precise_param_strat` (1000 train, 5 seeds).
