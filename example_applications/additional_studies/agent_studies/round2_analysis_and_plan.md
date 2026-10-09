# PEDS Agent Studies — Round 2 Analysis and Round 3 Plan

**Rounds analysed:** R1 = job 37726895 (s01–s10), R2 = job 37729632 (s11–s20)
**Setup:** 500 train / 300 val / 300 test, 3 seeds, `PEDS_agent.py`
**Status:** no new runs submitted — plan for review first.

---

## 0. Methodological correction applied to this analysis

Two problems with how R1 was read, now fixed:

1. **`val_mean_pcm` was read as the raw per-epoch minimum.** That cherry-picks a noisy
   dip. The model actually deployed is chosen by an EMA-smoothed criterion, recorded in
   `study_summary.csv`. For `s12_elu_long` seed 0 the raw minimum is 624 pcm but the
   deployed checkpoint is 671 pcm — a 47 pcm optimistic bias. All numbers below use the
   deployed checkpoint.

2. **The validation split moves with the seed.** In `PEDS_agent.py`:
   `TRAIN_SEED = VAL_SEED = SEED`, while `TEST_SEED = 0` is fixed. So changing seed changes
   the training set, the validation set *and* the initialisation together. Validation
   numbers are therefore not comparable across seeds. **The fixed 300-sample test set is
   the only unbiased cross-study metric** and is used as primary below.

---

## 1. Results on the fixed test set (mean ± std over 3 seeds)

| Study | test mean\|Δρ\| | std | test frac<650 | per-seed test pcm |
|-------|------------|-----|-----------|-------------------|
| s01_baseline (ReLU, MSE) | 727.8 | 48.1 | 0.654 | 720 / 684 / 779 |
| s06_residual | 699.9 | 29.0 | 0.666 | 687 / 679 / 733 |
| s13_residual_long | 688.1 | 42.4 | 0.681 | 660 / 667 / 737 |
| s15_gelu | 665.4 | 41.8 | 0.707 | 655 / 630 / 711 |
| s18_elu_res_log | 640.6 | 29.2 | 0.696 | 669 / 611 / 642 |
| s11_elu_residual | 636.8 | 37.2 | 0.704 | 674 / 599 / 637 |
| s16_elu_log_loss | 633.3 | 45.5 | 0.714 | 680 / 589 / 632 |
| s07_elu | 625.9 | 41.9 | 0.730 | 669 / 585 / 624 |
| s17_elu_logreg | 625.6 | 44.5 | 0.729 | 667 / 578 / 632 |
| s19_elu_res_log_long | 624.7 | 36.0 | 0.713 | 665 / 595 / 615 |
| s20_elu_res_logreg_long | 610.6 | 35.3 | 0.724 | 647 / 576 / 609 |
| s14_elu_res_long | 611.2 | 34.7 | 0.726 | 649 / 581 / 604 |
| **s12_elu_long** | **595.8** | **33.0** | **0.730** | 632 / 568 / 587 |

### Paired deltas vs baseline (same seed ⇒ same train set and init)

| Study | seed0 | seed1 | seed2 | mean |
|-------|-------|-------|-------|------|
| s06_residual | −33 | −5 | −46 | **−28** |
| s13_residual_long | −60 | −17 | −42 | **−40** |
| s15_gelu | −65 | −54 | −68 | **−62** |
| s18_elu_res_log | −51 | −73 | −137 | **−87** |
| s11_elu_residual | −46 | −85 | −142 | **−91** |
| s16_elu_log_loss | −41 | −95 | −148 | **−94** |
| s07_elu | −52 | −99 | −155 | **−102** |
| s17_elu_logreg | −53 | −106 | −148 | **−102** |
| s19_elu_res_log_long | −55 | −89 | −165 | **−103** |
| s14_elu_res_long | −71 | −103 | −175 | **−117** |
| s20_elu_res_logreg_long | −74 | −108 | −170 | **−117** |
| s12_elu_long | −88 | −116 | −192 | **−132** |

---

## 2. Why the two problems were not fixed

### 2.1 ELU accounts for essentially the entire gain; nothing stacks on top of it

- ELU alone: **−102 pcm**
- ELU + 100 epochs: −132 (the only genuine addition, +30)
- ELU + log-ratio L2 penalty: −102 → **exactly zero effect**
- ELU + log-keff loss: −94 → **zero effect, slightly negative**
- ELU + residual (narrower trunk): −91 → **worse than ELU alone**

Both R2 interventions aimed at the two stated problems (log-keff loss for Problem 1,
log-ratio penalty for Problem 2) produced **no measurable change**. The loss function is
not the bottleneck.

*(Caveat: residual arms use `[128,128,128]` while non-residual use `[128,256,128]`, so
"residual" is confounded with "narrower". The combination loses either way, so there is no
reason to pursue it, but the clean attribution is untested.)*

### 2.2 The generalization gap is pinned at ~220 pcm and will not move

| Study | train@ckpt | test | gap |
|-------|-----------|------|-----|
| s01_baseline | 236 | 728 | 492 |
| s06_residual | 379 | 700 | 321 |
| s07_elu | 409 | 626 | **217** |
| s12_elu_long | 371 | 596 | **225** |
| s14_elu_res_long | 390 | 611 | **221** |
| s20_elu_res_logreg_long | 386 | 611 | **225** |

Every ELU variant lands on the same gap, 217–225 pcm, regardless of loss, penalty,
residual connections or run length. This is a hard floor, not something the four
mechanisms tested can reach.

**Half the apparent gap reduction was underfitting, not generalization.**
Baseline → s12: the gap fell 492 → 225 (−267). Of that, train error *rose* by 135 pcm
(236 → 371) and test error fell by 132 pcm (728 → 596). Almost exactly 50/50. We removed
half the gap by making the model fit its training data worse.

### 2.3 The dominant source of variance is *which 500 samples you train on*

On the **fixed** test set, averaged over all 13 studies:

| seed (⇒ train set) | mean test pcm |
|---|---|
| seed 0 | 667 |
| seed 1 | 611 |
| seed 2 | 657 |

Seed 0 is the worst arm in 8 of 10 R2 studies and 3 of 4 R1 studies. Since the test set is
identical, this cannot be a validation-split artifact — a different choice of 500 training
samples is worth **56 pcm**, versus **132 pcm** for every architectural change we made
combined. Confirmed that the splits are *not* differently difficult at baseline:
pre-PEDS val error is 5414 / 5355 / 5401 pcm and `frac(keff<0.95)` is 0.273 for all three
seeds. The splits are equally hard; the *models* trained on them differ.

**Conclusion: the model is data-limited, not regularization-limited.** It fits 500 samples
to 371 pcm but tests at 596 pcm. Adding constraints (dropout, weight decay, L2 on
corrections, residual paths) cannot manufacture information that is not in 500 samples.
This is why nothing closed the gap.

### 2.4 Problem 1 is a consequence of the error profile, not of loss allocation

Per keff bin, validation, averaged over 3 seeds:

| bin | keff | before | baseline after | impr% | s12 after | impr% |
|-----|------|--------|-------|-------|-------|-------|
| 0 | 0.820 | 10312 | 1641 | 84.1 | 1453 | 85.9 |
| 1 | 0.860 | 8827 | 1278 | 85.5 | 1212 | 86.3 |
| 2 | 0.900 | 8107 | 1183 | 85.4 | 898 | 88.9 |
| 3 | 0.940 | 7077 | 861 | 87.8 | 736 | 89.6 |
| 4 | 0.980 | 5464 | 736 | 86.5 | 626 | 88.5 |
| 5 | 1.020 | 5011 | 559 | 88.9 | 483 | 90.4 |
| 6 | 1.060 | 4487 | 444 | 90.1 | 358 | 92.0 |
| 7 | 1.100 | 3917 | 428 | 89.1 | 360 | 90.8 |
| 8 | 1.140 | 3232 | 395 | 87.8 | 331 | 89.8 |
| 9 | 1.185 | 2803 | 285 | 89.8 | 252 | 91.0 |

low bins 0–2: 85.0% → 87.0%  high bins 3–9: 88.6% → 90.3%
**spread 3.6 pp → 3.3 pp: essentially unchanged.**

The model delivers a roughly **constant fractional** error reduction (86–91%) in every
bin. The after-error is therefore proportional to the before-error, and low-keff bins end
worse purely because they start ~3.7× worse. This is not a training-allocation failure —
which is exactly why up-weighting (s02, 1/k⁴) and oversampling (s10) both *hurt*. They
changed how much the model is penalised for low-keff samples, when the binding constraint
is how well it can model them at all.

To move the low bins you must change the **fractional** correction quality there, which
requires more information (data or features), not a different loss weighting.

---

## 3. The learning-rate schedule problem

`steps_per_epoch = ceil(500/32) = 16`, and:

```python
decay_steps  = decay_epochs * steps_per_epoch
lr_schedule  = optax.warmup_cosine_decay_schedule(
    init_value=lr_min, peak_value=lr_max,
    warmup_steps=int(0.1 * decay_steps),   # ← couples warmup to run length
    decay_steps=decay_steps, end_value=lr_min)
```

In optax, `decay_steps` is the **total** schedule length; the cosine runs for
`decay_steps − warmup_steps`. R2 scaled `decay_epochs` with `epochs`, which changed three
things simultaneously:

| epoch | LR @70ep | LR @100ep | ratio |
|-------|---------|----------|-------|
| 10 | 1.99e-04 | 2.00e-04 | 1.01× |
| 30 | 1.43e-04 | 1.77e-04 | 1.24× |
| 40 | 9.52e-05 | 1.51e-04 | 1.59× |
| 50 | 4.96e-05 | 1.19e-04 | **2.41×** |
| 60 | 1.69e-05 | 8.56e-05 | **5.07×** |
| 70 | 5.00e-06 | 5.37e-05 | **10.75×** |

- **Warmup drifted** 7 → 10 epochs. Warmup exists to stabilise Adam's moment estimates;
  it should be a fixed number of steps, not 10% of an arbitrary total.
- **Mid/late training LR rose up to 10×.**
- **Integrated LR (total optimisation budget) rose 43%**: 0.1148 → 0.1640.

So the +30 pcm that `s12_elu_long` gained over `s07_elu` **cannot be attributed to
duration**. It could be the larger budget, the higher late LR, the longer anneal, or any
mix. Supporting evidence that these runs are anneal-limited rather than converged:
`best_epoch` was 99 or 100 for several 100-epoch arms — still improving when the LR hit
its floor.

`lr_max` has never been varied in any of the 20 studies.

### Two separate questions, two different controls

1. *Does duration alone help?* Hold the integrated budget fixed while extending epochs.
   Computed values that keep the budget at 0.1148 with a fixed 7-epoch warmup:

   | epochs | lr_max | integrated |
   |--------|--------|-----------|
   | 70 | 2.0e-4 | 0.1148 |
   | 100 | 1.4e-4 | 0.1160 |
   | 150 | 9.3e-5 | 0.1176 |

2. *Is the budget itself too small?* Sweep `lr_max` at fixed duration. Never tested.

### A further trap for the data-scaling study

`steps_per_epoch` is derived from `train_size`, so at 1000 samples one epoch is 32 steps,
not 16. Running 500 and 1000 samples "for the same number of epochs" silently gives the
larger run **twice the optimiser steps and twice the integrated LR**. Any data-scaling
comparison must **hold total steps fixed**, not epochs, or the data effect is confounded
with the optimisation budget.

---

## 4. Round 3 plan

Ordered by expected value. Nothing submitted yet.

### Tier 1 — Scale the training set (highest confidence)

**What:** best current recipe (ELU, `[128,256,128]`, MSE, Adam) at
`train_size ∈ {500, 1000, 2000}`, **holding total optimiser steps constant** across arms
so the LR trajectory is identical in step space.

**Why it is different:** all 20 studies so far held `train_size = 500`. The evidence says
this is the binding constraint: the gap is pinned at 220 pcm across six different ELU
variants, and merely changing *which* 500 samples are used moves the fixed-test result by
56 pcm — 42% of the total gain from all architectural work combined. We have been
optimising the wrong axis.

**Expected:** the existing `precise_param_strat` reference (1000 train, ReLU/MSE) reached
~599 pcm val where our 500-train ReLU baseline reaches ~686 val, i.e. doubling data was
worth ~87 pcm with the *old* recipe. Applying that to the ELU recipe:

| train_size | expected test pcm | expected gap |
|-----------|------------------|--------------|
| 500 (ref) | 596 | 225 |
| 1000 | 500–520 | ~150 |
| 2000 | 440–470 | ~110 |

**Cost:** the NT solver dominates runtime and scales with sample count; the 2000-sample arm
is ~4× the per-epoch cost of 500. Needs a wall-time and node budget before submitting.

### Tier 2 — Fix and then actually tune the LR schedule (cheap, never done)

**What:** (a) fix `warmup_steps` to a constant 112 steps (7 epochs) instead of 10% of
total; (b) sweep `lr_max ∈ {1e-4, 2e-4, 4e-4}` at fixed duration; (c) lower `lr_min`
5e-6 → 1e-6 for a deeper final anneal; (d) optionally cosine to `lr_min` by 85% of the run
then hold, giving a genuine low-LR settling phase.

**Why it is different:** `lr_max` is the one high-leverage hyperparameter never varied,
and the current coupling means "more epochs" is not a clean knob. This also de-risks
Tier 1 — we should not spend 4× compute on 2000 samples at an untuned learning rate.

**Expected:** 20–50 pcm on its own, plus correct attribution of the duration effect.

### Tier 3 — Weight averaging (SWA) instead of picking one checkpoint

**What:** average weights over the last ~20 epochs (or the best-k epochs) and deploy that,
rather than selecting a single epoch by EMA-smoothed validation.

**Why it is different:** this changes *which point in weight space is deployed*, not the
loss or the architecture. It directly targets the oscillation ELU introduced: the val curve
swings ±100 pcm epoch to epoch, and the raw minimum sits ~45 pcm below the EMA-selected
checkpoint, so real performance is being lost to noisy single-epoch selection. It also
pairs naturally with the long low-LR tail in Tier 2(d).

**Expected:** 30–60 pcm, and a substantial cut in seed-to-seed spread.

### Tier 4 — Quantify seed ensembling (near-zero cost)

**What:** average the predictions of the 3 existing per-seed models on the fixed test set.

**Why it is different:** it exploits the variance we just diagnosed rather than fighting it.
Since the 3 models are trained on different 500-sample subsets, their errors are partly
independent.

**Expected:** ~540–560 pcm from a 596 pcm mean. Requires 3× inference at deployment, so
this is a decision for you, not automatically a win. Note it needs a small re-evaluation
script: per-sample logs are currently written for train/val only, and val splits differ by
seed, so the fixed test set has to be re-scored.

### Tier 5 — Physics features for Problem 1 (only untested mechanism)

**What:** add features that describe *why* the diffusion baseline fails, and how that
failure scales: the baseline keff itself (s05, whose dimension bug is now fixed but which
never actually ran), leakage fraction, buckling / migration area from the baseline solve,
and group-wise flux ratios or a spectral index.

**Why it is different:** s02 and s10 changed how heavily low-keff samples are *penalised*
and both hurt. This changes what the model *knows* about them. Section 2.4 shows the model
applies a near-constant fractional correction; a regime indicator is the kind of input that
could let the fraction itself vary with keff.

**Expected:** genuinely uncertain, 0–60 pcm concentrated in bins 0–2. Lowest confidence of
the five, but it is the only remaining untested mechanism for Problem 1.

### Do not retry — measured as inert or harmful

| Mechanism | Evidence |
|-----------|----------|
| AdamW weight decay | numerically identical to Adam (−28 vs −28 paired) |
| Dropout | gradient explosion vs physics custom-VJP; stopped ~ep23 |
| 1/k⁴ PCM weighting | +18 pcm worse than baseline |
| Balanced oversampling | +19 pcm worse than baseline |
| Wider ReLU trunk | gap 414–440 pcm, worst overfitting observed |
| log-keff loss | −94 vs −102 for ELU alone: no effect |
| log-ratio L2 penalty | −102 vs −102: exactly no effect |
| Residual + ELU | −91 vs −102, and −117 vs −132 at 100 epochs: consistently worse |
| GELU | −62 vs −102 for ELU: clearly worse |

---

## 5. Recommended sequence

1. **Tier 2 first** (cheap, fast, fixes a confound): fixed warmup + `lr_max` sweep at
   500 samples. Establishes the correct optimiser settings.
2. **Tier 1 next** with those settings, holding total steps fixed. This is the main event.
3. **Tier 3 folded into Tier 1** — SWA is nearly free to add to the same runs.
4. **Tier 4** as a cheap post-hoc analysis of whatever comes out.
5. **Tier 5** only if Problem 1's bin spread still matters after data scaling.

Target after Tiers 1–3: test mean\|Δρ\| **≈ 450–500 pcm**, frac<650 **≈ 0.80**, gap ≈ 120–150 pcm,
versus today's 596 pcm / 0.730 / 225 pcm.
