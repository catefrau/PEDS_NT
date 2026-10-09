# Round 3 results and how to proceed

**Job:** 37731518 — 6 studies × 3 seeds, 500 train, all 100 epochs completed.
**Primary metric:** fixed test set (`TEST_SEED=0`), deployed (window-5) checkpoint.

---

## Headline

| Study | test mean\|Δρ\| | std | frac<650 | vs s12 (paired) | vs ReLU baseline |
|-------|------------|-----|----------|-----------------|------------------|
| s01_baseline (ReLU) | 727.8 | 48 | 0.654 | — | — |
| s12_elu_long (prev best) | 595.8 | 33 | 0.730 | — | −132 |
| r01_ref (ELU, 2e-4, window-5) | 603.2 | 36 | 0.732 | **+7** | −125 |
| **r02_lr_hi (ELU, 4e-4)** | **570.8** | **34** | **0.749** | **−25** | **−157** |
| r03_lr_lo (1e-4) | 631.1 | 42 | 0.712 | +35 | −97 |
| r04_swa_flat | 630.2 | 37 | 0.706 | +34 | −98 |
| r05_regime_feat | 616.0 | 60 | 0.713 | +20 | −112 |
| r06_small [64,128,64] | 642.5 | 27 | 0.710 | +47 | −85 |

`lr_max=4e-4` is the first lever that **stacked on ELU**. The three-point sweep is monotonic: 1e-4 ≪ 2e-4 < 4e-4. Higher LR lowered **both** train error (~30 pcm) and test error (~25 pcm) — it is a better optimiser setting, not more overfitting.

r01 vs s12 (+7 pcm) is the cost of the more conservative window-5 checkpoint vs the old EMA. Worth paying: selection is no longer distorted by epoch-1's 3900 pcm.

---

## What each arm actually did

**SWA.** Dead on annealed runs (r01/r02/r05/r06): −4 to +4 pcm, noise. Helped only when the point checkpoint was bad:
- r03 seed 0: point picked at epoch 60, SWA used later weights → −22 pcm
- r04 (hold at 5e-5): −5 to −18 pcm, but the hold schedule itself is ~34 pcm worse than annealing, so SWA is rescuing a worse trajectory

Drop SWA from the annealed recipe.

**Regime features (r05).** Highest variance of the round (std 60 pcm). Seed 1 looked great (548); seed 2 collapsed (663). Train error dropped to 297–347 pcm and the gap opened to 201–357 pcm — extra features with 500 samples increased memorisation. Drop.

**Smaller net (r06).** More consistent (std 27) and worse (642). Capacity is not the problem; 73k params is fine. Drop.

**Problem 1 (low-keff bins).** Untouched. Fractional improvement still 87.3% (low) vs 90.6% (high), spread 3.4 pp — identical to s12. Higher LR helped every bin by roughly the same number of pcm.

**Gap.** Still 189–265 pcm on r02. Same floor as every other ELU run.

**Seed effect, still the largest remaining number.** Averaged over r01–r06: seed 0 = 644, seed 1 = 580, seed 2 = 623. A 64 pcm swing from *which 500 samples*, vs 25 pcm from the best hyperparameter change this round.

---

## Current training-set selection (what we already do)

`derive_split_indices_by_params` already:
1. Equal coverage of `fuel_r` quantile bins
2. Matches the **natural keff mix** across train/val/test (does not flatten low-keff)
3. Nested secondary-param diversity inside each (fuel_r, keff) cell
4. Locks the test set (`TEST_SEED=0`)

The 2648-sample pool is not badly duplicated: only 4.2% of points have a 6D neighbour closer than 0.08. Current 500-sample trains are already reasonably spaced (median 1-NN ≈ 0.24).

A greedy **maximin within keff bins**, holding the current keff histogram fixed and locking the test set, does this:

| | median 1-NN | p10 1-NN | worst hole (max dist from leftover pool) |
|--|--|--|--|
| current seed 0 | 0.244 | 0.163 | 0.554 |
| current seed 1 (best performer) | 0.246 | 0.166 | **0.487** |
| maximin, same keff counts | 0.271 | 0.190 | **0.440** |

Seed 1, the best training draw, already has the smallest hole. That is a weak but consistent signal that **coverage holes, not keff mix**, are what make seed 0 worse. Maximin shrinks the worst hole by ~20% without changing keff-range coverage at all (same 10-bin counts).

Overlap of maximin with seed-0 train is only 109/500, so it is a genuinely different 500, not a shuffle.

---

## How to proceed (recommended sequence)

Stay at 500 samples for **one more experiment**, then scale to 1000.

### 1. Freeze the recipe (no more architecture/loss/SWA/features at 500)

```
ELU, [128, 256, 128], MSE, Adam
lr_max = 4e-4, lr_min = 5e-6
warmup = 7 epochs (fixed)
decay_epochs = 100, steps_per_epoch = 16 (pinned)
checkpoint = 5-epoch window mean
no SWA, no residual, no extra features, no reweighting
```

This is r02. Everything else from rounds 1–3 is measured inert or harmful.

### 2. One data-selection study (the remaining 500-sample shot that can still move the needle)

**What:** greedy maximin in the 6D normalised parameter space, **inside each keff bin**, with the keff histogram locked to the current train histogram (so the 0.80–1.20 coverage is unchanged). Test set stays the locked 300. Val can stay seed-coupled or also be re-picked from leftovers after train; test comparison is what matters.

**Why it is different from previous failed “data” ideas:** s02/s10 changed how much low-keff is *penalised*. This changes *which geometries* occupy the 500 slots, without touching the keff mix. It directly attacks the 64 pcm seed-draw effect.

**Design, 3 seeds, cheap:**
- Seed still randomises init
- Train set is either (a) one deterministic maximin, 3 inits — cleanly separates init vs data — or (b) 3 jittered maximins (random first point per bin) to check robustness. Prefer (a) plus one jittered arm if you can spare it; if only 3 runs, do (a).

**Expected:** 15–40 pcm on the fixed test set if coverage holes are the seed-0 tax; ~0 if the seed effect is something the geometry metric does not capture. Either answer is useful. If it matches seed 1 (545 pcm) from a *single* deterministic 500, we have a selection rule to carry to 1000.

**Do not flatten the keff histogram.** That would change coverage of the range, which you asked to keep, and it is the same family of intervention as s10 (which hurt).

### 3. Then scale to 1000 — not before

With r02 (+ maximin if it wins), `STEPS_PER_EPOCH_REF=16` already pinned, so 1000 samples will not silently double the optimiser budget. Expected from the earlier scaling argument: test **~500–530 pcm** at 1000, gap ~150.

Do not spend another round on `lr_max=8e-4`, GELU, residuals, or more features at 500. The monotonic LR sweep could in principle go higher, but r02 is still barely moving at epoch 100 (val 652→649 over the last 20 epochs) and further LR is more likely to re-introduce the oscillation we just tamed.

---

## What not to retry

| Mechanism | Status |
|-----------|--------|
| AdamW, dropout, 1/k⁴, oversampling, wider ReLU, GELU, log-keff loss, log-ratio L2, residual+ELU | R1/R2: inert or harmful |
| SWA on annealed cosine | R3: 0 ± 4 pcm |
| SWA + hold schedule | R3: hold itself −34 pcm |
| Regime features | R3: +20 mean, std 60, larger gap |
| Smaller trunk | R3: +47 pcm |
| Window-5 vs old EMA | Keep window-5 (honesty); do not go back |
