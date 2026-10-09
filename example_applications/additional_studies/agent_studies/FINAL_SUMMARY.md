# PEDS Agent Studies — Final Summary

The per-run folders other than `r13_1k_elu6e4`, `r07_lr_6e4`, and `r11_lr6_warm14` were removed on 6 Oct 2026. Snapshots and the per-seed metrics are in `code_snapshots/`. The code-by-code record is `CODE_VERSIONS_AND_OUTCOMES.md`.

**Campaign:** 2026-08-18 → 2026-08-20  
**Code:** `modules/PEDS_agent.py` (main `PEDS.py` untouched)  
**Logs:** `modules/RUNS/agent_studies/`  
**Primary metric:** mean |Δρ| (pcm) on a **locked 300-sample test set** (`TEST_SEED=0`)  
**Secondary:** fraction of samples with |Δρ| < 650 pcm; train→test gap

This document is the end-to-end record: the two problems we started from, every round of experiments, the frozen recipe, the 1000-sample scale-up, and what is still open.

---

## 1. Starting point

From `precise_param_strat` (original PEDS: ReLU `[128,256,128]`, MSE, Adam, **1000 train / 5 seeds**):

| Split | mean \|Δρ\| after PEDS | frac < 650 | improvement vs diffusion baseline |
|-------|------------------------|------------|-----------------------------------|
| Train | ~299 pcm | 88.6% | ~94% |
| Val   | ~597 pcm | 69.8% | ~88.5% |
| Test  | **611 ± 38 pcm** | **69.0%** | ~88.5% |

Two problems:

1. **Keff-bin quality.** After correction, low-keff configurations remain much worse in *absolute* pcm (val bin 0: ~1430 pcm vs bin 9: ~276 pcm). Improvement is a nearly **constant fraction** (~85–91%) of the (much larger) diffusion baseline error, so the low-keff tail never catches up.
2. **Generalization gap.** Train improves ~94%; val/test stall ~88.5%. The gap is worse in bins 0–2 (val ~85% vs train ~93%).

The 500-sample screening baseline (same architecture, 3 seeds) was **728 ± 48 pcm** test, frac 0.654 — worse because of half the data, as expected.

---

## 2. Frozen recipe (what to keep)

```
activation     ELU
trunk          [128, 256, 128]
loss           MSE on keff
optimiser      Adam
lr_max         6e-4
lr_min         5e-6
warmup         7 epochs (fixed step count, not 10% of the run)
epochs         100
decay_epochs   100
steps/epoch    16  (pinned; independent of train_size)
batch          32
checkpoint     5-epoch trailing-window mean of val mean|Δρ|
split          param-stratified, TEST locked (TEST_SEED=0)
```

This is study **`r07_lr_6e4`** at 500 samples, then **`r13_1k_elu6e4`** at 1000.

---

## 3. Headline numbers (locked test set)

| Stage | Train N | Recipe | test mean\|Δρ\| | std | frac<650 | vs original 1000-train ReLU |
|-------|---------|--------|-----------------|-----|----------|------------------------------|
| Original `precise_param_strat` | 1000 | ReLU, 2e-4, 5 seeds | **611** | 38 | 0.690 | — |
| Screening baseline | 500 | ReLU, 2e-4 | 728 | 48 | 0.654 | — |
| Best 500-sample (r07) | 500 | ELU, 6e-4 | **558** | 27 | 0.743 | −53 pcm *with half the data* |
| **Scale-up (r13)** | **1000** | **ELU, 6e-4** | **509 ± 28** | 28 | **0.761** | **−102 pcm** |

r13 per seed (1000 train / 500 val / 300 test, 100 epochs):

| Seed | best epoch | train pcm* | val pcm | **test pcm** | test frac<650 |
|------|------------|------------|---------|--------------|---------------|
| 0 | 99 | 443 | 477 | 541 | 0.720 |
| 1 | 100 | 384 | 494 | 496 | 0.783 |
| 2 | 98 | 383 | 518 | 489 | 0.780 |
| **mean** | | **403** | **496** | **509** | **0.761** |

\*Train pcm is measured on the 16×32 = 512 samples seen that epoch (`steps_per_epoch` is pinned at 16). It is not a full-1000-set number; val and test are full-split evaluations.

**Val vs original 1000-train ReLU:** 496 vs 597 pcm (−101 pcm), frac 0.766 vs 0.698.

**Gap (test − train):** 492 pcm (ReLU 500) → 236 pcm (ELU 500) → **105 pcm (ELU 1000)**. Doubling the training set more than halved the remaining gap. Val (496) and test (509) now sit close together; the old 6-point train/val improvement split is largely gone at this scale.

---

## 4. Campaign map (what was tried)

All screening runs: 500 train / 300 val / 300 test, 3 seeds, same locked test.

### Round 1 — one change at a time

| ID | Change | Test vs ReLU 500 | Verdict |
|----|--------|------------------|---------|
| s01 | ReLU baseline | 728 pcm | reference |
| s02 | 1/k⁴ PCM-weighted loss | worse | drop |
| s03 | Dropout 0.15 | exploded | drop — incompatible with physics VJP |
| s04 | AdamW wd=1e-4 | identical to s01 | drop — Adam normalises it away |
| s05 | Baseline keff as input | crashed (dim bug) | later fixed; feature idea failed in r05 |
| s06 | Residual `[128,128,128]` | small, very consistent | implicit regulariser; lost when combined with ELU |
| **s07** | **ELU** | **626 (−102)** | **first real win** |
| s08 | Wider `[256,512,256]` | worse gap (414–440) | drop — more memorisation |
| s09 | Input noise σ=0.01 | neutral | drop |
| s10 | Oversample low-keff 2× | worse | drop — same family as s02 |

### Round 2 — stack on ELU

| ID | Change | vs ELU | Verdict |
|----|--------|--------|---------|
| s12 | 100 epochs (vs 70) | −30 | keep longer runs; **confounded with LR schedule** (see §6) |
| residual+ELU, log-keff loss, log-ratio L2, GELU | ~0 or worse | drop |

### Round 3 — optimiser schedule + failed extra ideas

| ID | Change | Test | vs s12 | Verdict |
|----|--------|------|--------|---------|
| r02 | `lr_max=4e-4`, warmup fixed 7 ep, window-5 ckpt | **571** | **−25** | first lever that *stacked* on ELU |
| r01 | 2e-4 + window-5 | 603 | +7 | window-5 is more honest than EMA; keep it |
| r03 | 1e-4 | 631 | worse | |
| r04–r06 | SWA / regime features / smaller net | worse or 0 | drop |

### Round 4 — finish the LR map + data selection

| ID | Change | Test | vs r02 | Verdict |
|----|--------|------|--------|---------|
| **r07** | **`lr_max=6e-4`, warmup 7** | **558** | **−13** | **peak of the LR sweep** |
| r08 | 8e-4, warmup 7 | 564 | −7 | past the peak (seed 2 +24) |
| r09 | 4e-4, warmup 3 | 575 | +4 | shorter warmup hurts |
| r10 | 4e-4, warmup 14 | 584 | +13 | longer warmup at 4e-4 hurts |
| r11 | 6e-4 × warmup 14 | 558 | −13 | same as r07; extra warmup not needed at the peak |
| r12 | maximin-within-keff-bin 500 | 587 | n/a | did not beat param-strat; init-only std still ~26 pcm |

LR sweep is no longer monotonic: **1e-4 ≪ 2e-4 < 4e-4 < 6e-4 ≤ 8e-4 (down).** Warmup 7 is correct at 6e-4.

### Round 5 — scale-up

`r13_1k_elu6e4`: r07 recipe, **1000 train / 500 val / 300 test**, 3 seeds, `steps_per_epoch` still 16.  
**509 ± 28 pcm test, frac 0.761.**

---

## 5. The two original problems, after the campaign

### Problem 2 — generalisation gap: largely closed at 1000 samples

| Setup | train pcm | val pcm | test pcm | test−train |
|-------|-----------|---------|----------|------------|
| ReLU 500 | 236 | 686 | 728 | 492 |
| ELU 500 (r07) | 322 | 552 | 558 | 236 |
| ELU 1000 (r13) | 403* | **496** | **509** | **105** |
| Original ReLU 1000 | 299 | 597 | 611 | 312 |

The 500-sample ELU gap of ~220 pcm was a **data-volume floor**, not a regularisation floor. That is why dropout, AdamW, log-ratio L2, residual+ELU, and maximin selection did not close it. Doubling the training set did.

Val and test now agree (496 vs 509). The remaining ~100 pcm is in the same range as seed-to-seed scatter (~28 pcm std) plus the fact that each epoch only updates on 512 of the 1000 points.

Seed 0 is still the weakest draw (test 541 vs 489–496). That pattern survived every architecture; it is which 500/1000 geometries you get, not init. Maximin (r12) did not remove it.

### Problem 1 — low-keff bins: improved in pcm, not in *fractional* terms

Validation set, r13 (3-seed mean), compared with original ReLU 1000-train val:

| Bin | keff | Δρ before | ReLU 1000 after | impr | **ELU 1000 after** | impr |
|-----|------|-----------|-----------------|------|-------------------|------|
| 0 | 0.82 | 9689 | 1433 | 85.0% | **1109** | 88.6% |
| 1 | 0.86 | 8708 | 1267 | 85.5% | **1037** | 88.1% |
| 2 | 0.90 | 7983 | 1012 | 87.3% | **827** | 89.6% |
| 3 | 0.94 | 7034 | 799 | 88.5% | **679** | 90.3% |
| 4 | 0.98 | 5604 | 570 | 90.0% | **474** | 91.5% |
| 5 | 1.02 | 4976 | 464 | 90.7% | **416** | 91.6% |
| 6 | 1.06 | 4470 | 411 | 90.8% | **294** | 93.4% |
| 7 | 1.10 | 3915 | 353 | 90.9% | **309** | 92.1% |
| 8 | 1.14 | 3231 | 307 | 90.6% | **268** | 91.7% |
| 9 | 1.18 | 2748 | 276 | 89.9% | **248** | 91.0% |

Low bins 0–2: 85.9% → **88.8%**. High bins 3–9: 90.2% → **91.7%**. Spread **4.3 pp → 2.9 pp**.

Absolute low-keff error is better (bin 0: 1433 → 1109 pcm) but still ~4× bin 9. The model still applies a roughly **constant fractional** correction. Reweighting (s02, s10) tried to change that by force and made the average worse. Extra regime features (r05) increased memorisation. This is the main remaining scientific issue: the diffusion baseline is 3–4× worse at low keff, and PEDS inherits that shape.

Val frac<650 by bin still runs ~0.33 (bin 0) → ~0.94 (bins 6–9). Overall val frac 0.766 vs original 0.698.

---

## 6. Lessons that cost compute to learn (do not retry)

| Mechanism | Why it failed |
|-----------|----------------|
| AdamW / L2 weight decay | Adam’s denominator cancels small `wd` |
| Dropout | Stochastic masks × sparse physics VJP → explosions |
| 1/k⁴ loss, low-keff oversampling | Gradient allocation was not the bottleneck |
| Wider ReLU | More capacity, more memorisation |
| GELU, residual+ELU, log-keff MSE, log-ratio L2 | Zero or negative vs ELU+MSE |
| SWA on annealed cosine | Tail iterates do not move; 0 ± 4 pcm |
| SWA + constant-LR hold | Hold trajectory itself is worse |
| Regime features (k_base + spectral ratios) | Extra channels, 500 samples → larger gap |
| Smaller trunk `[64,128,64]` | Underfit (−47 pcm) |
| Maximin geometry selection | Same keff mix, no test gain; init variance still ~26 pcm |
| Scaling `decay_epochs` with `epochs` | Silently raised mid-run LR up to 10× and integrated LR 43%. Always pin `steps_per_epoch` and a **fixed** warmup length. |
| EMA checkpoint (`α=0.1` from epoch 1) | Stays biased by the 3000+ pcm start; window-5 mean is the honest selector |

**What actually moved the needle, in order:**

1. ELU instead of ReLU (−102 pcm at 500 samples)
2. `lr_max` 2e-4 → 4e-4 → **6e-4** (−25 then −13)
3. 70 → 100 epochs, with a correctly pinned schedule
4. **500 → 1000 training samples** (−49 pcm test, gap 236 → 105)

Regularisation and loss-shape tricks did not.

---

## 7. Method notes for anyone reading the folders

- **Trust `study_summary.csv` from the training job** for test pcm. A later `testset_results/` pass on r13 reported ~4300 pcm on test (uncorrected / wrong forward path). Train/val rows in that same CSV look sane; **do not use its TEST numbers**.
- The locked test set is the same 300 configurations as the 500-sample studies, so 509 vs 558 vs 728 vs original 611 is a fair comparison on that split. Original `precise_param_strat` also used this test lock.
- `VAL_SEED = SEED`, so validation sets move with the seed. Do not average val pcm across seeds as if it were one split. Test is the comparable number.
- `steps_per_epoch = 16` is deliberate. At 1000 samples the model does **not** get 2× optimiser steps. The gain from r13 is extra unique geometries in the pool, seen across shuffled epochs, not a larger learning budget.

---

## 8. Where things stand

**Delivered**

- A recipe that beats the original 1000-train ReLU PEDS by **~100 pcm** on the locked test set (611 → 509) and lifts frac<650 from 0.69 to 0.76, with 3 seeds instead of 5.
- The same recipe already beats that original model **at 500 samples** (558 vs 611).
- The train/val/test gap is no longer the 6-point “94% vs 88.5%” failure mode; val and test agree.
- A clean negative result list so those axes are not re-opened.

**Still open**

1. **Low-keff absolute error.** Fractional correction is ~89% everywhere; bin 0 is still ~1100 pcm on val. Next attempts should change *what the model knows about the failing baseline* or *the physics correction itself*, not the loss weights. That work was not successful at 500 samples (r05); it may be worth one careful retry **at 1000**, not another 500-sample feature lottery.
2. **Seed-0 tax.** ~50 pcm between the worst and best 1000-sample draw. A better 1000-subset rule is still plausible; maximin-at-500 was the wrong scale to judge it.
3. **Pinned 16 steps/epoch at 1000 samples** leaves training pcm higher than the original ReLU 1000-train run (403 vs 299) because each epoch only sees half the set. A controlled experiment — same 6e-4 recipe, `steps_per_epoch` 16 vs 32, **matched integrated LR** (lower `lr_max` on the 32-step arm) — would show whether extra unique-geometry coverage per epoch is worth anything. Do not raise steps without cutting `lr_max`; that is the Round-2 confound again.
4. Full 5-seed error bar on r13, if this model is the one that goes into inverse design.

**Not recommended:** another architecture/loss/SWA round at 500 samples; flattening the keff histogram; `lr_max` above 6e-4 without a new schedule.

---

## 9. File index

| Path | What |
|------|------|
| `modules/PEDS_agent.py` | All study configs; r13 is `r13_1k_elu6e4` |
| `modules/PEDS.py` | Unchanged production model |
| `RUNS/agent_studies/r13_1k_elu6e4/train_1000_seed_{0,1,2}/` | Checkpoints, `study_summary.csv`, epoch logs |
| `RUNS/agent_studies/r07_lr_6e4/` | Best 500-sample run |
| `RUNS/agent_studies/round2_analysis_and_plan.md` | R1–R2 critique |
| `RUNS/agent_studies/round3_analysis.md` | R3 numbers and LR-schedule diagnosis |
| `RUNS/precise_param_strat/` | Original ReLU 1000-train reference |
