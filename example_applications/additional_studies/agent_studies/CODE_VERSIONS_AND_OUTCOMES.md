# Agent studies — code versions and outcomes

Campaign 18–20 Aug 2026. Every number below is the **deployed-checkpoint** test error from `study_summary.csv` (mean |Δρ| in pcm on the locked 300-sample test set, `TEST_SEED=0`). Seeds are 0 / 1 / 2. Sample standard deviation is over those three seeds.

The per-run logs, checkpoints, and plots were removed on 6 Oct 2026, except the three best studies. One code snapshot per study is in `code_snapshots/`. Per-seed rows are in `code_snapshots/metrics_all_runs.csv`.

## What was held fixed

- Geometry / cross-section dataset: `data/highfidelity/17jul_0.8_1.2.npz`.
- Split: parameter-stratified, 5 bins. `TRAIN_SEED = VAL_SEED = SEED`, so the validation set moves with the seed. **Only the test set is comparable across studies.**
- Batch 32. Adam. `lr_min = 5e-6`. Log-ratio clip `[−1.8, 0.5]`.
- Screening size: 500 train / 300 val / 300 test. The scale-up (`r13`) is 1000 / 500 / 300, same test lock.
- Reference outside this folder: original ReLU PEDS in `precise_param_strat`, 1000 train, 5 seeds, test **611 ± 38 pcm**, fraction below 650 pcm **0.69**.

Two questions the campaign was built to answer:

1. Low-keff bins stay much worse in absolute pcm, because the diffusion baseline is 3–4× worse there and PEDS applies a nearly constant fractional correction.
2. Train improves more than val/test (about 94% vs 88.5% error reduction on the original 1000-sample ReLU run).

## Six trainers, not 34

Seeds of one study share one snapshot. Studies launched from the same script revision are byte-identical; the study name is `PEDS_STUDY_NAME` and only selects a row of `STUDY_CONFIGS`.

| Generation | Snapshot sha256 (10) | Lines | Studies |
|---|---|---|---|
| R1 | `9f2a3537c7` | 1068 | s01–s10 |
| R2 | `0a2a05806a` | 1301 | s11–s20 |
| R3 smoke | `5ef90f7a5d` | 1707 | r00 |
| R3 | `98fc698ddd` | 1710 | r01–r06 |
| R4 | `3b29a780ad` | 1847 | r07–r12 |
| R5 | `3e5714f3f7` | 1870 | r13 |

### R1 — first fork of PEDS (`s01`–`s10`)

New relative to production `PEDS.py`: a `STUDY_CONFIGS` table, 70 epochs, `decay_epochs` default 70, learning rate 2e-4, and these optional paths:

- `loss_scheme="pcm_weighted"`: MSE weighted by `1/k^4`.
- Dropout 0.15 on the trunk.
- AdamW via `weight_decay=1e-4`.
- `use_keff_input`: diffusion keff appended as a 7th geometry feature.
- Residual trunk, which forces equal widths `[128,128,128]` so the skip matches.
- ELU instead of ReLU.
- Wider ReLU trunk `[256,512,256]`.
- Gaussian input noise σ = 0.01 on the geometry.
- `use_balanced`: oversample the bottom 30% of keff twice per epoch.

Checkpointing in this generation is the old EMA of validation pcm. Warmup length is 10% of `decay_steps`, so it moves if the run is lengthened.

### R2 — stack on ELU (`s11`–`s20`)

Adds, per study: `epochs`, `decay_epochs_override`, `patience`, `logratio_penalty`, activation `"gelu"`, and `loss_scheme="log_keff"` (`MSE(log k_pred, log k_ref)`). The log-ratio penalty is `λ * mean(log_ratios²)` with λ = 0.001. Residual arms are still the narrower `[128,128,128]` trunk, so “residual” and “less capacity” are the same change.

Important confound: lengthening the run also lengthens the cosine. At epoch 70 the learning rate on a 100-epoch schedule is about **10×** the learning rate on a 70-epoch schedule, and the integrated learning rate rises about 43%. The gain of `s12` over `s07` is therefore not “30 more epochs” alone.

### R3 — schedule, SWA, features (`r00`–`r06`)

Pins `STEPS_PER_EPOCH_REF = 16`, so the optimiser step count no longer grows with the training-set size. Checkpoint selection becomes a 5-epoch trailing mean of validation pcm (`sel_window=5`) instead of EMA. Adds stochastic weight averaging (`swa`, `swa_start`, `swa_eval_every`), `lr_mode="hold"` (cosine down to a floor, then constant), and `extra_feat_mode="regime"` (baseline keff plus three spectral ratios). `patience=999` disables early stopping. `r00` is a 32-epoch smoke of the regime-feature and SWA paths on the small trunk; it is not a result.

### R4 — warmup is an explicit knob; SWA off (`r07`–`r12`)

`warmup_epochs` is a fixed epoch count, no longer 10% of the schedule. SWA is off. `r12` sets `split_method="maximin"` and `lock_split_seeds=True`: greedy maximin inside each keff bin, keff histogram unchanged, test set still locked. The recipe otherwise matches `r02` (ELU, 4e-4, warmup 7), not the later 6e-4 peak.

### R5 — scale-up (`r13`)

Same weights as `r07`. `VAL_SIZE` and `TEST_SIZE` are read from the environment so the launch script can set 1000 train / 500 val / 300 test. `steps_per_epoch` stays 16, so 1000 samples do **not** double the number of Adam steps. Each epoch still updates on 512 examples; the extra data is a larger pool, reshuffled across epochs.

## Outcomes

Fraction is the share of the locked test set with |Δρ| < 650 pcm.

### Round 1 — one change at a time, 70 epochs, ReLU unless noted

| Study | What the code changed | Test pcm (seeds 0, 1, 2) | Mean | Frac | Verdict |
|---|---|---|---|---:|---|
| s01_baseline | ReLU `[128,256,128]`, MSE, Adam 2e-4 | 720, 684, 779 | 728 | 0.654 | Screening reference |
| s02_weighted_loss | loss `1/k^4` | 715, 727, 775 | 739 | 0.649 | Worse. Gradient for k=0.8 is ~25× the gradient for k=1.2 |
| s03_dropout | dropout 0.15 | — | — | — | No `study_summary`. Stopped near epoch 23; physics VJP exploded |
| s04_adamw | AdamW, wd=1e-4 | 722, 683, 780 | 728 | 0.652 | Identical to s01. Adam’s denominator cancels a small decay |
| s05_keff_input | baseline keff as input | — | — | — | Crashed: feature dimension mismatch. Later fixed; the idea itself failed as r05 |
| s06_residual | residual `[128,128,128]` | 687, 679, 733 | 700 | 0.666 | Small, very consistent gain. Implicit regulariser |
| **s07_elu** | **ELU, same trunk** | **669, 585, 624** | **626** | **0.730** | **First real win, −102 pcm vs s01** |
| s08_wider | ReLU `[256,512,256]` | 752, 697, 757 | 736 | 0.649 | More capacity, more memorisation |
| s09_noise_aug | input noise σ=0.01 | 709, 685, 776 | 723 | 0.652 | Neutral |
| s10_balanced | oversample low-keff 2× | 719, 709, 798 | 742 | 0.637 | Worse. Same family as the weighted loss |

### Round 2 — combinations on top of ELU

| Study | What the code changed | Test pcm | Mean | Frac | Verdict |
|---|---|---|---|---:|---|
| s11_elu_residual | ELU + residual, 70 ep | 674, 599, 637 | 637 | 0.704 | Worse than ELU alone |
| **s12_elu_long** | **ELU, 100 ep, decay 100** | **632, 568, 587** | **596** | **0.730** | **Best of R2, −30 vs s07. Confounded with the longer cosine** |
| s13_residual_long | ReLU residual, 100 ep | 660, 667, 737 | 688 | 0.681 | Residual without ELU stays near the ReLU floor |
| s14_elu_res_long | ELU + residual, 100 ep | 649, 581, 604 | 611 | 0.726 | Residual still does not beat plain ELU |
| s15_gelu | GELU, 70 ep | 655, 630, 711 | 665 | 0.707 | Between ReLU and ELU |
| s16_elu_log_loss | ELU + log-keff MSE | 680, 589, 632 | 633 | 0.714 | No gain over ELU |
| s17_elu_logreg | ELU + log-ratio L2 λ=0.001 | 667, 578, 632 | 626 | 0.729 | Identical to s07 |
| s18_elu_res_log | ELU + residual + log-keff | 669, 611, 642 | 641 | 0.696 | Worse |
| s19_elu_res_log_long | same, 100 ep | 665, 595, 615 | 625 | 0.713 | No better than s07 |
| s20_elu_res_logreg_long | ELU + residual + L2, 100 ep | 647, 576, 609 | 611 | 0.724 | Still short of s12 |

Nothing aimed at the two original problems (log-keff for the low-keff bins, log-ratio L2 for the gap) moved the test number once ELU was in place. Every ELU variant sat on a train-to-test gap of about 220 pcm.

### Round 3 — learning rate, SWA, features. Window-5 checkpoint, 100 epochs

| Study | What the code changed | Test pcm | Mean | Frac | vs s12 |
|---|---|---|---|---:|---|
| r01_ref | ELU, 2e-4, window-5, SWA logged | 639, 568, 603 | 603 | 0.732 | +7. Cost of the honest checkpoint vs EMA |
| **r02_lr_hi** | **lr_max = 4e-4** | **609, 545, 558** | **571** | **0.749** | **−25. First change that stacked on ELU** |
| r03_lr_lo | lr_max = 1e-4 | 678, 596, 619 | 631 | 0.712 | Worse |
| r04_swa_flat | cosine to 5e-5 by ep 55, then hold; SWA from 55 | 670, 597, 624 | 630 | 0.706 | The hold itself is worse. SWA only helps a bad trajectory |
| r05_regime_feat | + baseline keff and 3 spectral ratios | 636, 548, 663 | 616 | 0.713 | Highest variance (seed 2 collapses). Extra inputs, 500 samples, more memorisation |
| r06_small | trunk `[64,128,64]` | 631, 624, 673 | 642 | 0.710 | Underfit. Capacity was not the limit |

On the annealed runs, SWA changed the test number by about 0 ± 4 pcm. Drop it.

### Round 4 — finish the learning-rate map

All of these are ELU, `[128,256,128]`, MSE, 100 epochs, window-5, no SWA, steps/epoch pinned at 16.

| Study | What the code changed | Test pcm | Mean | Frac | vs r02 (571) |
|---|---|---|---|---:|---|
| **r07_lr_6e4** | **lr_max = 6e-4, warmup 7** | **584, 530, 560** | **558** | **0.743** | **−13. Peak of the sweep** |
| r08_lr_8e4 | 8e-4, warmup 7 | 579, 531, 582 | 564 | 0.762 | Past the peak. Seed 2 is +24 vs r07 |
| r09_warm_3 | 4e-4, warmup 3 | 602, 547, 576 | 575 | 0.744 | Shorter warmup hurts |
| r10_warm_14 | 4e-4, warmup 14 | 610, 540, 602 | 584 | 0.750 | Longer warmup at 4e-4 hurts |
| **r11_lr6_warm14** | **6e-4, warmup 14** | **587, 539, 548** | **558** | **0.757** | **Same mean as r07. Extra warmup is not required at 6e-4** |
| r12_maximin | r02 recipe, maximin-within-keff-bin 500 | 597, 557, 607 | 587 | 0.708 | Does not beat param-stratified selection |

The learning-rate curve is 1e-4 ≪ 2e-4 < 4e-4 < **6e-4** ≤ 8e-4. Warmup of 7 epochs is the right setting at 6e-4.

### Round 5 — the same recipe at 1000 samples

| Study | Setup | Test pcm | Mean ± std | Frac |
|---|---|---|---|---|
| **r13_1k_elu6e4** | r07 recipe, 1000 train / 500 val / 300 test, 100 epochs, 16 steps/epoch | 541, 496, 489 | **509 ± 28** | **0.761** |

Against the original 1000-sample ReLU PEDS (611 pcm, frac 0.69) this is about **−100 pcm** and +7 points of fraction-below-650, with 3 seeds instead of 5. The same recipe already beats that original model at 500 samples (r07, 558 vs 611).

Train pcm on r13 (about 403) is not a full pass over the 1000 points; it is the 512 samples updated that epoch. Validation (~496) and test (509) agree. The train-to-test gap fell from 492 pcm (ReLU, 500) to 236 (ELU, 500, r07) to about 105 (ELU, 1000). Doubling the training pool did what dropout, AdamW, log-ratio L2, residuals, and maximin did not.

## What actually moved the test error

In order:

1. ELU instead of ReLU (−102 pcm at 500 samples, s07 vs s01).
2. `lr_max` from 2e-4 to 4e-4 to **6e-4** (−25 then −13), with warmup fixed at 7 epochs and the cosine no longer tied to the run length.
3. 70 to 100 epochs, only after the schedule is pinned. The raw s12 gain mixed duration with a higher late learning rate.
4. 500 to 1000 training samples (−49 pcm, r07 to r13), without increasing the number of optimiser steps.

## What not to rerun

| Mechanism | Evidence |
|---|---|
| AdamW / small L2 | s04 identical to s01 |
| Dropout | s03 exploded; masks fight the sparse physics VJP |
| `1/k^4` loss, low-keff oversampling | s02, s10 worse. The model already corrects those samples; the weight just destabilises Adam |
| Wider ReLU | s08 larger train/test gap |
| GELU, residual+ELU, log-keff MSE, log-ratio L2 | s11–s20: zero or negative once ELU is present |
| SWA on an annealed cosine | r01–r03, r05, r06: about 0 ± 4 pcm |
| SWA plus a constant-LR hold | r04: the hold is ~30 pcm worse than annealing |
| Regime features | r05: mean worse, std ~49 pcm, larger gap |
| Trunk `[64,128,64]` | r06: +70 pcm vs r02 |
| Maximin geometry selection at 500 | r12: 587 vs 571 for the same optimiser |
| Letting `decay_epochs` scale with `epochs` | Silently raises mid-run learning rate by up to 10× |
| EMA checkpoint from epoch 1 | Stays biased by the 3000+ pcm start. Window-5 is the selector to keep |

Low-keff absolute error is still open. On r13 validation, bin 0 (keff ~0.82) is still about 1100 pcm after correction versus about 250 pcm in the highest bin. The fractional improvement is ~89% in every bin. Reweighting did not change that shape.

Seed 0 remains the weak draw (r13 test 541 vs 489–496). That is which geometries land in the training set, not the initialisation. Maximin at 500 samples did not remove it.

## The three runs kept on disk

Primary metric is mean test |Δρ|. These are the three lowest:

| Kept folder | Why |
|---|---|
| `r13_1k_elu6e4/` | Best result in the campaign. 509 ± 28 pcm, frac 0.761. This is the model to reload |
| `r07_lr_6e4/` | Best 500-sample run and the recipe that was scaled. 558 pcm, frac 0.743 |
| `r11_lr6_warm14/` | Tied with r07 on the mean (558 pcm) and higher fraction below 650 (0.757). Shows that warmup 14 does not beat warmup 7 |

Each kept folder still has its three seeds: checkpoints, `study_summary.csv`, epoch logs, and the original snapshot. Everything else in this directory that was a `train_*` tree has been deleted.

`r08_lr_8e4` is the next distinct point (564 pcm, frac 0.762). It was not kept; its snapshot is enough to repeat it.

## How to rerun a deleted study

`code_snapshots/<study>.py` is the trainer that produced that study, copied from seed 0. Files that share a sha256 in the table above are the same program.

The snapshot computes paths from `__file__`. It expects to live one directory below the repo root (historically `modules/PEDS_agent.py`), so that `data/highfidelity/17jul_0.8_1.2.npz` resolves and the neutron-transport imports are on `sys.path`. Copy the chosen file there, then:

```bash
export PEDS_STUDY_NAME=<study>     # key inside STUDY_CONFIGS
export PEDS_SEED=0                 # 0, 1, or 2
export PEDS_TRAIN_SIZE=500         # 1000 for r13
export PEDS_VAL_SIZE=300           # 500 for r13
export PEDS_TEST_SIZE=300
export PEDS_TEST_SEED=0
# R1 70-epoch arms also set:
# export PEDS_DECAY_EPOCHS=70
python PEDS_agent.py
```

`r13` is the only study that needs `PEDS_TRAIN_SIZE=1000` and `PEDS_VAL_SIZE=500`. The launch scripts still in this folder (`run_agent_studies.sh`, `run_agent_studies_r2.sh`, `run_agent_studies_r3.sh`, `run_agent_studies_r4.sh`, `run_agent_studies_r5_1k.sh`) are the original Slurm arrays and list the same environment variables.

Do not trust a later `analysis/testset_results/` pass on r13: that evaluation reported ~4300 pcm on the test split (wrong forward path). The training job’s `study_summary.csv` is the number used in this note.
