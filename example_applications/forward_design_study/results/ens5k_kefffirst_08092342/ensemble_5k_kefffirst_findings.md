# Ensemble LHS 5k (extend) + keff-first HF verify — findings

**Run directory:** `modules/inverse_design/lhs_peds_baseline/results/lhs_peds_ens5k_kefffirst_08092342`

## Scope

5-seed ensemble PEDS screen of **5000** LHS designs (prior 3000 + extended
2000), gate `|Δρ(k_mean,1)| ≤ 1000 pcm`, ranked **keff-first** then PPF.
Top **30** HF-verified with OpenMC (9 reused); reported HF-feasible if
`|Δρ_HF| ≤ 1500 pcm`.

## Main numbers

| Item | Value |
|---|---|
| Screen size | 5000 (3000 + 2000) |
| PEDS-feasible (`≤1000` pcm) | **154** (3.08%) |
| HF shortlist / reused / new | 30 / 9 / 21 |
| HF-feasible @ **1500** pcm | **28 / 30** (93%) |
| Mean / median `|Δρ|_HF` (pcm) | 515 / 339 |
| Mean / median HF PPF | 1.480 / 1.366 |
| Mean `|Δρ(PEDS, HF)|` (pcm) | 541 |
| Mean \|PPF_PEDS − PPF_HF\| | 1.263 |
| Mean ensemble `keff_std` on feasible | 0.00346 (~346 pcm-scale) |

## Screening cost (full 5000)

| Stage | Wall time |
|---|---|
| Prior 3000-design ensemble screen | **2m 50.4s** (170.4 s) |
| Extra 2000-design ensemble screen | **1m 55.6s** (115.6 s) |
| **Full 5000-design screen (sum)** | **4m 46.0s** (286.0 s) |
| Mean / design (5 seeds) | 57.2 ms |
| HF verify this job (**21 new** OpenMC runs) | **27m 19s** (1639 s) |
| HF cost of all **30** shortlist runs (21 new + 9 reused) | **37m 47s** (2267 s) |
| Mean OpenMC wall / HF run (n=30) | **75.6 s** |
| Mean OpenMC wall / new run (n=21) | **78.1 s** |

## Ensemble std (feasible set, n=154)

Mean / median `keff_peds_std`: 0.00346 / 0.00317 (≈346 / 317 pcm-scale);
83% below 500 pcm-scale. Mean / median `ppf_diffusion_std`: 0.156 / 0.137.
On the verified shortlist, higher std mildly tracks larger `|Δρ(PEDS,HF)|`
(low vs high std tercile: 543 vs 708 pcm). Means were used for gating/ranking;
std is kept as a reliability signal.

## Target-distance inflation

On the selected shortlist,
`mean|Δρ|_HF / mean|Δρ|_PEDS ≈ 515 / 65.5 ≈ 7.86×`.
This is selection optimism under keff-first (tiny PEDS distances), **not**
PEDS↔HF accuracy (that is ~541 pcm). Prefer absolute HF distances.

## Diffusion-only check

Same 154 PEDS-feasible designs, gate 1000 pcm, uncorrected diffusion:
**0 / 154** feasible. On the 30 HF-verified rows: mean `|Δρ|` vs HF is
**5744 pcm** (PEDS: 541 pcm); mean `|ΔPPF|` 1.335 vs 1.263.
Runtime 1.2 s.

Parity plots:
`publication_plots/peds_hf_parity_ens5k_kefffirst.png`,
`publication_plots/peds_diffonly_hf_parity_ens5k_kefffirst.png`.

## Top HF-feasible designs (≤1500 pcm), by HF PPF

| HF rank | PEDS rank | keff_HF | \|Δρ\|_HF | PPF_HF | PPF_PEDS | \|Δρ\|_PEDS↔HF |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 27 | 1.00090 | 90 | 1.2353 | 2.3294 | 218 |
| 2 | 8 | 1.01336 | 1319 | 1.2906 | 2.0785 | 1351 |
| 3 | 25 | 1.00042 | 42 | 1.2991 | 4.5732 | 78 |
| 4 | 19 | 1.00576 | 573 | 1.3150 | 3.0494 | 646 |
| 5 | 24 | 1.01378 | 1359 | 1.3206 | 2.1706 | 1240 |
| 6 | 9 | 1.00557 | 554 | 1.3349 | 2.7334 | 517 |
| 7 | 15 | 1.00358 | 357 | 1.3400 | 3.6794 | 415 |
| 8 | 5 | 1.01226 | 1211 | 1.3522 | 2.1980 | 1221 |
| 9 | 7 | 1.00046 | 46 | 1.3551 | 1.7769 | 65 |
| 10 | 20 | 1.00120 | 120 | 1.3561 | 3.0245 | 211 |

## vs OpenMC-only baseline (brief)

| | OpenMC LHS | This ensemble study |
|---|---|---|
| Evaluations | 500 HF | 5000 PEDS + 30 HF |
| HF-feasible | 28 / 500 (5.6%) | 28 / 30 verified |
| Screen wall | 8h 58m | **4m 46s** (full 5k) |
| Best HF PPF (feasible) | 1.253 | 1.235 |

## Takeaway

Full 5000-design ensemble screen cost **4m 46s**. Keff-first shortlist
yielded **28/30** HF-feasible designs—matching the 500-run OpenMC baseline
count with far fewer HF calls. Ensemble std is informative (~346 pcm-scale
on feasible); diffusion-only alone cannot pass the gate (0/154).
