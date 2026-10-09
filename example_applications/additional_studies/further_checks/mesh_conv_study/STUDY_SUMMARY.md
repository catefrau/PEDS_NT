# Mesh Convergence Study — Summary for Downstream Context

**Run directory:** `modules/MOREstudies/0small_studies/mesh_conv_08091523/`  
**Date:** 2026-08-09  
**Scripts:** `run_mesh_convergence_study.py`, `run_mesh_convergence_study.sh`  
**Slurm job:** 36789251  
**Follow-on flux overlays:** `flux_mesh_overlay/` (job 36813548)

This is the **model-consistent** mesh study. It is the follow-up to
`mesh_conv_08091423/`, which mixed baseline “worst” labels with a
precise_param_strat checkpoint. Here, case selection and PEDS XS extraction
use the **same** model.

Two XS sources are swept at every mesh:

- **poly** — polynomial-regression MGXS (`predict_xs`), i.e. the uncorrected
  diffusion baseline.
- **peds** — NN-corrected MGXS from the trained PEDS checkpoint.

---

## 1. Purpose

Question to answer:

> For this trained PEDS model’s own hardest test geometries, is the large
> keff error vs OpenMC caused (or strongly worsened) by the spatial mesh being
> too coarse? Or does refining the mesh leave the HF error essentially unchanged?

The same question is asked for the **polynomial** baseline, so one can see
whether mesh error is large compared with the poly→PEDS correction itself.

Secondary goals:

- Measure how keff and wall time change with mesh size (poly and PEDS).
- Quantify mesh discretisation error relative to a fine “converged” mesh (0.01 cm).
- Quantify signed reactivity error vs the OpenMC (HF) target as mesh is refined.
- Confirm that mesh=1 PEDS keff matches the test-log `keff_peds_ref`.

---

## 2. Method (high level)

### Why not vary mesh inside the trained PEDS JAX model?

PEDS was trained at `mesh_size = 1 cm` (`modules/NTcode_config_data/config_run.py`).
The training forward path pads flux vectors to a fixed `N_FLAT_MAX` sized for that
mesh. Changing mesh inside the live PEDS callback is therefore unsafe / invalid.

Regional cross sections (poly-regression **and** NN-corrected) are
**mesh-independent**. So the correct approach is:

1. Fix geometry + XS.
2. Re-run only the diffusion eigenvalue solve at each mesh size.

### Workflow used

1. **Select cases:** top-10 worst *test* geometries of
   `precise_param_strat` / `train_1000_seed_1` by final-epoch
   `|Δρ(k_PEDS, k_OpenMC)|` from
   `modules/RUNS/precise_param_strat/testset_results/run_train1000_seed1_keff_comparison.csv`.
2. **Extract XS once at training mesh (1 cm):**
   - `poly` — `predict_xs(geo)` polynomial regression.
   - `peds` — NN-corrected XS from `model.compute_xs` on
     `modules/pretrained_models/train_1000_seed_1`.
3. **Mesh sweep** of the forward diffusion eigenvalue solve only, with fixed XS:
   meshes = `{2, 1, 0.5, 0.2, 0.1, 0.05, 0.01}` cm.
4. Record keff + wall time; plot vs mesh and vs HF / vs finest mesh.

Signed reactivity difference (same as study CSVs):

\[
\Delta\rho\ [\mathrm{pcm}]
= \frac{k_a - k_b}{k_a\, k_b}\times 10^5
\]

---

## 3. Models / data involved

| Role | Path / source |
|------|----------------|
| Case selection | `RUNS/precise_param_strat/testset_results/run_train1000_seed1_keff_comparison.csv` (final epoch) |
| Geometry / HF keff | `data/highfidelity/17jul_0.8_1.2.npz` (params checked vs CSV) |
| PEDS checkpoint | `modules/pretrained_models/train_1000_seed_1` (= `RUNS/precise_param_strat/train_1000_seed_1`, seed 1, best) |
| Poly XS | `solvers/NTdiffusion/diffusion_solver.predict_xs` |
| Training mesh | 1 cm |
| Solves | `DiffusionEigenvalue_MG` (forward only) |

**Selected cases** (`selected_cases.csv`), all `precise_param_strat_seed1`:

| rank | sample_idx | k_PEDS (test log) | k_OpenMC | \|Δρ\| pcm |
|------|------------|-------------------|----------|-----------|
| 1 | 2053 | 0.8825 | 0.8521 | 4050 |
| 2 | 2019 | 0.8684 | 0.8965 | 3615 |
| 3 | 2347 | 0.9024 | 0.9298 | 3270 |
| 4 | 1535 | 0.8399 | 0.8198 | 2921 |
| 5 | 2200 | 1.0004 | 1.0282 | 2703 |
| 6 | 210 | 0.8905 | 0.8699 | 2654 |
| 7 | 1967 | 0.8312 | 0.8490 | 2516 |
| 8 | 1842 | 0.9809 | 1.0034 | 2288 |
| 9 | 1807 | 0.7919 | 0.8056 | 2144 |
| 10 | 1955 | 0.8716 | 0.8560 | 2090 |

### Sanity: mesh=1 PEDS keff matches the test log

`xs_extract_summary.csv` column `pcm_recalc_minus_ref`:

- mean `|Δρ(recalc − ref)| ≈ 0.23 pcm`
- max ≈ **1.85 pcm** (sample 1535)

Example (sample 2053, mesh=1): test-log `keff_peds_ref = 0.88253492`,
recalc `0.88253492`. This run is **model-consistent**; the 08091423 mismatch
(`0.916` vs `0.943`) does not apply here.

Poly at mesh=1 is the uncorrected baseline (e.g. sample 2053: `k_poly = 0.9398`
vs OpenMC `0.8521` → ~11 000 pcm). That is the starting error PEDS is meant to
reduce.

---

## 4. Relation to `mesh_conv_08091423`

| | 08091423 | **08091523 (this run)** |
|---|----------|-------------------------|
| “Worst” source | baseline orig/alt test logs | precise_param_strat seed 1 test log |
| Samples | 1605, 645, 760, … | 2053, 2019, 2347, … |
| Dataset NPZ | `LHS_0.8_newbounds.npz` | `17jul_0.8_1.2.npz` |
| PEDS XS | same pretrained checkpoint | same checkpoint |
| mesh=1 vs `keff_peds_ref` | ~1650 pcm mean mismatch | ~0.2 pcm (match) |

Use **this folder** for claims about mesh vs HF error on this model’s actual
worst test cases.

---

## 5. Main numerical findings

Artifacts under `mesh_conv_08091523/`:

| File | Contents |
|------|----------|
| `selected_cases.csv` | Top-10 worst cases |
| `xs_tensors.npz` | Saved poly + peds XS per sample |
| `xs_extract_summary.csv` | Extraction sanity (mesh=1 match) |
| `mesh_convergence.csv` | Full keff / timing table |
| `signed_pcm_vs_openmc_by_mesh.csv` | Signed Δρ vs OpenMC |
| `pcm_vs_converged_by_mesh.csv` | Δρ vs finest mesh (0.01 cm), meshes ≤ 1 cm |
| `plots/*.png` | Figures |
| `flux_mesh_overlay/` | Top-5 flux overlays (PEDS meshes + OpenMC) |

### 5.1 Error vs OpenMC vs mesh (poly **and** PEDS)

Plots: `plots/signed_pcm_vs_openmc_{peds,poly}.png`.

Qualitative result for **both** XS sources:

- As mesh is refined, signed Δρ vs OpenMC **plateaus**; it does **not** go to 0.
- Remaining HF bias is **not removed by mesh refinement alone**.

**PEDS-corrected XS.** Mean `|Δρ|` vs OpenMC is **smallest near ~0.2 cm**
on this worst-10 set, not exactly at the 1 cm training mesh, but 1 cm is
already in the same ballpark as finer meshes. Going from 1 cm → 0.01 cm does
**not** systematically kill the ~2–4k pcm case-wise errors that defined
“worst.” Mean signed Δρ trends slightly more negative at the finest mesh.

**Polynomial XS.** Errors stay **much larger** than PEDS at every mesh
(~8–10k pcm mean `|Δρ|`, almost all **positive**: poly over-predicts keff).
Refining the mesh does not bring poly in line with OpenMC. The poly–HF gap is
several times the PEDS–HF gap, so the dominant error for the baseline is the
XS regression, not the 1 cm mesh.

Mean `|Δρ|` vs OpenMC (10 worst cases):

| mesh (cm) | PEDS | poly |
|-----------|------|------|
| 2.0 | ~3675 | ~9601 |
| 1.0 | ~2825 | ~8352 |
| 0.5 | ~1889 | ~8482 |
| 0.2 | ~1632 | ~8633 |
| 0.1 | ~1759 | ~8489 |
| 0.05 | ~1867 | ~8267 |
| 0.01 | ~2099 | ~7995 |

At training mesh (1 cm), per-case PEDS `|Δρ|` vs OpenMC matches the test-log
ranking (~2089–4050 pcm). The same geometries with **poly** XS at 1 cm span
about **1.6k to 14.6k pcm** (sample 2019 is the only poly case with a modest
error; several others exceed 10k pcm).

### 5.2 Error vs converged mesh (meshes ≤ 1 cm)

Plots: `plots/pcm_vs_converged_{peds,poly}.png`, `plots/dpcm_vs_finest_mesh.png`.  
Reference = finest mesh in the sweep (0.01 cm).

Mean `|Δρ|` vs converged:

| mesh (cm) | PEDS | poly |
|-----------|------|------|
| 1.0 | ~1363 | ~1272 |
| 0.5 | ~994 | ~951 |
| 0.2 | ~744 | ~686 |
| 0.1 | ~609 | ~557 |
| 0.05 | ~309 | ~272 |
| 0.01 | 0 | 0 |

Interpretation:

- Relative to a very fine diffusion mesh, **1 cm is not fully converged** on
  these geometries (~1.3k pcm mean |Δρ| to 0.01 cm for both XS sources).
- Poly and PEDS **discretisation** errors vs 0.01 cm are similar (same solver,
  different regional XS). The huge poly-vs-OpenMC bias is therefore **not** a
  mesh effect.
- Most of the remaining mesh error drops between 0.5 → 0.05 cm; by 0.05 cm the
  residual vs 0.01 cm is ~300 pcm mean.
- Mesh discretisation (~1.3k pcm at 1 cm vs finest) is **smaller** than the
  PEDS-vs-OpenMC worst-case errors (~2–4k pcm) and **much smaller** than
  poly-vs-OpenMC (~8k pcm mean). Mesh is a real but secondary contributor.

### 5.3 Cost

Plot: `plots/time_vs_mesh.png`.  
Forward-solve wall time is essentially the same for poly and PEDS (same
eigenproblem size). Mean ~0.003 s at 1 cm vs ~5.7 s at 0.01 cm on these cases.
Mesh 0.01 cm dominates runtime.

### 5.4 Flux overlays (top 5, PEDS XS)

`flux_mesh_overlay/`: diffusion fluxes at 1, 0.5, 0.2, 0.1, 0.01 cm plus
OpenMC from `MC_solver.run_mc`. PNGs in `flux_mesh_overlay/flux_plots/`.
Each curve is peak-normalised to 1. Mesh-to-mesh diffusion shapes converge;
OpenMC shape differences remain (especially thermal), consistent with XS /
transport error rather than unresolved spatial mesh.

---

## 6. Bottom-line conclusions

1. **This run is internally consistent:** worst cases, dataset, and checkpoint
   all belong to precise_param_strat seed 1. Mesh=1 PEDS keff matches the test
   log (~0.2 pcm mean).
2. **XS can be frozen and mesh varied**; do not change mesh inside the trained
   PEDS JAX padding path.
3. **Mesh refinement does not drive keff → OpenMC** for either XS source.
   Curves plateau away from zero.
4. **Polynomial baseline** stays ~8k pcm mean `|Δρ|` vs OpenMC at all meshes
   (typically high-k). That error is XS-dominated.
5. **PEDS** reduces that to ~2–4k pcm on these worst cases; further mesh
   refinement does not remove the remainder (mean `|Δρ|` vs HF even slightly
   larger at 0.01 cm than at 0.2 cm).
6. **1 cm is only partially converged** vs 0.01 cm (~1.3k pcm mean |Δρ| for
   both poly and PEDS). Finer than ~0.05 cm is diminishing returns vs 0.01 cm
   (~300 pcm). That mesh error is smaller than the PEDS–HF gap on this worst
   set and much smaller than the poly–HF gap.

---

## 7. How to replot (no solver)

From `modules/MOREstudies/0small_studies/`:

```bash
conda activate jax-env
python - <<'EOF'
from pathlib import Path
import pandas as pd
from run_mesh_convergence_study import plot_error_evolutions
out = Path("mesh_conv_08091523")
df = pd.read_csv(out / "mesh_convergence.csv")
plot_error_evolutions(df[df["status"] == "ok"].copy(), out)
EOF
```

Legend font size for the vs-OpenMC figures: `ax.legend(fontsize=…)` in
`plot_error_evolutions` in `run_mesh_convergence_study.py`.

---

## 8. Key paths (absolute)

- Study outs: `/global/home/users/caterinafrau/PEDS_NT/modules/MOREstudies/0small_studies/mesh_conv_08091523/`
- Driver: `/global/home/users/caterinafrau/PEDS_NT/modules/MOREstudies/0small_studies/run_mesh_convergence_study.py`
- Pretrained model: `/global/home/users/caterinafrau/PEDS_NT/modules/pretrained_models/train_1000_seed_1`
- Test metrics used for worst-case selection:  
  `/global/home/users/caterinafrau/PEDS_NT/modules/RUNS/precise_param_strat/testset_results/run_train1000_seed1_keff_comparison.csv`
- Mixed-model predecessor (do not conflate):  
  `/global/home/users/caterinafrau/PEDS_NT/modules/MOREstudies/0small_studies/mesh_conv_08091423/`
