# Guide to using PEDS × NT

Practical, end-to-end guide to this repository: how the three pieces (OpenMC
high-fidelity model → 1D diffusion low-fidelity solver → PEDS neural network)
fit together, which file to edit for each decision, and how to either
**reproduce the thesis results** or **set up a new problem**.

For the physics and the results themselves, see
[the thesis PDF](thesis_PDF/Caterina_Frau_Master_Thesis.pdf) and
[`README.md`](README.md). This document is about *operating* the code.

---

## How to skim this guide

Every section ends with a **`▶ Bottom line`** box: the 3–5 things you actually
need to remember. Lines marked **⚠️** are traps that will silently give you
wrong numbers. Lines marked **📌** are the single file you edit for that
decision.

| If you want to… | Jump to |
|---|---|
| Run a training job today | [§0 Cheat sheet](#0-cheat-sheet) |
| Generate new OpenMC data | [§1 OpenMC model](#1-the-openmc-high-fidelity-model) |
| Change the geometry/case | [§1.2](#12-where-the-case-is-defined) + [§2.1](#21-mirroring-the-case-in-the-lf-config) |
| Understand/retrain the XS regression | [§2.3](#23-the-polynomial-regression-baseline) |
| Run diffusion with exact OpenMC MGXS | [§2.4](#24-running-the-lf-solver-on-its-own) |
| Understand the NN | [§3.3](#33-the-neural-network-in-detail) |
| Launch / monitor PEDS | [§3.5](#35-launching-monitoring-and-where-results-appear) |
| Post-process results | [§3.8](#38-analysis-tools) |
| Reproduce the thesis | [§5](#5-reproducing-the-thesis-results) |
| Start a brand-new problem | [§6](#6-setting-up-a-new-problem-checklist) |
| Know what's broken/stale | [§7 Known traps](#7-known-traps-and-stale-code) |

---

## 0. Cheat sheet

### Environments

Two conda environments, **never mixed**:

| Env | Used for | Activate |
|---|---|---|
| `jax-env` | Everything PEDS / diffusion / analysis | `source ~/miniconda3/etc/profile.d/conda.sh && conda activate jax-env` |
| `mc-env` | Everything OpenMC | same, with `conda activate mc-env` |

OpenMC additionally needs:

```bash
export OPENMC_CROSS_SECTIONS=/global/scratch/users/caterinafrau/openmc_data/endfb-viii.0-hdf5/cross_sections.xml
```

> **⚠️ Nothing compute-heavy runs on the login node.** All workloads go through
> `sbatch`. The repo-wide rule lives in `.cursor/rules/slurm-only-compute.mdc`.
> Login node is for editing, `sbatch`/`squeue`/`scancel`, reading CSVs.

### The five commands you actually use

```bash
# 1. Generate high-fidelity OpenMC training data  (hours–days)
cd solvers/HF_openMC      && sbatch run_many_openmc.sh

# 2. Smoke-test the whole PEDS chain               (~minutes, writes to RUNS/_smoke_test)
cd config_and_run         && sbatch smoke_test_peds.sh

# 3. Train PEDS (5-seed array by default)          (~hours per seed)
cd config_and_run         && sbatch run_peds.sh

# 4. Held-out test-set evaluation of trained runs
cd utils                  && sbatch run_testset.sh

# 5. Full post-processing suite (plots + stats)
cd utils                  && sbatch run_analysis.sh
```

Monitor with `squeue -u $USER`; logs land in the `slurm_logs/` folder *next to
the script you submitted*.

### Pipeline map

```
         solvers/HF_openMC/config_run.py      ← case definition (HF)
                      │  CFG_ROD
                      ▼
       TD_generation.py  (LHS over BOUNDS)    ← sbatch run_many_openmc.sh
                      │
                      ▼  LHS_full_dataset.csv  (6 params + keff + 36 MGXS cols)
         ┌────────────┴─────────────┐
         ▼                          ▼
 polynomial XS regression     data/highfidelity/*.npz   ← PEDS training set
 (.pkl under solvers/LF_diffusion/              │
  regression_model/<name>/polyreg_model/)       │
         │  Σ_base(p)                           │  (params, params_raw, keffs)
         ▼                                      ▼
 solvers/LF_diffusion/  ◄───── × exp(δ) ───── models/PEDS.py  (GeneratorNN)
   diffusion_solver.py                          ▲
   core/MG1D_eigenvalue_nregions.py             │ ∂k/∂Σ via adjoint flux
         │  k_eff, φ, φ†                        │ (custom VJP)
         └──────────────────────────────────────┘
                      │
                      ▼
   config_and_run/RUNS/<study>/train_<N>_seed_<S>/   ← logs, checkpoints, XS
                      │
                      ▼
   utils/analysis/*  →  <study>/analysis/{testset_results,param_error,
                                          pub_keff,xs_stats,training_diagnostics}
```

> **▶ Bottom line**
> - `jax-env` for PEDS/diffusion, `mc-env` for OpenMC; always `sbatch`.
> - The six-parameter vector **`p = [b4c_r, cr_frac, fuel_r, enrichment, f_mod, water_r]`**
>   is the common currency of the whole pipeline, in that order, everywhere.
> - Three config files own all experiment decisions:
>   `solvers/HF_openMC/config_run.py`, `solvers/LF_diffusion/config_run.py`,
>   `config_and_run/config_peds.py`.

---

## 1. The OpenMC high-fidelity model

**Folder: `solvers/HF_openMC/`**

| File | Role |
|---|---|
| `config_def.py` | Type definitions: `RegionSpec`, `EnergyGroupSpec`, `SolverSettings`, `MCConfig`, `ParamSpec`, `SweepConfig`. Read this for what every knob means. |
| `config_run.py` | **📌 The actual cases.** `CFG_ROD` (the thesis case), plus `CFG_HOM`, `CFG_HET`, `CFG_STACY_APPROX`. |
| `MC_solver.py` | The engine: materials, geometry, MGXS library, tallies, convergence diagnostics, CSV writers. ~1600 lines, rarely edited. |
| `TD_generation.py` | **📌 The LHS training-data campaign**: bounds, sample count, output folder. |
| `run_many_openmc.sh` | Slurm launcher for `TD_generation.py`. |
| `design_api.py` | Thin API to run OpenMC on a single `params_raw` 6-vector (used by the forward-design verification stage). |

### 1.1 Geometry constraint: it must be reproducible by the 1D diffusion solver

This is the hard constraint of the whole project. The LF solver
(`solvers/LF_diffusion/core/MG1D_eigenvalue_nregions.py`) is a **1D radial,
multigroup, finite-volume diffusion eigenvalue solver on concentric
regions**. So the HF model may only use geometries that collapse to that:

- **`geometry` ∈ `{'spherical', 'cylindrical', 'slab'}`** — nothing else.
  `MC_solver.build_geometry()` builds concentric `Sphere` / `ZCylinder` /
  mirrored `XPlane` surfaces only.
- **Concentric, ordered centre → outside.** `cfg.regions` is a tuple of
  `RegionSpec`, each with an `outer_radius`. Region *i* is the shell between
  radius *i−1* and *i*. No lattices, no azimuthal or axial structure.
- **The outer surface is `vacuum` in OpenMC**, so the LF boundary condition
  must be `bc_type='vacuum'` to match (see [§2.1](#21-mirroring-the-case-in-the-lf-config)).
- **Cylinders are axially infinite in effect**: `axial_half_height` planes at
  ±H are **reflective**, which is what makes the problem genuinely 1D radial.
  Changing them to vacuum would introduce axial leakage the LF solver cannot see.
- **Heterogeneity inside a region is homogenised away.** The thesis fuel region
  uses the `'mix'` material preset (homogeneous U + water at volume fraction
  `f_mod`) precisely so that one MGXS set per region is physically meaningful.

The thesis case `CFG_ROD`: **1D cylinder, 3 concentric regions, 2 energy groups**

| # | Region name | Material preset | Swept knobs |
|---|---|---|---|
| 0 | `b4c_rod` | `b4c` (control rod) | `outer_radius`, `cr_fraction` |
| 1 | `fuel_annulus` | `mix` (homogenised UO₂ + water) | `outer_radius`, `enrichment`, `f_mod` |
| 2 | `water` | `water` (reflector) | `outer_radius` |

Energy groups: `EnergyGroupSpec(G=2, boundaries=(0., 0.625, 20.0e6))` eV —
thermal / fast. Mesh: `mesh_size = 1.0` cm.

### 1.2 Where the case is defined

Everything is a `NamedTuple`, so a "case" is one literal in `config_run.py`:

```python
CFG_ROD = MCConfig(
    geometry      = 'cylindrical',
    regions       = (RegionSpec(name=…, outer_radius=…, material=…, …), …),
    energy_groups = EnergyGroupSpec(G=2, boundaries=(0., 0.625, 20.0e6)),
    settings      = SolverSettings(batches=…, inactive=…, particles=…, …),
    mesh_size     = 1.,
)
```

Key `SolverSettings` knobs (all documented inline in `config_def.py`):

| Knob | Thesis value | Meaning |
|---|---|---|
| `particles` | 20 000 | particles per batch |
| `batches` / `inactive` | 200 / 50 | starting batches; 50 discarded for source convergence |
| `keff_trigger_std` | `4e-4` | **OpenMC extends the run in place** until σ(k) < 40 pcm |
| `trigger_max_batches` | 400 | ceiling on that in-run extension |
| `axial_half_height` | 50.0 | ±z reflective planes (cylinder) |
| `entropy_mesh_dim` | (20,20,1) | Shannon-entropy mesh for convergence diagnostics |
| `cmfd_on` | `False` | CMFD acceleration available but off |
| `xs_output_path` | — | cumulative results CSV, one row appended per run |

Materials are **presets**, built in `MC_solver.build_region_material()`:
`'fuel'`, `'water'`, `'b4c'`, `'mix'`, `'void'`, or `'custom'` with a
fully-formed `openmc.Material` (see `CFG_STACY_APPROX` for an example of
`'custom'`).

### 1.3 Defining the parameter sweep (bounds)

> **⚠️ Bounds live in two places and only one of them is used.**
> `SWEEP_ROD` / `ParamSpec` in `config_run.py` is **dead code** — nothing
> imports it. The LHS campaign reads the **`BOUNDS` dict at the top of
> `TD_generation.py`**. Edit that one.

```python
# solvers/HF_openMC/TD_generation.py
BOUNDS = {
    'r0_b4c_rod_outer_radius':      (1.0,  6.0),
    'r0_b4c_rod_cr_fraction':       (0.0,  1.0),
    'r1_fuel_annulus_outer_radius': (10.0, 40.0),
    'r1_fuel_annulus_enrichment':   (1,    10.0),
    'r1_fuel_annulus_f_mod':        (0.40, 0.8),
    'r2_water_outer_radius':        (40.0, 80.0),
}
```

- **Key naming is load-bearing**: `r{region_index}_{region_name}_{knob}`.
  `MC_solver.apply_sample()` parses these generically, and the *same strings*
  become the regression input column names and the `param_names` in the `.npz`.
  Known knobs: `outer_radius`, `enrichment`, `f_mod`, `cr_fraction`.
- Comment out a line to freeze that parameter at its nominal value.
- `N_SAMPLES` and `LHS_SEED` are read from the environment (set in
  `run_many_openmc.sh`); sampling is Latin Hypercube with `criterion='maximin'`
  via `pyDOE3`.
- Output folder: `DATA_DIR` near the top of `TD_generation.py`, currently
  `training_data/2_aug/seed_<LHS_SEED>/`.

Helpers for choosing bounds already exist: `data/reports/suggest_bounds.py`
→ `data/reports/suggested_bounds_report.md` and `bounds_A1…B2.csv`.

### 1.4 Outputs of interest

Per accepted run, `MC_solver.save_results()` appends one row to
`LHS_full_dataset.csv` containing **the 6 parameters + `keff` + `keff_std` +
every MGXS value**. That single CSV is the paired
`(p-vector, keff, MGXS)` dataset the rest of the project consumes.

**MGXS.** `build_mgxs_library()` tallies, per cell domain (= per region),
`domain_type='cell'`, `by_nuclide=False`:

```
diffusion-coefficient, absorption, nu-fission, scatter matrix, chi
```

`extract_mgxs_vector()` flattens them to keys
**`{region_name}_{xs_type}_g{i}`**, 1-indexed, with the scatter matrix stored as
`scatter matrix_g1…g{G²}` in **row-major (from-group × to-group)** order.
For G=2 and 3 regions that is `(4·2 + 2²) = 12` values per region → **36 columns**.

**Other artefacts in the campaign folder:**

| File | Contents |
|---|---|
| `LHS_full_dataset.csv` | **the dataset**: params + keff + keff_std + 36 MGXS |
| `<csv>_meta.json` | column schema sidecar written by `save_meta()`: `input_cols`, `output_cols`, `chi_cols`. **The LF regression reads this** |
| `checkpoint.csv` | lightweight per-run summary, rewritten every 10 runs |
| `diagnostics.csv` | convergence diagnostics (entropy, k windows, flags) |
| `timings.csv` | wall time, OpenMC transport/active/inactive times, retry history |
| `study_scalar_summary.csv` | scalars only: keff, PPF, power densities, runtimes, flags |
| `generation_log.txt` | human-readable log, one block per run |
| `runs/run_XXXX/` | per-run OpenMC XML + statepoint + convergence plots |
| `flux_plots/` | radial flux profile PNG per run |

Derived quantities worth knowing about, computed from the flux mesh tally:
`radial_power_peaking_factor` (fuel-only PPF, used as the second objective in
the forward-design study), `fuel_{avg,max}_power_density`,
`flux_peak_factor_full`, plus `converged_flag` / `keff_std_converged_flag`.

### 1.5 Running a campaign

```bash
cd solvers/HF_openMC
# edit BOUNDS, DATA_DIR in TD_generation.py; N_SAMPLES / LHS_SEED in the .sh
sbatch run_many_openmc.sh          # mc-env, 32 cpus, 24 h, savio3
squeue -u $USER
tail -f slurm_logs/td_generation_openmc_<jobid>.out
```

Cost reference: ~65 s per OpenMC solve at thesis settings, so ~500 samples per
24 h job per seed. Run several seeds in parallel as separate jobs and merge.

> **⚠️ `TD_generation.generate_data()` deletes `LHS_full_dataset.csv` and all
> sidecar files at startup.** Always point a new campaign at a fresh
> `DATA_DIR`, or you will lose a previous sweep.

**Merging campaigns into datasets.** Drop CSVs anywhere under `data/files/` and
run `python data/files/build_merged_datasets.py` (light, login-node fine). It
de-duplicates cases by the rounded 6-parameter key across all historical
exports and writes:

- `dataset_keff_xs.csv` — cases with the complete 36-column MGXS set
- `dataset_keff.csv` — every case with keff + keff_std
- `dataset_incomplete.csv` — cases with missing fields (named in `missing_fields`)
- `dataset_extras.csv` — optional columns: `solver_keff`, `delta_pcm`, PPF,
  power densities, runtimes

Reports on these live in `data/reports/` (`analyse_dataset_keff.py`,
`analyse_highfidelity.py` → `*_report.md`).

**Converting to the PEDS `.npz`.** The script is
`archive/standalone_tools/NTcode_config_data_scripts/csv_cut_npz.py` (archived
but functional, and the only place this conversion is written down). It does
two things: `cut_dataset()` applies inclusion filters (e.g. keep
0.8 ≤ keff ≤ 1.2, drop `b4c_r > 6`, `f_mod < 0.4`, `fuel_r < 10`), then
`csv_to_npz()` writes the six arrays PEDS expects:

| Array | Shape | Meaning |
|---|---|---|
| `params` | (N, 6) f32 | parameters **min–max normalised to [0,1]** → the NN input |
| `params_raw` | (N, 6) f32 | physical values → fed to the solver |
| `keffs` | (N,) f32 | OpenMC reference |
| `param_names` | (6,) str | column names |
| `bounds_lo` / `bounds_hi` | (6,) f32 | normalisation bounds |

> **⚠️ `bounds_lo/hi` are the min/max of the *filtered dataset*, not the LHS
> bounds.** Two datasets filtered differently have different normalisations, so
> a checkpoint is only valid for the `.npz` it was trained on.

> **▶ Bottom line for §1**
> - Only concentric 1D geometries (sphere/cylinder/slab) with a vacuum outer
>   surface and reflective axial planes are admissible.
> - 📌 Case: `solvers/HF_openMC/config_run.py`. 📌 Sweep bounds + campaign
>   folder: `solvers/HF_openMC/TD_generation.py`. `SWEEP_ROD` is dead code.
> - Output of record is one CSV row per run: **6 params + keff + keff_std + 36
>   MGXS columns**, with the `*_meta.json` sidecar describing the schema.
> - A campaign wipes its output folder on start. Use a fresh `DATA_DIR`.

---

## 2. The 1D diffusion low-fidelity model

**Folder: `solvers/LF_diffusion/`**

| File | Role |
|---|---|
| `config_def.py` | Types: `GeometryConfig`, `MaterialSpec`, `BoundarySpec`, `BoundaryCondition`, `MatProperties`. All five boundary-condition types are documented inline. |
| `config_run.py` | **📌 The LF cases**: `GEO_CYL` (thesis), `GEO_HET`, `GEO_HOM`, `GEO_STACY`. |
| `diffusion_solver.py` | Entry point: XS prediction (`predict_xs`), XS tensor layout, callable builders, `run_diffusion_solver`, plotting. **📌 Also where the regression model folder is chosen.** |
| `core/MG1D_eigenvalue_nregions.py` | Finite-volume multigroup diffusion matrices + forward/adjoint eigenvalue solves. |
| `core/inverse_power.py` | Inverse power iteration for the generalised eigenproblem. |
| `regression_model/<name>/` | Training CSV + fitted `polyreg_model/` artefacts. |

### 2.1 Mirroring the case in the LF config

`GEO_CYL` in `config_run.py` is the LF twin of `CFG_ROD`. These must agree:

| Must match | HF (`MCConfig`) | LF (`GeometryConfig`) |
|---|---|---|
| geometry kind | `geometry='cylindrical'` | `geometry='cylindrical'` |
| number & order of regions | `regions=(…)` centre→out | `regions=(MaterialSpec(id, region_index), …)` |
| radii | `RegionSpec.outer_radius` | `BoundarySpec.radius` |
| energy groups | `EnergyGroupSpec.G` | `G` |
| spatial mesh | `mesh_size` | `mesh_size` |
| outer boundary | OpenMC `vacuum` surface | `BoundaryCondition(bc_type='vacuum')` |

**⚠️ `MaterialSpec.region_ID` must equal the HF `RegionSpec.name`** (`b4c_rod`,
`fuel_annulus`, `water`). Those strings are the prefixes of the MGXS column
names, and `predict_xs()` looks the regression outputs up by
`f'{region_ID}_{xs_type}_g{g}'`. A typo yields a silent zero cross-section.

Boundary-condition options (all in `config_def.BoundaryCondition`, converted to
`[A,B,C]` coefficients by `bc_to_coeffs()` for `A·φ(R) + B·φ'(R) = C`):
`reflective`, `vacuum`, `dirichlet`, `albedo` (needs `alpha`), `partial_current`
(needs `Jin`). The thesis uses `vacuum`.

**Geometry as a function of the parameter vector.** At training/inference time
the geometry is not read from `config_run.py` — it is rebuilt per sample by
`PEDS_subdivision.context.update_geo(geo, params_raw)`, which maps

```
params_raw = [b4c_r, cr_frac, fuel_r, enrichment, f_mod, water_r]
   →  boundaries = (CR_outer=p[0], core_outer=p[2], moderator_outer=p[5])
   →  mat_properties = (cr_fraction=p[1], enrichment=p[3], f_mod=p[4])
```

**⚠️ `update_geo` is hard-coded to the 3-region / 6-parameter case.** It is the
first function to rewrite for a new problem.

### 2.2 The XS tensor layout

The single data structure the NN, the solver and the VJP all share:

```
xs_tensor : np.ndarray, shape (N_regions, 4·G + G²)
```

For G=2 → 12 slots per region, indexed by `xs_layout(G)`:

| Index (G=2) | Slot |
|---|---|
| 0, 1 | `D` — diffusion coefficient, per group |
| 2, 3 | `Sigma_a` — absorption |
| 4, 5 | `nuSigma_f` — ν·fission |
| 6, 7, 8, 9 | `Sigma_s` — scatter matrix, row-major `[from, to]` |
| 10, 11 | `chi` — fission spectrum |

Rows are ordered by `MaterialSpec.region_index` (centre → outside).
`fn_xs_per_region(G)` and `xs_layout(G)` are the only correct way to index it —
never hard-code offsets.

### 2.3 The polynomial regression baseline

**Role.** Monte Carlo cannot run at inference time, so a polynomial regression
provides the *baseline* cross-sections Σ_base(p) that the NN then corrects:

```
Σ_final = exp(δ_NN(p)) · Σ_base(p)
```

It is the **epoch-0 state of PEDS**: with δ = 0 the model is exactly the
coarse diffusion solver, which is why the training curves start from the
"diffusion-only" error level and why the NN is learning a *correction*, not a
cross-section from scratch.

**Pipeline** (`diffusion_solver.predict_xs`, loaded once at import):

1. `_regression_inputs(geo)` builds the input row. Column order comes from
   `input_cols` in the **`xs_model_meta.json` sidecar**, and each column name is
   parsed by `_parse_knob()` (`r{i}_{name}_{knob}`) and resolved against the
   live `GeometryConfig` via `_KNOB_EXTRACTORS`. No feature names are hard-coded.
2. `xs_scaler.pkl` (standardisation) → `xs_poly_transformer.pkl` (polynomial
   feature expansion) → `xs_regression_model.pkl` (linear fit). All three are
   `joblib.load`-ed at module import.
3. Predictions are mapped to the named columns `output_cols` minus `chi_cols`.
4. `_enforce_chi()` **overrides** the regressed χ with the physical convention:
   χ = 1 in the fastest group and 0 elsewhere for any region with
   νΣ_f > 0, and χ ≡ 0 for non-fissile regions.
5. Values are packed into the `(N_regions, 12)` tensor.

**Which model is loaded — and this is a real trap:**

```python
# solvers/LF_diffusion/diffusion_solver.py  (~line 32)
DATA_FOLDER   = PROJECT_ROOT / "solvers" / "LF_diffusion" / "regression_model" / "2k_samples_bounded"
REG_MODEL_PATH = DATA_FOLDER / "polyreg_model"
```

> **⚠️ The regression model is selected by editing this hard-coded path**, not by
> config. Available folders: `2k_samples_bounded` (**active**, the thesis
> baseline), `17jul`, `CR_1000samples`, `heterogeneous_model`,
> `homogeneous_model`. Note the asymmetry worth remembering: the *regression* is
> fitted on `2k_samples_bounded/LHS_XS_0.8-1.2.csv` while the *NN* is trained on
> `data/highfidelity/17jul_0.8_1.2.npz`. Changing this path silently changes
> every baseline, every epoch-0 metric, and invalidates existing checkpoints.

**Fit quality** is recorded per cross-section in
`regression_model/<name>/polyreg_model/xs_model_metrics.csv` (R², RMSE, χ², p).
The best-fit entries exceed R² = 0.999; sort that file ascending to find the
weakest ones — they are where the NN correction does the most work.
`xs_polynomial_coefficients.csv` holds the fitted coefficients.

> **⚠️ The script that *fits* the regression is not in this repository.** Only
> the artefacts (`xs_scaler.pkl`, `xs_poly_transformer.pkl`,
> `xs_regression_model.pkl`, `xs_model_meta.json`, `xs_model_metrics.csv`,
> `xs_polynomial_coefficients.csv`) are versioned. To fit a new one you must
> rewrite it: `StandardScaler` → `PolynomialFeatures` → `LinearRegression` on an
> `LHS_full_dataset.csv`, dumping the three pickles plus a `xs_model_meta.json`
> with `input_cols` / `output_cols` / `chi_cols`. Keep that schema exactly —
> `predict_xs()` is driven by it.

### 2.4 Running the LF solver on its own

**Baseline (regression) XS, one case:**

```bash
conda activate jax-env
cd /global/home/users/caterinafrau/PEDS_NT
python -m solvers.LF_diffusion.diffusion_solver     # uses GEO_CYL as written
```

The `__main__` block is exactly the two-step API you should reuse:

```python
from solvers.LF_diffusion.config_run import GEO_CYL as GEO
from solvers.LF_diffusion.diffusion_solver import get_xs_basedon_geo, run_diffusion_solver
from PEDS_subdivision.context import update_geo

geo = update_geo(GEO, params_raw)          # geometry for this p-vector
xs  = get_xs_basedon_geo(geo)              # == predict_xs(geo), shape (3, 12)
k, phi_fwd, phi_adj = run_diffusion_solver(xs, geo)
```

`run_diffusion_solver` runs **both** the forward and the adjoint eigenvalue
problem (`DiffusionEigenvalue_MG` / `..._MG_adjoint`) and returns fluxes
**volume-normalised** so that Σ_g Σ_i φ·V = 1 — that normalisation is what makes
the adjoint-based gradients physically meaningful, so don't renormalise.

**Feeding exact OpenMC MGXS for a specific parameter vector.** This is the
"is the error from the XS or from the diffusion approximation?" experiment, and
there is a finished script for it:

```
example_applications/additional_studies/further_checks/exactXS_vs_poly_full/true_xs_study.py
          (launcher: run_true_xs_study.sh;  summary: FULL_STUDY_SUMMARY.md)
```

It matches each row of `data/files/dataset_keff_xs.csv` to its LHS sample, packs
the 36 true MGXS columns into the `(3, 12)` tensor using `xs_layout(G)`, and
runs diffusion with true XS and with regression XS against the same OpenMC
keff. **Copy this script as the template whenever you want to bypass
`predict_xs()`** — the only thing you ever need to do is build the tensor
yourself and call `run_diffusion_solver(xs_tensor, geo)`.

Related ready-made studies:

- `further_checks/mesh_conv_study/` — mesh-size convergence (`mesh_size` sweep);
  run this when you change `mesh_size`, since the LF↔HF bias depends on it.
- `further_checks/three_flux_plots/` — forward/adjoint/baseline flux overlays.

### 2.5 What the NN actually optimises

The NN's output space is the **`xs_tensor`**, not keff. Concretely:

- the NN emits one log-ratio δ per `(region, XS slot)` → 3 × 12 = **36 numbers**;
- `XS_MASK` (built in `context.build_xs_mask`) zeroes the slots that must stay
  exactly zero (e.g. νΣ_f in the control rod, χ in non-fissile regions);
- χ is renormalised inside the model (`PEDSModel._normalize_chi`);
- the corrected tensor goes into the *unchanged* diffusion solver, and the
  resulting keff is compared with OpenMC.

That is the whole PEDS idea: **the network never sees keff directly, it only
reshapes a physically-structured cross-section tensor**, and the physics solver
turns that into the prediction.

> **▶ Bottom line for §2**
> - 📌 LF case: `solvers/LF_diffusion/config_run.py`; it must mirror the HF
>   `MCConfig` region-for-region, with identical `region_ID` ↔ `RegionSpec.name`.
> - The shared data structure is `xs_tensor` of shape `(N_regions, 4G+G²)`,
>   always indexed via `xs_layout(G)`.
> - The polynomial regression supplies Σ_base = the epoch-0 PEDS state. Its
>   folder is **hard-coded in `diffusion_solver.py`** (`2k_samples_bounded`), and
>   **no fitting script is versioned** — only the pickles.
> - To test exact MGXS, build the tensor yourself and call
>   `run_diffusion_solver`; use `true_xs_study.py` as the template.

---

## 3. PEDS and the neural network

**Folders: `models/` (implementation), `config_and_run/` (configuration + launch)**

| File | Role |
|---|---|
| `config_and_run/config_peds.py` | **📌 The only file you edit to configure a run.** Every knob is also overridable by a `PEDS_*` environment variable. |
| `config_and_run/run_peds.sh` | Slurm launcher + sweep over (train size × seed × decay epochs). |
| `config_and_run/smoke_test_peds.sh` | Tiny end-to-end run; use after any refactor. |
| `models/PEDS.py` | `GeneratorNN`, `PEDSModel`, the training loop, checkpointing, run logging. |
| `models/matrix_JAX_optimized.py` | JAX diffusion matrix assembly and the adjoint VJP kernels. |
| `models/PEDS_subdivision/context.py` | Shared derived constants and caches; `update_geo`, `build_xs_mask`. |
| `models/PEDS_subdivision/physics_solver.py` | Forward solve + **custom VJP** (`NTdiff_solver`, `NTdiff_solver_batch`), parallel pre-solve worker. |
| `models/PEDS_subdivision/data_loading.py` | Splits (3 strategies), split cache, φ features, XS baselines. |
| `models/PEDS_subdivision/metrics.py` | reactivity discrepancy in pcm and the metric dict. |
| `models/PEDS_subdivision/logging_csv.py`, `plotting.py` | CSV logs and figures. |
| `models/PEDS_core/diagnostics.py` | Custom-VJP vs finite-difference gradient check. |
| `models/run_mlp_baseline_compare.py`, `run_mlp_augmented_compare.py` | Pure-MLP and augmented-MLP baselines (no embedded solver). |

### 3.1 The training set

**Default and most-used: `data/highfidelity/17jul_0.8_1.2.npz`** — 2648 samples,
filtered to 0.8 ≤ keff ≤ 1.2, six parameters. This is the dataset behind the
thesis results.

```python
# config_and_run/config_peds.py
DATA_FILENAME = os.environ.get("PEDS_DATA_FILENAME", "17jul_0.8_1.2.npz")
TRAIN_SIZE, VAL_SIZE, TEST_SIZE = 1000, 500, 300
```

**Train = 1000 is the thesis operating point** (performance saturates there;
see `example_applications/additional_studies/trainsize_study/`).

Other datasets in `data/highfidelity/`: `LHS_0.8_newbounds.npz`,
`merged_lhs_0.8_1.2.npz`, `complete_LHS_2900.npz`, `MC_1315.npz`, `MC_2200.npz`,
plus `different_keffrange/`, `parameters_study/`, `other_DS/`.
`split_cache/` holds cached split indices.

**Splitting.** Three strategies, selected in `config_peds.py`; the first one
wins if several flags are on:

| Flag | Function | Behaviour |
|---|---|---|
| `USE_PARAM_STRATIFIED=True` **(default)** | `load_data_param_stratified` | Stratifies on the *parameters* (fuel radius primary, `PARAM_STRAT_BINS=5` bins per parameter), keff-matched within each fuel-radius bin |
| `USE_BALANCED_RANGES` | `load_data_balanced_ranges` | Fixed keff ranges, balanced counts |
| neither | `load_data` | keff quantile-stratified (10 bins), optional disk cache via `USE_SPLIT_CACHE` |

**Seeds:** `SEED` (= `TRAIN_SEED` = `VAL_SEED`) varies across a study;
**`TEST_SEED` is fixed at 0 so the test set is identical across all runs** and
cross-run comparisons are meaningful. The realised split is written to
`split_log.csv` in the run folder, which is what the analysis tools read back.

### 3.2 The PEDS config file

`config_and_run/config_peds.py` is self-documenting; the knobs that matter:

```python
# data & sizes
DATA_FILENAME, TRAIN_SIZE=1000, VAL_SIZE=500, TEST_SIZE=300
# split
USE_PARAM_STRATIFIED=True, PARAM_STRAT_BINS=5, USE_SPLIT_CACHE=False
# optimisation
BATCH_SIZE=32, EPOCHS=70, DECAY_EPOCHS=70, LR_MAX=2e-4, LR_MIN=5e-6
# checkpoint selection / early stop
EMA_ALPHA=0.1, MIN_SAVE_EPOCH=30, PATIENCE=15
# architecture
HIDDEN_SIZES=[128,256,128], N_REGIONS=3, N_GROUPS=2
LOG_RATIO_CLIP_LO=-1.8, LOG_RATIO_CLIP_HI=0.5
# diagnostics & output
RUN_GRAD_CHECK=True, XS_HEATMAP_EPOCHS, SUBPLOT_SAMPLE_INDICES=[22,44,55,68,82]
STUDY_NAME="precise_param_strat", RUNS_ROOT=config_and_run/RUNS
EXP_NAME=f"train_{TRAIN_SIZE}_seed_{SEED}"
LOG_DIR=RUNS_ROOT/STUDY_NAME/EXP_NAME
```

Two conventions to respect:

- **Every knob reads an environment variable first** (`PEDS_TRAIN_SIZE`,
  `PEDS_SEED`, `PEDS_EPOCHS`, …), which is how `run_peds.sh` sweeps without
  editing files. `python config_and_run/config_peds.py` prints the resolved
  configuration — do this before submitting.
- **⚠️ Keep the `train_<N>_seed_<S>` folder-name convention.** The analysis
  scripts parse train size and seed out of the directory name; rename it and
  post-processing finds nothing.

### 3.3 The neural network in detail

Defined in `models/PEDS.py` as `GeneratorNN` (Flax NNX), wrapped by `PEDSModel`.

**Inputs** (per sample, concatenated):

| Block | Dim | Source |
|---|---|---|
| normalised geometry parameters | 6 | `params` from the `.npz` (min–max to [0,1]) |
| φ features | `G·3` = 6 | `compute_phi_features()`: region-averaged baseline flux, per group per region, z-scored on train statistics |
| → trunk input | **12** | `input_dim = 6 + n_phi_feats` |
| log-baselines (injected at the head) | 3×12 = **36** | `log(Σ_base)`, z-scored per slot on train statistics |

The φ features are the mean baseline flux in each region for each group
(order: g0-CR, g0-core, g0-moderator, g1-CR, …), computed by running the
*baseline* diffusion solve once per sample before training. They give the
network a cheap sense of the flux shape without another solve.

**Architecture:**

```
trunk:  12 → 128 → 256 → 128     (Linear + ReLU, He/Kaiming init)
head :  concat(trunk_128, log_baseline_36) = 164  →  36   (Linear, ZERO init)
out  :  reshape → (3 regions, 12 XS slots)  = log-ratio offsets δ
```

≈ 73.5 k parameters. Two design choices worth understanding:

- **The head is zero-initialised**, so δ = 0 at epoch 0 and PEDS *starts exactly
  at the diffusion baseline*. Training can only improve on it.
- **The log-baselines are fed to the head, not the trunk** (a skip connection),
  so the correction is conditioned on the magnitude of the quantity it scales.

**Output path** (`PEDSModel.__call__`):

```
δ  =  generator(geoms, log_baselines, phi)
δ  ←  clip(δ, LOG_RATIO_CLIP_LO=-1.8, LOG_RATIO_CLIP_HI=+0.5)   # hard clip, fwd AND bwd
Σ  =  exp(δ) · Σ_base · XS_MASK
Σ  ←  _normalize_chi(Σ)                      # Σ_g χ_g = 1 if fissile, else 0
k  =  NTdiff_solver_batch(Σ, params_raw)     # the 1D diffusion eigenvalue solve
```

The clip is a **hard** clip by deliberate choice (`clip_log_ratios`, with the
reasoning in its docstring): a straight-through estimator would keep feeding
Adam unbounded gradients and the bound would effectively vanish. In physical
terms δ ∈ [−1.8, 0.5] means corrections from ×0.165 to ×1.65.

> **⚠️ The docstrings of `GeneratorNN` are stale** — they say "`[batch, 7]` →
> 128 → 128 → 64". The real shapes come from `HIDDEN_SIZES` and
> `n_phi_feats` as above. Trust the config, not the comment.

**Gradients — the part that makes this work.** The diffusion solver is NumPy/
SciPy, so it is wrapped in `jax.pure_callback` and given a **custom VJP**
(`physics_solver.py`: `_NTdiff_fwd` / `_NTdiff_bwd`, batched as
`_NT_batch_fwd` / `_NT_batch_bwd`). ∂k/∂Σ is computed analytically from the
**adjoint flux** via the standard perturbation formula, with the adjoint
normalised so ⟨φ†, Fφ⟩ = 1. The kernels live in
`models/matrix_JAX_optimized.py` and are `jit`+`vmap`-ed once per run over
padded arrays (`N_FLAT_MAX`, `I_MAX_SCAN` from `context.init`), so XLA compiles
one kernel for all batches.

`RUN_GRAD_CHECK=True` (default) validates this against finite differences
*before* training starts and writes
`gradient_FDcheck/grad_check_{per_sample,summary}.csv`.
**Check that file first whenever training diverges or a new geometry is wired
in** — a broken VJP looks exactly like bad hyperparameters.

### 3.4 Training, validation and testing

Training loop in `PEDS.train()`:

1. **Load + split** as per §3.1; write `split_log.csv`; print keff-bin counts
   and representation diagnostics for all three sets.
2. **Gradient check** (optional) against finite differences.
3. **Pre-compute**: φ-feature normalisation (train mean/std), baseline XS for
   train/val (`compute_batch_baselines`), log-baseline mean/std per slot
   (train only).
4. **Epoch 0 baseline**: run the diffusion solver with *pure regression* XS on
   train and val and log it as epoch 0. **This is your reference line for
   every "how much did PEDS improve?" statement.**
5. **Per epoch**: shuffle → per batch: NN forward to concrete XS → **parallel
   pre-solve** of the eigenvalue problems across `N_WORKERS` processes
   (`min(SLURM_CPUS_PER_TASK, BATCH_SIZE)`, results cached in
   `_PRESOLVE_CACHE`) → one batched forward+backward → Adam update.
   - **Loss: plain MSE on keff**, `mean((k_pred − k_ref)²)`. No regularisation.
   - **Optimiser:** Adam, `clip_by_global_norm(0.5)` applied *before* Adam,
     `warmup_cosine_decay_schedule` (10 % warmup, cosine `LR_MAX→LR_MIN` over
     `DECAY_EPOCHS`).
6. **Validation every epoch** → full metric set; log-ratio saturation
   diagnostics; per-sample CSV rows.
7. **Checkpointing:** best model selected on an **EMA (α = 0.1) of the
   validation mean |Δρ|**, only from `MIN_SAVE_EPOCH=30` onward (skips early
   oscillation). **Early stop** after `PATIENCE=15` epochs without EMA
   improvement. Figures (XS subplots, flux plots) at `XS_HEATMAP_EPOCHS`.
8. **At the end:** reload the best checkpoint, dump final XS CSVs for
   train/val/test, evaluate the **held-out test set** (untouched until this
   point), print the 10 worst validation samples with their geometries, write
   `training_history.png` and a timing report.

**Metrics** (`PEDS_subdivision/metrics.py`). The headline quantity is the
reactivity discrepancy in pcm:

```
|Δρ| = |k_pred − k_ref| / (k_pred · k_ref) × 1e5
```

reported as `mean_pcm`, `median_pcm`, `p95_pcm`, `std_pcm`,
`frac_below_650` (the β_eff-scale threshold), `frac_below_100`, plus `mse_k`
and `MAE_k`. The analysis stage can alternatively report
`|Δk| = |k_pred − k_ref| × 1e5` (`USE_DELTA_K=true` in `run_analysis.sh`),
which is mode-insensitive; thesis tables use that variant in the `*_dk` folders.

### 3.5 Launching, monitoring and where results appear

```bash
# 0. Sanity-check the resolved config (login node, instant)
conda activate jax-env && python config_and_run/config_peds.py

# 1. Smoke test after any code change (~minutes)
cd config_and_run && sbatch smoke_test_peds.sh     # → RUNS/_smoke_test/
#    success looks like "SMOKE TEST COMPLETED OK" at the end of the .out log

# 2. The real study (default: 5 seeds as a Slurm array)
cd config_and_run && sbatch run_peds.sh

# 3. Sweep without editing files
PEDS_SWEEP_TRAIN_SIZES="500 1000 1500" PEDS_SWEEP_SEEDS="0 1 2" \
  sbatch --array=0-8 run_peds.sh
```

`run_peds.sh` flattens `(TRAIN_SIZES × SEEDS × DECAY_EPOCHS_LIST)` onto
`SLURM_ARRAY_TASK_ID` and exports `PEDS_TRAIN_SIZE`, `PEDS_SEED`,
`PEDS_DECAY_EPOCHS`. Default resources: `savio3`, 32 CPUs, 10 h,
account `fc_neutronix`. **⚠️ Set `--array=0-(TOTAL−1)` to match your sweep** —
the script aborts on a mismatch.

**Is it running?**

```bash
squeue -u $USER
tail -f config_and_run/slurm_logs/peds_<arrayjob>_<task>.out
tail -f config_and_run/RUNS/<STUDY_NAME>/train_1000_seed_0/train_log_train_1000_seed_0.txt
```

Both logs carry the same stream (`PEDS.py` tees stdout). Per epoch you see the
learning rate, gradient-norm statistics with the clipped fraction, and one
`TRAIN`/`VAL` metric line each; new best checkpoints are announced with
`✓ new best val_mean_pcm = … — saved → …`.

**Per-run output folder** (`LOG_DIR = config_and_run/RUNS/<study>/train_<N>_seed_<S>/`):

| Path | Contents |
|---|---|
| `train_log_<exp>.txt` | full stdout |
| `epoch_metrics.csv` | **one row per epoch, all aggregate metrics** — the file to plot |
| `keff_epoch_log_{train,val}.csv` | one row per sample per epoch: `keff_openmc`, `keff_peds`, `delta_rho_pcm` |
| `split_log.csv` | realised train/val/test indices + keffs |
| `checkpoints/best_model.pkl` | best model state (EMA-selected) + metadata |
| `checkpoints/last_checkpoint.pkl` | last-epoch model **and** optimiser state (for resuming) |
| `XS/final_xs_{train,val}.csv`, `XS/test_final_xs.csv` | final corrected cross-sections per sample |
| `XS/logratio_saturation.csv` | per-epoch clipping statistics |
| `gradient_FDcheck/` | custom-VJP vs finite-difference check |
| `xs_subplots/`, `flux_plots/` | per-sample XS and flux figures at `XS_HEATMAP_EPOCHS` |
| `training_history.png`, `val_reactivity_logscale.png` | training curves |
| `code_snapshot_<exp>.py`, `config_snapshot_<exp>.py`, `config_resolved_<exp>.txt` | **exact code + config used**, saved at launch |

The code snapshot is not decoration: `evaluate_test_metrics.py` re-`exec`s each
run's snapshot by default, so an old run is evaluated with the code it was
trained with. Pass `--use-current-peds` to override.

**Resuming** is manual: set `RESUME_FROM = LAST_CKPT_PATH` inside
`PEDS.train()` (currently `None`).

### 3.6 What results to focus on

In priority order:

1. **Validation / test `mean_pcm` compared with the epoch-0 row** of the same
   run. The thesis headline — ~90 % reduction of the diffusion-vs-OpenMC
   reactivity discrepancy (88.1 % across stratified bins) — is exactly this
   ratio. An absolute pcm number means nothing without its epoch-0 reference.
2. **`median_pcm` and `p95_pcm`.** Median shows the typical case, p95 the tail.
   Mean improving while p95 does not means you fixed the bulk and left a hard
   subpopulation — look at the "TOP 10 WORST SAMPLES" block in the log and at
   `analysis/param_error/` to see which parameter region it is.
3. **`frac_below_650`** — fraction within one β_eff-scale; the "is this useful
   to a designer?" metric.
4. **Test vs validation gap.** Test is touched once, at the end, with a fixed
   seed. A large gap means the split or the early-stopping criterion is leaking.
5. **Cross-seed spread.** Always read the 5 seeds together
   (`analysis/training_diagnostics/cross_run_train_val_mean_std_summary.png`);
   single-seed differences of a few tens of pcm are noise.
6. **Gradient-check summary and log-ratio saturation.** If a large fraction of
   δ sits at the clip bounds, the network is asking for corrections the bounds
   forbid — either the baseline regression is wrong for that region or the
   bounds are too tight.

### 3.7 Analysis tools

All post-processing lives in **`utils/analysis/`** (see also
`utils/analysis/ANALYSIS_ORGANIZATION.md`), driven by two Slurm launchers in
`utils/`.

**(a) Held-out test evaluation —** `sbatch utils/run_testset.sh`

Runs `utils/analysis/evaluate_test_metrics.py <LOGS_ROOT> --with-val-study`.
Edit the `LOGS_ROOTS` array in the script (or set `PEDS_LOGS_ROOTS`), one Slurm
array task per study root. For every run it reads `split_log.csv`, reloads
`checkpoints/best_model.pkl`, re-evaluates the test set and writes to
`<study>/analysis/testset_results/`:

- `run_train<N>_seed<S>_keff_comparison.csv` — per-sample predicted vs reference
- `test_metrics_all_runs.csv`, `new_test_metrics_summary_by_train_size.csv`
- `keff_dist_*` tables and plots (train/val/test keff distributions)
- `representative_plots/` for the representative run

Useful flags: `--use-current-peds`, `--with-error-scatter` (worst 5 %/10 %
highlighted), `--keff-dist-only` (rebuild distribution tables without
re-evaluating checkpoints), `--split-log alt_split_log.csv`.

**(b) Full suite —** `sbatch utils/run_analysis.sh`

Runs `utils/analysis/run_study_analysis.py`. Edit the `CONFIG (EDIT)` block:
`STUDY_PARENT_FOLDER`, `STUDY_FOLDER`, `XS_SOURCE_RUN`, `USE_DELTA_K`, `TASKS`.
Tasks (`--tasks all` or a comma-separated subset) and what each produces under
`<study>/analysis/`:

| Task | Module | Outputs |
|---|---|---|
| `test-metrics` | `evaluate_test_metrics.py` | as in (a) → `testset_results/` (`_dk/` when `USE_DELTA_K=true`) |
| `param-error` | `param_error_analysis.py` | `param_error/`: error vs each parameter, correlations per keff bin, binned error, parameter importance, distributions by split, `train/ val/ test/` subfolders |
| `pub-keff` | `pub_keff_plots.py` | `pub_keff/`: `keff_parity_{baseline,peds}.png`, `rho_hist_{baseline,peds}.png` — the publication parity plots |
| `xs-stats` | `xs_stats_report.py` | `xs_stats/`: correction magnitude per XS type/region (heatmap, boxplot), clipping summaries per epoch and per region, `xs_statistics_summary.md` |
| `training-diagnostics` | `training_diagnostics.py` | `training_diagnostics/`: per-run curves, `final_epoch_metrics_by_run.csv`, cross-run mean±std summaries |

Defaults for paths live in `utils/analysis/config.py` (`STUDY_PARENT_FOLDER`,
`STUDY_FOLDER`, `XS_SOURCE_RUN`) and are overridden by the CLI flags the
launchers pass. Logs: `utils/slurm_logs/study_analysis_<jobid>.{out,err}`.

Also available: `utils/analysis/plots_all.py`, and dataset-level figures in
`utils/inclusion_criteria_plots/`.

> **▶ Bottom line for §3**
> - 📌 Configure in `config_and_run/config_peds.py` (or `PEDS_*` env vars);
>   launch with `sbatch run_peds.sh` from `config_and_run/`; smoke-test first.
> - NN: `12 → 128 → 256 → 128`, head `(128+36) → 36`, **zero-initialised** so
>   epoch 0 = pure diffusion. Output = 36 clipped log-ratio corrections to the
>   cross-section tensor, not keff.
> - Loss is plain MSE on keff; gradients come from the **adjoint flux via a
>   custom VJP**; `RUN_GRAD_CHECK` validates it against finite differences.
> - Headline result = val/test `mean_pcm` **relative to the epoch-0 row**; read
>   median and p95 alongside it, and always across the 5 seeds.
> - Post-process with `sbatch run_testset.sh` then `sbatch run_analysis.sh`
>   from `utils/`. Never rename `train_<N>_seed_<S>` folders.

---

## 4. Applications and stored studies

**`example_applications/main_results/`** — the thesis runs.
`PEDS/train_1000_seed_{0..4}/` plus the complete `analysis/` tree
(`testset_results`, `testset_results_dk`, `param_error`, `param_error_dk`,
`pub_keff`, `pub_keff_dk`, `xs_stats`, `training_diagnostics`, `valset_results`),
and `mlp_baseline/` for the pure-MLP comparison.

**`example_applications/forward_design_study/`** — the inverse/forward design
demonstration (5000 candidate geometries screened for near-criticality and low
radial PPF). See its own `README.md`. Phases, all drivable through
`run_forward_design.sh {screen|verify|diffusion|report|plot|all}`:

| Script | Phase |
|---|---|
| `peds_ensemble_screen.py` | LHS sampling + 5-seed PEDS ensemble + feasibility gate + ranking → `keff_scan_top_candidates.csv` |
| `openmc_verify_topk.py` | OpenMC verification of the shortlist (`mc-env`) |
| `diffusion_only_recheck.py` | the same shortlist with diffusion only, no NN — quantifies what PEDS added |
| `build_report.py` | `forward_design_report.md` |
| `publication_plots/plot_peds_hf_parity.py` | the parity figure |
| `openmc_baseline/`, `bayes_openmc_wip/` | brute-force OpenMC baseline; unfinished Bayesian-optimisation branch |

> **⚠️ `peds_ensemble_screen.py` still has `RUN_DIR_TEMPLATE` pointing at
> `modules/RUNS/precise_param_strat/train_1000_seed_{seed}`.** The `modules/`
> folder no longer exists — repoint it at
> `example_applications/main_results/PEDS/train_1000_seed_{seed}` (or your own
> study) before rerunning.

**`example_applications/additional_studies/`**

| Folder | Question it answers |
|---|---|
| `trainsize_study/` | how many training samples are needed (→ 1000) |
| `different_ranges/` | `subcritical`, `supercritical`, `lhs_0.85_1.15_balanced` keff ranges |
| `grad_check/` | custom-VJP vs finite-difference validation runs |
| `further_checks/exactXS_vs_poly_full/` | true MGXS vs regression XS in the diffusion solver |
| `further_checks/mesh_conv_study/` | spatial mesh convergence |
| `further_checks/three_flux_plots/`, `param_analysis/` | flux overlays, parameter sensitivity |
| `agent_studies/` | learning-rate / activation hyperparameter probes |

**`archive/`** — superseded code kept for reference: the original
Boltzmann/heat-transport PEDS pipeline (`boltzmann_pipeline/`), unused PEDS
modules, and standalone tools. Note that two still-useful utilities live only
here: `csv_cut_npz.py` (CSV → `.npz`) and the split-cache helpers in
`NTcode_config_data_scripts/`.

---

## 5. Reproducing the thesis results

No new OpenMC data is needed — the dataset is committed.

```bash
# 1. Verify the chain
cd config_and_run && sbatch smoke_test_peds.sh

# 2. Confirm the configuration matches the thesis
#    config_peds.py: 17jul_0.8_1.2.npz, 1000/500/300, param-stratified, 5 bins,
#    batch 32, 70 epochs, lr 2e-4→5e-6, hidden [128,256,128], clip [-1.8,0.5]
#    diffusion_solver.py: regression_model/2k_samples_bounded
python config_peds.py

# 3. Train the 5 seeds
sbatch run_peds.sh                      # --array=0-4, seeds 0..4
# NOTE: PEDS_SWEEP_TRAIN_SIZES defaults to 1500 in run_peds.sh;
#       export PEDS_SWEEP_TRAIN_SIZES=1000 for the thesis operating point.

# 4. Evaluate and post-process (point LOGS_ROOTS / STUDY_* at your study)
cd ../utils
sbatch run_testset.sh
sbatch run_analysis.sh
```

Compare against the stored outputs in
`example_applications/main_results/PEDS/analysis/`. The thesis tables use the
`*_dk` variants (`USE_DELTA_K=true`).

To re-derive the published numbers *without* retraining, run the analysis
launchers directly against `example_applications/main_results/PEDS` — the
checkpoints and logs are all there.

---

## 6. Setting up a new problem: checklist

In dependency order. Each step is cheap to verify before moving on.

**A. High-fidelity model**

1. Add a new `MCConfig` to `solvers/HF_openMC/config_run.py`. Respect the 1D
   concentric constraint ([§1.1](#11-geometry-constraint-it-must-be-reproducible-by-the-1d-diffusion-solver)).
   If you need a material the presets don't cover, use `material='custom'` with
   an `openmc.Material`.
2. Edit `BOUNDS`, `DATA_DIR`, `N_SAMPLES`, `LHS_SEED` in `TD_generation.py`;
   point `generate_data(...)` at your config. Keep the
   `r{i}_{region_name}_{knob}` key convention.
3. Run **a handful of samples first** (`N_SAMPLES=5`) and inspect
   `generation_log.txt`, `keff_std`, and the flux plots before committing to a
   full campaign.
4. Full campaign: `sbatch run_many_openmc.sh`.

**B. Low-fidelity model**

5. Add the mirror `GeometryConfig` to `solvers/LF_diffusion/config_run.py`:
   same geometry kind, same number/order of regions, radii, `G`, `mesh_size`,
   and `region_ID` strings **identical** to the HF region names.
6. Fit a new polynomial regression on your `LHS_full_dataset.csv` (script not
   versioned — see the recipe in [§2.3](#23-the-polynomial-regression-baseline)),
   save the three pickles plus `xs_model_meta.json` under
   `regression_model/<your_name>/polyreg_model/`, and **update `DATA_FOLDER` in
   `diffusion_solver.py`**.
7. Verify the LF solver standalone: `python -m solvers.LF_diffusion.diffusion_solver`,
   then run a mesh-convergence check
   (`further_checks/mesh_conv_study/`) for your new `mesh_size`.
8. Sanity-check the baseline against OpenMC on a few cases using
   `true_xs_study.py` as a template — you want to know the diffusion-only error
   level *before* training anything.

**C. PEDS**

9. If the parameter vector changed, rewrite
   **`PEDS_subdivision/context.update_geo()`** (hard-coded to 3 regions and 6
   parameters) and update `PARAM_NAMES` in `config_peds.py`.
10. Build the `.npz` from your CSV with
    `archive/standalone_tools/NTcode_config_data_scripts/csv_cut_npz.py`,
    choosing the inclusion filters deliberately — they set `bounds_lo/hi` and
    hence the input normalisation.
11. Set `DATA_FILENAME`, `N_REGIONS`, `N_GROUPS`, `TRAIN/VAL/TEST_SIZE`,
    `HIDDEN_SIZES` and a new `STUDY_NAME` in `config_peds.py`.
12. Run `smoke_test_peds.sh`. **Then read
    `gradient_FDcheck/grad_check_summary.csv`** — if the custom VJP doesn't
    match finite differences on the new geometry, stop and fix that first.
13. Check the clip bounds: if `XS/logratio_saturation.csv` shows heavy
    saturation on a short run, widen `LOG_RATIO_CLIP_{LO,HI}` or revisit the
    baseline regression.
14. Launch the real study, then post-process with the `utils/` launchers.

**Also worth touching for a genuinely new case:** `N_REGIONS`/`N_GROUPS`
propagate into `xs_layout`, `XS_MASK` and the padded VJP shapes automatically,
but `SUBPLOT_SAMPLE_INDICES` must stay inside your val-set size, and
`PARAM_NAMES` is used for every axis label in the analysis plots.

---

## 7. Known traps and stale code

Collected from the sections above, worst first.

| # | Trap | Where |
|---|---|---|
| 1 | The polynomial regression folder is **hard-coded**, and the *regression* dataset (`2k_samples_bounded`) differs from the *NN* dataset (`17jul_0.8_1.2.npz`). Changing it silently changes every baseline and invalidates checkpoints. | `solvers/LF_diffusion/diffusion_solver.py` ~line 32 |
| 2 | **No script fits the regression** — only the `.pkl`/`.json` artefacts are versioned. You must rewrite the fit for a new case. | `regression_model/*/polyreg_model/` |
| 3 | `update_geo()` is hard-coded to 3 regions / 6 parameters. | `models/PEDS_subdivision/context.py` |
| 4 | **Sweep bounds exist twice**; `SWEEP_ROD`/`ParamSpec` in `config_run.py` is dead code. The live bounds are `BOUNDS` in `TD_generation.py`. | `solvers/HF_openMC/` |
| 5 | A campaign **deletes its output CSVs on start**. Use a fresh `DATA_DIR`. | `TD_generation.generate_data()` |
| 6 | `GeneratorNN` docstrings describe an older architecture (input 7, trunk 128→128→64). | `models/PEDS.py` |
| 7 | `run_peds.sh` defaults `PEDS_SWEEP_TRAIN_SIZES=1500`, while `config_peds.py` defaults `TRAIN_SIZE=1000`. The shell export wins. | `config_and_run/` |
| 8 | Analysis scripts parse train size and seed from the folder name — the `train_<N>_seed_<S>` convention is mandatory. | `utils/analysis/*` |
| 9 | `peds_ensemble_screen.py` points at the removed `modules/RUNS/...` path. | `example_applications/forward_design_study/` |
| 10 | `README.md` references a `requirements.txt` that does not exist; reproduce environments from the `jax-env` / `mc-env` conda envs instead. | repo root |
| 11 | `.npz` normalisation bounds come from the *filtered dataset*, so a checkpoint is tied to its exact `.npz`. | `csv_cut_npz.py` |
| 12 | `data/highfidelity/split_cache/` is only used when `USE_SPLIT_CACHE=True` (off by default); a stale cache will silently fix your splits if you turn it on. | `config_peds.py` |

---

## Appendix: file index by question

| Question | File |
|---|---|
| What does this HF knob mean? | `solvers/HF_openMC/config_def.py` |
| Which HF case am I running? | `solvers/HF_openMC/config_run.py` |
| How is the OpenMC geometry/material/tally built? | `solvers/HF_openMC/MC_solver.py` |
| How is the LHS campaign defined? | `solvers/HF_openMC/TD_generation.py` |
| How do I run OpenMC on one design point? | `solvers/HF_openMC/design_api.py` |
| What does this LF knob mean? | `solvers/LF_diffusion/config_def.py` |
| Which LF geometry am I solving? | `solvers/LF_diffusion/config_run.py` |
| How are XS predicted and packed? | `solvers/LF_diffusion/diffusion_solver.py` |
| How is the diffusion eigenproblem assembled/solved? | `solvers/LF_diffusion/core/` |
| What are the training hyperparameters? | `config_and_run/config_peds.py` |
| How do I launch training? | `config_and_run/run_peds.sh` |
| What is the NN and the training loop? | `models/PEDS.py` |
| How are gradients computed through the solver? | `models/PEDS_subdivision/physics_solver.py`, `models/matrix_JAX_optimized.py` |
| How are splits and φ features made? | `models/PEDS_subdivision/data_loading.py` |
| How is the gradient validated? | `models/PEDS_core/diagnostics.py` |
| How do I evaluate / plot results? | `utils/run_testset.sh`, `utils/run_analysis.sh`, `utils/analysis/` |
| Where are the thesis numbers? | `example_applications/main_results/PEDS/analysis/` |
| How do I merge CSVs into datasets? | `data/files/build_merged_datasets.py` |
| How do I inspect dataset coverage? | `data/reports/` |
