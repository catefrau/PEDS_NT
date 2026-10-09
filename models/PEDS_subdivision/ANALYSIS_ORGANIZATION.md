# Study analysis organization

All analysis code lives under `models/PEDS_subdivision/analysis/`.

## Package modules

| File | Purpose |
|------|---------|
| `config.py` | Central path/config defaults (`STUDY_PARENT_FOLDER`, `STUDY_FOLDER`, …) |
| `param_error_analysis.py` | Parameter–error correlations, scatter plots, interactions |
| `pub_keff_plots.py` | Publication keff parity + reactivity histograms |
| `xs_stats_report.py` | XS correction + clipping statistics |
| `evaluate_test_metrics.py` | Held-out test evaluation → `analysis/testset_results/` |
| `run_evaluate_test_metrics.py` | Test evaluation only |

## Runners (invoke directly or via Slurm)

| File | What it runs |
|------|----------------|
| `run_study_analysis.py` | Full suite |
| `run_param_error_analysis.py` | Param error only |
| `run_pub_keff_plots.py` | Pub keff plots only |
| `run_xs_stats_report.py` | XS stats only |
| `run_training_diagnostics.py` | Training diagnostics only |
| `run_evaluate_test_metrics.py` | Test-set evaluation only |
| `run_analysis.sh` | **Slurm batch script** (conda + config block) |

## Slurm: submit and find logs

From the **repo root** (`PEDS_NT`):

```bash
sbatch models/PEDS_subdivision/analysis/run_analysis.sh
```

**Logs are written here** (not in repo-root `slurm_logs/`):

```
models/PEDS_subdivision/analysis/slurm_logs/study_analysis_<JOBID>.out
models/PEDS_subdivision/analysis/slurm_logs/study_analysis_<JOBID>.err
```

After submitting job `38358212`, check:

```bash
cat models/PEDS_subdivision/analysis/slurm_logs/study_analysis_38358212.err
cat models/PEDS_subdivision/analysis/slurm_logs/study_analysis_38358212.out
```

Edit the `CONFIG` block inside `run_analysis.sh` before submitting.

## Configuration defaults

Edit `config.py`:

- `STUDY_PARENT_FOLDER = "RUNS"` or `"RESULTS"`
- `STUDY_FOLDER = "precise_param_strat"`
- `XS_SOURCE_RUN = "train_1000_seed_2"`

Paths resolve to `config_and_run/<parent>/<study>/...`.

## CLI arguments (orchestrator)

```bash
python models/PEDS_subdivision/analysis/run_study_analysis.py \
  --study-parent-folder RESULTS \
  --study-folder complete_strat \
  --xs-source-run train_1000_seed_2
```

Skip flags: `--skip-test-metrics`, `--skip-param-error`, `--skip-pub-keff`, `--skip-xs-stats`, `--skip-training-diagnostics`

## Analysis outputs

Written under `config_and_run/<parent>/<study>/`:

- `analysis/testset_results/` — from test-set evaluation (metrics CSVs, keff comparisons, keff-dist plots)
- `analysis/testset_results_dk/` — same evaluation reported as Δk
- `analysis/valset_results/` — validation representative outputs (when enabled)
- `analysis/param_error/`
- `analysis/pub_keff/`
- `analysis/xs_stats/`
- `analysis/training_diagnostics/` (includes `cross_run_train_val_mean_std_summary.png`)

## Legacy imports (optional — delete when ready)

Thin re-exports kept at parent level for old import paths:

- `models/PEDS_subdivision/analysis/evaluate_test_metrics.py`
- `models/PEDS_subdivision/analysis/param_error_analysis.py`
- `archive/standalone_tools/plot_functions_scripts/plot_test_pub_keff.py` (archived)
- `models/PEDS_subdivision/analysis/xs_stats_report.py`
- `models/PEDS_subdivision/study_analysis_config.py`

Implementations are in this `analysis/` package.
