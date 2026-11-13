# Study analysis organization

All analysis code lives under `utils/analysis/`.

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

Slurm launchers live one level up in `utils/`:

| File | What it runs |
|------|----------------|
| `../run_analysis.sh` | Full study analysis via `run_study_analysis.py` |
| `../run_testset.sh` | Held-out test metrics only (`evaluate_test_metrics.py`) |

## Slurm: submit and find logs

From `utils/`:

```bash
cd utils
sbatch run_analysis.sh    # full post-processing pipeline
sbatch run_testset.sh     # test-set evaluation only
```

**Logs:**

```
utils/slurm_logs/study_analysis_<JOBID>.out
utils/slurm_logs/study_analysis_<JOBID>.err
utils/slurm_logs/eval_metrics_<ARRAY_JOB>_<TASK>.out
```

Edit the `CONFIG` block inside `utils/run_analysis.sh` (or `LOGS_ROOTS` in `run_testset.sh`) before submitting.

## Configuration defaults

Edit `config.py`:

- `STUDY_PARENT_FOLDER = "RUNS"` or `"RESULTS"`
- `STUDY_FOLDER = "precise_param_strat"`
- `XS_SOURCE_RUN = "train_1000_seed_2"`

Paths resolve to `config_and_run/<parent>/<study>/...`.

## CLI arguments (orchestrator)

```bash
python utils/analysis/run_study_analysis.py \
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

- `utils/analysis/evaluate_test_metrics.py`
- `utils/analysis/param_error_analysis.py`
- `archive/standalone_tools/plot_functions_scripts/plot_test_pub_keff.py` (archived)
- `utils/analysis/xs_stats_report.py`
- `utils/analysis/study_analysis_config.py`

Implementations are in this `analysis/` package.
