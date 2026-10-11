# forward_design_study (cleaned)

This folder is the consolidated, minimal workflow for your forward-design study.

## Main scripts

- `peds_ensemble_screen.py`
  - Generates LHS points, runs 5-seed PEDS ensemble, applies `TOL_PCM`, ranks candidates, writes `keff_scan_top_candidates.csv`.
  - Config block keeps the key knobs you highlighted (`SEEDS`, `RUN_DIR_TEMPLATE`, `TRAIN_SIZE`, `N_LHS`, `LHS_SEED`, `TARGET_KEFF`, `TOL_PCM`, `TOP_K`, `NN_BATCH`).
  - `RANK_MODE` (`keff_first` or `ppf_first`) is set directly in this file's config block.

- `openmc_verify_topk.py`
  - OpenMC verification of shortlisted candidates.
  - Reads `keff_scan_top_candidates.csv`, writes:
    - `keff_scan_hf_diagnostics.csv`
    - `keff_scan_final_summary.csv`
    - `keff_scan_hf_timing.txt`

- `diffusion_only_recheck.py`
  - Runs diffusion-only (no PEDS NN) on the shortlisted/feasible cases.
  - Writes `diffusion_only_check/` outputs.

- `build_report.py`
  - Writes `forward_design_report.md` and updates summary tables from verified outputs.

- `publication_plots/plot_peds_hf_parity.py`
  - Recreates `publication_plots/peds_hf_parity_ens5k_kefffirst.png`.

- `run_forward_design.sh`
  - Single entrypoint to run phases separately or all at once:
    - `screen`, `verify`, `diffusion`, `report`, `plot`, `all`

## Stored outputs

- `results/ens5k_kefffirst_08092342/`
  - Working copy used by the cleaned pipeline.
  - Contains `RESULTS_README.md` with core-vs-optional file guidance.

## OpenMC baseline (separate)

- `openmc_baseline/build_openmc_baseline_ranked.py`
- `openmc_baseline/lhs_feasible_ranked.csv`

## Unfinished BO project retained

- `bayes_openmc_wip/keff_scan_design_openmc.py`

## Suggested Slurm usage

Do not run heavy phases on login node. Submit with `sbatch --wrap` or put this launcher inside your preferred Slurm script.
