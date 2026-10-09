# Code snapshots for the agent studies

One file per study: the seed-0 `code_snapshot_*.py` that trained it. Studies that share a sha256 prefix are the same trainer; `PEDS_STUDY_NAME` selects the row in `STUDY_CONFIGS`.

`metrics_all_runs.csv` is every `study_summary.csv` row collected before the run folders were deleted.

Which file belongs to which code generation, the test numbers, and how to relaunch a study are in `../CODE_VERSIONS_AND_OUTCOMES.md`.

The three studies still stored in full (checkpoints and logs) are `r13_1k_elu6e4`, `r07_lr_6e4`, and `r11_lr6_warm14`.
