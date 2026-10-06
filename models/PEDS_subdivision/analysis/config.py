"""Central configuration for study-level analysis modules.

Paths are built as:
`config_and_run/<STUDY_PARENT_FOLDER>/<STUDY_FOLDER>/...`
"""

from __future__ import annotations

from pathlib import Path


MODELS_DIR = Path(__file__).resolve().parents[2]        # models/
PROJECT_ROOT = MODELS_DIR.parent
# Study outputs live next to the launchers, not next to the model code.
CONFIG_RUN_DIR = PROJECT_ROOT / "config_and_run"

# Supports both config_and_run/RUNS/<study>/... and config_and_run/RESULTS/<study>/...
STUDY_PARENT_FOLDER = "RUNS"
STUDY_FOLDER = "precise_param_strat"

# Representative run used by XS-statistics inputs.
XS_SOURCE_RUN = "train_1000_seed_2"


def study_parent_dir(parent_folder: str = STUDY_PARENT_FOLDER) -> Path:
    return CONFIG_RUN_DIR / parent_folder


def study_dir(
    study_folder: str = STUDY_FOLDER,
    parent_folder: str = STUDY_PARENT_FOLDER,
) -> Path:
    return study_parent_dir(parent_folder) / study_folder


def analysis_dir(
    study_folder: str = STUDY_FOLDER,
    parent_folder: str = STUDY_PARENT_FOLDER,
) -> Path:
    return study_dir(study_folder, parent_folder) / "analysis"


def param_error_outdir(
    study_folder: str = STUDY_FOLDER,
    parent_folder: str = STUDY_PARENT_FOLDER,
) -> Path:
    return analysis_dir(study_folder, parent_folder) / "param_error"


def pub_keff_outdir(
    study_folder: str = STUDY_FOLDER,
    parent_folder: str = STUDY_PARENT_FOLDER,
    use_delta_k: bool = False,
) -> Path:
    dirname = "pub_keff_dk" if use_delta_k else "pub_keff"
    return analysis_dir(study_folder, parent_folder) / dirname


def xs_stats_outdir(
    study_folder: str = STUDY_FOLDER,
    parent_folder: str = STUDY_PARENT_FOLDER,
) -> Path:
    return analysis_dir(study_folder, parent_folder) / "xs_stats"


def training_diagnostics_outdir(
    study_folder: str = STUDY_FOLDER,
    parent_folder: str = STUDY_PARENT_FOLDER,
) -> Path:
    return analysis_dir(study_folder, parent_folder) / "training_diagnostics"


def testset_results_dir(
    study_folder: str = STUDY_FOLDER,
    parent_folder: str = STUDY_PARENT_FOLDER,
    use_delta_k: bool = False,
) -> Path:
    dirname = "testset_results_dk" if use_delta_k else "testset_results"
    return study_dir(study_folder, parent_folder) / dirname


def testset_csv_glob(
    study_folder: str = STUDY_FOLDER,
    parent_folder: str = STUDY_PARENT_FOLDER,
    use_delta_k: bool = False,
) -> str:
    return str(testset_results_dir(study_folder, parent_folder, use_delta_k) / "run_train*_keff_comparison.csv")


def representative_test_csv(
    study_folder: str = STUDY_FOLDER,
    parent_folder: str = STUDY_PARENT_FOLDER,
    use_delta_k: bool = False,
) -> Path:
    return testset_results_dir(study_folder, parent_folder, use_delta_k) / "rep_train1000_seed2_keff_comparison.csv"


def train_run_glob(
    study_folder: str = STUDY_FOLDER,
    parent_folder: str = STUDY_PARENT_FOLDER,
) -> str:
    return str(study_dir(study_folder, parent_folder) / "train_1000_seed_*")


def all_train_run_glob(
    study_folder: str = STUDY_FOLDER,
    parent_folder: str = STUDY_PARENT_FOLDER,
) -> str:
    return str(study_dir(study_folder, parent_folder) / "train_*_seed_*")


def xs_source_run_dir(
    study_folder: str = STUDY_FOLDER,
    parent_folder: str = STUDY_PARENT_FOLDER,
    run_name: str = XS_SOURCE_RUN,
) -> Path:
    return study_dir(study_folder, parent_folder) / run_name / "XS"


def xs_final_csv(
    study_folder: str = STUDY_FOLDER,
    parent_folder: str = STUDY_PARENT_FOLDER,
    run_name: str = XS_SOURCE_RUN,
) -> Path:
    return xs_source_run_dir(study_folder, parent_folder, run_name) / "test_final_xs.csv"


def xs_logratio_csv(
    study_folder: str = STUDY_FOLDER,
    parent_folder: str = STUDY_PARENT_FOLDER,
    run_name: str = XS_SOURCE_RUN,
) -> Path:
    return xs_source_run_dir(study_folder, parent_folder, run_name) / "logratio_saturation.csv"

