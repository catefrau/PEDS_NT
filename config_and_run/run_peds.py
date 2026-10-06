#!/usr/bin/env python3
"""Launch a PEDS training run described by ``config_peds.py``.

This is the entry point ``run_jobby.sh`` calls. It only wires up import paths,
echoes the resolved configuration, and hands over to the model implementation in
``models/PEDS.py`` -- all experiment knobs live in ``config_peds.py``.

Never invoke this directly on a login node; submit it through ``run_jobby.sh``.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

CONFIG_RUN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CONFIG_RUN_DIR.parent

# config_and_run/ -> config_peds, NTcode_config_data, plot_functions
# PROJECT_ROOT    -> solvers, models, data
# models/         -> PEDS_core, PEDS_subdivision, matrix_JAX_optimized
for _path in (str(CONFIG_RUN_DIR), str(PROJECT_ROOT), str(PROJECT_ROOT / "models")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import config_peds as CFG  # noqa: E402


def main() -> None:
    print(CFG.summary(), flush=True)
    if not CFG.DATA_FILEPATH.exists():
        raise FileNotFoundError(f"Dataset not found: {CFG.DATA_FILEPATH}")
    if not CFG.MODEL_SCRIPT.exists():
        raise FileNotFoundError(f"Model script not found: {CFG.MODEL_SCRIPT}")
    runpy.run_path(str(CFG.MODEL_SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()
