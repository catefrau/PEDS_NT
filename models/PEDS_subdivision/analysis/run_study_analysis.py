#!/usr/bin/env python3
"""Single entry point for all study analyses.

Examples
--------
Run everything (default)::

    python models/PEDS_subdivision/analysis/run_study_analysis.py

Run one task::

    python models/PEDS_subdivision/analysis/run_study_analysis.py --tasks param-error

Run several tasks::

    python models/PEDS_subdivision/analysis/run_study_analysis.py \\
        --tasks test-metrics,pub-keff --use-delta-k

Skip tasks when running all::

    python models/PEDS_subdivision/analysis/run_study_analysis.py --skip-xs-stats
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[2]
if str(MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(MODELS_DIR))

from PEDS_subdivision.analysis.config import (
    STUDY_FOLDER,
    STUDY_PARENT_FOLDER,
    XS_SOURCE_RUN,
    representative_test_csv,
    study_dir,
    testset_csv_glob,
    pub_keff_outdir,
    training_diagnostics_outdir,
    xs_final_csv,
    xs_logratio_csv,
    xs_stats_outdir,
)
from PEDS_subdivision.analysis.evaluate_test_metrics import evaluate_study
from PEDS_subdivision.analysis.param_error_analysis import main as run_param_error
from PEDS_subdivision.analysis.pub_keff_plots import generate_all as run_pub_keff
from PEDS_subdivision.analysis.training_diagnostics import generate_training_diagnostics
from PEDS_subdivision.analysis.xs_stats_report import generate_xs_stats

ALL_TASKS = (
    "test-metrics",
    "param-error",
    "pub-keff",
    "xs-stats",
    "training-diagnostics",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--study-folder", default=STUDY_FOLDER)
    parser.add_argument("--study-parent-folder", default=STUDY_PARENT_FOLDER)
    parser.add_argument("--xs-source-run", default=XS_SOURCE_RUN)
    parser.add_argument(
        "--tasks",
        default="all",
        metavar="TASKS",
        help=(
            "Comma-separated task list or 'all'. "
            f"Choices: {', '.join(ALL_TASKS)}"
        ),
    )
    parser.add_argument("--skip-test-metrics", action="store_true")
    parser.add_argument("--skip-param-error", action="store_true")
    parser.add_argument("--skip-pub-keff", action="store_true")
    parser.add_argument("--skip-xs-stats", action="store_true")
    parser.add_argument("--skip-training-diagnostics", action="store_true")
    parser.add_argument(
        "--use-delta-k",
        action="store_true",
        help=(
            "Report |Δk| = |k_pred - k_ref| × 10^5 pcm instead of |Δρ|. "
            "Outputs go to separate *_dk folders."
        ),
    )
    parser.add_argument(
        "--no-val-study",
        action="store_true",
        help="Skip validation-set representative outputs (test-metrics only)",
    )
    parser.add_argument(
        "--run-glob",
        default=None,
        help="Optional run-dir glob for training-diagnostics only",
    )
    return parser.parse_args()


def resolve_tasks(args: argparse.Namespace) -> list[str]:
    if args.tasks == "all":
        tasks = list(ALL_TASKS)
    else:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
        unknown = sorted(set(tasks) - set(ALL_TASKS))
        if unknown:
            raise SystemExit(f"Unknown task(s): {unknown}. Choose from: {', '.join(ALL_TASKS)}")

    skip = {
        "test-metrics": args.skip_test_metrics,
        "param-error": args.skip_param_error,
        "pub-keff": args.skip_pub_keff,
        "xs-stats": args.skip_xs_stats,
        "training-diagnostics": args.skip_training_diagnostics,
    }
    return [t for t in tasks if not skip[t]]


def _step_test_metrics(args: argparse.Namespace, logs_root: Path) -> None:
    use_dk = args.use_delta_k
    suffix = "_dk" if use_dk else ""
    print(f"\n[test-metrics] Test-set evaluation (analysis/testset_results{suffix}/)")
    try:
        out_dir = evaluate_study(
            logs_root,
            with_val_study=not args.no_val_study,
            use_delta_k=use_dk,
        )
        print(f"  Wrote test metrics to {out_dir}")
    except Exception as exc:
        print(f"  WARNING: test-set evaluation failed: {exc}")


def _step_param_error(args: argparse.Namespace) -> None:
    use_dk = args.use_delta_k
    has_test_csvs = bool(
        glob.glob(testset_csv_glob(args.study_folder, args.study_parent_folder, use_dk))
    )
    print("\n[param-error] Parameter error analysis")
    if not has_test_csvs:
        print("  NOTE: no testset comparison CSVs found; running train/val only.")
    run_param_error(args.study_folder, args.study_parent_folder, use_delta_k=use_dk)


def _step_pub_keff(args: argparse.Namespace) -> None:
    use_dk = args.use_delta_k
    suffix = "_dk" if use_dk else ""
    rep_csv = representative_test_csv(args.study_folder, args.study_parent_folder, use_dk)
    print(f"\n[pub-keff] Publication keff plots (analysis/pub_keff{suffix}/)")
    if rep_csv.exists():
        run_pub_keff(
            csv_path=rep_csv,
            out_dir=pub_keff_outdir(args.study_folder, args.study_parent_folder, use_dk),
        )
    else:
        print(f"  SKIP: missing representative test CSV: {rep_csv}")


def _step_xs_stats(args: argparse.Namespace) -> None:
    xs_final = xs_final_csv(args.study_folder, args.study_parent_folder, args.xs_source_run)
    xs_logratio = xs_logratio_csv(args.study_folder, args.study_parent_folder, args.xs_source_run)
    print("\n[xs-stats] XS correction statistics")
    if xs_final.exists() and xs_logratio.exists():
        generate_xs_stats(
            final_xs_csv=xs_final,
            logratio_csv=xs_logratio,
            outdir=xs_stats_outdir(args.study_folder, args.study_parent_folder),
        )
    else:
        print("  SKIP: missing XS inputs:")
        print(f"    final xs: {xs_final} (exists={xs_final.exists()})")
        print(f"    logratio: {xs_logratio} (exists={xs_logratio.exists()})")


def _step_training_diagnostics(args: argparse.Namespace) -> None:
    print("\n[training-diagnostics] Training curves and cross-run summaries")
    generate_training_diagnostics(
        study_folder=args.study_folder,
        study_parent_folder=args.study_parent_folder,
        outdir=training_diagnostics_outdir(args.study_folder, args.study_parent_folder),
        run_glob=args.run_glob,
    )


TASK_STEPS = {
    "test-metrics": _step_test_metrics,
    "param-error": _step_param_error,
    "pub-keff": _step_pub_keff,
    "xs-stats": _step_xs_stats,
    "training-diagnostics": _step_training_diagnostics,
}


def main() -> None:
    args = parse_args()
    tasks = resolve_tasks(args)
    if not tasks:
        raise SystemExit("No tasks selected (check --tasks and --skip-* flags).")

    logs_root = study_dir(args.study_folder, args.study_parent_folder)
    metric_mode = "delta_k (|Δk| pcm)" if args.use_delta_k else "delta_rho (|Δρ| pcm)"
    print(
        f"Study: config_and_run/{args.study_parent_folder}/{args.study_folder}\n"
        f"  xs-source-run={args.xs_source_run}\n"
        f"  metric={metric_mode}\n"
        f"  tasks={', '.join(tasks)}"
    )

    n = len(tasks)
    for i, task in enumerate(tasks, start=1):
        print(f"\n--- [{i}/{n}] {task} ---")
        step = TASK_STEPS[task]
        if task == "test-metrics":
            step(args, logs_root)
        else:
            step(args)

    print("\nDone.")


if __name__ == "__main__":
    main()
