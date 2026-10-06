#!/usr/bin/env python3
"""Train/evaluate a vanilla NN baseline against PEDS complete_strat runs.

Baseline intent (Boltzmann-app analogue)
----------------------------------------
- Inputs: only the 6 geometry features from dataset ``params``.
- Target: direct keff prediction (solver removed).
- Architecture: swap PEDS generator's final hidden->XS layer for hidden->1,
  i.e. ``6 -> 128 -> 256 -> 128 -> 1`` (same hidden stack as PEDS trunk width).
- Ensemble: one model per seed, reusing each PEDS split_log.csv exactly.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from flax import nnx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "config_and_run" / "RESULTS" / "complete_strat"
DEFAULT_OUTPUT_DIR = RESULTS_ROOT / "mlp_baseline" / "mlp_geom_only"
DATA_PATH = PROJECT_ROOT / "data" / "highfidelity" / "17jul_0.8_1.2.npz"
PEDS_ROOT = RESULTS_ROOT / "PEDS"
PEDS_METRICS_DK = PEDS_ROOT / "testset_results_dk" / "test_metrics_all_runs.csv"
PEDS_METRICS_RHO = PEDS_ROOT / "testset_results" / "test_metrics_all_runs.csv"
SPLIT_TEMPLATE = PEDS_ROOT / "train_1000_seed_{seed}" / "split_log.csv"

# Same hidden stack as PEDS generator trunk; final layer maps hidden -> 1
# (Boltzmann-app analogue: hidden -> resolution^2 becomes hidden -> 1).
GEOM_DIM = 6
HIDDEN_SIZES = [128, 256, 128]
SEEDS = [0, 1, 2, 3, 4]

BATCH_SIZE = 32
EPOCHS = 70
LR_MAX = 2e-4
LR_MIN = 5e-6
MIN_SAVE_EPOCH = 10


@dataclass
class SplitData:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


class VanillaMLP(nnx.Module):
    """Direct MLP baseline: geom trunk with hidden->1 output layer."""

    def __init__(self, rngs: nnx.Rngs):
        super().__init__()
        dims = [GEOM_DIM] + HIDDEN_SIZES + [1]
        self.layers = nnx.List(
            [
                nnx.Linear(
                    in_features=dims[i],
                    out_features=dims[i + 1],
                    kernel_init=nnx.initializers.kaiming_normal(),
                    bias_init=nnx.initializers.constant(0.0),
                    rngs=rngs,
                )
                for i in range(len(dims) - 1)
            ]
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = x
        for layer in self.layers[:-1]:
            h = nnx.relu(layer(h))
        out = self.layers[-1](h)
        return jnp.squeeze(out, axis=-1)


def linear_param_count(in_dim: int, out_dim: int) -> int:
    return in_dim * out_dim + out_dim


def peds_generator_param_count(
    geom_dim: int = 6,
    phi_dim: int = 6,
    hidden_sizes: list[int] | tuple[int, ...] = (128, 256, 128),
    n_regions: int = 3,
    xs_per_region: int = 12,
) -> int:
    trunk_dims = [geom_dim + phi_dim] + list(hidden_sizes)
    total = 0
    for i in range(len(trunk_dims) - 1):
        total += linear_param_count(trunk_dims[i], trunk_dims[i + 1])
    total_xs = n_regions * xs_per_region
    head_in = hidden_sizes[-1] + total_xs
    total += linear_param_count(head_in, total_xs)
    return total


def nnx_param_count(model: nnx.Module) -> int:
    state = nnx.state(model, nnx.Param)
    leaves = jax.tree_util.tree_leaves(state)
    return int(sum(int(np.prod(np.asarray(v).shape)) for v in leaves))


def load_dataset() -> tuple[np.ndarray, np.ndarray]:
    data = np.load(DATA_PATH, allow_pickle=True)
    geoms = np.asarray(data["params"], dtype=np.float32)
    keffs = np.asarray(data["keffs"], dtype=np.float32)
    if geoms.shape[1] != GEOM_DIM:
        raise ValueError(f"Expected {GEOM_DIM} geometry features, got {geoms.shape[1]}")
    return geoms, keffs


def read_split(seed: int) -> SplitData:
    split_path = Path(str(SPLIT_TEMPLATE).format(seed=seed))
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")
    train_idx, val_idx, test_idx = [], [], []
    with split_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = int(row["sample_idx"])
            split = row["split"].strip().lower()
            if split == "train":
                train_idx.append(idx)
            elif split == "val":
                val_idx.append(idx)
            elif split == "test":
                test_idx.append(idx)
    return SplitData(
        train_idx=np.asarray(train_idx, dtype=np.int64),
        val_idx=np.asarray(val_idx, dtype=np.int64),
        test_idx=np.asarray(test_idx, dtype=np.int64),
    )


def to_metrics(pred: np.ndarray, ref: np.ndarray) -> dict:
    abs_diff = np.abs(pred - ref)
    abs_frac = abs_diff / np.clip(np.abs(ref), 1e-12, None)
    delta_k_pcm = abs_diff * 1e5
    delta_rho_pcm = abs_diff / np.clip(pred * ref, 1e-12, None) * 1e5
    return {
        "mse_k": float(np.mean((pred - ref) ** 2)),
        "mae_k": float(np.mean(abs_diff)),
        "mean_frac_error": float(np.mean(abs_frac)),
        "median_frac_error": float(np.median(abs_frac)),
        "mean_delta_k_pcm": float(np.mean(delta_k_pcm)),
        "median_delta_k_pcm": float(np.median(delta_k_pcm)),
        "p95_delta_k_pcm": float(np.percentile(delta_k_pcm, 95)),
        "std_delta_k_pcm": float(np.std(delta_k_pcm)),
        "mean_delta_rho_pcm": float(np.mean(delta_rho_pcm)),
        "median_delta_rho_pcm": float(np.median(delta_rho_pcm)),
        "p95_delta_rho_pcm": float(np.percentile(delta_rho_pcm, 95)),
        "std_delta_rho_pcm": float(np.std(delta_rho_pcm)),
        "frac_below_650_delta_k": float(np.mean(delta_k_pcm < 650.0)),
        "frac_below_650_delta_rho": float(np.mean(delta_rho_pcm < 650.0)),
    }


def data_loader(x: np.ndarray, y: np.ndarray, batch_size: int) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    n = len(x)
    for i in range(0, n, batch_size):
        yield x[i : i + batch_size], y[i : i + batch_size]


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def train_one_seed(
    seed: int,
    geoms: np.ndarray,
    keffs: np.ndarray,
    out_dir: Path,
) -> dict:
    split = read_split(seed)
    run_dir = out_dir / f"train_1000_seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    x_train = geoms[split.train_idx]
    y_train = keffs[split.train_idx]
    x_val = geoms[split.val_idx]
    y_val = keffs[split.val_idx]
    x_test = geoms[split.test_idx]
    y_test = keffs[split.test_idx]

    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0) + 1e-8
    x_train_n = ((x_train - x_mean) / x_std).astype(np.float32)
    x_val_n = ((x_val - x_mean) / x_std).astype(np.float32)
    x_test_n = ((x_test - x_mean) / x_std).astype(np.float32)

    model = VanillaMLP(rngs=nnx.Rngs(seed))
    steps_per_epoch = max(int(np.ceil(len(x_train_n) / BATCH_SIZE)), 1)
    decay_steps = EPOCHS * steps_per_epoch
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=LR_MIN,
        peak_value=LR_MAX,
        warmup_steps=max(1, int(0.1 * decay_steps)),
        decay_steps=decay_steps,
        end_value=LR_MIN,
    )
    optimizer = nnx.Optimizer(
        model,
        optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(lr_schedule),
        ),
        wrt=nnx.Param,
    )

    def loss_fn(mod: VanillaMLP, xb: jnp.ndarray, yb: jnp.ndarray):
        pred = mod(xb)
        return jnp.mean((pred - yb) ** 2), pred

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)

    rng = np.random.default_rng(seed)
    history_rows: list[dict] = []
    best_val = float("inf")
    best_epoch = 1
    best_state = jax.tree_util.tree_map(np.asarray, nnx.state(model))

    for epoch in range(1, EPOCHS + 1):
        perm = rng.permutation(len(x_train_n))
        x_ep = x_train_n[perm]
        y_ep = y_train[perm]

        train_pred_all, train_ref_all = [], []
        epoch_loss = 0.0
        n_batches = 0

        for xb, yb in data_loader(x_ep, y_ep, BATCH_SIZE):
            xb_j = jnp.asarray(xb)
            yb_j = jnp.asarray(yb)
            (loss, pred), grads = grad_fn(model, xb_j, yb_j)
            optimizer.update(model, grads)

            epoch_loss += float(loss)
            n_batches += 1
            train_pred_all.append(np.asarray(pred))
            train_ref_all.append(np.asarray(yb))

        train_pred = np.concatenate(train_pred_all)
        train_ref = np.concatenate(train_ref_all)
        train_m = to_metrics(train_pred, train_ref)

        val_pred = np.asarray(model(jnp.asarray(x_val_n)))
        val_m = to_metrics(val_pred, y_val)
        val_obj = val_m["mean_delta_k_pcm"]

        if epoch >= MIN_SAVE_EPOCH and val_obj < best_val:
            best_val = val_obj
            best_epoch = epoch
            best_state = jax.tree_util.tree_map(np.asarray, nnx.state(model))

        history_rows.append(
            {
                "epoch": epoch,
                "lr": float(lr_schedule(optimizer.step.value)),
                "train_mse_k": train_m["mse_k"],
                "val_mse_k": val_m["mse_k"],
                "train_mean_delta_k_pcm": train_m["mean_delta_k_pcm"],
                "val_mean_delta_k_pcm": val_m["mean_delta_k_pcm"],
                "train_mean_delta_rho_pcm": train_m["mean_delta_rho_pcm"],
                "val_mean_delta_rho_pcm": val_m["mean_delta_rho_pcm"],
                "train_mean_frac_error": train_m["mean_frac_error"],
                "val_mean_frac_error": val_m["mean_frac_error"],
                "epoch_loss_mean": epoch_loss / max(n_batches, 1),
            }
        )

    nnx.update(model, jax.tree_util.tree_map(jnp.asarray, best_state))
    test_pred = np.asarray(model(jnp.asarray(x_test_n)))
    test_m = to_metrics(test_pred, y_test)

    pd.DataFrame(history_rows).to_csv(run_dir / "epoch_metrics.csv", index=False)
    pd.DataFrame(
        {
            "sample_idx": split.test_idx,
            "split": "test",
            "keff_openmc": y_test,
            "keff_mlp": test_pred,
            "frac_error": np.abs(test_pred - y_test) / np.clip(np.abs(y_test), 1e-12, None),
            "delta_k_pcm": np.abs(test_pred - y_test) * 1e5,
            "delta_rho_pcm": np.abs(test_pred - y_test) / np.clip(test_pred * y_test, 1e-12, None) * 1e5,
        }
    ).to_csv(run_dir / "test_predictions.csv", index=False)

    save_json(
        run_dir / "run_metadata.json",
        {
            "seed": seed,
            "train_size": int(len(split.train_idx)),
            "val_size": int(len(split.val_idx)),
            "test_size": int(len(split.test_idx)),
            "best_epoch": int(best_epoch),
            "best_val_mean_delta_k_pcm": float(best_val),
            "peds_generator_param_count_estimate": peds_generator_param_count(),
            "baseline_param_count": nnx_param_count(model),
            "architecture": [GEOM_DIM] + HIDDEN_SIZES + [1],
        },
    )

    return {
        "run_id": f"train_1000_seed_{seed}",
        "train_size": int(len(split.train_idx)),
        "seed": seed,
        "best_epoch": best_epoch,
        "test_mse_k": test_m["mse_k"],
        "test_mae_k": test_m["mae_k"],
        "test_mean_frac_error": test_m["mean_frac_error"],
        "test_median_frac_error": test_m["median_frac_error"],
        "test_mean_delta_k_pcm": test_m["mean_delta_k_pcm"],
        "test_median_delta_k_pcm": test_m["median_delta_k_pcm"],
        "test_p95_delta_k_pcm": test_m["p95_delta_k_pcm"],
        "test_std_delta_k_pcm": test_m["std_delta_k_pcm"],
        "test_mean_delta_rho_pcm": test_m["mean_delta_rho_pcm"],
        "test_median_delta_rho_pcm": test_m["median_delta_rho_pcm"],
        "test_p95_delta_rho_pcm": test_m["p95_delta_rho_pcm"],
        "test_std_delta_rho_pcm": test_m["std_delta_rho_pcm"],
        "test_frac_below_650_delta_k": test_m["frac_below_650_delta_k"],
        "test_frac_below_650_delta_rho": test_m["frac_below_650_delta_rho"],
    }


def summarize_baseline(metrics_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    mean_row = metrics_df.mean(numeric_only=True)
    std_row = metrics_df.std(numeric_only=True)
    summary = pd.DataFrame(
        {
            "metric": mean_row.index,
            "mean_across_seeds": mean_row.values,
            "std_across_seeds": std_row.reindex(mean_row.index).values,
        }
    )
    summary.to_csv(out_dir / "testset_results" / "baseline_summary_by_metric.csv", index=False)
    return summary


def build_comparison_markdown(baseline_df: pd.DataFrame, out_dir: Path) -> None:
    peds_dk_df = pd.read_csv(PEDS_METRICS_DK)
    peds_rho_df = pd.read_csv(PEDS_METRICS_RHO)

    b_mean_dk = float(baseline_df["test_mean_delta_k_pcm"].mean())
    b_mean_rho = float(baseline_df["test_mean_delta_rho_pcm"].mean())
    b_mean_frac = float(baseline_df["test_mean_frac_error"].mean())
    b_frac650_dk = float(baseline_df["test_frac_below_650_delta_k"].mean())
    b_frac650_rho = float(baseline_df["test_frac_below_650_delta_rho"].mean())

    p_mean_dk = float(peds_dk_df["test_mean_pcm"].mean())
    p_mean_rho = float(peds_rho_df["test_mean_pcm"].mean())
    p_frac650_dk = float(peds_dk_df["test_frac_below_650"].mean())
    p_frac650_rho = float(peds_rho_df["test_frac_below_650"].mean())

    rel_gap_dk = (b_mean_dk - p_mean_dk) / p_mean_dk if p_mean_dk != 0.0 else np.nan
    rel_gap_rho = (b_mean_rho - p_mean_rho) / p_mean_rho if p_mean_rho != 0.0 else np.nan

    peds_params = peds_generator_param_count()
    baseline_model = VanillaMLP(rngs=nnx.Rngs(0))
    baseline_params = nnx_param_count(baseline_model)

    lines = [
        "# MLP baseline vs PEDS (complete_strat)",
        "",
        "## What was done",
        "- Trained a vanilla NN-only baseline ensemble with 5 runs (seeds 0..4).",
        "- For each seed, used the exact split from `train_1000_seed_<seed>/split_log.csv`.",
        "- Inputs are only the 6 geometry features (`params`); target is direct `keff`.",
        "- Baseline architecture: `6 -> 128 -> 256 -> 128 -> 1` "
        "(Boltzmann-app analogue: final hidden->XS layer swapped for hidden->1).",
        "- Capacity check: PEDS generator estimate "
        f"`{peds_params}` params vs baseline `{baseline_params}` params.",
        "",
        "## Test comparison (mean over seeds)",
        f"- Mean absolute fractional error (baseline): `{b_mean_frac:.5f}`.",
        f"- Mean |Δk| (pcm): baseline `{b_mean_dk:.1f}` vs PEDS `{p_mean_dk:.1f}` "
        f"({rel_gap_dk * 100.0:+.1f}%).",
        f"- Mean |Δρ| (pcm): baseline `{b_mean_rho:.1f}` vs PEDS `{p_mean_rho:.1f}` "
        f"({rel_gap_rho * 100.0:+.1f}%).",
        f"- Fraction below 650 pcm in |Δk|: baseline `{b_frac650_dk:.3f}` vs PEDS `{p_frac650_dk:.3f}`.",
        f"- Fraction below 650 pcm in |Δρ|: baseline `{b_frac650_rho:.3f}` vs PEDS `{p_frac650_rho:.3f}`.",
        "",
        "## Interpretation on data sufficiency",
        "- If baseline and PEDS metrics are close, 1000 training points are likely enough "
        "for a pure NN surrogate at this geometry-only setting.",
        "- If baseline remains noticeably worse, the PEDS solver-informed structure is "
        "extracting useful inductive bias not recovered by data alone.",
        "- Use `<output_dir>/testset_results/test_metrics_all_runs.csv` for per-seed inspection.",
    ]
    (out_dir / "COMPARISON_WITH_PEDS.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train geometry-only MLP baseline.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for run outputs (default: mlp_baseline/mlp_geom_only).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "testset_results").mkdir(parents=True, exist_ok=True)

    geoms, keffs = load_dataset()
    run_rows = []
    for seed in SEEDS:
        print(f"[run] seed={seed}")
        row = train_one_seed(seed=seed, geoms=geoms, keffs=keffs, out_dir=output_dir)
        run_rows.append(row)
        print(
            f"  test mean |Δk|={row['test_mean_delta_k_pcm']:.1f} pcm | "
            f"mean frac err={row['test_mean_frac_error']:.5f}"
        )

    metrics_df = pd.DataFrame(run_rows).sort_values(["train_size", "seed"]).reset_index(drop=True)
    out_csv = output_dir / "testset_results" / "test_metrics_all_runs.csv"
    metrics_df.to_csv(out_csv, index=False)
    summarize_baseline(metrics_df, output_dir)
    build_comparison_markdown(metrics_df, output_dir)
    print(f"\nSaved baseline run metrics: {out_csv}")
    print(f"Saved comparison note: {output_dir / 'COMPARISON_WITH_PEDS.md'}")


if __name__ == "__main__":
    main()
