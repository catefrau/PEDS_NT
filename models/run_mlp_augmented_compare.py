#!/usr/bin/env python3
"""Augmented MLP baseline with geom + phi + XS inputs.

Boltzmann-app analogue
----------------------
PEDS generator: trunk(geom+phi) -> hidden, head (hidden+XS) -> 36, solver -> keff.
This baseline:  trunk(geom+phi) -> hidden, single layer (hidden+XS) -> 1.

Same extra inputs as PEDS (6 geom, 6 phi flux features, 36 log baseline XS).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

MODELS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODELS_DIR.parent
# NTcode_config_data/plot_functions live under config_and_run/, while this script and
# PEDS_subdivision now live under models/.
CONFIG_RUN_DIR = PROJECT_ROOT / "config_and_run"
for _p in (str(PROJECT_ROOT), str(MODELS_DIR), str(CONFIG_RUN_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from flax import nnx

from NTcode_config_data.config_run import GEO_CYL as GEO
from PEDS_subdivision import context as peds_context
from PEDS_subdivision.data_loading import compute_batch_baselines, compute_phi_features

RESULTS_ROOT = PROJECT_ROOT / "config_and_run" / "RESULTS" / "complete_strat"
BASELINE_PARENT = RESULTS_ROOT / "mlp_baseline"
DEFAULT_OUTPUT_DIR = BASELINE_PARENT / "mlp_with_phi_xs"
DEFAULT_GEOM_ONLY_METRICS = BASELINE_PARENT / "mlp_geom_only" / "testset_results" / "test_metrics_all_runs.csv"

DATA_PATH = PROJECT_ROOT / "data" / "highfidelity" / "17jul_0.8_1.2.npz"
PEDS_ROOT = RESULTS_ROOT / "PEDS"
SPLIT_TEMPLATE = PEDS_ROOT / "train_1000_seed_{seed}" / "split_log.csv"
PEDS_METRICS_DK = PEDS_ROOT / "testset_results_dk" / "test_metrics_all_runs.csv"

SEEDS = [0, 1, 2, 3, 4]
GEOM_DIM = 6
PHI_DIM = GEO.G * 3  # 2 groups x 3 regions
XS_DIM = 3 * 12      # 36 XS slots
TRUNK_INPUT_DIM = GEOM_DIM + PHI_DIM  # 12, same as PEDS generator trunk input
TRUNK_HIDDEN = [128, 256, 128]        # same as PEDS generator trunk

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


@dataclass
class FeatureBundle:
    geoms: np.ndarray
    phi: np.ndarray
    xs_log: np.ndarray
    keffs: np.ndarray


def linear_param_count(in_dim: int, out_dim: int) -> int:
    return in_dim * out_dim + out_dim


def count_mlp_stack(dims: list[int]) -> int:
    return sum(linear_param_count(dims[i], dims[i + 1]) for i in range(len(dims) - 1))


def peds_generator_param_count(
    trunk_hidden: list[int] = TRUNK_HIDDEN,
    trunk_input: int = TRUNK_INPUT_DIM,
    xs_dim: int = XS_DIM,
) -> int:
    dims = [trunk_input] + trunk_hidden
    total = count_mlp_stack(dims)
    total += linear_param_count(trunk_hidden[-1] + xs_dim, xs_dim)
    return total


def baseline_param_count(trunk_hidden: list[int] = TRUNK_HIDDEN) -> int:
    dims = [TRUNK_INPUT_DIM] + trunk_hidden
    total = count_mlp_stack(dims)
    total += linear_param_count(trunk_hidden[-1] + XS_DIM, 1)
    return total


class AugmentedMLPBaseline(nnx.Module):
    """Trunk on geom+phi; single (hidden+XS) -> 1 layer parallel to PEDS XS head."""

    def __init__(self, rngs: nnx.Rngs):
        super().__init__()
        he_init = nnx.initializers.kaiming_normal()

        trunk_dims = [TRUNK_INPUT_DIM] + TRUNK_HIDDEN
        self.trunk_layers = nnx.List(
            [
                nnx.Linear(
                    in_features=trunk_dims[i],
                    out_features=trunk_dims[i + 1],
                    kernel_init=he_init,
                    bias_init=nnx.initializers.constant(0.0),
                    rngs=rngs,
                )
                for i in range(len(trunk_dims) - 1)
            ]
        )

        trunk_out = TRUNK_HIDDEN[-1]
        self.output = nnx.Linear(
            in_features=trunk_out + XS_DIM,
            out_features=1,
            kernel_init=he_init,
            bias_init=nnx.initializers.constant(0.0),
            rngs=rngs,
        )

    def __call__(self, geom: jnp.ndarray, phi: jnp.ndarray, xs_log: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([geom, phi], axis=-1)
        for layer in self.trunk_layers:
            x = nnx.relu(layer(x))

        xs_flat = jnp.reshape(xs_log, (geom.shape[0], XS_DIM))
        feat = jnp.concatenate([x, xs_flat], axis=-1)
        keff = self.output(feat)
        return jnp.squeeze(keff, axis=-1)


def nnx_param_count(model: nnx.Module) -> int:
    state = nnx.state(model, nnx.Param)
    leaves = jax.tree_util.tree_leaves(state)
    return int(sum(int(np.prod(np.asarray(v).shape)) for v in leaves))


def read_split(seed: int) -> SplitData:
    split_path = Path(str(SPLIT_TEMPLATE).format(seed=seed))
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")
    train_idx, val_idx, test_idx = [], [], []
    with split_path.open(newline="") as f:
        for row in csv.DictReader(f):
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


def ensure_peds_context(data_filepath: Path | str = DATA_PATH) -> None:
    """Initialize PEDS runtime context required by XS/phi feature builders.

    ``compute_batch_baselines`` needs ``context.XS_MASK`` and
    ``compute_phi_features`` needs ``context.N_FLAT_MAX`` / caches from
    ``PEDS_subdivision.context.init`` (same setup PEDS.py does at import).
    """
    peds_context.init(
        data_filepath=str(data_filepath),
        log_dir=str(DEFAULT_OUTPUT_DIR / "_context_tmp"),
        xs_dir=str(DEFAULT_OUTPUT_DIR / "_context_tmp" / "XS"),
        param_names=["b4c_r", "cr_frac", "fuel_r", "enrichment", "f_mod", "water_r"],
        param_strat_bins=5,
        log_ratio_clip_lo=-1.8,
        log_ratio_clip_hi=0.5,
        batch_size=BATCH_SIZE,
    )


def build_log_xs_features(rawparams: np.ndarray) -> np.ndarray:
    xs_baselines = np.asarray(compute_batch_baselines(rawparams, GEO))
    safe = np.where(xs_baselines > 1e-10, xs_baselines, np.ones_like(xs_baselines))
    return np.log(safe).astype(np.float32)


def build_feature_bundle(
    geoms: np.ndarray,
    rawparams: np.ndarray,
    keffs: np.ndarray,
    cache_dir: Path | None = None,
) -> FeatureBundle:
    cache_path = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / "feature_cache_geom_phi_xslog.npz"
        if cache_path.exists():
            print(f"Loading cached features from {cache_path}")
            cached = np.load(cache_path)
            return FeatureBundle(
                geoms=np.asarray(cached["geoms"], dtype=np.float32),
                phi=np.asarray(cached["phi"], dtype=np.float32),
                xs_log=np.asarray(cached["xs_log"], dtype=np.float32),
                keffs=np.asarray(cached["keffs"], dtype=np.float32),
            )

    ensure_peds_context(DATA_PATH)
    print("Precomputing phi features for all samples...")
    phi = compute_phi_features(rawparams).astype(np.float32)
    print("Precomputing log baseline XS features for all samples...")
    xs_log = build_log_xs_features(rawparams)
    bundle = FeatureBundle(
        geoms=geoms.astype(np.float32),
        phi=phi,
        xs_log=xs_log,
        keffs=keffs.astype(np.float32),
    )
    if cache_path is not None:
        np.savez_compressed(
            cache_path,
            geoms=bundle.geoms,
            phi=bundle.phi,
            xs_log=bundle.xs_log,
            keffs=bundle.keffs,
        )
        print(f"Saved feature cache to {cache_path}")
    return bundle


def normalize_split_features(
    bundle: FeatureBundle,
    train_idx: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    geom_mean = bundle.geoms[train_idx].mean(axis=0)
    geom_std = bundle.geoms[train_idx].std(axis=0) + 1e-8
    phi_mean = bundle.phi[train_idx].mean(axis=0)
    phi_std = bundle.phi[train_idx].std(axis=0) + 1e-8
    xs_mean = bundle.xs_log[train_idx].mean(axis=0)
    xs_std = bundle.xs_log[train_idx].std(axis=0) + 1e-8

    def norm(arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        return ((arr - mean) / std).astype(np.float32)

    arrays = {
        "geom": norm(bundle.geoms, geom_mean, geom_std),
        "phi": norm(bundle.phi, phi_mean, phi_std),
        "xs_log": norm(bundle.xs_log, xs_mean, xs_std),
        "keff": bundle.keffs,
    }
    stats = {
        "geom_mean": geom_mean.astype(np.float32),
        "geom_std": geom_std.astype(np.float32),
        "phi_mean": phi_mean.astype(np.float32),
        "phi_std": phi_std.astype(np.float32),
        "xs_log_mean": xs_mean.astype(np.float32),
        "xs_log_std": xs_std.astype(np.float32),
    }
    return arrays, stats


def data_loader(
    geom: np.ndarray,
    phi: np.ndarray,
    xs_log: np.ndarray,
    y: np.ndarray,
    batch_size: int,
) -> Iterable[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    n = len(geom)
    for start in range(0, n, batch_size):
        sl = slice(start, start + batch_size)
        yield geom[sl], phi[sl], xs_log[sl], y[sl]


def to_metrics(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
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


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def train_one_seed(seed: int, bundle: FeatureBundle, out_dir: Path) -> dict:
    split = read_split(seed)
    run_dir = out_dir / f"train_1000_seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    arrays, norm_stats = normalize_split_features(bundle, split.train_idx)

    g_train = arrays["geom"][split.train_idx]
    p_train = arrays["phi"][split.train_idx]
    x_train = arrays["xs_log"][split.train_idx]
    y_train = arrays["keff"][split.train_idx]

    g_val = arrays["geom"][split.val_idx]
    p_val = arrays["phi"][split.val_idx]
    x_val = arrays["xs_log"][split.val_idx]
    y_val = arrays["keff"][split.val_idx]

    g_test = arrays["geom"][split.test_idx]
    p_test = arrays["phi"][split.test_idx]
    x_test = arrays["xs_log"][split.test_idx]
    y_test = arrays["keff"][split.test_idx]

    model = AugmentedMLPBaseline(rngs=nnx.Rngs(seed))
    steps_per_epoch = max(int(np.ceil(len(g_train) / BATCH_SIZE)), 1)
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
        optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr_schedule)),
        wrt=nnx.Param,
    )

    def loss_fn(
        mod: AugmentedMLPBaseline,
        geom_b: jnp.ndarray,
        phi_b: jnp.ndarray,
        xs_b: jnp.ndarray,
        yb: jnp.ndarray,
    ):
        pred = mod(geom_b, phi_b, xs_b)
        return jnp.mean((pred - yb) ** 2), pred

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    rng = np.random.default_rng(seed)

    history_rows: list[dict] = []
    best_val = float("inf")
    best_epoch = 1
    best_state = jax.tree_util.tree_map(np.asarray, nnx.state(model))

    for epoch in range(1, EPOCHS + 1):
        perm = rng.permutation(len(g_train))
        epoch_loss = 0.0
        n_batches = 0
        train_preds, train_refs = [], []

        for bg, bp, bx, by in data_loader(
            g_train[perm], p_train[perm], x_train[perm], y_train[perm], BATCH_SIZE
        ):
            (loss, pred), grads = grad_fn(
                model, jnp.asarray(bg), jnp.asarray(bp), jnp.asarray(bx), jnp.asarray(by)
            )
            optimizer.update(model, grads)
            epoch_loss += float(loss)
            n_batches += 1
            train_preds.append(np.asarray(pred))
            train_refs.append(np.asarray(by))

        train_m = to_metrics(np.concatenate(train_preds), np.concatenate(train_refs))
        val_pred = np.asarray(model(jnp.asarray(g_val), jnp.asarray(p_val), jnp.asarray(x_val)))
        val_m = to_metrics(val_pred, y_val)

        if epoch >= MIN_SAVE_EPOCH and val_m["mean_delta_k_pcm"] < best_val:
            best_val = val_m["mean_delta_k_pcm"]
            best_epoch = epoch
            best_state = jax.tree_util.tree_map(np.asarray, nnx.state(model))

        history_rows.append(
            {
                "epoch": epoch,
                "lr": float(lr_schedule(optimizer.step.value)),
                "train_mse_k": train_m["mse_k"],
                "val_mse_k": val_m["mse_k"],
                "train_mae_k": train_m["mae_k"],
                "val_mae_k": val_m["mae_k"],
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
    test_pred = np.asarray(model(jnp.asarray(g_test), jnp.asarray(p_test), jnp.asarray(x_test)))
    test_m = to_metrics(test_pred, y_test)

    pd.DataFrame(history_rows).to_csv(run_dir / "epoch_metrics.csv", index=False)
    pd.DataFrame(
        {
            "sample_idx": split.test_idx,
            "split": "test",
            "keff_openmc": y_test,
            "keff_mlp_aug": test_pred,
            "frac_error": np.abs(test_pred - y_test) / np.clip(np.abs(y_test), 1e-12, None),
            "delta_k_pcm": np.abs(test_pred - y_test) * 1e5,
            "delta_rho_pcm": np.abs(test_pred - y_test) / np.clip(test_pred * y_test, 1e-12, None) * 1e5,
        }
    ).to_csv(run_dir / "test_predictions.csv", index=False)

    peds_target = peds_generator_param_count()
    baseline_target = baseline_param_count()
    actual_params = nnx_param_count(model)

    save_json(
        run_dir / "run_metadata.json",
        {
            "seed": seed,
            "train_size": int(len(split.train_idx)),
            "val_size": int(len(split.val_idx)),
            "test_size": int(len(split.test_idx)),
            "best_epoch": int(best_epoch),
            "best_val_mean_delta_k_pcm": float(best_val),
            "feature_set": ["geom_6", "phi_6", "xs_log_36"],
            "generator_trunk": [TRUNK_INPUT_DIM] + TRUNK_HIDDEN,
            "output_layer": f"({TRUNK_HIDDEN[-1]}+{XS_DIM})->1",
            "peds_generator_reference": [TRUNK_INPUT_DIM] + TRUNK_HIDDEN,
            "peds_head_reference": f"({TRUNK_HIDDEN[-1]}+{XS_DIM})->36",
            "peds_generator_param_count": peds_target,
            "baseline_param_count_estimate": baseline_target,
            "baseline_param_count": actual_params,
            "normalization": "train-split z-score for geom/phi/log-xs",
            "architecture_note": (
                "Boltzmann-app analogue: trunk on geom+phi, then single "
                "(hidden+XS)->1 layer parallel to PEDS XS head wiring."
            ),
            "norm_stats": {k: v.tolist() for k, v in norm_stats.items()},
        },
    )

    return {
        "run_id": f"train_1000_seed_{seed}",
        "train_size": int(len(split.train_idx)),
        "seed": seed,
        "best_epoch": int(best_epoch),
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
        "baseline_param_count": actual_params,
    }


def summarize_by_metric(metrics_df: pd.DataFrame, out_dir: Path) -> None:
    mean_row = metrics_df.mean(numeric_only=True)
    std_row = metrics_df.std(numeric_only=True)
    pd.DataFrame(
        {
            "metric": mean_row.index,
            "mean_across_seeds": mean_row.values,
            "std_across_seeds": std_row.reindex(mean_row.index).values,
        }
    ).to_csv(out_dir / "testset_results" / "baseline_summary_by_metric.csv", index=False)


def write_comparison_markdown(
    aug_df: pd.DataFrame,
    out_dir: Path,
    geom_only_metrics: Path,
) -> None:
    if not geom_only_metrics.exists():
        print(f"[warn] geom-only metrics not found at {geom_only_metrics}; skipping geom column in summary.")
        geom_df = None
    else:
        geom_df = pd.read_csv(geom_only_metrics)
    peds_df = pd.read_csv(PEDS_METRICS_DK)

    a = aug_df.mean(numeric_only=True)
    p = peds_df.mean(numeric_only=True)

    peds_params = peds_generator_param_count()
    baseline_params = int(a.get("baseline_param_count", baseline_param_count()))

    lines = [
        "# Augmented MLP (geom+phi+XS) vs geometry-only MLP and PEDS",
        "",
        "## Architecture (Boltzmann-app analogue)",
        "- **PEDS generator:** trunk `12 -> 128 -> 256 -> 128`, head `(128+36)->36`, "
        f"then physics solver ({peds_params:,} generator params).",
        "- **This baseline:**",
        f"  - trunk: `12 -> {' -> '.join(map(str, TRUNK_HIDDEN))}` on geom+phi",
        f"  - output: `({TRUNK_HIDDEN[-1]}+36)->1` (parallel to XS head wiring, single layer)",
        f"- **Baseline params:** `{baseline_params:,}`.",
        "",
        "## Inputs included",
        "- `6` geometry features (`params`)",
        "- `6` low-fidelity flux features (`phi`, same construction as PEDS)",
        "- `36` baseline XS slots (`log(xs_baseline)`, train-normalized like PEDS)",
        "",
        "## Test means across seeds",
        "| Metric | MLP geom-only | MLP geom+phi+XS | PEDS |",
        "|---|---:|---:|---:|",
    ]
    if geom_df is not None:
        g = geom_df.mean(numeric_only=True)
        lines.extend(
            [
                f"| MSE(k) | {g['test_mse_k']:.4g} | {a['test_mse_k']:.4g} | {p['test_mse_k']:.4g} |",
                f"| Mean |Δk| (pcm) | {g['test_mean_delta_k_pcm']:.1f} | {a['test_mean_delta_k_pcm']:.1f} | {p['test_mean_pcm']:.1f} |",
                f"| Median |Δk| (pcm) | {g['test_median_delta_k_pcm']:.1f} | {a['test_median_delta_k_pcm']:.1f} | {p['test_median_pcm']:.1f} |",
                f"| Mean fractional error | {g['test_mean_frac_error']:.4f} | {a['test_mean_frac_error']:.4f} | — |",
                f"| Frac(|Δk|<650 pcm) | {g['test_frac_below_650_delta_k']:.3f} | {a['test_frac_below_650_delta_k']:.3f} | {p['test_frac_below_650']:.3f} |",
            ]
        )
    else:
        lines.extend(
            [
                f"| MSE(k) | — | {a['test_mse_k']:.4g} | {p['test_mse_k']:.4g} |",
                f"| Mean |Δk| (pcm) | — | {a['test_mean_delta_k_pcm']:.1f} | {p['test_mean_pcm']:.1f} |",
                f"| Median |Δk| (pcm) | — | {a['test_median_delta_k_pcm']:.1f} | {p['test_median_pcm']:.1f} |",
                f"| Mean fractional error | — | {a['test_mean_frac_error']:.4f} | — |",
                f"| Frac(|Δk|<650 pcm) | — | {a['test_frac_below_650_delta_k']:.3f} | {p['test_frac_below_650']:.3f} |",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "- Augmented baseline uses the same extra inputs as PEDS with a single-layer "
            "solver swap (hidden+XS -> 1).",
            "- Remaining gap vs PEDS isolates the value of the physics solver and "
            "XS-correction training path.",
        ]
    )
    (out_dir / "COMPARISON_WITH_PEDS.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train geom+phi+XS MLP baseline.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for run outputs (default: mlp_baseline/mlp_with_phi_xs).",
    )
    parser.add_argument(
        "--geom-only-metrics",
        type=Path,
        default=DEFAULT_GEOM_ONLY_METRICS,
        help="CSV with geometry-only baseline metrics for comparison table.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    geom_only_metrics = args.geom_only_metrics.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "testset_results").mkdir(parents=True, exist_ok=True)

    peds_target = peds_generator_param_count()
    baseline_target = baseline_param_count()
    print(
        "Architecture:\n"
        f"  PEDS generator params : {peds_target:,}\n"
        f"  Baseline params (est): {baseline_target:,}"
    )

    data = np.load(DATA_PATH, allow_pickle=True)
    bundle = build_feature_bundle(
        geoms=np.asarray(data["params"], dtype=np.float32),
        rawparams=np.asarray(data["params_raw"], dtype=np.float32),
        keffs=np.asarray(data["keffs"], dtype=np.float32),
        cache_dir=output_dir,
    )

    rows: list[dict] = []
    for seed in SEEDS:
        print(f"[run] seed={seed}")
        row = train_one_seed(seed=seed, bundle=bundle, out_dir=output_dir)
        rows.append(row)
        print(
            f"  params={row['baseline_param_count']:,} | "
            f"mean |Δk|={row['test_mean_delta_k_pcm']:.1f} pcm | "
            f"frac<650={row['test_frac_below_650_delta_k']:.3f}"
        )

    metrics_df = pd.DataFrame(rows).sort_values(["train_size", "seed"]).reset_index(drop=True)
    out_csv = output_dir / "testset_results" / "test_metrics_all_runs.csv"
    metrics_df.to_csv(out_csv, index=False)
    summarize_by_metric(metrics_df, output_dir)
    write_comparison_markdown(metrics_df, output_dir, geom_only_metrics)
    print(f"Saved: {out_csv}")
    print(f"Saved: {output_dir / 'COMPARISON_WITH_PEDS.md'}")


if __name__ == "__main__":
    main()
