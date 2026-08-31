"""PEDS_agent.py — agent study variant of PEDS.py.

Investigates two problems identified in precise_param_strat results:
  1. Low-keff configurations improve less after PEDS correction.
  2. Train/val generalization gap (~94% vs ~88.5% improvement).

10 study variants, selected via PEDS_STUDY_NAME env var.
Runs with TRAIN_SIZE=500, VAL_SIZE=300, TEST_SIZE=300, 3 seeds.
Results go to RUNS/agent_studies/{STUDY_NAME}/train_{N}_seed_{S}/.
"""

import multiprocessing
import os

THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={os.cpu_count()}"
os.environ["XLA_CPU_ENABLE_FAST_MATH"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"
multiprocessing.set_start_method("spawn", force=True)

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import optax
import pickle
from flax import nnx
import matplotlib.pyplot as plt
import csv
import time
import threading
import resource
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from collections import OrderedDict
import sys
import traceback
from datetime import datetime
import glob

sys.path.insert(0, PARENT_DIR)

ctx = mp.get_context("spawn")
_csv_lock = threading.Lock()

soft, hard = resource.getrlimit(resource.RLIMIT_AS)
print(f"Memory limit: soft={soft/1e9:.1f}GB hard={hard/1e9:.1f}GB")

from matrix_JAX_optimized import diffusion_setup_jax, Aphi_Fphi_scan, Aphi_Fphi_vjp
from NTcode_config_data.config_def import (GeometryConfig, MaterialSpec,
    BoundarySpec, BoundaryCondition, MatProperties)
from NTcode_config_data.config_run import GEO_CYL as GEO
from solvers.NTdiffusion.core.MG1D_eigenvalue_nregions import diffusion_setup, region_index
from solvers.NTdiffusion.diffusion_solver import (
    get_xs_basedon_geo, run_diffusion_solver, is_homogeneous,
    predict_xs, precompute_geometry, xs_layout, _plot_fluxes,
    build_xs_callables, fn_xs_per_region, bc_to_coeffs, GEOMETRY_CODE)
from PEDS_core.timing_utils import timer, print_timing_report, _TIMINGS
from plot_functions.xs_heatmap import plot_xs_subplots


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Study configuration
# ─────────────────────────────────────────────────────────────────────────────

STUDY_CONFIGS = {
    # Study 01: Baseline — same architecture/loss as PEDS.py, smaller dataset
    "s01_baseline": dict(
        hidden_sizes        = [128, 256, 128],
        activation          = "relu",
        use_dropout         = False,
        dropout_rate        = 0.0,
        use_residual        = False,
        loss_scheme         = "mse",          # plain MSE on keff
        weight_decay        = 0.0,            # no L2 reg (Adam)
        use_keff_input      = False,          # don't add baseline keff as feature
        input_noise_sigma   = 0.0,            # no input noise
        use_balanced        = False,          # no oversampling
        lr_max              = 2e-4,
        lr_min              = 5e-6,
        description         = "Baseline: [128,256,128] ReLU, MSE, Adam",
    ),
    # Study 02: PCM-weighted loss — upweight low-keff via 1/k^4 weighting
    "s02_weighted_loss": dict(
        hidden_sizes        = [128, 256, 128],
        activation          = "relu",
        use_dropout         = False,
        dropout_rate        = 0.0,
        use_residual        = False,
        loss_scheme         = "pcm_weighted",  # MSE weighted by 1/k_true^4
        weight_decay        = 0.0,
        use_keff_input      = False,
        input_noise_sigma   = 0.0,
        use_balanced        = False,
        lr_max              = 2e-4,
        lr_min              = 5e-6,
        description         = "PCM-weighted loss: w=1/k^4",
    ),
    # Study 03: Dropout 0.15 — explicit regularization to reduce generalization gap
    "s03_dropout": dict(
        hidden_sizes        = [128, 256, 128],
        activation          = "relu",
        use_dropout         = True,
        dropout_rate        = 0.15,
        use_residual        = False,
        loss_scheme         = "mse",
        weight_decay        = 0.0,
        use_keff_input      = False,
        input_noise_sigma   = 0.0,
        use_balanced        = False,
        lr_max              = 2e-4,
        lr_min              = 5e-6,
        description         = "Dropout 0.15 on trunk",
    ),
    # Study 04: AdamW weight decay 1e-4 — L2 regularization
    "s04_adamw": dict(
        hidden_sizes        = [128, 256, 128],
        activation          = "relu",
        use_dropout         = False,
        dropout_rate        = 0.0,
        use_residual        = False,
        loss_scheme         = "mse",
        weight_decay        = 1e-4,           # AdamW weight decay
        use_keff_input      = False,
        input_noise_sigma   = 0.0,
        use_balanced        = False,
        lr_max              = 2e-4,
        lr_min              = 5e-6,
        description         = "AdamW with weight_decay=1e-4",
    ),
    # Study 05: Baseline keff as extra input — helps model adapt to starting point
    "s05_keff_input": dict(
        hidden_sizes        = [128, 256, 128],
        activation          = "relu",
        use_dropout         = False,
        dropout_rate        = 0.0,
        use_residual        = False,
        loss_scheme         = "mse",
        weight_decay        = 0.0,
        use_keff_input      = True,           # add k_base_norm as 7th geom input
        input_noise_sigma   = 0.0,
        use_balanced        = False,
        lr_max              = 2e-4,
        lr_min              = 5e-6,
        description         = "Baseline keff added as 7th input feature",
    ),
    # Study 06: Residual trunk [128,128,128] — skip connections for better gradient flow
    "s06_residual": dict(
        hidden_sizes        = [128, 128, 128], # same-width for residuals
        activation          = "relu",
        use_dropout         = False,
        dropout_rate        = 0.0,
        use_residual        = True,           # add h_{l-1} skip to h_l (same dim)
        loss_scheme         = "mse",
        weight_decay        = 0.0,
        use_keff_input      = False,
        input_noise_sigma   = 0.0,
        use_balanced        = False,
        lr_max              = 2e-4,
        lr_min              = 5e-6,
        description         = "Residual trunk [128,128,128]",
    ),
    # Study 07: ELU activation — smoother gradients for non-saturating learning
    "s07_elu": dict(
        hidden_sizes        = [128, 256, 128],
        activation          = "elu",
        use_dropout         = False,
        dropout_rate        = 0.0,
        use_residual        = False,
        loss_scheme         = "mse",
        weight_decay        = 0.0,
        use_keff_input      = False,
        input_noise_sigma   = 0.0,
        use_balanced        = False,
        lr_max              = 2e-4,
        lr_min              = 5e-6,
        description         = "ELU activation instead of ReLU",
    ),
    # Study 08: Wider network [256,512,256] — more capacity for complex low-keff patterns
    "s08_wider": dict(
        hidden_sizes        = [256, 512, 256],
        activation          = "relu",
        use_dropout         = False,
        dropout_rate        = 0.0,
        use_residual        = False,
        loss_scheme         = "mse",
        weight_decay        = 0.0,
        use_keff_input      = False,
        input_noise_sigma   = 0.0,
        use_balanced        = False,
        lr_max              = 2e-4,
        lr_min              = 5e-6,
        description         = "Wider trunk [256,512,256]",
    ),
    # Study 09: Input noise augmentation sigma=0.01 — implicit regularization
    "s09_noise_aug": dict(
        hidden_sizes        = [128, 256, 128],
        activation          = "relu",
        use_dropout         = False,
        dropout_rate        = 0.0,
        use_residual        = False,
        loss_scheme         = "mse",
        weight_decay        = 0.0,
        use_keff_input      = False,
        input_noise_sigma   = 0.01,           # Gaussian noise on geom inputs
        use_balanced        = False,
        lr_max              = 2e-4,
        lr_min              = 5e-6,
        description         = "Input noise augmentation sigma=0.01",
    ),
    # Study 10: Balanced sampling — oversample bottom-30%-keff 2x per epoch
    "s10_balanced": dict(
        hidden_sizes        = [128, 256, 128],
        activation          = "relu",
        use_dropout         = False,
        dropout_rate        = 0.0,
        use_residual        = False,
        loss_scheme         = "mse",
        weight_decay        = 0.0,
        use_keff_input      = False,
        input_noise_sigma   = 0.0,
        use_balanced        = True,
        lr_max              = 2e-4,
        lr_min              = 5e-6,
        logratio_penalty    = 0.0,
        epochs              = 70,
        decay_epochs_override = 70,
        patience            = 15,
        description         = "Balanced sampling: low-keff (bot-30%) 2× per epoch",
    ),

    # ── Round-2 studies (all build on ELU as the clear round-1 winner) ─────────

    # Study 11: ELU + Residual — combine the two best individual improvements
    "s11_elu_residual": dict(
        hidden_sizes        = [128, 128, 128],
        activation          = "elu",
        use_dropout         = False,
        dropout_rate        = 0.0,
        use_residual        = True,
        loss_scheme         = "mse",
        weight_decay        = 0.0,
        use_keff_input      = False,
        input_noise_sigma   = 0.0,
        use_balanced        = False,
        lr_max              = 2e-4,
        lr_min              = 5e-6,
        logratio_penalty    = 0.0,
        epochs              = 70,
        decay_epochs_override = 70,
        patience            = 15,
        description         = "ELU + Residual [128,128,128]",
    ),
    # Study 12: ELU + longer training — ELU hadn't converged at epoch 70
    "s12_elu_long": dict(
        hidden_sizes        = [128, 256, 128],
        activation          = "elu",
        use_dropout         = False,
        dropout_rate        = 0.0,
        use_residual        = False,
        loss_scheme         = "mse",
        weight_decay        = 0.0,
        use_keff_input      = False,
        input_noise_sigma   = 0.0,
        use_balanced        = False,
        lr_max              = 2e-4,
        lr_min              = 5e-6,
        logratio_penalty    = 0.0,
        epochs              = 100,
        decay_epochs_override = 100,
        patience            = 20,
        description         = "ELU + 100 epochs (decay 100, patience 20)",
    ),
    # Study 13: Residual + longer training
    "s13_residual_long": dict(
        hidden_sizes        = [128, 128, 128],
        activation          = "relu",
        use_dropout         = False,
        dropout_rate        = 0.0,
        use_residual        = True,
        loss_scheme         = "mse",
        weight_decay        = 0.0,
        use_keff_input      = False,
        input_noise_sigma   = 0.0,
        use_balanced        = False,
        lr_max              = 2e-4,
        lr_min              = 5e-6,
        logratio_penalty    = 0.0,
        epochs              = 100,
        decay_epochs_override = 100,
        patience            = 20,
        description         = "Residual [128,128,128] + 100 epochs",
    ),
    # Study 14: ELU + Residual + longer training — full structural combo + full convergence
    "s14_elu_res_long": dict(
        hidden_sizes        = [128, 128, 128],
        activation          = "elu",
        use_dropout         = False,
        dropout_rate        = 0.0,
        use_residual        = True,
        loss_scheme         = "mse",
        weight_decay        = 0.0,
        use_keff_input      = False,
        input_noise_sigma   = 0.0,
        use_balanced        = False,
        lr_max              = 2e-4,
        lr_min              = 5e-6,
        logratio_penalty    = 0.0,
        epochs              = 100,
        decay_epochs_override = 100,
        patience            = 20,
        description         = "ELU + Residual [128,128,128] + 100 epochs",
    ),
    # Study 15: GELU — alternative smooth activation (used in GPT/BERT)
    "s15_gelu": dict(
        hidden_sizes        = [128, 256, 128],
        activation          = "gelu",
        use_dropout         = False,
        dropout_rate        = 0.0,
        use_residual        = False,
        loss_scheme         = "mse",
        weight_decay        = 0.0,
        use_keff_input      = False,
        input_noise_sigma   = 0.0,
        use_balanced        = False,
        lr_max              = 2e-4,
        lr_min              = 5e-6,
        logratio_penalty    = 0.0,
        epochs              = 70,
        decay_epochs_override = 70,
        patience            = 15,
        description         = "GELU activation",
    ),
    # Study 16: ELU + log-keff loss — scale-invariant loss, naturally weights low-keff
    # Loss = MSE(log k_pred, log k_ref). Gradient ∝ 1/k, larger for subcritical.
    # Gentler than 1/k^4 weighting (round-1 s02 failed with that).
    "s16_elu_log_loss": dict(
        hidden_sizes        = [128, 256, 128],
        activation          = "elu",
        use_dropout         = False,
        dropout_rate        = 0.0,
        use_residual        = False,
        loss_scheme         = "log_keff",
        weight_decay        = 0.0,
        use_keff_input      = False,
        input_noise_sigma   = 0.0,
        use_balanced        = False,
        lr_max              = 2e-4,
        lr_min              = 5e-6,
        logratio_penalty    = 0.0,
        epochs              = 70,
        decay_epochs_override = 70,
        patience            = 15,
        description         = "ELU + log-keff MSE loss (scale-invariant)",
    ),
    # Study 17: ELU + log-ratio L2 penalty — forces smaller corrections → less memorisation
    # penalty = λ * mean(log_ratios²) added to main loss.
    "s17_elu_logreg": dict(
        hidden_sizes        = [128, 256, 128],
        activation          = "elu",
        use_dropout         = False,
        dropout_rate        = 0.0,
        use_residual        = False,
        loss_scheme         = "mse",
        weight_decay        = 0.0,
        use_keff_input      = False,
        input_noise_sigma   = 0.0,
        use_balanced        = False,
        lr_max              = 2e-4,
        lr_min              = 5e-6,
        logratio_penalty    = 0.001,          # L2 on log_ratios
        epochs              = 70,
        decay_epochs_override = 70,
        patience            = 15,
        description         = "ELU + log-ratio L2 penalty λ=0.001",
    ),
    # Study 18: ELU + Residual + log-keff loss
    "s18_elu_res_log": dict(
        hidden_sizes        = [128, 128, 128],
        activation          = "elu",
        use_dropout         = False,
        dropout_rate        = 0.0,
        use_residual        = True,
        loss_scheme         = "log_keff",
        weight_decay        = 0.0,
        use_keff_input      = False,
        input_noise_sigma   = 0.0,
        use_balanced        = False,
        lr_max              = 2e-4,
        lr_min              = 5e-6,
        logratio_penalty    = 0.0,
        epochs              = 70,
        decay_epochs_override = 70,
        patience            = 15,
        description         = "ELU + Residual + log-keff loss",
    ),
    # Study 19: ELU + Residual + log-keff + 100 epochs — best structural + loss + convergence
    "s19_elu_res_log_long": dict(
        hidden_sizes        = [128, 128, 128],
        activation          = "elu",
        use_dropout         = False,
        dropout_rate        = 0.0,
        use_residual        = True,
        loss_scheme         = "log_keff",
        weight_decay        = 0.0,
        use_keff_input      = False,
        input_noise_sigma   = 0.0,
        use_balanced        = False,
        lr_max              = 2e-4,
        lr_min              = 5e-6,
        logratio_penalty    = 0.0,
        epochs              = 100,
        decay_epochs_override = 100,
        patience            = 20,
        description         = "ELU + Residual + log-keff loss + 100 epochs",
    ),
    # Study 20: ELU + Residual + log-ratio L2 + 100 epochs — structural + soft constraint + convergence
    "s20_elu_res_logreg_long": dict(
        hidden_sizes        = [128, 128, 128],
        activation          = "elu",
        use_dropout         = False,
        dropout_rate        = 0.0,
        use_residual        = True,
        loss_scheme         = "mse",
        weight_decay        = 0.0,
        use_keff_input      = False,
        input_noise_sigma   = 0.0,
        use_balanced        = False,
        lr_max              = 2e-4,
        lr_min              = 5e-6,
        logratio_penalty    = 0.001,
        epochs              = 100,
        decay_epochs_override = 100,
        patience            = 20,
        description         = "ELU + Residual + log-ratio L2 λ=0.001 + 100 epochs",
    ),

    # ── Round-3 studies ───────────────────────────────────────────────────────
    # All arms share the corrected infrastructure:
    #   * steps_per_epoch pinned to STEPS_PER_EPOCH_REF (independent of train_size)
    #   * fixed 7-epoch warmup (previously drifted with run length)
    #   * checkpoint selected on a 5-epoch trailing-window mean, not a single epoch
    #   * SWA weight averaging over the training tail, reported next to the point model
    #   * patience high so runs complete and the SWA phase is never truncated
    # Mechanisms measured inert in rounds 1-2 (log-keff loss, log-ratio penalty,
    # residual+ELU, GELU, AdamW, dropout, reweighting, oversampling) are all dropped.

    # r01: reference for round 3 — isolates the infrastructure change vs s12_elu_long
    "r01_ref": dict(
        hidden_sizes        = [128, 256, 128],
        activation          = "elu",
        use_dropout         = False, dropout_rate = 0.0, use_residual = False,
        loss_scheme         = "mse",
        weight_decay        = 0.0,
        use_keff_input      = False,
        input_noise_sigma   = 0.0,
        use_balanced        = False,
        lr_max              = 2e-4,
        lr_min              = 5e-6,
        logratio_penalty    = 0.0,
        epochs              = 100,
        decay_epochs_override = 100,
        patience            = 999,
        sel_window          = 5,
        swa                 = True,
        swa_start           = 70,
        swa_eval_every      = 10,
        description         = "R3 reference: ELU [128,256,128], lr_max 2e-4, SWA tail",
    ),
    # r02 / r03: lr_max sweep. Never varied in any of the 20 previous studies, and the
    # single hyperparameter most likely to matter. Fixed warmup + fixed steps/epoch make
    # this a clean 3-point sweep with r01 as the centre.
    "r02_lr_hi": dict(
        hidden_sizes        = [128, 256, 128],
        activation          = "elu",
        use_dropout         = False, dropout_rate = 0.0, use_residual = False,
        loss_scheme         = "mse",
        weight_decay        = 0.0,
        use_keff_input      = False,
        input_noise_sigma   = 0.0,
        use_balanced        = False,
        lr_max              = 4e-4,
        lr_min              = 5e-6,
        logratio_penalty    = 0.0,
        epochs              = 100,
        decay_epochs_override = 100,
        patience            = 999,
        sel_window          = 5,
        swa                 = True, swa_start = 70, swa_eval_every = 10,
        description         = "R3 lr_max=4e-4 (2x reference)",
    ),
    "r03_lr_lo": dict(
        hidden_sizes        = [128, 256, 128],
        activation          = "elu",
        use_dropout         = False, dropout_rate = 0.0, use_residual = False,
        loss_scheme         = "mse",
        weight_decay        = 0.0,
        use_keff_input      = False,
        input_noise_sigma   = 0.0,
        use_balanced        = False,
        lr_max              = 1e-4,
        lr_min              = 5e-6,
        logratio_penalty    = 0.0,
        epochs              = 100,
        decay_epochs_override = 100,
        patience            = 999,
        sel_window          = 5,
        swa                 = True, swa_start = 70, swa_eval_every = 10,
        description         = "R3 lr_max=1e-4 (0.5x reference)",
    ),
    # r04: proper SWA schedule — anneal to lr_max/10 by epoch 55, then hold constant so
    # the iterates keep exploring and the weight average has real signal to average over.
    # Annealing to ~0 (r01-r03) makes SWA collapse onto the final weights.
    "r04_swa_flat": dict(
        hidden_sizes        = [128, 256, 128],
        activation          = "elu",
        use_dropout         = False, dropout_rate = 0.0, use_residual = False,
        loss_scheme         = "mse",
        weight_decay        = 0.0,
        use_keff_input      = False,
        input_noise_sigma   = 0.0,
        use_balanced        = False,
        lr_max              = 2e-4,
        lr_min              = 5e-6,
        lr_mode             = "hold",
        # lr_hold = lr_max/4. SWA only helps if the iterates keep moving enough to have
        # variance worth averaging; the smoke run at lr_max/10 produced a +0.1 pcm delta
        # because consecutive epochs were nearly identical.
        lr_hold             = 5e-5,
        logratio_penalty    = 0.0,
        epochs              = 100,
        decay_epochs_override = 100,
        patience            = 999,
        sel_window          = 5,
        swa                 = True, swa_start = 55, swa_eval_every = 10,
        description         = "R3 SWA schedule: cosine->5e-5 by ep55 then hold, SWA from 55",
    ),
    # r05: regime features. Problem 1 is a near-constant *fractional* correction across
    # keff bins, so reweighting cannot fix it (s02/s10 both hurt). This instead gives the
    # model explicit information about the physical regime: baseline keff plus the
    # per-region fast/thermal flux ratio, which an MLP cannot easily form by division.
    "r05_regime_feat": dict(
        hidden_sizes        = [128, 256, 128],
        activation          = "elu",
        use_dropout         = False, dropout_rate = 0.0, use_residual = False,
        loss_scheme         = "mse",
        weight_decay        = 0.0,
        use_keff_input      = False,
        extra_feat_mode     = "regime",
        input_noise_sigma   = 0.0,
        use_balanced        = False,
        lr_max              = 2e-4,
        lr_min              = 5e-6,
        logratio_penalty    = 0.0,
        epochs              = 100,
        decay_epochs_override = 100,
        patience            = 999,
        sel_window          = 5,
        swa                 = True, swa_start = 70, swa_eval_every = 10,
        description         = "R3 + regime features (k_base + 3 spectral ratios)",
    ),
    # r06: reduce capacity. ~73k parameters for 500 samples is heavily over-parameterised,
    # and the one width experiment we ran (s08, wider) produced the worst gap of all
    # (414-440 pcm). Going the other direction has never been tested: [64,128,64] is ~19k.
    "r06_small": dict(
        hidden_sizes        = [64, 128, 64],
        activation          = "elu",
        use_dropout         = False, dropout_rate = 0.0, use_residual = False,
        loss_scheme         = "mse",
        weight_decay        = 0.0,
        use_keff_input      = False,
        input_noise_sigma   = 0.0,
        use_balanced        = False,
        lr_max              = 2e-4,
        lr_min              = 5e-6,
        logratio_penalty    = 0.0,
        epochs              = 100,
        decay_epochs_override = 100,
        patience            = 999,
        sel_window          = 5,
        swa                 = True, swa_start = 70, swa_eval_every = 10,
        description         = "R3 smaller trunk [64,128,64] (~19k params vs ~73k)",
    ),
    # Smoke test: exercises every new code path (regime features, SWA accumulate/eval/save,
    # window selection, test-time SWA comparison) in ~1/3 of a full run. Not a result arm.
    "r00_smoke": dict(
        hidden_sizes        = [64, 128, 64],
        activation          = "elu",
        use_dropout         = False, dropout_rate = 0.0, use_residual = False,
        loss_scheme         = "mse",
        weight_decay        = 0.0,
        use_keff_input      = False,
        extra_feat_mode     = "regime",
        input_noise_sigma   = 0.0,
        use_balanced        = False,
        lr_max              = 2e-4,
        lr_min              = 5e-6,
        lr_mode             = "hold",
        lr_hold             = 2e-5,
        logratio_penalty    = 0.0,
        epochs              = 32,
        decay_epochs_override = 32,
        patience            = 999,
        sel_window          = 5,
        swa                 = True, swa_start = 26, swa_eval_every = 3,
        description         = "SMOKE TEST — validates regime features + SWA paths end to end",
    ),

    # ── Round 4: finish the LR/warmup map, then test space-filling data selection ──
    # Recipe freeze from r02: ELU [128,256,128] MSE Adam, 100 ep, window-5, no SWA.
    # r02 already showed 1e-4 ≪ 2e-4 < 4e-4; these arms ask whether 4e-4 is the peak
    # and whether the inherited 7-epoch warmup is itself a constraint.

    "r07_lr_6e4": dict(
        hidden_sizes        = [128, 256, 128],
        activation          = "elu",
        use_dropout         = False, dropout_rate = 0.0, use_residual = False,
        loss_scheme         = "mse",
        weight_decay        = 0.0,
        use_keff_input      = False,
        input_noise_sigma   = 0.0,
        use_balanced        = False,
        lr_max              = 6e-4,
        lr_min              = 5e-6,
        warmup_epochs       = 7,
        logratio_penalty    = 0.0,
        epochs              = 100,
        decay_epochs_override = 100,
        patience            = 999,
        sel_window          = 5,
        swa                 = False,
        description         = "R4 lr_max=6e-4, warmup=7 (is 4e-4 still climbing?)",
    ),
    "r08_lr_8e4": dict(
        hidden_sizes        = [128, 256, 128],
        activation          = "elu",
        use_dropout         = False, dropout_rate = 0.0, use_residual = False,
        loss_scheme         = "mse",
        weight_decay        = 0.0,
        use_keff_input      = False,
        input_noise_sigma   = 0.0,
        use_balanced        = False,
        lr_max              = 8e-4,
        lr_min              = 5e-6,
        warmup_epochs       = 7,
        logratio_penalty    = 0.0,
        epochs              = 100,
        decay_epochs_override = 100,
        patience            = 999,
        sel_window          = 5,
        swa                 = False,
        description         = "R4 lr_max=8e-4, warmup=7 (past the peak?)",
    ),
    "r09_warm_3": dict(
        hidden_sizes        = [128, 256, 128],
        activation          = "elu",
        use_dropout         = False, dropout_rate = 0.0, use_residual = False,
        loss_scheme         = "mse",
        weight_decay        = 0.0,
        use_keff_input      = False,
        input_noise_sigma   = 0.0,
        use_balanced        = False,
        lr_max              = 4e-4,
        lr_min              = 5e-6,
        warmup_epochs       = 3,
        logratio_penalty    = 0.0,
        epochs              = 100,
        decay_epochs_override = 100,
        patience            = 999,
        sel_window          = 5,
        swa                 = False,
        description         = "R4 lr_max=4e-4, warmup=3 (more time at peak)",
    ),
    "r10_warm_14": dict(
        hidden_sizes        = [128, 256, 128],
        activation          = "elu",
        use_dropout         = False, dropout_rate = 0.0, use_residual = False,
        loss_scheme         = "mse",
        weight_decay        = 0.0,
        use_keff_input      = False,
        input_noise_sigma   = 0.0,
        use_balanced        = False,
        lr_max              = 4e-4,
        lr_min              = 5e-6,
        warmup_epochs       = 14,
        logratio_penalty    = 0.0,
        epochs              = 100,
        decay_epochs_override = 100,
        patience            = 999,
        sel_window          = 5,
        swa                 = False,
        description         = "R4 lr_max=4e-4, warmup=14 (stabler Adam moments)",
    ),
    "r11_lr6_warm14": dict(
        hidden_sizes        = [128, 256, 128],
        activation          = "elu",
        use_dropout         = False, dropout_rate = 0.0, use_residual = False,
        loss_scheme         = "mse",
        weight_decay        = 0.0,
        use_keff_input      = False,
        input_noise_sigma   = 0.0,
        use_balanced        = False,
        lr_max              = 6e-4,
        lr_min              = 5e-6,
        warmup_epochs       = 14,
        logratio_penalty    = 0.0,
        epochs              = 100,
        decay_epochs_override = 100,
        patience            = 999,
        sel_window          = 5,
        swa                 = False,
        description         = "R4 lr_max=6e-4 × warmup=14 (higher LR often needs longer warmup)",
    ),
    "r12_maximin": dict(
        hidden_sizes        = [128, 256, 128],
        activation          = "elu",
        use_dropout         = False, dropout_rate = 0.0, use_residual = False,
        loss_scheme         = "mse",
        weight_decay        = 0.0,
        use_keff_input      = False,
        input_noise_sigma   = 0.0,
        use_balanced        = False,
        lr_max              = 4e-4,
        lr_min              = 5e-6,
        warmup_epochs       = 7,
        logratio_penalty    = 0.0,
        epochs              = 100,
        decay_epochs_override = 100,
        patience            = 999,
        sel_window          = 5,
        swa                 = False,
        split_method        = "maximin",
        lock_split_seeds    = True,
        description         = "R4 maximin-within-keff-bin train/val, locked test, r02 recipe, 3 inits",
    ),

    # Round 5: scale the R4 winner (r07: ELU, lr_max=6e-4, warmup=7) to 1000 train.
    # steps_per_epoch stays pinned at 16 so the LR trajectory is unchanged vs 500-sample runs.
    "r13_1k_elu6e4": dict(
        hidden_sizes        = [128, 256, 128],
        activation          = "elu",
        use_dropout         = False, dropout_rate = 0.0, use_residual = False,
        loss_scheme         = "mse",
        weight_decay        = 0.0,
        use_keff_input      = False,
        input_noise_sigma   = 0.0,
        use_balanced        = False,
        lr_max              = 6e-4,
        lr_min              = 5e-6,
        warmup_epochs       = 7,
        logratio_penalty    = 0.0,
        epochs              = 100,
        decay_epochs_override = 100,
        patience            = 999,
        sel_window          = 5,
        swa                 = False,
        description         = "R5 scale-up: r07 recipe, 1000 train / 500 val, lr_max=6e-4 warmup=7",
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0: Global constants
# ─────────────────────────────────────────────────────────────────────────────
STUDY_NAME   = os.environ.get("PEDS_STUDY_NAME", "s01_baseline")
if STUDY_NAME not in STUDY_CONFIGS:
    raise ValueError(f"Unknown STUDY_NAME={STUDY_NAME!r}. Choose from: {list(STUDY_CONFIGS)}")

cfg = STUDY_CONFIGS[STUDY_NAME]
print(f"\n=== PEDS Agent Study: {STUDY_NAME} ===")
print(f"    {cfg['description']}\n")

TRAIN_SIZE   = int(os.environ.get("PEDS_TRAIN_SIZE", 500))
VAL_SIZE     = int(os.environ.get("PEDS_VAL_SIZE", 300))
TEST_SIZE    = int(os.environ.get("PEDS_TEST_SIZE", 300))
SEED         = int(os.environ.get("PEDS_SEED", 0))
TRAIN_SEED   = SEED
VAL_SEED     = SEED
TEST_SEED    = int(os.environ.get("PEDS_TEST_SEED", 0))   # fixed across all runs
HOLDOUT_SEED = TEST_SEED
BATCH_SIZE   = 32
EPOCHS       = 70
DECAY_EPOCHS = int(os.environ.get("PEDS_DECAY_EPOCHS", 70))

# Reference optimiser steps per epoch. Pinned so that the LR schedule and the number
# of updates per epoch do not change when train_size changes. ceil(500/32) = 16 keeps
# behaviour at train_size=500 identical to rounds 1-2, so results stay comparable.
STEPS_PER_EPOCH_REF = 16
WARMUP_EPOCHS_REF   = 7      # fixed warmup length in epochs (was 10% of total steps)

EXP_NAME  = f"train_{TRAIN_SIZE}_seed_{SEED}"
LOG_DIR   = os.path.join(THIS_DIR, "RUNS", "agent_studies", STUDY_NAME, EXP_NAME)

_DATA_FILEPATH     = os.path.join(PARENT_DIR, "data", "highfidelity", "17jul_0.8_1.2.npz")
USE_SPLIT_CACHE    = False
USE_BALANCED_RANGES = False
USE_PARAM_STRATIFIED = True
PARAM_STRAT_BINS   = 5
CACHE_DIR_NAME     = "split_cache/.split_cache_"

RUN_GRAD_CHECK     = False   # skip for speed in study runs

LOG_RATIO_CLIP_LO  = -1.8
LOG_RATIO_CLIP_HI  = 0.5

XS_HEATMAP_EPOCHS        = [EPOCHS]           # only final epoch (saves I/O)
SUBPLOT_SAMPLE_INDICES   = [0, 10, 20, 30, 40]
TRACKED_SAMPLES          = [0, 1, 2, 3, 4]
PARAM_NAMES              = ['b4c_r', 'cr_frac', 'fuel_r', 'enrichment', 'f_mod', 'water_r']

N_WORKERS = min(int(os.environ.get("SLURM_CPUS_PER_TASK", 32)), BATCH_SIZE)
print(f"Using {N_WORKERS} parallel workers")

executor = None

XS_DIR = os.path.join(LOG_DIR, "XS")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(XS_DIR, exist_ok=True)

CHECKPOINT_DIR = os.path.join(LOG_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
BEST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pkl")
LAST_CKPT_PATH = os.path.join(CHECKPOINT_DIR, "last_checkpoint.pkl")
SWA_CKPT_PATH  = os.path.join(CHECKPOINT_DIR, "swa_model.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Context init + imports from PEDS_subdivision
# ─────────────────────────────────────────────────────────────────────────────
from PEDS_subdivision import context
context.init(
    data_filepath     = _DATA_FILEPATH,
    log_dir           = LOG_DIR,
    xs_dir            = XS_DIR,
    param_names       = PARAM_NAMES,
    param_strat_bins  = PARAM_STRAT_BINS,
    log_ratio_clip_lo = LOG_RATIO_CLIP_LO,
    log_ratio_clip_hi = LOG_RATIO_CLIP_HI,
    batch_size        = BATCH_SIZE,
)

from PEDS_subdivision.context import update_geo
GEO_DATA        = context.GEO_DATA
SLAY            = context.SLAY
N_FLAT_MAX      = context.N_FLAT_MAX
XS_MASK         = context.XS_MASK
XS_MASK_J       = context.XS_MASK_J
_PRESOLVE_CACHE = context._PRESOLVE_CACHE
_GEO_DATA_CACHE = context._GEO_DATA_CACHE

from PEDS_subdivision.physics_solver import (
    _run_NT_solver, _NTdiff_fwd, _NTdiff_bwd, NTdiff_solver,
    _NT_batch_fwd, _NT_batch_bwd, NTdiff_solver_batch,
    _build_padded_geo_arrays, _solve_sample_worker,
)
from PEDS_subdivision.data_loading import (
    data_loader, compute_batch_baselines, compute_phi_features,
    derive_split_indices, derive_split_indices_by_params,
    _save_split_cache, load_or_create_split_cache,
    load_data, load_data_balanced_ranges, load_data_param_stratified,
    print_keff_bin_counts, print_param_bin_counts,
    check_keff_split_representation, check_keff_bin_geometry_dominance,
)
from PEDS_subdivision.metrics import (
    compute_delta_rho_pcm, compute_metrics, print_metrics,
)
from PEDS_subdivision import logging_csv
from PEDS_subdivision.logging_csv import (
    init_csv_logs, close_csv_logs, log_epoch_stats, log_keff_batch,
    log_logratio_saturation, log_xs_history_samples, log_splits,
    save_final_xs_csv, val_log_path,
)
from PEDS_subdivision.plotting import (
    _plot_xs_heatmap, _save_xs_subplots_for_samples, _plot_history, _save_flux_plots,
)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Neural network
# ─────────────────────────────────────────────────────────────────────────────

def clip_log_ratios(x, lo, hi):
    return jnp.clip(x, lo, hi)


def _get_activation(name: str):
    if name == "relu":
        return nnx.relu
    if name == "elu":
        return nnx.elu
    if name == "gelu":
        return nnx.gelu
    raise ValueError(f"Unknown activation: {name!r}")


class GeneratorNN(nnx.Module):
    """Trunk + single head, extended with study-specific options.

    Supports:
      - `activation`: "relu" | "elu"
      - `use_dropout` + `dropout_rate`: dropout after each trunk activation
      - `use_residual`: skip connections between same-width layers
      - `use_keff_input`: keff_base_norm [batch,1] appended to geoms/phi
    """

    def __init__(self, layer_sizes: list, n_regions: int, xs_per_region: int,
                 rngs: nnx.Rngs, *,
                 activation: str = "relu",
                 use_dropout: bool = False,
                 dropout_rate: float = 0.0,
                 use_residual: bool = False):
        super().__init__()
        he_init   = nnx.initializers.kaiming_normal()
        zero_init = nnx.initializers.zeros

        self.act         = _get_activation(activation)
        self.use_residual = use_residual
        self.use_dropout  = use_dropout

        self.layers = nnx.List([
            nnx.Linear(
                in_features  = layer_sizes[i],
                out_features = layer_sizes[i + 1],
                kernel_init  = he_init,
                bias_init    = nnx.initializers.constant(0.0),
                rngs         = rngs,
            )
            for i in range(len(layer_sizes) - 1)
        ])

        if use_dropout and dropout_rate > 0.0:
            self.dropouts = nnx.List([
                nnx.Dropout(rate=dropout_rate, rngs=rngs)
                for _ in range(len(layer_sizes) - 1)
            ])
        else:
            self.dropouts = None

        trunk_out  = layer_sizes[-1]
        total_xs   = n_regions * xs_per_region

        self.head = nnx.Linear(
            in_features  = trunk_out + total_xs,
            out_features = total_xs,
            kernel_init  = zero_init,
            bias_init    = zero_init,
            rngs         = rngs,
        )

        self.n_regions     = n_regions
        self.xs_per_region = xs_per_region
        # Store sizes for residual check
        self._layer_sizes  = layer_sizes

    def __call__(self, geoms: jnp.ndarray, xs_baselines_log: jnp.ndarray,
                 phi_norm: jnp.ndarray, keff_base_norm: jnp.ndarray = None,
                 training: bool = False) -> jnp.ndarray:
        parts = [geoms, phi_norm]
        if keff_base_norm is not None:
            parts.append(keff_base_norm)
        else:
            # Guard every call path (including external diagnostic/plotting helpers that
            # do not know about the extra-feature channel): pad to the trunk's expected
            # input width with zeros, which is the training mean for standardised feats.
            missing = self._layer_sizes[0] - geoms.shape[-1] - phi_norm.shape[-1]
            if missing > 0:
                parts.append(jnp.zeros((geoms.shape[0], missing), dtype=geoms.dtype))
        x = jnp.concatenate(parts, axis=-1)

        prev_x = None
        for i, layer in enumerate(self.layers):
            x_new = self.act(layer(x))
            if self.use_dropout and self.dropouts is not None:
                x_new = self.dropouts[i](x_new, deterministic=not training)
            # Residual skip when current output matches previous output in width
            if (self.use_residual and prev_x is not None
                    and x_new.shape[-1] == prev_x.shape[-1]):
                x_new = x_new + prev_x
            prev_x = x_new
            x = x_new

        log_base_flat = jnp.reshape(xs_baselines_log, (geoms.shape[0], -1))
        feat = jnp.concatenate([x, log_base_flat], axis=-1)
        out  = self.head(feat)
        return jnp.reshape(out, (geoms.shape[0], self.n_regions, self.xs_per_region))


def compute_phi_features_ext(rawparams: np.ndarray) -> tuple:
    """Single-pass variant of ``compute_phi_features`` that also returns regime features.

    The upstream helper already runs the NT solver for every sample and discards its
    baseline keff, so these extra descriptors cost no additional solves.

    Returns:
        phi_features : [N, G*3]  identical to ``compute_phi_features``
        regime       : [N, 4]    [k_base, phi_g1/phi_g0 per region (3)]
    """
    phi_features, regime = [], []

    for i, p in enumerate(rawparams):
        geo_i = update_geo(GEO, p)
        xs_i  = np.array(predict_xs(geo_i), dtype=np.float32)
        k, phi_fwd_padded, _, _ = _run_NT_solver(xs_i, p, np.array([i], dtype=np.int32))

        R       = geo_i.boundaries[-1].radius
        I       = int(R / geo_i.mesh_size)
        Delta_r = geo_i.mesh_size
        G       = geo_i.G
        region_radii = [b.radius for b in geo_i.boundaries]

        feats = []
        for g in range(G):
            phi_g  = phi_fwd_padded[g * (I + 1): g * (I + 1) + I]
            r_prev = 0.0
            for r_reg in region_radii:
                centres = np.array([(ic + 0.5) * Delta_r for ic in range(I)])
                mask    = (centres >= r_prev) & (centres < r_reg)
                feats.append(float(np.mean(phi_g[mask])) if mask.any() else 0.0)
                r_prev = r_reg
        phi_features.append(feats)

        # Spectral index per region: fast/thermal balance. An MLP cannot easily form
        # a ratio from the two absolute fluxes, so supply it explicitly.
        n_reg = len(region_radii)
        ratios = []
        for ridx in range(n_reg):
            g0 = feats[ridx]
            g1 = feats[n_reg + ridx] if (n_reg + ridx) < len(feats) else 0.0
            ratios.append(float(g1 / g0) if abs(g0) > 1e-12 else 0.0)
        regime.append([float(k)] + ratios)

        if i % 20 == 0:
            print(f"  phi_reg+regime precompute {i}/{len(rawparams)}", flush=True)

    return (np.array(phi_features, dtype=np.float32),
            np.array(regime, dtype=np.float32))


class PEDSModel(nnx.Module):
    """GeneratorNN + physics solver, extended for agent studies."""

    def __init__(self, hidden_sizes: list, n_regions: int, G: int,
                 n_phi_feats: int, rngs: nnx.Rngs, *,
                 use_keff_input: bool = False,
                 extra_feats_dim: int = 0,
                 activation: str = "relu",
                 use_dropout: bool = False,
                 dropout_rate: float = 0.0,
                 use_residual: bool = False):
        super().__init__()
        xs_region = fn_xs_per_region(G)
        # extra_feats carried through the keff_base_norm channel:
        # 1 for legacy keff-only input, or extra_feats_dim for regime features
        extra_feats = extra_feats_dim if extra_feats_dim > 0 else (1 if use_keff_input else 0)
        input_dim   = 6 + n_phi_feats + extra_feats
        layer_sizes = [input_dim] + hidden_sizes
        self.generator = GeneratorNN(
            layer_sizes, n_regions, xs_region, rngs,
            activation   = activation,
            use_dropout  = use_dropout,
            dropout_rate = dropout_rate,
            use_residual = use_residual,
        )
        self.n_regions      = n_regions
        self.G              = G
        self.use_keff_input = use_keff_input
        self._extra_dim     = extra_feats

    def _log_baselines(self, xs_baselines, log_xs_mean=None, log_xs_std=None):
        if log_xs_mean is not None:
            fallback = jnp.exp(log_xs_mean)
        else:
            fallback = jnp.ones_like(xs_baselines)
        safe   = jnp.where(xs_baselines > 1e-10, xs_baselines, fallback)
        log_xs = jnp.log(safe)
        if log_xs_mean is not None and log_xs_std is not None:
            log_xs = (log_xs - log_xs_mean) / log_xs_std
        return log_xs

    def _normalize_chi(self, xs_final, xs_baselines):
        lay   = xs_layout(self.G)
        chi_sl = lay['chi']
        nuf_sl = lay['nuSigma_f']
        eps    = 1e-12

        chi         = jnp.maximum(xs_final[:, :, chi_sl], 0.0)
        chi_sum     = jnp.sum(chi, axis=-1, keepdims=True)
        chi_norm    = chi / jnp.where(chi_sum > eps, chi_sum, 1.0)

        base_chi    = jnp.maximum(xs_baselines[:, :, chi_sl], 0.0)
        base_sum    = jnp.sum(base_chi, axis=-1, keepdims=True)
        base_norm   = base_chi / jnp.where(base_sum > eps, base_sum, 1.0)
        uniform     = jnp.ones_like(base_chi) / float(self.G)
        chi_fallback = jnp.where(base_sum > eps, base_norm, uniform)

        fissile   = jnp.sum(jnp.maximum(xs_baselines[:, :, nuf_sl], 0.0), axis=-1, keepdims=True) > eps
        chi_final = jnp.where(fissile,
                              jnp.where(chi_sum > eps, chi_norm, chi_fallback),
                              jnp.zeros_like(chi))
        return xs_final.at[:, :, chi_sl].set(chi_final)

    def compute_xs(self, geoms, xs_baselines, phi_norm,
                   log_xs_mean=None, log_xs_std=None, keff_base_norm=None):
        # Fallback: if model expects extra inputs but none supplied, use zeros
        if self._extra_dim > 0 and keff_base_norm is None:
            keff_base_norm = jnp.zeros((geoms.shape[0], self._extra_dim), dtype=jnp.float32)
        log_base   = self._log_baselines(xs_baselines, log_xs_mean, log_xs_std)
        log_ratios = self.generator(geoms, log_base, phi_norm,
                                    keff_base_norm=keff_base_norm, training=False)
        log_ratios = clip_log_ratios(log_ratios, LOG_RATIO_CLIP_LO, LOG_RATIO_CLIP_HI)
        xs_final   = jnp.exp(log_ratios) * xs_baselines * XS_MASK_J
        xs_final   = self._normalize_chi(xs_final, xs_baselines)
        return np.array(xs_final)

    def compute_log_ratios(self, geoms, xs_baselines, phi_norm,
                           log_xs_mean=None, log_xs_std=None, keff_base_norm=None):
        if self._extra_dim > 0 and keff_base_norm is None:
            keff_base_norm = jnp.zeros((geoms.shape[0], self._extra_dim), dtype=jnp.float32)
        log_base   = self._log_baselines(xs_baselines, log_xs_mean, log_xs_std)
        log_ratios = self.generator(geoms, log_base, phi_norm,
                                    keff_base_norm=keff_base_norm, training=False)
        log_ratios = clip_log_ratios(log_ratios, LOG_RATIO_CLIP_LO, LOG_RATIO_CLIP_HI)
        return np.array(log_ratios)

    def __call__(self, geoms, params_raw, xs_baselines, phi_norm,
                 training: bool = False, sample_id_offset: int = 0,
                 log_xs_mean=None, log_xs_std=None, keff_base_norm=None):
        batch_size = geoms.shape[0]
        log_base   = self._log_baselines(xs_baselines, log_xs_mean, log_xs_std)
        phi        = jnp.reshape(phi_norm, (batch_size, -1))

        log_ratios = self.generator(geoms, log_base, phi,
                                    keff_base_norm=keff_base_norm, training=training)
        log_ratios = clip_log_ratios(log_ratios, LOG_RATIO_CLIP_LO, LOG_RATIO_CLIP_HI)

        xs_final     = jnp.exp(log_ratios) * xs_baselines * XS_MASK_J
        xs_final     = self._normalize_chi(xs_final, xs_baselines)

        params_raw_j = jnp.array(params_raw, dtype=jnp.float32)
        sample_ids_j = jnp.arange(batch_size, dtype=jnp.int32) + sample_id_offset

        with timer("forward: solver loop (all samples)", verbose=False):
            keffs = NTdiff_solver_batch(xs_final, params_raw_j, sample_ids_j)

        return keffs, xs_final, log_ratios


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_code_snapshot(logdir, expname, log_handle=None):
    import shutil, hashlib
    src = os.path.abspath(__file__)
    snapshot_path = os.path.join(logdir, f"code_snapshot_{expname}.py")
    shutil.copy2(src, snapshot_path)
    with open(src, "r", encoding="utf-8") as f:
        code_text = f.read()
    code_hash = hashlib.sha256(code_text.encode("utf-8")).hexdigest()
    if log_handle is not None:
        log_handle.write("\n" + "="*80 + "\nCODE SNAPSHOT\n")
        log_handle.write(f"Source : {src}\nSHA256 : {code_hash}\n")
        log_handle.write("="*80 + "\n")
        log_handle.flush()
    return snapshot_path, code_hash


def save_checkpoint(state_obj, filepath, metadata=None):
    np_state = jax.tree_util.tree_map(np.asarray, state_obj)
    with open(filepath, "wb") as f:
        pickle.dump({"state": np_state, "metadata": metadata or {}}, f)


def load_checkpoint(filepath):
    with open(filepath, "rb") as f:
        payload = pickle.load(f)
    jax_state = jax.tree_util.tree_map(jnp.asarray, payload["state"])
    return jax_state, payload["metadata"]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: Training
# ─────────────────────────────────────────────────────────────────────────────

def train(filepath, train_size, val_size, test_size, batch_size, epochs, lr_max, lr_min,
          decay_epochs, hidden_sizes, n_regions, G, seed, study_cfg: dict):

    init_csv_logs()

    # Adapt diagnostic epochs to the actual run length
    _run_epochs = epochs
    _heatmap_epochs = {_run_epochs}   # only final epoch

    # ── 5.1 Data ──────────────────────────────────────────────────────────────
    print("Loading data …")
    split_method = study_cfg.get("split_method", "param_strat")
    if study_cfg.get("lock_split_seeds", False):
        # Init seed still varies; train/val/test are the same 500/300/300.
        data_train_seed, data_val_seed = 0, 0
        print(f"  lock_split_seeds: TRAIN/VAL seeds forced to 0 (init seed={seed})")
    else:
        data_train_seed, data_val_seed = TRAIN_SEED, VAL_SEED
    (train_idx, train_geoms, train_keffs, train_rawparams, train_phi_features), \
    (val_idx,   val_geoms,   val_keffs,   val_rawparams,   val_phi_features), \
    (test_idx,  test_geoms,  test_keffs,  test_rawparams,  test_phi_features) = \
        load_data_param_stratified(
            filepath, train_size, val_size, test_size,
            train_seed=data_train_seed, val_seed=data_val_seed, test_seed=TEST_SEED,
            holdout_seed=HOLDOUT_SEED,
            n_bins_per_param=PARAM_STRAT_BINS,
            balanced=False,
            split_method=split_method,
        )
    train_size = len(train_geoms)
    val_size   = len(val_geoms)
    test_size  = len(test_geoms)
    log_splits(train_idx, train_keffs, val_idx, val_keffs, test_idx, test_keffs)
    print(f"Actual  train={train_size}  val={val_size}  test={test_size}")
    print_keff_bin_counts("TRAIN", train_keffs)
    print_keff_bin_counts("VAL",   val_keffs)

    # ── phi normalisation ─────────────────────────────────────────────────────
    phi_mean       = train_phi_features.mean(axis=0)
    phi_std        = train_phi_features.std(axis=0) + 1e-8
    train_phi_norm = ((train_phi_features - phi_mean) / phi_std).astype(np.float32)
    val_phi_norm   = ((val_phi_features   - phi_mean) / phi_std).astype(np.float32)
    test_phi_norm  = ((test_phi_features  - phi_mean) / phi_std).astype(np.float32)

    n_phi_feats = GEO.G * 3

    # ── regime features (optional) ────────────────────────────────────────────
    # [k_base, phi_g1/phi_g0 per region]. The baseline keff is already computed and
    # discarded inside the phi-feature pass, so these cost no extra solver calls.
    extra_mode = study_cfg.get("extra_feat_mode", None)
    if extra_mode == "regime":
        print("Computing regime features (k_base + spectral ratios) …")
        _, train_regime = compute_phi_features_ext(train_rawparams)
        _, val_regime   = compute_phi_features_ext(val_rawparams)
        reg_mean  = train_regime.mean(axis=0)
        reg_std   = train_regime.std(axis=0) + 1e-8
        train_extra = ((train_regime - reg_mean) / reg_std).astype(np.float32)
        val_extra   = ((val_regime   - reg_mean) / reg_std).astype(np.float32)
        extra_dim   = train_extra.shape[1]
        print(f"  regime features: dim={extra_dim}  mean={reg_mean}  std={reg_std}")
    else:
        train_extra = val_extra = None
        reg_mean = reg_std = None
        extra_dim = 0

    # ── 5.2 Model ────────────────────────────────────────────────────────────
    rngs  = nnx.Rngs(seed)
    model = PEDSModel(
        hidden_sizes  = hidden_sizes,
        n_regions     = n_regions,
        G             = G,
        n_phi_feats   = n_phi_feats,
        rngs          = rngs,
        use_keff_input = study_cfg["use_keff_input"],
        extra_feats_dim = extra_dim,
        activation    = study_cfg["activation"],
        use_dropout   = study_cfg["use_dropout"],
        dropout_rate  = study_cfg["dropout_rate"],
        use_residual  = study_cfg["use_residual"],
    )

    # ── LR schedule, decoupled from train_size ────────────────────────────────
    # steps_per_epoch is pinned to a reference value so that the schedule (and the
    # number of optimiser updates per epoch) is identical regardless of train_size.
    # Previously it was ceil(train_size/batch_size), which silently doubled both the
    # step count and the integrated LR when the dataset grew.
    steps_per_epoch  = STEPS_PER_EPOCH_REF
    natural_spe      = max(int(np.ceil(train_size / batch_size)), 1)
    decay_steps      = int(decay_epochs) * steps_per_epoch
    # Fixed warmup: warmup stabilises Adam's moment estimates, so it must be a fixed
    # number of steps, not a fraction of an arbitrary run length.
    warmup_epochs    = int(study_cfg.get("warmup_epochs", WARMUP_EPOCHS_REF))
    warmup_steps     = warmup_epochs * steps_per_epoch
    lr_mode          = study_cfg.get("lr_mode", "anneal")
    swa_start        = int(study_cfg.get("swa_start", 0))

    if lr_mode == "hold":
        # SWA-style schedule: cosine down to lr_hold by swa_start, then hold constant
        # so the iterates keep exploring the basin and weight averaging has signal.
        lr_hold    = study_cfg.get("lr_hold", lr_max / 10.0)
        hold_start = max(swa_start * steps_per_epoch, warmup_steps + 1)
        lr_schedule = optax.join_schedules(
            schedules=[
                optax.linear_schedule(lr_min, lr_max, warmup_steps),
                optax.cosine_decay_schedule(lr_max, hold_start - warmup_steps,
                                            alpha=lr_hold / lr_max),
                optax.constant_schedule(lr_hold),
            ],
            boundaries=[warmup_steps, hold_start],
        )
    else:
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value   = lr_min,
            peak_value   = lr_max,
            warmup_steps = warmup_steps,
            decay_steps  = decay_steps,
            end_value    = lr_min,
        )
    print(f"LR schedule: mode={lr_mode}  steps/epoch={steps_per_epoch} "
          f"(natural would be {natural_spe})  warmup={warmup_steps} steps "
          f"({warmup_epochs} ep)  total={decay_steps} steps  "
          f"lr_max={lr_max:.2e} lr_min={lr_min:.2e}")
    wd = study_cfg["weight_decay"]
    if wd > 0.0:
        base_opt = optax.adamw(lr_schedule, weight_decay=wd)
    else:
        base_opt = optax.adam(lr_schedule)
    optimizer = nnx.Optimizer(
        model,
        optax.chain(optax.clip_by_global_norm(0.5), base_opt),
        wrt=nnx.Param,
    )

    # ── 5.3 Loss ─────────────────────────────────────────────────────────────
    loss_scheme      = study_cfg["loss_scheme"]
    logratio_penalty = study_cfg.get("logratio_penalty", 0.0)

    def loss_fn(model, geoms, keffs_true, rawparams_batch,
                xs_baselines_batch, phi_norm_batch,
                log_xs_mean_j, log_xs_std_j, keff_base_norm_batch=None):
        keff_pred, _, log_ratios = model(
            geoms, rawparams_batch, xs_baselines_batch, phi_norm_batch,
            training=True, log_xs_mean=log_xs_mean_j, log_xs_std=log_xs_std_j,
            keff_base_norm=keff_base_norm_batch,
        )
        if loss_scheme == "pcm_weighted":
            w = 1.0 / jnp.clip(keffs_true, 0.5, 2.0) ** 4
            w = w / jnp.mean(w)
            loss = jnp.mean(w * (keff_pred - keffs_true) ** 2)
        elif loss_scheme == "log_keff":
            # Scale-invariant loss: MSE(log k_pred, log k_ref).
            # Gradient ∝ 1/k → naturally upweights subcritical (low-keff) samples
            # without the instability of hard 1/k^4 weighting.
            lp = jnp.log(jnp.clip(keff_pred,  0.3, 5.0))
            lt = jnp.log(jnp.clip(keffs_true, 0.3, 5.0))
            loss = jnp.mean((lp - lt) ** 2)
        else:
            loss = jnp.mean((keff_pred - keffs_true) ** 2)
        if logratio_penalty > 0.0:
            # L2 penalty on the magnitude of XS corrections.
            # Forces the model to prefer smaller, more generalizable corrections
            # rather than extreme sample-specific adjustments.
            loss = loss + logratio_penalty * jnp.mean(log_ratios ** 2)
        return loss, keff_pred

    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)

    # ── 5.4 Pre-compute XS baselines ─────────────────────────────────────────
    print("Precomputing XS baselines …")
    train_xs_baselines = compute_batch_baselines(train_rawparams, GEO)
    val_xs_baselines   = compute_batch_baselines(val_rawparams,   GEO)
    print("Done.")

    # Log-baseline normalisation stats
    _train_xs_np = np.array(train_xs_baselines)
    _safe        = np.where(_train_xs_np > 1e-10, _train_xs_np, np.ones_like(_train_xs_np))
    _log_train   = np.log(_safe)
    log_xs_mean  = _log_train.mean(axis=0).astype(np.float32)
    log_xs_std   = (_log_train.std(axis=0) + 1e-8).astype(np.float32)
    log_xs_mean_j = jnp.array(log_xs_mean)
    log_xs_std_j  = jnp.array(log_xs_std)

    # ── 5.5 Epoch-0 baseline keffs (also used as input feature for study_05) ─
    print("Computing epoch-0 baselines …")
    kp_train_baseline, kp_val_baseline = [], []
    for i in range(len(train_rawparams)):
        xs_i = np.array(predict_xs(update_geo(GEO, train_rawparams[i])), dtype=np.float32)
        k, _, _, _ = _run_NT_solver(xs_i, train_rawparams[i], np.array([i], dtype=np.int32))
        kp_train_baseline.append(float(k))
    for i in range(len(val_rawparams)):
        xs_i = np.array(predict_xs(update_geo(GEO, val_rawparams[i])), dtype=np.float32)
        k, _, _, _ = _run_NT_solver(xs_i, val_rawparams[i], np.array([i], dtype=np.int32))
        kp_val_baseline.append(float(k))

    log_keff_batch(logging_csv.train_writer, logging_csv.train_logfile, epoch=0,
                   k_pred=np.array(kp_train_baseline), k_ref=train_keffs,
                   avg_train_loss=0.0, avg_val_loss=0.0)
    log_keff_batch(logging_csv.val_writer, logging_csv.val_logfile, epoch=0,
                   k_pred=np.array(kp_val_baseline), k_ref=val_keffs,
                   avg_train_loss=0.0, avg_val_loss=0.0)

    # Normalise baseline keff for use as NN input (study_05)
    kp_train_base_arr = np.array(kp_train_baseline, dtype=np.float32)
    kp_val_base_arr   = np.array(kp_val_baseline,   dtype=np.float32)
    keff_base_mean = kp_train_base_arr.mean()
    keff_base_std  = kp_train_base_arr.std() + 1e-8
    if extra_dim > 0:
        # Regime features ride the same channel as the legacy keff input
        train_keff_base_norm = train_extra
        val_keff_base_norm   = val_extra
    else:
        train_keff_base_norm = ((kp_train_base_arr - keff_base_mean) / keff_base_std)[:, None]
        val_keff_base_norm   = ((kp_val_base_arr   - keff_base_mean) / keff_base_std)[:, None]

    log_xs_history_samples(
        model=model, epoch=0,
        sample_indices=TRACKED_SAMPLES,
        geom_sample_all=train_geoms,
        rawparams_sample_all=train_rawparams,
        xs_baselines_sample_all=train_xs_baselines,
        phi_norm_sample_all=train_phi_norm,
        log_xs_mean_j=log_xs_mean_j, log_xs_std_j=log_xs_std_j,
    )

    # ── 5.6 Training loop ────────────────────────────────────────────────────
    history = {k: [] for k in
               ["train_mse","val_mse","train_mae","val_mae",
                "val_mean_pcm","val_median_pcm","val_p95_pcm",
                "val_std_pcm","val_frac_below_650"]}

    best_val_mean_pcm      = float("inf")
    best_epoch             = 0
    EMA_ALPHA              = 0.1
    MIN_SAVE_EPOCH         = 25
    PATIENCE               = study_cfg.get("patience", 15)
    epochs_since_improvement = 0
    ema_val_mean_pcm       = None
    train_shuffle_rng      = np.random.default_rng(seed)

    # ── Robust checkpoint selection ───────────────────────────────────────────
    # Selection on a trailing-window mean of val mean_pcm rather than a single epoch,
    # so one lucky epoch cannot win. The val curve swings by ~100 pcm epoch to epoch,
    # and the raw per-epoch minimum sat ~45 pcm below the deployed EMA checkpoint.
    SEL_WINDOW  = int(study_cfg.get("sel_window", 5))
    val_pcm_hist = []

    # ── SWA (stochastic weight averaging) ─────────────────────────────────────
    # Averages the weights themselves over the tail of training, instead of merely
    # smoothing the selection metric. Directly targets the observed oscillation.
    swa_enabled    = bool(study_cfg.get("swa", False))
    swa_eval_every = int(study_cfg.get("swa_eval_every", 10))
    swa_params     = None
    swa_count      = 0
    best_swa_pcm   = float("inf")
    best_swa_epoch = 0

    # Balanced sampling: identify low-keff indices (bottom 30%)
    low_keff_thresh  = np.percentile(train_keffs, 30)
    low_keff_indices = np.where(train_keffs < low_keff_thresh)[0]
    print(f"Balanced sampling: {len(low_keff_indices)} low-keff samples (k < {low_keff_thresh:.3f})")

    use_keff_input  = study_cfg["use_keff_input"] or extra_dim > 0
    input_noise_sig = study_cfg["input_noise_sigma"]
    use_balanced    = study_cfg["use_balanced"]

    def _eval_split(eval_model, geoms_, keffs_, raw_, xsb_, phi_, kbase_, id_offset):
        """Run the physics solver over a split with the given model; return (k_pred, k_ref)."""
        _PRESOLVE_CACHE.clear()
        kp, kr = [], []
        for b_i, (bg, bk, br, bb, bphi, bkbase) in enumerate(data_loader(
                geoms_, keffs_, raw_, np.array(xsb_), phi_, kbase_,
                batch_size=batch_size)):
            gstart   = b_i * batch_size
            bkbase_j = jnp.array(bkbase) if use_keff_input else None
            xs_np = eval_model.compute_xs(
                jnp.array(bg), jnp.array(bb), jnp.array(bphi),
                log_xs_mean_j, log_xs_std_j, keff_base_norm=bkbase_j,
            )
            args_list = [(i, xs_np[i], np.array(br[i]), id_offset + i + gstart, -1)
                         for i in range(len(bg))]
            for i, k, _pf, _pa, _gd, _ep, w_el in executor.map(_solve_sample_worker, args_list):
                kp.append(float(k)); kr.append(float(bk[i]))
                _TIMINGS["fwd val worker sample"].append(w_el)
        return np.array(kp), np.array(kr)

    for epoch in range(1, epochs + 1):
        # Build epoch index (with optional oversampling of low-keff)
        if use_balanced:
            base_idx  = np.arange(train_size)
            extra_idx = low_keff_indices                        # 1 extra copy
            all_idx   = np.concatenate([base_idx, extra_idx])
            epoch_idx = train_shuffle_rng.permutation(len(all_idx))
            idx       = all_idx[epoch_idx]
        else:
            idx = train_shuffle_rng.permutation(train_size)

        t_geoms   = train_geoms[idx]
        t_keffs   = train_keffs[idx]
        t_raw     = train_rawparams[idx]
        t_base    = np.array(train_xs_baselines)[idx]
        t_phi     = train_phi_norm[idx]
        t_kbase   = train_keff_base_norm[idx]          # [N,1], used only if keff_input
        t_sample_ids = np.arange(train_size)[idx % train_size]

        current_lr = float(lr_schedule(optimizer.step.value))
        print(f"Epoch {epoch:4d} | LR = {current_lr:.2e}")

        epoch_loss = 0.0
        all_kp_train, all_kr_train, all_sample_ids = [], [], []
        epoch_grad_norms = []
        n_steps_done = 0

        n_eff = len(idx)
        for batch_idx, batch_data in enumerate(data_loader(
                t_geoms, t_keffs, t_raw, t_base, t_phi, t_kbase, t_sample_ids,
                batch_size=batch_size)):
            # Fixed number of optimiser updates per epoch, independent of train_size.
            # At train_size=500/bs=32 the natural count is already 16, so this is a
            # no-op there; for larger datasets the remaining batches roll into the
            # next epoch via the reshuffle, so all data is still seen over time.
            if batch_idx >= STEPS_PER_EPOCH_REF:
                break
            bg, bk, br, bb, bphi, bkbase, bsid = batch_data

            # Optional input noise on geometry inputs (study_09)
            if input_noise_sig > 0.0:
                noise = np.random.normal(0.0, input_noise_sig, bg.shape).astype(np.float32)
                bg    = np.clip(bg + noise, 0.0, 1.0)

            bp      = jnp.array(bg)
            bkj     = jnp.array(bk)
            bxs     = jnp.array(bb)
            b_phi_j = jnp.array(bphi)
            bkbase_j = jnp.array(bkbase) if use_keff_input else None

            xs_np = model.compute_xs(bp, bxs, b_phi_j, log_xs_mean_j, log_xs_std_j,
                                     keff_base_norm=bkbase_j)

            args_list = [(i, xs_np[i], np.array(br[i]), i, epoch)
                         for i in range(len(bg))]
            _t_batch = time.perf_counter()
            results  = list(executor.map(_solve_sample_worker, args_list))
            _TIMINGS["fwd pre-solve batch wall"].append(time.perf_counter() - _t_batch)
            _PRESOLVE_CACHE.clear()
            for i, k, phi_fwd, phi_adj, geo_data, _, w_elapsed in results:
                _PRESOLVE_CACHE[i] = (k, phi_fwd, phi_adj, geo_data)
                _TIMINGS["fwd pre-solve worker sample"].append(w_elapsed)

            with timer("train step: forward+backward+optiupd", verbose=False):
                (loss, keff_preds_batch), grads = grad_fn(
                    model, bp, bkj, br, bxs, b_phi_j,
                    log_xs_mean_j, log_xs_std_j,
                    bkbase_j,
                )
            grad_norm = float(optax.global_norm(grads))
            epoch_grad_norms.append(grad_norm)
            optimizer.update(model, grads)

            epoch_loss    += float(loss)
            n_steps_done  += 1
            all_kp_train.extend(np.array(keff_preds_batch).tolist())
            all_kr_train.extend(bk.tolist())
            all_sample_ids.extend(bsid.tolist())

        ids   = np.asarray(all_sample_ids, dtype=int)
        order = np.argsort(ids)
        log_keff_batch(
            logging_csv.train_writer, logging_csv.train_logfile, epoch,
            np.array(all_kp_train)[order],
            np.array(all_kr_train)[order],
            epoch_loss / max(1, n_steps_done),
            0.0,
            sample_id_offset=0,
            sample_ids_override=ids[order],
        )

        log_xs_history_samples(
            model=model, epoch=epoch,
            sample_indices=TRACKED_SAMPLES,
            geom_sample_all=train_geoms,
            rawparams_sample_all=train_rawparams,
            xs_baselines_sample_all=train_xs_baselines,
            phi_norm_sample_all=train_phi_norm,
            log_xs_mean_j=log_xs_mean_j, log_xs_std_j=log_xs_std_j,
        )

        # ── validation ───────────────────────────────────────────────────────
        gn = np.array(epoch_grad_norms)
        print(f"  grad_norm: min={gn.min():.3f} mean={gn.mean():.3f} max={gn.max():.3f}")

        _PRESOLVE_CACHE.clear()
        all_kp_val, all_kr_val = [], []
        with timer("validation step"):
            for batch_idx, (bg, bk, br, bb, bphi, bkbase) in enumerate(data_loader(
                    val_geoms, val_keffs, val_rawparams,
                    np.array(val_xs_baselines), val_phi_norm,
                    val_keff_base_norm,
                    batch_size=batch_size)):

                global_start = batch_idx * batch_size
                bkbase_j = jnp.array(bkbase) if use_keff_input else None
                xs_np = model.compute_xs(
                    jnp.array(bg), jnp.array(bb), jnp.array(bphi),
                    log_xs_mean_j, log_xs_std_j, keff_base_norm=bkbase_j,
                )
                VAL_OFFSET = train_size + 10000
                args_list  = [(i, xs_np[i], np.array(br[i]),
                               VAL_OFFSET + i + global_start, -1)
                              for i in range(len(bg))]
                results = list(executor.map(_solve_sample_worker, args_list))
                for i, k, _pf, _pa, _gd, _ep, w_elapsed in results:
                    all_kp_val.append(float(k))
                    all_kr_val.append(float(bk[i]))
                    _TIMINGS["fwd val worker sample"].append(w_elapsed)

        val_log_ratios = model.compute_log_ratios(
            jnp.array(val_geoms), jnp.array(val_xs_baselines),
            jnp.array(val_phi_norm), log_xs_mean_j, log_xs_std_j,
            keff_base_norm=jnp.array(val_keff_base_norm) if use_keff_input else None,
        )
        log_logratio_saturation(epoch, val_log_ratios)

        kp_val = np.array(all_kp_val)
        kr_val = np.array(all_kr_val)
        kp_tr  = np.array(all_kp_train)
        kr_tr  = np.array(all_kr_train)

        train_m = compute_metrics(kp_tr, kr_tr)
        val_m   = compute_metrics(kp_val, kr_val)
        print_metrics(epoch, "TRAIN", train_m)
        print_metrics(epoch, "VAL",   val_m)

        if ema_val_mean_pcm is None:
            ema_val_mean_pcm = val_m["mean_pcm"]
        else:
            ema_val_mean_pcm = EMA_ALPHA * val_m["mean_pcm"] + (1 - EMA_ALPHA) * ema_val_mean_pcm

        # Selection metric: trailing-window mean over the last SEL_WINDOW epochs.
        # Requires a sustained good stretch rather than one fortunate epoch.
        val_pcm_hist.append(val_m["mean_pcm"])
        sel_metric = float(np.mean(val_pcm_hist[-SEL_WINDOW:]))
        print(f"  select: window{SEL_WINDOW}-mean={sel_metric:.1f}  "
              f"ema={ema_val_mean_pcm:.1f}  raw={val_m['mean_pcm']:.1f}")

        if epoch >= MIN_SAVE_EPOCH and sel_metric < best_val_mean_pcm:
            best_val_mean_pcm = sel_metric
            best_epoch = epoch
            save_checkpoint(nnx.state(model), BEST_CKPT_PATH,
                            metadata={"epoch": epoch, "val_mean_pcm": best_val_mean_pcm,
                                      "hidden_sizes": hidden_sizes, "n_regions": n_regions,
                                      "G": G, "n_phi_feats": n_phi_feats,
                                      "activation": study_cfg.get("activation", "relu"),
                                      "use_residual": bool(study_cfg.get("use_residual", False)),
                                      "use_dropout": bool(study_cfg.get("use_dropout", False)),
                                      "dropout_rate": float(study_cfg.get("dropout_rate", 0.0)),
                                      "use_keff_input": bool(study_cfg.get("use_keff_input", False)),
                                      "extra_feats_dim": int(model._extra_dim),
                                      "study_name": STUDY_NAME})
            print(f"  ✓ best window-mean val_mean_pcm = {best_val_mean_pcm:.1f} pcm (epoch {epoch})")
            epochs_since_improvement = 0
        elif epoch >= MIN_SAVE_EPOCH:
            epochs_since_improvement += 1

        # ── SWA: accumulate the running weight average over the tail ───────────
        if swa_enabled and epoch >= swa_start:
            cur = jax.tree_util.tree_map(
                lambda x: jnp.asarray(x, dtype=jnp.float32), nnx.state(model, nnx.Param))
            if swa_params is None:
                swa_params, swa_count = cur, 1
            else:
                swa_count += 1
                swa_params = jax.tree_util.tree_map(
                    lambda a, x: a + (x - a) / swa_count, swa_params, cur)

        log_keff_batch(logging_csv.val_writer, logging_csv.val_logfile, epoch,
                       kp_val, kr_val, train_m["mse_k"], val_m["mse_k"])
        log_epoch_stats(epoch, train_m, val_m)

        history["train_mse"].append(train_m["mse_k"])
        history["val_mse"].append(val_m["mse_k"])
        history["train_mae"].append(train_m["MAE_k"])
        history["val_mae"].append(val_m["MAE_k"])
        history["val_mean_pcm"].append(val_m["mean_pcm"])
        history["val_median_pcm"].append(val_m["median_pcm"])
        history["val_p95_pcm"].append(val_m["p95_pcm"])
        history["val_std_pcm"].append(val_m["std_pcm"])
        history["val_frac_below_650"].append(val_m["frac_below_650"])

        if epoch in _heatmap_epochs:
            _plot_xs_heatmap(
                model, train_geoms[:8], np.array(train_xs_baselines)[:8],
                train_phi_norm[:8], log_xs_mean_j, log_xs_std_j,
                epoch=epoch, n_show=8,
            )
            _save_xs_subplots_for_samples(
                model=model, sample_indices=SUBPLOT_SAMPLE_INDICES,
                geoms_all=val_geoms, keffs_all=val_keffs, rawparams_all=val_rawparams,
                xs_baselines_all=np.array(val_xs_baselines), phi_norm_all=val_phi_norm,
                log_xs_mean_j=log_xs_mean_j, log_xs_std_j=log_xs_std_j, epoch=epoch,
            )

        # ── SWA evaluation (end of epoch, after all cache-dependent logging) ───
        if swa_enabled and swa_params is not None and swa_count >= 2:
            if epoch % swa_eval_every == 0 or epoch == epochs:
                live = nnx.state(model, nnx.Param)
                nnx.update(model, swa_params)
                kp_s, kr_s = _eval_split(model, val_geoms, val_keffs, val_rawparams,
                                         val_xs_baselines, val_phi_norm,
                                         val_keff_base_norm, train_size + 90000)
                swa_m = compute_metrics(kp_s, kr_s)
                nnx.update(model, live)     # restore live weights; training continues
                _PRESOLVE_CACHE.clear()
                print(f"  SWA(n={swa_count}) val mean_pcm={swa_m['mean_pcm']:.1f}  "
                      f"frac<650={swa_m['frac_below_650']:.3f}  "
                      f"(point={val_m['mean_pcm']:.1f})")
                if swa_m["mean_pcm"] < best_swa_pcm:
                    best_swa_pcm, best_swa_epoch = swa_m["mean_pcm"], epoch
                    save_checkpoint(swa_params, SWA_CKPT_PATH,
                                    metadata={"epoch": epoch, "val_mean_pcm": best_swa_pcm,
                                              "swa_count": swa_count,
                                              "hidden_sizes": hidden_sizes,
                                              "n_regions": n_regions, "G": G,
                                              "n_phi_feats": n_phi_feats})
                    print(f"  ✓ best SWA val_mean_pcm = {best_swa_pcm:.1f} pcm")

        if epoch >= MIN_SAVE_EPOCH and epochs_since_improvement >= PATIENCE:
            print(f"Early stopping at epoch {epoch}: no improvement for {PATIENCE} epochs")
            break

        if epoch % 20 == 0:
            jax.clear_caches()

    # ── post-training ─────────────────────────────────────────────────────────
    save_checkpoint(nnx.state(optimizer), LAST_CKPT_PATH,
                    metadata={"epoch": epochs, "val_mean_pcm": val_m["mean_pcm"]})

    best_state, best_meta = load_checkpoint(BEST_CKPT_PATH)
    nnx.update(model, best_state)
    print(f"Loaded best checkpoint from epoch {best_meta['epoch']} "
          f"(val_mean_pcm={best_meta['val_mean_pcm']:.1f})")

    save_final_xs_csv(model, train_geoms, train_rawparams, np.array(train_xs_baselines),
                      train_phi_norm, train_keffs, log_xs_mean_j, log_xs_std_j,
                      file_path=os.path.join(XS_DIR, "final_xs_train.csv"), tag="train")
    save_final_xs_csv(model, val_geoms, val_rawparams, np.array(val_xs_baselines),
                      val_phi_norm, val_keffs, log_xs_mean_j, log_xs_std_j,
                      file_path=os.path.join(XS_DIR, "final_xs_val.csv"), tag="val")

    # ── TEST evaluation ───────────────────────────────────────────────────────
    print("\n=== Final TEST evaluation ===")
    test_xs_baselines = compute_batch_baselines(test_rawparams, GEO)

    # Compute test baseline keff for keff_input study
    if extra_dim > 0:
        _, test_regime = compute_phi_features_ext(test_rawparams)
        test_keff_base_norm = ((test_regime - reg_mean) / reg_std).astype(np.float32)
    elif use_keff_input:
        kp_test_base = []
        for i in range(len(test_rawparams)):
            xs_i = np.array(predict_xs(update_geo(GEO, test_rawparams[i])), dtype=np.float32)
            k, _, _, _ = _run_NT_solver(xs_i, test_rawparams[i], np.array([i], dtype=np.int32))
            kp_test_base.append(float(k))
        test_keff_base_norm = ((np.array(kp_test_base, dtype=np.float32) - keff_base_mean)
                               / keff_base_std)[:, None]
    else:
        test_keff_base_norm = np.zeros((test_size, 1), dtype=np.float32)

    all_kp_test, all_kr_test = [], []
    for bg, bk, br, bb, bphi, bkbase in data_loader(
            test_geoms, test_keffs, test_rawparams,
            np.array(test_xs_baselines), test_phi_norm,
            test_keff_base_norm, batch_size=batch_size):
        bkbase_j = jnp.array(bkbase) if use_keff_input else None
        xs_np = model.compute_xs(jnp.array(bg), jnp.array(bb), jnp.array(bphi),
                                 log_xs_mean_j, log_xs_std_j, keff_base_norm=bkbase_j)
        for i in range(len(bg)):
            k, _, _, _ = _run_NT_solver(xs_np[i], np.array(br[i]), np.array([i], dtype=np.int32))
            all_kp_test.append(float(k))
            all_kr_test.append(float(bk[i]))

    test_m = compute_metrics(np.array(all_kp_test), np.array(all_kr_test))
    print_metrics(epochs, "TEST", test_m)

    # ── SWA model evaluated on the same fixed test set ────────────────────────
    swa_test_m = None
    if swa_enabled and os.path.exists(SWA_CKPT_PATH):
        swa_state, swa_meta = load_checkpoint(SWA_CKPT_PATH)
        point_state = nnx.state(model, nnx.Param)
        nnx.update(model, swa_state)
        kp_st, kr_st = _eval_split(model, test_geoms, test_keffs, test_rawparams,
                                   test_xs_baselines, test_phi_norm,
                                   test_keff_base_norm, 200000)
        swa_test_m = compute_metrics(kp_st, kr_st)
        print(f"\n=== SWA TEST (epoch {swa_meta['epoch']}, n={swa_meta['swa_count']}) ===")
        print_metrics(epochs, "TEST-SWA", swa_test_m)
        print(f"  point-checkpoint test mean_pcm = {test_m['mean_pcm']:.1f}")
        print(f"  SWA             test mean_pcm = {swa_test_m['mean_pcm']:.1f}  "
              f"(delta {swa_test_m['mean_pcm'] - test_m['mean_pcm']:+.1f})")
        if swa_test_m["mean_pcm"] > test_m["mean_pcm"]:
            nnx.update(model, point_state)   # keep whichever generalises better

    # Write brief per-study summary CSV
    summary_path = os.path.join(LOG_DIR, "study_summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["study", "seed", "best_epoch", "val_mean_pcm", "val_frac_below_650",
                    "test_mean_pcm", "test_frac_below_650",
                    "swa_epoch", "swa_val_mean_pcm",
                    "swa_test_mean_pcm", "swa_test_frac_below_650"])
        w.writerow([STUDY_NAME, seed, best_meta["epoch"],
                    f"{best_meta['val_mean_pcm']:.2f}",
                    f"{history['val_frac_below_650'][best_meta['epoch']-1]:.4f}",
                    f"{test_m['mean_pcm']:.2f}",
                    f"{test_m['frac_below_650']:.4f}",
                    best_swa_epoch if swa_test_m else "",
                    f"{best_swa_pcm:.2f}" if swa_test_m else "",
                    f"{swa_test_m['mean_pcm']:.2f}" if swa_test_m else "",
                    f"{swa_test_m['frac_below_650']:.4f}" if swa_test_m else ""])

    _plot_history(history, EXP_NAME)
    print_timing_report()
    close_csv_logs()
    return model, history


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
class _Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for fh in self.files:
            fh.write(data)
            fh.flush()
    def flush(self):
        for fh in self.files:
            fh.flush()


if __name__ == "__main__":
    _orig_stdout = sys.stdout
    _orig_stderr = sys.stderr
    loggy_file = open(os.path.join(LOG_DIR, f"train_log_{EXP_NAME}.txt"), "w", buffering=1)
    snapshot_path, code_hash = save_code_snapshot(LOG_DIR, EXP_NAME)
    sys.stdout = _Tee(_orig_stdout, loggy_file)
    sys.stderr = _Tee(_orig_stderr, loggy_file)
    start_perf = time.time()
    print(f"Started at   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Study        : {STUDY_NAME} — {cfg['description']}")

    executor = ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=ctx)

    try:
        print(f"N_FLAT_MAX = {N_FLAT_MAX}")
        HP = dict(
            filepath     = _DATA_FILEPATH,
            train_size   = TRAIN_SIZE,
            val_size     = VAL_SIZE,
            test_size    = TEST_SIZE,
            batch_size   = BATCH_SIZE,
            epochs       = cfg.get("epochs", EPOCHS),
            lr_max       = cfg["lr_max"],
            lr_min       = cfg["lr_min"],
            decay_epochs = cfg.get("decay_epochs_override", DECAY_EPOCHS),
            hidden_sizes = cfg["hidden_sizes"],
            n_regions    = 3,
            G            = 2,
            seed         = SEED,
            study_cfg    = cfg,
        )
        model, history = train(**HP)
        print("\nDone.")
        print(f"  Final val mean |Δρ|   : {history['val_mean_pcm'][-1]:.1f} pcm")
        print(f"  Final val frac <650   : {history['val_frac_below_650'][-1]*100:.1f}%")

    except Exception:
        traceback.print_exc(file=loggy_file)
        traceback.print_exc(file=_orig_stderr)
        raise

    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        elapsed = time.time() - start_perf
        print(f"Finished at  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Elapsed      : {elapsed/3600:.2f} hours")
        sys.stdout = _orig_stdout
        sys.stderr = _orig_stderr
        loggy_file.close()
