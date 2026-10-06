"""Plotting helpers: per-sample XS subplots, training-history figures,
and flux-shape plots."""
import os

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from NTcode_config_data.config_run import GEO_CYL as GEO
from solvers.NTdiffusion.diffusion_solver import run_diffusion_solver, _plot_fluxes
from plot_functions.xs_heatmap import plot_xs_subplots
from PEDS_subdivision import context
from PEDS_subdivision.context import update_geo
from PEDS_subdivision.physics_solver import NTdiff_solver


def _save_xs_subplots_for_samples(
    model,
    sample_indices,
    geoms_all,
    keffs_all,
    rawparams_all,
    xs_baselines_all,
    phi_norm_all,
    log_xs_mean_j,
    log_xs_std_j,
    epoch: int,
):
    """
    For each index in sample_indices, run a lightweight NN forward pass and
    save one plot_xs_subplots figure per sample.

    Files land in context.LOG_DIR/xs_subplots/epoch_{E:04d}_sample{IDX:03d}.png
    """
    subplots_dir = os.path.join(context.LOG_DIR, "xs_subplots")
    os.makedirs(subplots_dir, exist_ok=True)
    epoch_label = f"Epoch {epoch}"

    for idx in sample_indices:
        idx = int(idx)
        if idx >= len(geoms_all):
            print(f"  [subplot] sample_idx={idx} out of range, skipping")
            continue

        # ── per-sample baseline ───────────────────────────────────────────────
        geo_i    = update_geo(GEO, rawparams_all[idx])
        baseline = np.array(xs_baselines_all[idx], dtype=np.float32)   # (3, 12)

        xs_b  = jnp.array(xs_baselines_all[idx:idx+1], dtype=jnp.float32)   # (1,3,12)
        geom  = jnp.array(geoms_all[idx:idx+1],        dtype=jnp.float32)   # (1,6)

        # ── NN forward (no gradient needed) ──────────────────────────────────
        log_base   = model._log_baselines(xs_b, log_xs_mean_j, log_xs_std_j)   
        phi = jnp.array(phi_norm_all[idx:idx+1], dtype=jnp.float32)   # (1, nphifeats)                   # (1,3,12)
        log_ratios = np.array(
            model.generator(geom, log_base, phi, training=False)[0]  # (3,12)
        )
        final_xs = np.exp(log_ratios) * baseline                     # (3,12)

        final_xs = model.compute_xs(
            jnp.array(geoms_all[idx:idx+1], dtype=jnp.float32),
            jnp.array(xs_baselines_all[idx:idx+1], dtype=jnp.float32),
            jnp.array(phi_norm_all[idx:idx+1], dtype=jnp.float32),
            log_xs_mean_j,
            log_xs_std_j,
        )[0]
        keff_pred_val = None
        try:
            keff_pred_val = float(NTdiff_solver(
                jnp.array(final_xs, dtype=jnp.float32),
                jnp.array(rawparams_all[idx], dtype=jnp.float32),
                jnp.array([idx], dtype=jnp.int32),
            ))
        except Exception as e:
            print(f"  [subplot] solver failed for sample {idx} epoch {epoch}: {e}")

        save_path = os.path.join(
            subplots_dir, f"epoch_{epoch:04d}_sample{idx:03d}.png"
        )
        plot_xs_subplots(
            baseline    = baseline,
            final_xs    = final_xs,
            G           = GEO.G,
            save_path   = save_path,
            epoch_label = epoch_label,
            sample_idx  = idx,
            geo_params  = rawparams_all[idx],
            param_names = context.PARAM_NAMES,
            keff_ref    = float(keffs_all[idx]),
            keff_pred   = keff_pred_val,
        )
        print(f"  [subplot] epoch {epoch}  sample {idx}  →  {save_path}")


def _plot_history(history: dict, exp_name: str):
    """5-panel summary plot saved to context.LOG_DIR/training_history.png

    Panels:
      [0,0] MSE loss (log scale)
      [0,1] Val reactivity error in pcm
      [1,0] MAE in k-units (log scale)   ← NEW: correct label, log y-axis
      [1,1] % samples below 650 pcm
      [2,0] MAE in pcm (log scale)        ← NEW: physics-unit MAE
    """
    epochs_range = range(1, len(history["train_mse"]) + 1)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"PEDS training history — {exp_name}", fontsize=13)

    # ── [0,0] MSE ─────────────────────────────────────────────────────────────
    axes[0, 0].semilogy(epochs_range, history["train_mse"], label="train MSE")
    axes[0, 0].semilogy(epochs_range, history["val_mse"],   label="val MSE")
    axes[0, 0].set_title("MSE loss (k-units²)")
    axes[0, 0].set_ylabel("MSE  [k²]")
    axes[0, 0].legend()

    # ── [0,1] Reactivity error pcm ───────────────────────────────────────────
    axes[0, 1].plot(epochs_range, history["val_mean_pcm"],   label="mean |Δρ|")
    axes[0, 1].plot(epochs_range, history["val_median_pcm"], label="median |Δρ|")
    axes[0, 1].axhline(650, ls="--", color="grey", label="β_eff = 650 pcm")
    axes[0, 1].set_title("Val reactivity error (pcm)")
    axes[0, 1].set_ylabel("|Δρ|  [pcm]")
    axes[0, 1].legend()

    # ── [1,0] MAE in k-units, log scale ──────────────────────────────────────
    # MAE_k is in k-units (dimensionless k-eigenvalue differences).
    # Log scale is appropriate because it spans several orders of magnitude
    # during training and makes early improvement visible.
    axes[1, 0].semilogy(epochs_range, history["train_mae"], label="train MAE")
    axes[1, 0].semilogy(epochs_range, history["val_mae"],   label="val MAE")
    axes[1, 0].set_title("MAE  (k-units, log scale)")
    axes[1, 0].set_ylabel("MAE  [Δk]")
    axes[1, 0].legend()

    # ── [1,1] Fraction below 650 pcm ─────────────────────────────────────────
    axes[1, 1].plot(epochs_range, [x * 100 for x in history["val_frac_below_650"]])
    axes[1, 1].axhline(95, ls="--", color="grey", label="95% target")
    axes[1, 1].set_ylim(0, 105)
    axes[1, 1].set_title("% samples below 650 pcm")
    axes[1, 1].set_ylabel("fraction  [%]")
    axes[1, 1].legend()

    for ax in axes.flat:
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(context.LOG_DIR, "training_history.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"History plot saved → {path}")

    # ── Separate plot: val mean |Δρ| in pcm on log scale ─────────────────────
    # Useful when early epochs have very large errors that compress the linear scale.
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.semilogy(epochs_range, history["val_mean_pcm"],   label="mean |Δρ|")
    ax2.semilogy(epochs_range, history["val_median_pcm"], label="median |Δρ|")
    ax2.semilogy(epochs_range, history["val_p95_pcm"],    label="p95 |Δρ|",
                 ls="--", alpha=0.7)
    ax2.axhline(650, ls=":", color="grey", label="β_eff = 650 pcm")
    ax2.axhline(100, ls=":", color="navy", label="100 pcm target")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("|Δρ|  [pcm]")
    ax2.set_title(f"Val reactivity error — {exp_name}  (log scale)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    path2 = os.path.join(context.LOG_DIR, "val_reactivity_logscale.png")
    plt.savefig(path2, dpi=150)
    plt.close()
    print(f"Reactivity log-scale plot saved → {path2}")

def _save_flux_plots(
    sample_indices: list,
    val_geoms: np.ndarray,
    val_keffs: np.ndarray,
    val_rawparams: np.ndarray,
    val_xs_baselines: np.ndarray,
    log_xs_mean_j: jnp.ndarray,
    log_xs_std_j: jnp.ndarray,
    val_phi_norm: np.ndarray,
    model,
    epoch: int,
):
    """
    For each sample index, run a fresh forward solve with the current NN
    XS corrections and plot the flux shape using _plot_fluxes.

    Files land in context.LOG_DIR/flux_plots/epoch_{E:04d}_sample{IDX:03d}.png

    This uses unshuffled arrays so sample identity is stable across epochs.
    """
    flux_dir = os.path.join(context.LOG_DIR, "flux_plots")
    os.makedirs(flux_dir, exist_ok=True)

    for idx in sample_indices:
        idx = int(idx)
        if idx >= len(val_geoms):
            print(f"  [flux_plot] sample_idx={idx} out of range, skipping")
            continue

        # ── Geometry for this sample ──────────────────────────────────────────
        geo_i    = update_geo(GEO, val_rawparams[idx])
        R        = geo_i.boundaries[-1].radius
        I        = int(R / geo_i.mesh_size)
        Delta_r  = geo_i.mesh_size

        # ── Get NN-corrected XS (no gradient needed) ──────────────────────────
        xs_b  = jnp.array(val_xs_baselines[idx:idx+1], dtype=jnp.float32)  # (1,3,12)
        geom  = jnp.array(val_geoms[idx:idx+1],        dtype=jnp.float32)  # (1,6)
        phi   = jnp.array((val_phi_norm[idx:idx+1]), dtype=jnp.float32) 

        log_base   = model._log_baselines(xs_b)
        log_ratios = model.generator(geom, log_base, phi, training=False)  # (1,3,12)
        #xs_corrected = np.array(jnp.exp(log_ratios[0]) * xs_b[0])        # (3,12)
        # TODO check
        xs_corrected = model.compute_xs(
                jnp.array(val_geoms[idx:idx+1], dtype=jnp.float32),
                jnp.array(val_xs_baselines[idx:idx+1], dtype=jnp.float32),
                jnp.array(val_phi_norm[idx:idx+1], dtype=jnp.float32),
                log_xs_mean_j,
                log_xs_std_j,
            )[0]
        # ── Run the solver directly (NumPy, no JAX trace) ─────────────────────
        # We call run_diffusion_solver because it returns phi_fwd and phi_adj
        # in (G, I) shape, which is exactly what _plot_fluxes expects.
        try:
            k_pred, phi_fwd, phi_adj = run_diffusion_solver(xs_corrected, geo_i)
        except Exception as e:
            print(f"  [flux_plot] solver failed for sample {idx} epoch {epoch}: {e}")
            continue

        # phi_fwd and phi_adj already have shape (G, I) — no reshaping needed.
        # run_diffusion_solver volume-normalises them before returning.

        # ── Build the radial grid x (cell centres) ────────────────────────────
        # _plot_fluxes uses x as the horizontal axis. Cell centres are at
        # r = (i + 0.5) * Delta_r for i in 0..I-1.
        x = np.array([(i + 0.5) * Delta_r for i in range(I)])  # shape (I,)

        # ── Also run baseline (no NN) for comparison ──────────────────────────
        xs_base_np = np.array(val_xs_baselines[idx], dtype=np.float32)  # (3,12)
        try:
            k_base, phi_fwd_base, phi_adj_base = run_diffusion_solver(xs_base_np, geo_i)
        except Exception as e:
            print(f"  [flux_plot] baseline solver failed for sample {idx}: {e}")
            phi_fwd_base = None
            phi_adj_base = None
            k_base       = None

        # ── Plot ──────────────────────────────────────────────────────────────
        save_path = os.path.join(
            flux_dir, f"epoch_{epoch:04d}_sample{idx:03d}.png"
        )

        # _plot_fluxes from diffusion_solver.py takes:
        #   x              (I,)     radial grid
        #   geo            GeometryConfig
        #   phi_fwd_norm   (G, I)   forward flux
        #   phi_adj_norm   (G, I)   adjoint flux
        #   an_fwd_groups  optional — we use this slot for the baseline flux
        #   an_adj_groups  optional
        #   plot_output    str path
        #
        # We pass phi_fwd_base as an_fwd_groups so the plot shows both
        # the corrected flux (solid line) and the baseline flux (circles)
        # on the same axes. The legend labels them "Analytic Fwd" but
        # you can rename them inside _plot_fluxes if you prefer.

        _plot_fluxes(
            x            = x,
            geo          = geo_i,
            phi_fwd_norm = phi_fwd,
            phi_adj_norm = phi_adj,
            an_fwd_groups = phi_fwd_base,   # baseline for comparison
            an_adj_groups = None,
            plot_output  = save_path,
        )

        k_ref = float(val_keffs[idx])
        dr    = abs(k_pred - k_ref) / (k_pred * k_ref) * 1e5
        print(
            f"  [flux_plot] epoch {epoch}  sample {idx}  "
            f"k_ref={k_ref:.5f}  k_pred={k_pred:.5f}  |Δρ|={dr:.1f} pcm  → {save_path}"
        )

