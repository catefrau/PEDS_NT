"""Plotting helpers: per-sample XS subplots, training-history figures,
and flux-shape plots."""
import os

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import LogFormatter

from NTcode_config_data.config_run import GEO_CYL as GEO
from solvers.NTdiffusion.diffusion_solver import run_diffusion_solver, _plot_fluxes
from PEDS_subdivision import context
from PEDS_subdivision.context import update_geo
from PEDS_subdivision.physics_solver import NTdiff_solver

_XS_REGION_NAMES = ["B4C rod", "Fuel", "Water"]


def _build_col_map(G):
    col, result = 0, {}
    result["D"] = list(range(col, col + G)); col += G
    result["Sigma_a"] = list(range(col, col + G)); col += G
    result["nuSigma_f"] = list(range(col, col + G)); col += G
    diag, offdiag = [], []
    for g1 in range(G):
        for g2 in range(G):
            (diag if g1 == g2 else offdiag).append(col); col += 1
    result["Sigma_s_diag"] = diag
    result["Sigma_s_offdiag"] = offdiag
    result["chi"] = list(range(col, col + G))
    return result


def _make_grid(xs, cols, n_regions):
    g = np.zeros((n_regions, len(cols)), dtype=np.float64)
    for j, c in enumerate(cols):
        g[:, j] = xs[:, c]
    return g


def _safe_norm(vmin, vmax):
    if vmin > 0 and vmax > 0:
        return LogNorm(vmin=vmin * 0.9, vmax=vmax * 1.1)
    return Normalize(vmin=vmin - abs(vmin)*0.1,
                     vmax=vmax + abs(vmax)*0.1 + 1e-12)


def _col_labels(xs_key, G):
    if xs_key == "Sigma_s_diag":
        return [f"g{g+1}->g{g+1}" for g in range(G)]
    if xs_key == "Sigma_s_offdiag":
        return [f"g{g1+1}->g{g2+1}"
                for g1 in range(G) for g2 in range(G) if g1 != g2]
    return [f"g{g+1}" for g in range(G)]


def _fmt_val(v):
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 10:   return f"{v:.2f}"
    if a >= 1:    return f"{v:.3f}"
    if a >= 0.01: return f"{v:.4f}"
    return f"{v:.2e}"


def plot_xs_subplots(
    baseline,
    final_xs,
    G=2,
    save_path="xs_subplots.png",
    suptitle="XS: Baseline vs Final (NN corrected)",
    epoch_label=None,
    sample_idx=None,
    geo_params=None,
    param_names=None,
    keff_ref=None,
    keff_pred=None,
    weight=None):
    n_reg   = baseline.shape[0]
    col_map = _build_col_map(G)

    PORTRAIT_ORDER = [
        ("D",               "D (diffusion coeff.)",   "Blues"),
        ("Sigma_a",         "Σ_a (absorption)",        "Oranges"),
        ("nuSigma_f",       "νΣ_f (fission)",          "Purples"),
        ("chi",             "χ (fission spectrum)",    "Reds"),
        ("Sigma_s_diag",    "Σ_s g→g (self-scatter)",  "Greens"),
        ("Sigma_s_offdiag", "Σ_s g→g' (cross-scatter)","YlGn"),
    ]

    active = []
    for xs_key, label, cmap_name in PORTRAIT_ORDER:
        bg = _make_grid(baseline, col_map[xs_key], n_reg)
        fg = _make_grid(final_xs, col_map[xs_key], n_reg)
        if np.any(bg != 0) or np.any(fg != 0):
            active.append((xs_key, label, cmap_name, bg, fg))

    n_t   = len(active)
    NCOLS = 2
    n_row = (n_t + NCOLS - 1) // NCOLS

    # ── sizing constants (bumped cell size to fit larger text) ────────────────
    cell_w   = 1.55   # was 1.40
    cell_h   = 0.88   # was 0.78
    cb_w_in  = 0.22
    cb_pad   = 0.12
    ylabel_w = 1.20   # was 1.10 — room for bigger region-name text
    xlabel_h = 0.65   # was 0.55 — room for bigger x-tick text
    title_h  = 0.60   # was 0.55
    sp_hgap  = 0.30
    sp_wgap  = 0.60

    sp_data_w = G * cell_w * 2
    sp_data_h = n_reg * cell_h

    sp_w = ylabel_w + sp_data_w + cb_pad + cb_w_in
    sp_h = title_h  + sp_data_h + xlabel_h

    fig_w = NCOLS * sp_w + (NCOLS - 1) * sp_wgap + 0.15
    fig_h = n_row  * sp_h + (n_row  - 1) * sp_hgap + 0.45

    has_info = any(x is not None for x in [geo_params, keff_ref, keff_pred])
    info_h   = 1.5 if has_info else 0.0
    fig_h   += info_h

    fig = plt.figure(figsize=(fig_w, fig_h))

    # ── suptitle removed (title placed in thesis caption instead) ────────────
    epoch_note  = f"  —  {epoch_label}"          if epoch_label  is not None else ""
    sample_note = f"  —  Sample {sample_idx}"    if sample_idx   is not None else ""

    # ── metadata info box ("legend") ──────────────────────────────────────────
    if has_info:
        parts = []
        if keff_ref is not None and keff_pred is not None:
            delta_rho = abs(keff_pred - keff_ref) / (keff_pred * keff_ref) * 1e5
            parts.append(
                f"k_ref={keff_ref:.5f}   k_pred={keff_pred:.5f}"
                f"   Δρ={delta_rho:.0f} pcm"
            )
            if weight is not None:
                parts.append(f"weight={weight:.2f}")
        elif keff_ref is not None:
            parts.append(f"k_ref={keff_ref:.5f}")

        if geo_params is not None:
            names  = param_names if param_names is not None \
                    else [f"p{j}" for j in range(len(geo_params))]

            UNITS = {
                "b4c_r":      "cm",
                "cr_frac":    "",       # dimensionless fraction
                "fuel_r":     "cm",
                "enrichment": "wt%",
                "f_mod":      "",       # dimensionless fraction
                "water_r":    "cm",
            }

            pairs = "   ".join(
                f"{n}={float(v):.2g}{UNITS.get(n, '')}"
                for n, v in zip(names, geo_params)
            )
            parts.append(pairs)

        info_text = "\n".join(parts)
        fig.text(
            0.5, 1.0 - 0.4 / fig_h,
            info_text,
            ha="center", va="top",
            fontsize=20,               # was 17
            linespacing=2.0,
            color="#222222",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5",
                      edgecolor="#bbbbbb", linewidth=0.8),
        )

    for i, (xs_key, label, cmap_name, bg, fg) in enumerate(active):
        row_i = i // NCOLS
        col_i = i %  NCOLS

        sp_x0_fig = (col_i * (sp_w + sp_wgap) + ylabel_w) / fig_w
        sp_y0_fig = ((n_row - 1 - row_i) * (sp_h + sp_hgap) + xlabel_h + 0.35) / fig_h
        sp_w_fig  = sp_data_w / fig_w
        sp_h_fig  = sp_data_h / fig_h

        ax = fig.add_axes([sp_x0_fig, sp_y0_fig, sp_w_fig, sp_h_fig])

        nx          = bg.shape[1]
        combo       = np.concatenate([bg, fg], axis=1)
        total_xcols = combo.shape[1]

        vals = np.concatenate([bg.flatten(), fg.flatten()])
        pos  = vals[vals > 0]
        pos  = pos if len(pos) > 0 else np.array([1e-10, 1.0])
        norm = _safe_norm(pos.min(), pos.max())

        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("#e4e4e4")

        im = ax.imshow(combo, cmap=cmap, norm=norm,
                       aspect="auto", interpolation="nearest")

        for r in range(n_reg):
            for c in range(total_xcols):
                v = combo[r, c]
                if np.isnan(v):
                    continue
                rgba = cmap(norm(v)) if v > 0 else cmap(0.0)
                lum  = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]
                tc   = "white" if lum < 0.45 else "black"

                is_final_col = c >= nx
                if is_final_col:
                    b_v = bg[r, c - nx]
                    pct = (v - b_v) / b_v * 100 if b_v != 0 else 0.0
                    pct_sign = "+" if pct >= 0 else ""
                    pct_str  = f"{pct_sign}{pct:.1f}%"

                    ax.text(c, r - 0.18, _fmt_val(v),
                            ha="center", va="center",
                            fontsize=16, fontweight="bold", color=tc)   # was 13
                    ax.text(c, r + 0.28, pct_str,
                            ha="center", va="center",
                            fontsize=15, color=tc,                       # was 12
                            style="italic")
                else:
                    ax.text(c, r, _fmt_val(v),
                            ha="center", va="center",
                            fontsize=16, fontweight="bold", color=tc)   # was 13

        gl     = _col_labels(xs_key, G)
        xticks = list(range(nx)) + list(range(nx, 2 * nx))
        xlbls  = [f"B {l}" for l in gl] + [f"F {l}" for l in gl]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlbls, fontsize=16, rotation=35, ha="right")  # was 13

        ax.set_yticks(range(n_reg))
        ax.set_yticklabels(_XS_REGION_NAMES[:n_reg], fontsize=16)             # was 13
        ax.tick_params(axis="y", length=0, pad=4)

        title_y_fig   = sp_y0_fig + sp_h_fig

        mid_b_data    = (nx - 1) / 2
        mid_b_fig     = sp_x0_fig + (mid_b_data + 0.5) / total_xcols * sp_w_fig
        mid_f_data    = nx + (nx - 1) / 2
        mid_f_fig     = sp_x0_fig + (mid_f_data + 0.5) / total_xcols * sp_w_fig

        subtitle_y    = title_y_fig + 0.012 / fig_h
        xs_title_y    = title_y_fig + (title_h * 0.62) / fig_h

        fig.text(mid_b_fig, subtitle_y, "Baseline",
                 ha="center", va="bottom", fontsize=17,                    # was 14
                 color="#1a5fa8", fontweight="bold")
        fig.text(mid_f_fig, subtitle_y, "Final (NN)  [val  Δ%]",
                 ha="center", va="bottom", fontsize=17,                    # was 14
                 color="#8b1a00", fontweight="bold")

        sp_cx_fig = sp_x0_fig + sp_w_fig / 2
        fig.text(sp_cx_fig, xs_title_y, label,
                 ha="center", va="bottom", fontsize=20, fontweight="bold") # was 16

        ax.axvline(x=nx - 0.5, color="white", linewidth=4, zorder=3)

        cb_x0_fig = sp_x0_fig + sp_w_fig + cb_pad / fig_w
        cax = fig.add_axes([cb_x0_fig, sp_y0_fig,
                             cb_w_in / fig_w, sp_h_fig])
        fmt = LogFormatter(labelOnlyBase=False) if isinstance(norm, LogNorm) else "%.3g"
        cb  = fig.colorbar(im, cax=cax, format=fmt)
        cb.ax.tick_params(labelsize=13)   # was 11

    dirn = os.path.dirname(save_path)
    if dirn:
        os.makedirs(dirn, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0.15)
    print(f"[xs_subplots] saved to: {save_path}")
    plt.show()
    plt.close()


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

