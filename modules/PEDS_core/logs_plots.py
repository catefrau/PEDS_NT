import os
import csv
import threading
import shutil
import hashlib

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from solvers.NTdiffusion.diffusion_solver import predict_xs, run_diffusion_solver, _plot_fluxes
from plot_functions.xs_heatmap import plot_xs_subplots
from PEDS_core.fwd_bwd_vjp import update_geo, _run_NT_solver, NTdiff_solver

_csv_lock = threading.Lock()
val_logfile = None
train_logfile = None
val_writer = None
train_writer = None
epoch_stats_file = None
epoch_stats_writer = None
logratio_stats_file = None
logratio_stats_writer = None
val_log_path = None
train_log_path = None
epoch_stats_path = None

LOG_DIR = None
PARAM_NAMES = None
LOG_RATIO_CLIP_LO = None
LOG_RATIO_CLIP_HI = None
GEO = None

def configure_training_helpers(log_dir, param_names, log_ratio_clip_lo, log_ratio_clip_hi, geo):
    global LOG_DIR, PARAM_NAMES, LOG_RATIO_CLIP_LO, LOG_RATIO_CLIP_HI, GEO
    global val_log_path, train_log_path, epoch_stats_path

    LOG_DIR = log_dir
    PARAM_NAMES = param_names
    LOG_RATIO_CLIP_LO = log_ratio_clip_lo
    LOG_RATIO_CLIP_HI = log_ratio_clip_hi
    GEO = geo

    val_log_path = os.path.join(LOG_DIR, "keff_epoch_log_val.csv")
    train_log_path = os.path.join(LOG_DIR, "keff_epoch_log_train.csv")
    epoch_stats_path = os.path.join(LOG_DIR, "epoch_metrics.csv")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: CSV logging
# ─────────────────────────────────────────────────────────────────────────────
def init_csv_logs():
    global val_logfile, train_logfile, val_writer, train_writer
    global epoch_stats_file, epoch_stats_writer
    global logratio_stats_file, logratio_stats_writer

    os.makedirs(LOG_DIR, exist_ok=True)

    for p in [val_log_path, train_log_path, epoch_stats_path,
              os.path.join(LOG_DIR, "logratio_saturation.csv")]:
        if os.path.exists(p):
            os.remove(p)

    val_logfile = open(val_log_path, "w", newline="", buffering=1)
    train_logfile = open(train_log_path, "w", newline="", buffering=1)

    val_writer = csv.writer(val_logfile)
    train_writer = csv.writer(train_logfile)

    header = ["epoch", "sample_idx", "keff_openmc", "keff_peds",
              "delta_rho_pcm", "train_loss", "val_loss"]
    val_writer.writerow(header)
    train_writer.writerow(header)
    val_logfile.flush()
    train_logfile.flush()

    epoch_stats_file = open(epoch_stats_path, "w", newline="", buffering=1)
    epoch_stats_writer = csv.writer(epoch_stats_file)
    epoch_header = [
        "epoch",
        "train_mse_k", "train_mae_k", "train_mean_pcm", "train_median_pcm",
        "train_p95_pcm", "train_std_pcm", "train_frac_below_650", "train_frac_below_100",
        "val_mse_k", "val_mae_k", "val_mean_pcm", "val_median_pcm",
        "val_p95_pcm", "val_std_pcm", "val_frac_below_650", "val_frac_below_100",
    ]
    epoch_stats_writer.writerow(epoch_header)
    epoch_stats_file.flush()

    logratio_stats_path = os.path.join(LOG_DIR, "logratio_saturation.csv")
    logratio_stats_file = open(logratio_stats_path, "w", newline="", buffering=1)
    logratio_stats_writer = csv.writer(logratio_stats_file)
    lr_header = ["epoch", "region", "xs_idx",
                 "min", "max", "mean", "std",
                 "frac_at_lower_clip", "frac_at_upper_clip"]
    logratio_stats_writer.writerow(lr_header)
    logratio_stats_file.flush()

def close_csv_logs():
    for fh in [val_logfile, train_logfile, epoch_stats_file, logratio_stats_file]:
        if fh is not None:
            fh.close()

def log_epoch_stats(epoch: int, train_m: dict, val_m: dict):
    """Write one row of aggregate stats per epoch to epoch_metrics.csv."""
    row = [epoch] + [
        train_m["mse_k"], train_m["MAE_k"], train_m["mean_pcm"], train_m["median_pcm"],
        train_m["p95_pcm"], train_m["std_pcm"], train_m["frac_below_650"], train_m["frac_below_100"],
        val_m["mse_k"],   val_m["MAE_k"],   val_m["mean_pcm"],   val_m["median_pcm"],
        val_m["p95_pcm"],   val_m["std_pcm"],   val_m["frac_below_650"],   val_m["frac_below_100"],
    ]
    with _csv_lock:
        epoch_stats_writer.writerow(row)
        epoch_stats_file.flush()


def log_keff_batch(split, epoch, k_pred, k_ref,
                   avg_train_loss, avg_val_loss, sample_id_offset=0):
    if split == "train":
        writer, filehandle = train_writer, train_logfile
    elif split == "val":
        writer, filehandle = val_writer, val_logfile
    else:
        raise ValueError(f"Unknown split: {split}")

    with _csv_lock:
        for sidx, (kr, kp) in enumerate(zip(k_ref, k_pred)):
            dr = abs(kp - kr) / (kp * kr) * 1e5
            writer.writerow([
                epoch, sidx + sample_id_offset,
                f"{kr:.6f}", f"{kp:.6f}", f"{dr:.1f}",
                avg_train_loss, avg_val_loss
            ])
        filehandle.flush()

_CLIP_EPS = 1e-4  # tolerance for "at the clip boundary"

region_names_lr = ["CR", "Core", "Moderator"]

def log_logratio_saturation(epoch: int, log_ratios_all: np.ndarray):
    """
    log_ratios_all: [N_samples, n_regions, xs_per_region]  (post-clip values)
    Writes one row per (region, xs_idx) summarizing saturation across all samples.
    """
    n_regions, xs_per_region = log_ratios_all.shape[1], log_ratios_all.shape[2]
    with _csv_lock:
        for r in range(n_regions):
            for x in range(xs_per_region):
                vals = log_ratios_all[:, r, x]
                frac_lo = float(np.mean(vals <= LOG_RATIO_CLIP_LO + _CLIP_EPS))
                frac_hi = float(np.mean(vals >= LOG_RATIO_CLIP_HI - _CLIP_EPS))
                logratio_stats_writer.writerow([
                    epoch, region_names_lr[r] if r < 3 else f"R{r}", x,
                    f"{vals.min():.5f}", f"{vals.max():.5f}",
                    f"{vals.mean():.5f}", f"{vals.std():.5f}",
                    f"{frac_lo:.3f}", f"{frac_hi:.3f}",
                ])
        logratio_stats_file.flush()

def save_final_xs_csv(model, geoms, raw_params, xs_baselines, k_reg_norm, phi_norm,
                      keffs_ref, log_xs_mean, log_xs_std, file_path, tag="train"):
    region_names = ["CR", "Core", "Mod"]
    xs_labels    = ["D1","D2","Sa1","Sa2","nSf1","nSf2","Ss11","Ss22","Ss12","Ss21","chi1","chi2"]
    
    geo_header = PARAM_NAMES
    xs_header  = [f"{reg}_{xs}" for reg in region_names for xs in xs_labels]
    # ↓ added keff_pred and delta_rho_pcm to the header
    header = ["sample_idx"] + geo_header + ["keff_ref", "keff_pred", "delta_rho_pcm"] + xs_header

    rows = []
    N = geoms.shape[0]
    batch_size = 25

    for start in range(0, N, batch_size):
        sl = slice(start, start + batch_size)
        xsf = model.compute_xs(
            jnp.array(geoms[sl],         dtype=jnp.float32),
            jnp.array(xs_baselines[sl],  dtype=jnp.float32),
            jnp.array(k_reg_norm[sl],    dtype=jnp.float32),
            jnp.array(phi_norm[sl],      dtype=jnp.float32),
            log_xs_mean, log_xs_std,
        )  # (batch, 3, 12)

        for i, global_idx in enumerate(range(start, min(start + batch_size, N))):
            # ── run solver to get keff_pred ──────────────────────────────────
            keff_pred = None
            try:
                k, _, _, _ = _run_NT_solver(
                    xsf[i],                                     # (3,12) final XS
                    raw_params[global_idx],                     # (6,)   geometry
                    np.array([global_idx], dtype=np.int32),     # sample id
                )
                keff_pred = float(k)
            except Exception as e:
                print(f"  [save_final_xs_csv] solver failed for {tag} sample {global_idx}: {e}")

            # ── compute delta_rho_pcm ────────────────────────────────────────
            keff_ref_val = float(keffs_ref[global_idx])
            if keff_pred is not None:
                dr = abs(keff_pred - keff_ref_val) / (keff_pred * keff_ref_val) * 1e5
            else:
                dr = float("nan")   # solver failed → mark as NaN

            row = (
                [global_idx]
                + raw_params[global_idx].tolist()   # 6 geometry params
                + [keff_ref_val, keff_pred, round(dr, 2)]
                + xsf[i].flatten().tolist()          # 36 XS values
            )
            rows.append(row)

        if start % 100 == 0:
            print(f"  [{tag}] saving XS CSV: {start}/{N}", flush=True)

    with open(file_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Saved final XS ({tag}): {file_path}  [{N} samples]")


def save_code_snapshot(logdir, expname, src_path, log_handle=None):
    src = os.path.abspath(src_path)
    snapshot_path = os.path.join(logdir, f"code_snapshot_{expname}.py")

    # 1) Save a real copy of the script
    shutil.copy2(src, snapshot_path)

    # 2) Read code text
    with open(src, "r", encoding="utf-8") as f:
        code_text = f.read()

    # 3) Optional hash, useful to identify exact version
    code_hash = hashlib.sha256(code_text.encode("utf-8")).hexdigest()

    # 4) Optional: also append the full code into the txt log
    if log_handle is not None:
        log_handle.write("\n" + "=" * 100 + "\n")
        log_handle.write("CODE SNAPSHOT\n")
        log_handle.write(f"Source file : {src}\n")
        log_handle.write(f"Saved copy  : {snapshot_path}\n")
        log_handle.write(f"SHA256      : {code_hash}\n")
        log_handle.write("=" * 100 + "\n")
        log_handle.write(code_text)
        log_handle.write("\n" + "=" * 100 + "\n")
        log_handle.flush()

    return snapshot_path, code_hash
# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

# ── XS name labels — adjust to match your actual xs_layout ordering ──────────
# These are used as axis tick labels on the heatmap.
# G=2 → groups 1,2; typical order: D1 D2 Sa1 Sa2 nF1 nF2 Ss12 Ss21 chi1 chi2 ...
# (12 entries per region for G=2). Modify if your SLAY ordering differs.
_XS_LABELS = [
    "D₁","D₂",
    "Σa₁","Σa₂",
    "νΣf₁","νΣf₂",
    "Σs₁₂","Σs₂₁",
    "χ₁","χ₂",
    "XS₁₁","XS₁₂",   # placeholder names for remaining slots
]


def _save_xs_subplots_for_samples(
    model,
    sample_indices,
    geoms_all,
    keffs_all,
    rawparams_all,
    xs_baselines_all,
    k_reg_norm_all,
    phi_norm_all,
    log_xs_mean_j,
    log_xs_std_j,
    epoch: int,
):
    """
    For each index in sample_indices, run a lightweight NN forward pass and
    save one plot_xs_subplots figure per sample.

    Files land in LOG_DIR/xs_subplots/epoch_{E:04d}_sample{IDX:03d}.png

    Arguments mirror _plot_xs_heatmap so they can share the same call-site data.
    k_reg_norm_all: 1-D float32 array [N] — normalised Δk for each sample.
    """
    subplots_dir = os.path.join(LOG_DIR, "xs_subplots")
    os.makedirs(subplots_dir, exist_ok=True)
    epoch_label = f"Epoch {epoch}"

    for idx in sample_indices:
        idx = int(idx)
        if idx >= len(geoms_all):
            print(f"  [subplot] sample_idx={idx} out of range, skipping")
            continue

        # ── per-sample baseline ───────────────────────────────────────────────
        geo_i    = update_geo(GEO, rawparams_all[idx])
        baseline = np.array(predict_xs(geo_i), dtype=np.float32)   # (3, 12)

        xs_b  = jnp.array(xs_baselines_all[idx:idx+1], dtype=jnp.float32)   # (1,3,12)
        geom  = jnp.array(geoms_all[idx:idx+1],        dtype=jnp.float32)   # (1,6)
        dk    = jnp.array([[float(k_reg_norm_all[idx])]], dtype=jnp.float32)  # (1,1)

        # ── NN forward (no gradient needed) ──────────────────────────────────
        log_base   = model._log_baselines(xs_b)   
        phi = jnp.array(phi_norm_all[idx:idx+1], dtype=jnp.float32)   # (1, nphifeats)                   # (1,3,12)
        log_ratios = np.array(
            model.generator(geom, log_base, dk, phi, training=False)[0]  # (3,12)
        )
        final_xs = np.exp(log_ratios) * baseline                     # (3,12)

        final_xs = model.compute_xs(
            jnp.array(geoms_all[idx:idx+1], dtype=jnp.float32),
            jnp.array(xs_baselines_all[idx:idx+1], dtype=jnp.float32),
            jnp.array(k_reg_norm_all[idx:idx+1], dtype=jnp.float32),
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
            param_names = PARAM_NAMES,
            keff_ref    = float(keffs_all[idx]),
            keff_pred   = keff_pred_val,
        )
        print(f"  [subplot] epoch {epoch}  sample {idx}  →  {save_path}")


def _plot_xs_heatmap(model, geoms_batch, xs_baselines_batch, k_reg_norm_batch,
                     phi_norm_batch, log_xs_mean_j, log_xs_std_j, epoch: int, n_show: int = 8):
    """
    Save a heatmap of the NN-corrected XS for a small representative batch.

    Layout: rows = samples (up to n_show), columns = XS types.
            One sub-figure per region, side by side.
    Shows the ratio  xs_final / xs_baseline  so 1.0 = no correction.
    Saved to LOG_DIR/xs_heatmap/epoch_{epoch:04d}.png
    """
    heatmap_dir = os.path.join(LOG_DIR, "xs_heatmap")
    os.makedirs(heatmap_dir, exist_ok=True)

    # ── get corrected XS (numpy, no grad) ────────────────────────────────────
    n    = min(n_show, geoms_batch.shape[0])
    xs_b = jnp.array(xs_baselines_batch[:n], dtype=jnp.float32)
    xs_f = model.compute_xs(
        jnp.array(geoms_batch[:n], dtype=jnp.float32),
        xs_b,
        jnp.array(k_reg_norm_batch[:n], dtype=jnp.float32),
        jnp.array(phi_norm_batch[:n], dtype=jnp.float32),
        log_xs_mean_j, log_xs_std_j,
    )  # [n, n_regions, xs_per_region]

    # ratio relative to baseline; clip extreme values for display
    xs_base_np = np.array(xs_b)
    safe_base  = np.where(xs_base_np > 1e-10, xs_base_np, np.ones_like(xs_base_np))
    ratio      = xs_f / safe_base                   # [n, n_regions, xs_per_region]
    ratio      = np.clip(ratio, 0.5, 2.0)           # display range

    n_regions    = ratio.shape[1]
    xs_per_region = ratio.shape[2]
    labels = _XS_LABELS[:xs_per_region]

    region_names = ["CR", "Core", "Moderator"]

    fig, axes = plt.subplots(1, n_regions, figsize=(5 * n_regions, 0.5 * n + 1.5),
                             squeeze=False)
    fig.suptitle(f"XS correction ratio  (epoch {epoch})\n"
                 f"colour = xs_final / xs_baseline   [clipped 0.5–2.0]", fontsize=10)

    for r in range(n_regions):
        ax  = axes[0, r]
        mat = ratio[:, r, :]            # [n, xs_per_region]
        im  = ax.imshow(mat, aspect="auto", vmin=0.5, vmax=2.0, cmap="RdBu_r")
        ax.set_xticks(range(xs_per_region))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(n))
        ax.set_yticklabels([f"s{i}" for i in range(n)], fontsize=7)
        ax.set_title(region_names[r] if r < len(region_names) else f"Region {r}", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    path = os.path.join(heatmap_dir, f"epoch_{epoch:04d}.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"XS heatmap saved → {path}")


def _plot_history(history: dict, exp_name: str):
    """5-panel summary plot saved to LOG_DIR/training_history.png

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
    path = os.path.join(LOG_DIR, "training_history.png")
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
    path2 = os.path.join(LOG_DIR, "val_reactivity_logscale.png")
    plt.savefig(path2, dpi=150)
    plt.close()
    print(f"Reactivity log-scale plot saved → {path2}")

def _save_flux_plots(
    sample_indices: list,
    test_geoms: np.ndarray,
    test_keffs: np.ndarray,
    test_rawparams: np.ndarray,
    test_xs_baselines: np.ndarray,
    test_k_reg_norm: np.ndarray,
    test_phi_norm: np.ndarray,
    model,
    epoch: int,
):
    """
    For each sample index, run a fresh forward solve with the current NN
    XS corrections and plot the flux shape using _plot_fluxes.

    Files land in LOG_DIR/flux_plots/epoch_{E:04d}_sample{IDX:03d}.png

    This uses unshuffled arrays so sample identity is stable across epochs.
    """
    flux_dir = os.path.join(LOG_DIR, "flux_plots")
    os.makedirs(flux_dir, exist_ok=True)

    for idx in sample_indices:
        idx = int(idx)
        if idx >= len(test_geoms):
            print(f"  [flux_plot] sample_idx={idx} out of range, skipping")
            continue

        # ── Geometry for this sample ──────────────────────────────────────────
        geo_i    = update_geo(GEO, test_rawparams[idx])
        R        = geo_i.boundaries[-1].radius
        I        = int(R / geo_i.mesh_size)
        Delta_r  = geo_i.mesh_size

        # ── Get NN-corrected XS (no gradient needed) ──────────────────────────
        xs_b  = jnp.array(test_xs_baselines[idx:idx+1], dtype=jnp.float32)  # (1,3,12)
        geom  = jnp.array(test_geoms[idx:idx+1],        dtype=jnp.float32)  # (1,6)
        dk    = jnp.array([[float(test_k_reg_norm[idx])]], dtype=jnp.float32)  # (1,1)
        phi   = jnp.array((test_phi_norm[idx:idx+1]), dtype=jnp.float32) 

        log_base   = model._log_baselines(xs_b)
        log_ratios = model.generator(geom, log_base, dk, phi, training=False)  # (1,3,12)
        xs_corrected = np.array(jnp.exp(log_ratios[0]) * xs_b[0])        # (3,12)

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
        xs_base_np = np.array(test_xs_baselines[idx], dtype=np.float32)  # (3,12)
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

        k_ref = float(test_keffs[idx])
        dr    = abs(k_pred - k_ref) / (k_pred * k_ref) * 1e5
        print(
            f"  [flux_plot] epoch {epoch}  sample {idx}  "
            f"k_ref={k_ref:.5f}  k_pred={k_pred:.5f}  |Δρ|={dr:.1f} pcm  → {save_path}"
        )