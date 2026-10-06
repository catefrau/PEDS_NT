"""CSV logging: per-epoch keff logs, epoch metrics, log-ratio saturation,
tracked-sample XS history, split log, and final-XS CSV export.

File handles and log-file paths live at module scope here; paths are derived
from the run's LOG_DIR / XS_DIR (held in PEDS_subdivision.context).
"""
import os
import csv
import threading

import numpy as np
import jax.numpy as jnp

from PEDS_subdivision import context
from PEDS_subdivision.physics_solver import _run_NT_solver

_csv_lock = threading.Lock()

# CSV file handles + writers (populated by init_csv_logs)
val_logfile = None
train_logfile = None
val_writer = None
train_writer = None
epoch_stats_file = None
epoch_stats_writer = None
logratio_stats_file = None
logratio_stats_writer = None
xs_history_file = None
xs_history_writer = None
xs_proposal_stats_file = None
xs_proposal_stats_writer = None
split_logfile = None
split_writer = None

# ── log-file paths (derived from the run's LOG_DIR / XS_DIR) ──────────────────
val_log_path           = os.path.join(context.LOG_DIR, "keff_epoch_log_val.csv")
train_log_path         = os.path.join(context.LOG_DIR, "keff_epoch_log_train.csv")
epoch_stats_path       = os.path.join(context.LOG_DIR, "epoch_metrics.csv")
xs_proposal_stats_path = os.path.join(context.LOG_DIR, "xs_proposal_stats.csv")
xs_history_path        = os.path.join(context.XS_DIR,  "history_first5.csv")
logratio_stats_path    = os.path.join(context.XS_DIR,  "logratio_saturation.csv")
split_log_path         = os.path.join(context.LOG_DIR, "split_log.csv")


def init_csv_logs():
    global val_logfile, train_logfile, val_writer, train_writer
    global epoch_stats_file, epoch_stats_writer
    global logratio_stats_file, logratio_stats_writer
    global xs_history_file, xs_history_writer
    global split_logfile, split_writer

    os.makedirs(context.LOG_DIR, exist_ok=True)

    for p in [val_log_path, train_log_path, epoch_stats_path, xs_history_path, split_log_path,
              logratio_stats_path]:
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

    logratio_stats_file = open(logratio_stats_path, "w", newline="", buffering=1)
    logratio_stats_writer = csv.writer(logratio_stats_file)
    lr_header = ["epoch", "region", "xs_idx",
                 "min", "max", "mean", "std",
                 "frac_at_lower_clip", "frac_at_upper_clip"]
    logratio_stats_writer.writerow(lr_header)
    logratio_stats_file.flush()

    xs_history_file = open(xs_history_path, "w", newline="", buffering=1)
    xs_history_writer = csv.writer(xs_history_file)

    region_names = ["CR", "Core", "Mod"]
    xs_labels = ["D1","D2","Sa1","Sa2","nSf1","nSf2","Ss11","Ss22","Ss12","Ss21","chi1","chi2"]
    xs_header = [f"{reg}_{xs}" for reg in region_names for xs in xs_labels]

    xs_history_writer.writerow(["epoch", "sample_idx", *context.PARAM_NAMES, *xs_header])
    xs_history_file.flush()

    split_logfile = open(split_log_path, "w", newline="", buffering=1)
    split_writer = csv.writer(split_logfile)
    split_writer.writerow(["sample_idx", "split", "keff"])
    split_logfile.flush()

def close_csv_logs():
    for fh in [val_logfile, train_logfile, epoch_stats_file, logratio_stats_file, split_logfile]:
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


def log_keff_batch(writer, filehandle, epoch, k_pred, k_ref,
                   avg_train_loss, avg_val_loss, sample_id_offset=0, sample_ids_override=None):
    with _csv_lock:
        for sidx, (kr, kp) in enumerate(zip(k_ref, k_pred)):
            dr = abs(kp - kr) / (kp * kr) * 1e5
            sample_idx = int(sample_ids_override[sidx]) if sample_ids_override is not None else sidx + sample_id_offset
            writer.writerow([epoch, sample_idx,
                              f"{kr:.6f}", f"{kp:.6f}", f"{dr:.1f}",
                              avg_train_loss, avg_val_loss])
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
                frac_lo = float(np.mean(vals <= context.LOG_RATIO_CLIP_LO + _CLIP_EPS))
                frac_hi = float(np.mean(vals >= context.LOG_RATIO_CLIP_HI - _CLIP_EPS))
                logratio_stats_writer.writerow([
                    epoch, region_names_lr[r] if r < 3 else f"R{r}", x,
                    f"{vals.min():.5f}", f"{vals.max():.5f}",
                    f"{vals.mean():.5f}", f"{vals.std():.5f}",
                    f"{frac_lo:.3f}", f"{frac_hi:.3f}",
                ])
        logratio_stats_file.flush()

def log_xs_history_samples(model, epoch, sample_indices,
                           geom_sample_all, rawparams_sample_all, xs_baselines_sample_all, phi_norm_sample_all,
                           log_xs_mean_j, log_xs_std_j):
    global xs_history_file, xs_history_writer

    if len(sample_indices) == 0:
        return

    idx = np.array(sample_indices, dtype=int)

    xsf = model.compute_xs(
        jnp.array(geom_sample_all[idx], dtype=jnp.float32),
        jnp.array(xs_baselines_sample_all[idx], dtype=jnp.float32),
        jnp.array(phi_norm_sample_all[idx], dtype=jnp.float32),
        log_xs_mean_j, log_xs_std_j
    )  # shape (nsamples, 3, 12)

    with _csv_lock:
        for j, s in enumerate(idx):
            row = [epoch, int(s), *rawparams_sample_all[j].tolist(), *xsf[j].reshape(-1).tolist()]
            xs_history_writer.writerow(row)
        xs_history_file.flush()


def log_splits(train_idx, train_keffs, valid_idx, valid_keffs, test_idx, test_keffs):
    global split_logfile, split_writer
    with _csv_lock:
        for label, idx_arr, k_arr in [
            ("train", train_idx, train_keffs),
            ("val",   valid_idx, valid_keffs),
            ("test",  test_idx,  test_keffs),
        ]:
            if idx_arr is None or k_arr is None:
                continue
            idx_arr = np.asarray(idx_arr)
            k_arr   = np.asarray(k_arr)
            order = np.argsort(idx_arr)          # ascending sample_idx within this split
            for i in order:
                split_writer.writerow([int(idx_arr[i]), label, float(k_arr[i])])
        split_logfile.flush()

def save_final_xs_csv(model, geoms, raw_params, xs_baselines, phi_norm,
                      keffs_ref, log_xs_mean, log_xs_std, file_path, tag="train"):
    region_names = ["CR", "Core", "Mod"]
    xs_labels    = ["D1","D2","Sa1","Sa2","nSf1","nSf2","Ss11","Ss22","Ss12","Ss21","chi1","chi2"]
    
    geo_header = context.PARAM_NAMES
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
