import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import jax.numpy as jnp
import numpy as np


def visualise_results(
    model,
    test_geoms:   np.ndarray,
    test_keffs:   np.ndarray,
    train_losses: list,
    val_losses:   list,
    val_pct_errs: list,
    sample_idx:   int = 0,
    save_path:    str = "./experiments/coding/figures/MYNT_results.png",
):
    """Four-panel figure:
       (a) pore geometry for one test sample
       (b) generated conductivity field for that sample
       (c) training vs validation MSE loss curve
       (d) validation percentage-error curve
    """
    # ── Run one sample through the model ─────────────────────────────────
    pore_sample = jnp.array(test_geoms[sample_idx:sample_idx + 1])  # [1, 6]
    kappa_true  = float(test_keffs[sample_idx])

    keff_pred_arr, cond_field = model(pore_sample)
    keff_pred = float(keff_pred_arr[0])
    cond_np   = np.array(cond_field[0])
    pore_np   = np.array(test_geoms[sample_idx]).reshape(5, 5)

    pct_err = abs(keff_pred - kappa_true) / abs(kappa_true) * 100

    # ── Layout ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # ── Panel (a): Pore geometry ──────────────────────────────────────────
    ax_pore = fig.add_subplot(gs[0, 0])
    im_p = ax_pore.imshow(pore_np, cmap="gray_r", interpolation="nearest",
                          vmin=0, vmax=1)
    ax_pore.set_title(f"(a) Pore Geometry (sample {sample_idx})", fontsize=12)
    ax_pore.set_xlabel("x");  ax_pore.set_ylabel("y")
    for i in range(pore_np.shape[0]):
        for j in range(pore_np.shape[1]):
            ax_pore.text(j, i, f"{int(pore_np[i,j])}", ha="center", va="center",
                         color="red", fontsize=9)
    fig.colorbar(im_p, ax=ax_pore, fraction=0.046, pad=0.04, label="Pore (1=solid)")

    # ── Panel (b): Generated conductivity field ───────────────────────────
    ax_cond = fig.add_subplot(gs[0, 1])
    im_c = ax_cond.imshow(cond_np, cmap="viridis", interpolation="nearest")
    ax_cond.set_title(
        f"(b) Generated Conductivity Field\n"
        f"κ_pred = {keff_pred:.2f}  |  κ_true = {kappa_true:.2f}  |  err = {pct_err:.1f}%",
        fontsize=11,
    )
    ax_cond.set_xlabel("x");  ax_cond.set_ylabel("y")
    for i in range(cond_np.shape[0]):
        for j in range(cond_np.shape[1]):
            ax_cond.text(j, i, f"{cond_np[i,j]:.1f}", ha="center", va="center",
                         color="white", fontsize=8)
    fig.colorbar(im_c, ax=ax_cond, fraction=0.046, pad=0.04, label="Conductivity")

    # ── Panel (c): MSE loss curves ────────────────────────────────────────
    epochs_range = np.arange(1, len(train_losses) + 1)

    ax_loss = fig.add_subplot(gs[1, 0])
    ax_loss.plot(epochs_range, train_losses, "b-",  lw=1.5, label="Train MSE")
    ax_loss.plot(epochs_range, val_losses,   "r--", lw=1.5, label="Val MSE")
    ax_loss.set_xlabel("Epoch", fontsize=11)
    ax_loss.set_ylabel("Mean Squared Loss", fontsize=11)
    ax_loss.set_title("(c) Training & Validation MSE", fontsize=12)
    ax_loss.set_yscale("log")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    # ── Panel (d): Validation percentage error ────────────────────────────
    ax_pct = fig.add_subplot(gs[1, 1])
    ax_pct.plot(epochs_range, val_pct_errs, "g-", lw=1.5, label="Val % Error")
    ax_pct.axhline(5.0, color="gray", linestyle="--", lw=1, label="5% target")
    ax_pct.set_xlabel("Epoch", fontsize=11)
    ax_pct.set_ylabel("Mean |κ_pred - κ_true| / |κ_true| × 100%", fontsize=10)
    ax_pct.set_title("(d) Validation Percentage Error", fontsize=12)
    ax_pct.legend()
    ax_pct.grid(True, alpha=0.3)

    plt.suptitle("PEDS Run — Single Device, No MPI", fontsize=14, y=1.01)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved to: {save_path}")