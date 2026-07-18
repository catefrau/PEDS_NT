"""Scalar metrics (reactivity error in pcm, MSE/MAE, threshold fractions)."""
import numpy as np


def compute_delta_rho_pcm(k_pred: np.ndarray, k_ref: np.ndarray) -> np.ndarray:
    """Reactivity error |Δρ| in pcm for each sample."""
    return np.abs(k_pred - k_ref) / (k_pred * k_ref) * 1e5


def compute_metrics(k_pred: np.ndarray, k_ref: np.ndarray) -> dict:
    """
    Returns a dict of scalar metrics for one epoch.

    Keys
    ----
    mse_k          : mean squared error in k units²
    MAE_k          : mean absolute error in k units
    mean_pcm       : mean |Δρ|  (pcm)  ← headline physics metric
    median_pcm     : median |Δρ| (pcm) ← robust to outliers
    p95_pcm        : 95th-percentile |Δρ| (pcm) ← worst-case tail
    std_pcm        : std dev of |Δρ|  (pcm) ← spread / consistency
    frac_below_650 : fraction of samples with |Δρ| < 650 pcm (β_eff threshold)
    frac_below_100 : fraction of samples with |Δρ| < 100 pcm (typical regulatory limit)
    """
    dr = compute_delta_rho_pcm(k_pred, k_ref)
    return dict(
        mse_k          = float(np.mean((k_pred - k_ref) ** 2)),
        MAE_k          = float(np.mean(np.abs(k_pred - k_ref))),
        mean_pcm       = float(np.mean(dr)),
        median_pcm     = float(np.median(dr)),
        p95_pcm        = float(np.percentile(dr, 95)),
        std_pcm        = float(np.std(dr)),
        frac_below_650 = float(np.mean(dr < 650.0)),
        frac_below_100 = float(np.mean(dr < 100.0)),
    )


def print_metrics(epoch: int, tag: str, m: dict):
    print(
        f"[Epoch {epoch:4d}] {tag:5s} | "
        f"MSE={m['mse_k']:.6f} | "
        f"MAE_k={m['MAE_k']:7.1f} | "
        f"mean={m['mean_pcm']:7.1f} pcm | "
        f"median={m['median_pcm']:7.1f} pcm | "
        f"p95={m['p95_pcm']:7.1f} pcm | "
        f"std={m['std_pcm']:6.1f} | "
        f"<650pcm={m['frac_below_650']*100:.1f}% | "
        f"<100pcm={m['frac_below_100']*100:.1f}%"
    )


