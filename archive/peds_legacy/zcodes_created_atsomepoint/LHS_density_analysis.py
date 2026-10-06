"""
density_analysis.py
--------------------
Analyzes the sample density of an existing (possibly non-uniform / cut) LHS
design, per parameter, relative to what a perfectly uniform LHS of the same
size would have produced.

This answers: "which ranges did I actually sample the most/least, given the
cuts I already applied?" It says nothing about the NN or keff -- that is a
separate, physics/model-driven question (see sensitivity_binning.py).

"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---- specs (edit these) ----
NPZ_PATH = "../data/highfidelity/MC_1315.npz"
N_BINS = 20
OUTDIR = "density_report/"

def load_dataset(npz_path):
    d = np.load(npz_path)
    X_raw = d["params_raw"]          # physical units, shape (N, P)
    names = [str(n) for n in d["param_names"]]
    bounds_lo = d["bounds_lo"]
    bounds_hi = d["bounds_hi"]
    keffs = d["keffs"] if "keffs" in d.files else None
    return X_raw, names, bounds_lo, bounds_hi, keffs


def density_table(X_raw, names, bounds_lo, bounds_hi, n_bins=10):
    """
    For each parameter, split [bounds_lo, bounds_hi] into n_bins equal-width
    bins and compare observed counts to the count expected under a uniform
    design (N / n_bins). Returns a dict per parameter with bin edges, counts,
    and a density ratio (observed / expected uniform).
    """
    N = X_raw.shape[0]
    expected_per_bin = N / n_bins
    results = {}
    for i, name in enumerate(names):
        edges = np.linspace(bounds_lo[i], bounds_hi[i], n_bins + 1)
        counts, _ = np.histogram(X_raw[:, i], bins=edges)
        ratio = counts / expected_per_bin
        results[name] = {
            "edges": edges.tolist(),
            "counts": counts.tolist(),
            "expected_uniform": expected_per_bin,
            "density_ratio": ratio.tolist(),
        }
    return results


def plot_density(results, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for name, r in results.items():
        edges = np.array(r["edges"])
        counts = np.array(r["counts"])
        centers = 0.5 * (edges[:-1] + edges[1:])
        widths = np.diff(edges)

        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(centers, counts, width=widths * 0.9, color="#4C72B0",
               edgecolor="white", align="center", label="observed")
        ax.axhline(r["expected_uniform"], color="#C44E52", ls="--",
                    label="expected (uniform LHS)")
        ax.set_title(name)
        ax.set_xlabel("parameter value")
        ax.set_ylabel("sample count")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(outdir / f"density_{name}.png", dpi=130)
        plt.close(fig)


def print_report(results):
    for name, r in results.items():
        print(f"\n=== {name} ===")
        edges = r["edges"]
        for j in range(len(edges) - 1):
            lo, hi = edges[j], edges[j + 1]
            cnt = r["counts"][j]
            ratio = r["density_ratio"][j]
            flag = ""
            if ratio < 0.5:
                flag = "  <-- strongly UNDER-sampled vs uniform"
            elif ratio > 1.5:
                flag = "  <-- strongly OVER-sampled vs uniform"
            print(f"  [{lo:9.4f}, {hi:9.4f})  n={cnt:4d}  ratio={ratio:5.2f}{flag}")




def main():
    X_raw, names, lo, hi, keffs = load_dataset(NPZ_PATH)
    results = density_table(X_raw, names, lo, hi, n_bins=N_BINS)
    print_report(results)
    plot_density(results, OUTDIR)

    with open(Path(OUTDIR) / "density_table.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved plots + density_table.json to {OUTDIR}/")


if __name__ == "__main__":
    main()
