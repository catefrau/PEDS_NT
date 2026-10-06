import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(THIS_DIR)

sys.path.insert(0, PARENT_DIR)

from NTcode_config_data.config_def import GeometryConfig, MaterialSpec, BoundarySpec, BoundaryCondition, MatProperties
from solvers.NTdiffusion.diffusion_solver import run_diffusion_solver, precompute_geometry, predict_xs, build_xs_callables, fn_xs_per_region, xs_layout
from solvers.NTdiffusion.core.MG1D_eigenvalue_nregions import DiffusionEigenvalue_MG
from NTcode_config_data.config_run import GEO_CYL  # to reuse G, geometry, bc, mesh_size



def geometry_from_row(row) -> GeometryConfig:
    """
    Build a GeometryConfig for the CR/fuel/water cylindrical case
    from one row of LHS_full_dataset.
    Assumes columns:
      - geometry
      - r0_b4c_rod_outer_radius
      - r1_fuel_annulus_outer_radius
      - r2_water_outer_radius
      - r1_fuel_annulus_enrichment
      - r1_fuel_annulus_f_mod
      - r0_b4c_rod_cr_fraction
    """
    # Use same solver settings as GEO_CYL
    G = GEO_CYL.G
    geometry = row["geometry"]  # should be "cylindrical"
    mesh_size = GEO_CYL.mesh_size
    bc = GEO_CYL.bc

    # Boundaries: centre → outside
    boundaries = (
        BoundarySpec(name="CR_outer",
                     radius=float(row["r0_b4c_rod_outer_radius"])),
        BoundarySpec(name="core_outer",
                     radius=float(row["r1_fuel_annulus_outer_radius"])),
        BoundarySpec(name="moderator_outer",
                     radius=float(row["r2_water_outer_radius"])),
    )

    # Regions: centre → outside
    regions = (
        MaterialSpec("b4c_rod",      region_index=0),
        MaterialSpec("fuel_annulus", region_index=1),
        MaterialSpec("water",        region_index=2),
    )

    # Global material knobs used by regression
    mat_props = MatProperties(
        enrichment=float(row["r1_fuel_annulus_enrichment"]),
        f_mod=float(row["r1_fuel_annulus_f_mod"]),
        cr_fraction=float(row["r0_b4c_rod_cr_fraction"]),
        # plutonium_fraction left as None
    )

    return GeometryConfig(
        G=G,
        regions=regions,
        boundaries=boundaries,
        geometry=geometry,
        mat_properties=mat_props,
        bc=bc,
        mesh_size=mesh_size,
    )


def xs_tensor_from_row(row, geo: GeometryConfig) -> np.ndarray:
    """
    Build xs_tensor directly from CSV XS columns for this geometry.
    """
    G = geo.G
    n = fn_xs_per_region(G)
    lay = xs_layout(G)
    xs_tensor = np.zeros((len(geo.regions), n))

    for mat in geo.regions:
        m = mat.region_ID  # e.g. 'b4c_rod'
        vec = np.zeros(n)

        # D
        vec[lay["D"]] = [
            row[f"{m}_diffusion-coefficient_g{g+1}"] for g in range(G)
        ]

        # Sigma_a
        vec[lay["Sigma_a"]] = [
            row[f"{m}_absorption_g{g+1}"] for g in range(G)
        ]

        # nuSigma_f
        vec[lay["nuSigma_f"]] = [
            row[f"{m}_nu-fission_g{g+1}"] for g in range(G)
        ]

        # chi — if present, else enforce convention: chi=1 in fastest fission group
        chi_vals = []
        for g in range(G):
            col = f"{m}_chi_g{g+1}"
            if col in row.index:
                chi_vals.append(row[col])
            else:
                # fallback physics: if any fission, chi_g1=1, others=0
                pass
        if chi_vals:
            vec[lay["chi"]] = chi_vals
        else:
            nu_vals = [row[f"{m}_nu-fission_g{g+1}"] for g in range(G)]
            if any(v > 0.0 for v in nu_vals):
                vec[lay["chi"]] = [1.0] + [0.0]*(G-1)
            else:
                vec[lay["chi"]] = [0.0]*G

        # Scatter matrix, stored as flat S[g_from → g_to]
        scatter_flat = [
            row[f"{m}_scatter matrix_g{k+1}"] for k in range(G**2)
        ]
        vec[lay["Sigma_s"]] = scatter_flat

        xs_tensor[mat.region_index] = vec

    return xs_tensor

KEFF_LABEL = r"$k_{\mathrm{eff}}$"


def scatter_plot_from_csv(df, plot_output="LOGS/zed/keff_scatter.png"):
    label_fs, tick_fs, legend_fs = 16, 13, 12
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(df["keff"], df["keff_lf"], s=8, alpha=0.4)
    ax.plot([df["keff"].min(), df["keff"].max()],
            [df["keff"].min(), df["keff"].max()],
            "k--", label="y = x")

    ax.set_xlabel(f"{KEFF_LABEL} (OpenMC, high fidelity)", fontsize=label_fs)
    ax.set_ylabel(f"{KEFF_LABEL} (diffusion, low fidelity)", fontsize=label_fs)
    ax.tick_params(axis="both", labelsize=tick_fs)
    ax.legend(fontsize=legend_fs)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(plot_output) or ".", exist_ok=True)
    fig.savefig(plot_output, dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_histogram(df, plot_output="LOGS/zed/keff_histogram.png",
                   keff_corrected_col=None):
    """
    Overlay OpenMC (HF), diffusion (LF), and optionally diffusion-corrected
    (PEDS) k_eff histograms. Pass ``keff_corrected_col`` to include the third.
    """
    label_fs, tick_fs, legend_fs = 18, 15, 14
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.hist(df["keff"], bins=40, alpha=0.55, label="OpenMC (HF)",
            color="#4C78A8", edgecolor="white", linewidth=0.4)
    ax.hist(df["keff_lf"], bins=40, alpha=0.55, label="Diffusion (LF)",
            color="#F58518", edgecolor="white", linewidth=0.4)
    if keff_corrected_col is not None and keff_corrected_col in df.columns:
        ax.hist(df[keff_corrected_col], bins=40, alpha=0.55,
                label="Diffusion corrected (PEDS)",
                color="#54A24B", edgecolor="white", linewidth=0.4)

    ax.set_xlabel(KEFF_LABEL, fontsize=label_fs)
    ax.set_ylabel("Count", fontsize=label_fs)
    ax.tick_params(axis="both", labelsize=tick_fs)
    ax.legend(fontsize=legend_fs, framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(plot_output) or ".", exist_ok=True)
    fig.savefig(plot_output, dpi=400, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    df = pd.read_csv("./FILES/1000_clean.csv")

    row = df.iloc[0]
    geo = geometry_from_row(row)

    keff_lf = []
    for idx, row in df.iterrows():
        geo = geometry_from_row(row)
        xs_tensor = xs_tensor_from_row(row, geo)
        k_lf, phi_fwd_norm, phi_adj_norm = run_diffusion_solver(xs_tensor, geo, plot_output="LOGS/geometry_plots/fluxes.png")
        keff_lf.append(k_lf)

    df["keff_lf"] = keff_lf
    xs_patterns = [
        "diffusion-coefficient",
        "absorption",
        "nu-fission",
        "scatter matrix",
        "chi_",
        "keff_std",
    ]
    xs_cols = [
        c for c in df.columns
        if any(pat in c for pat in xs_patterns)
    ]
    df_small = df.drop(columns=xs_cols)
    df_small.to_csv("./FILES/clean_with_diffusion_keff.csv", index=False)
    
    kh_min, kh_max = df_small["keff"].min(), df_small["keff"].max()
    kl_min, kl_max = df_small["keff_lf"].min(), df_small["keff_lf"].max()

    print("HF keff range :", kh_min, "→", kh_max)
    print("LF keff range :", kl_min, "→", kl_max)

    scatter_plot_from_csv(df_small)
    plot_histogram(df_small)




