
import openmc
from dataclasses import dataclass
try:
    from .config_def import RegionSpec, EnergyGroupSpec, SolverSettings, MCConfig, ParamSpec, SweepConfig
except ImportError:
    from config_def import RegionSpec, EnergyGroupSpec, SolverSettings, MCConfig, ParamSpec, SweepConfig

# ══════════════════════════════════════════════════════════════════════════════
#  P VECTORS 
# ══════════════════════════════════════════════════════════════════════════════

b4c_rod_outer_radius = ParamSpec(nom=8.0,  bounds=(4.0,  12.0))
b4c_rod_cr_fraction  = ParamSpec(nom=1.0,  bounds=(0.0,   1.0))
fuel_outer_radius    = ParamSpec(nom=30.0, bounds=(20.0, 45.0))
fuel_enrichment      = ParamSpec(nom=5.0,  bounds=(2.,  10.0))
fuel_f_mod           = ParamSpec(nom=0.5,  bounds=(0.3,   0.6))
water_outer_radius   = ParamSpec(nom=50.0, bounds=(50.0, 80.0))


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATIONS 
# ══════════════════════════════════════════════════════════════════════════════

# ── 3-region cylinder: B4C control rod → fuel annulus → water moderator ────
CFG_ROD = MCConfig(
    geometry = 'cylindrical',
    regions  = (
        RegionSpec(name='b4c_rod',      outer_radius= b4c_rod_outer_radius.nom,
                   material='b4c',      cr_fraction=b4c_rod_cr_fraction.nom),
        RegionSpec(name='fuel_annulus', outer_radius= fuel_outer_radius.nom,
                   material='mix',      enrichment=fuel_enrichment.nom,   f_mod =fuel_f_mod.nom),
        RegionSpec(name='water',        outer_radius= water_outer_radius.nom,
                   material='water'),
    ),
    energy_groups = EnergyGroupSpec(
        G          = 2,
        boundaries = (0., 0.625, 20.0e6),   # eV: thermal | fast
    ),
    settings = SolverSettings(
        batches   = 200,          # starting batch count; OpenMC may extend in-run
        inactive  = 50,
        particles = 20_000,
        generations_per_batch=1,
        statepoint_mid_fraction=0.75,
        entropy_mesh_dim=(20, 20, 1),
        convergence_output_dir='MC_sweep/convergence',
        copy_statepoint=False,
        axial_half_height = 50.0,
        xs_output_path = 'MC_sweep/full_results.csv',
        diagnostics_output_path = 'MC_sweep/diagnostics.csv',
        plot_output    = 'MC_sweep/openmc_rod_fluxes.png',
        verbose        = True,
        cmfd_on = False,                 # NEW
        cmfd_mesh_dim = (24, 24, 1),
        cmfd_tally_begin = 20,
        cmfd_solver_begin = 40,
        keff_trigger_std = 4e-4,        # stop in-run once σ(k_eff) < 50 pcm
        trigger_max_batches = 400,      # in-run extension ceiling (no restart)
    ),
    mesh_size = 1.,                 # cm per spatial cell
)

SWEEP_ROD = SweepConfig(
    base_cfg  = CFG_ROD,
    params    = (
        (b4c_rod_outer_radius, 'b4c_rod_outer_radius'),
        (b4c_rod_cr_fraction,  'b4c_rod_cr_fraction'),
        (fuel_outer_radius,    'fuel_outer_radius'),
        (fuel_enrichment,      'fuel_enrichment'),
        (fuel_f_mod,           'fuel_f_mod'),
        (water_outer_radius,   'water_outer_radius'),
    ),
    n_samples = 256,
    lhs_seed  = 2,
    data_dir  = 'CR/training_data',
)

# ── Single-region homogeneous sphere (mix preset) ──────────────────────────
CFG_HOM = MCConfig(
    geometry = 'spherical',
    regions  = (
        RegionSpec(name='mix', outer_radius=50.0,
                   material='mix', enrichment=4.0, f_mod=0.6),
    ),
    energy_groups = EnergyGroupSpec(
        G          = 2,
        boundaries = (0., 0.625, 20.0e6),
    ),
    settings = SolverSettings(
        batches   = 200,
        inactive  = 50,
        particles = 5_000,
        xs_output_path = 'try/full_results.csv',
        plot_output    = 'try/openmc_hom_fluxes.png',
        verbose        = True,
    ),
    mesh_size = 1, 
)

# ── 2-region heterogeneous sphere (fuel core + water moderator) ─────────────
CFG_HET = MCConfig(
    geometry = 'spherical',
    regions  = (
        RegionSpec(name='core',      outer_radius=30.0,
                   material='fuel',  enrichment=10.0),
        RegionSpec(name='moderator', outer_radius=60.0,
                   material='water'),
    ),
    energy_groups = EnergyGroupSpec(
        G          = 2,
        boundaries = (0., 0.625, 20.0e6),
    ),
    settings = SolverSettings(
        batches   = 200,
        inactive  = 50,
        particles = 5_000,
        xs_output_path = 'try/full_results.csv',
        plot_output    = 'try/openmc_het_fluxes.png',
        verbose        = True,
    ),
    mesh_size = 1, 
)


# Build STACY materials using the benchmark's exact atom densities
uranyl = openmc.Material(name='uranyl_nitrate')
uranyl.set_density('sum')
uranyl.add_nuclide('U234', 6.3833e-07)
uranyl.add_nuclide('U235', 7.9213e-05)
uranyl.add_nuclide('U238', 7.0556e-04)
uranyl.add_nuclide('H1',   5.6956e-02)
uranyl.add_nuclide('N14',  2.8778e-03)
uranyl.add_element('O',    3.8029e-02)
uranyl.add_s_alpha_beta('c_H_in_H2O')

water_stacy = openmc.Material(name='water')
water_stacy.set_density('sum')
water_stacy.add_nuclide('H1', 6.6658e-02)
water_stacy.add_element('O',  3.3329e-02)
water_stacy.add_s_alpha_beta('c_H_in_H2O')

# Approximate: 2-zone cylindrical model (fuel core + water reflector)
# Radii from benchmark: fuel r=29.5 cm, reflector r=59.8 cm
CFG_STACY_APPROX = MCConfig(
    geometry='cylindrical',
    regions=(
        RegionSpec(name='uranyl_fuel', outer_radius=29.5,
                   material='custom', custom_material=uranyl),
        RegionSpec(name='water_reflector', outer_radius=59.8,
                   material='custom', custom_material=water_stacy),
    ),
    energy_groups=EnergyGroupSpec(
        G=2,
        boundaries=(0., 0.625, 20.0e6),
    ),
    settings=SolverSettings(
        batches=200,
        inactive=100,
        particles=20_000,
        axial_half_height=41.53/2,  # half the fuel height
        xs_output_path='STACY/full_results.csv',
        plot_output='STACY/openmc_stacy_fluxes.png',
        verbose=True,
    ),
    mesh_size = 1, 
)