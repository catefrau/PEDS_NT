
from typing import NamedTuple, Optional
from dataclasses import dataclass, replace 



# ══════════════════════════════════════════════════════════════════════════════
#  NAMED TUPLES — full problem description
# ══════════════════════════════════════════════════════════════════════════════

class RegionSpec(NamedTuple):
    """
    One concentric zone, ordered centre → outside.

    name         : unique label used as cell name, MGXS domain key,
                   and CSV column prefix.
    outer_radius : outer edge of this zone in cm.
                   For slab geometry this is the half-thickness of the zone
                   measured from the origin.
    material     : preset name — one of:
                     'fuel'    UO2-like (U235 + U238 at 10.5 g/cm³)
                     'water'   H₂O at 1.0 g/cm³
                     'b4c'     Boron-carbide control-rod material
                     'mix'     Homogeneous U + water mixture (lattice cell)
                     'void'    Zero-density void region
                     'custom'  Pass a fully defined openmc.Material via
                               custom_material; all knobs below are ignored.

    Material knobs  (set the ones relevant to the chosen preset; rest ignored):
    enrichment     : float — U-235 atom % (used by 'fuel', 'mix')
    f_mod          : float — volume fraction of water  (used by 'mix')
    cr_fraction    : float — B4C mass fraction in the rod  (used by 'b4c')
    custom_material: openmc.Material instance  (required for 'custom')
    """
    name:            str
    outer_radius:    float
    material:        str
    enrichment:      float                   = 5.0
    f_mod:           float                   = 0.0
    cr_fraction:     float                   = 1.0
    custom_material: Optional[object]        = None


class EnergyGroupSpec(NamedTuple):
    """
    Energy group structure for MGXS tallies and the flux energy filter.

    G          : number of energy groups
    boundaries : tuple of G+1 bin edges in eV, low → high.
                 e.g. (0., 0.625, 20.0e6) for the classic 2-group split.
    """
    G:          int
    boundaries: tuple   # length G+1


class SolverSettings(NamedTuple):
    """
    Monte Carlo and I/O settings.

    batches           : total MC batches (active + inactive)
    inactive          : number of warm-up (inactive) batches
    particles         : particles per batch
    n_bins            : number of spatial flux-tally bins across the full domain
    axial_half_height : cm — axial (z) or transverse (y,z) extent for
                        cylindrical and slab geometries; reflective planes are
                        placed at ±axial_half_height.  Ignored for spherical.
    xs_output_path    : cumulative results CSV (one row appended per run)
    plot_output       : path for the flux-profile PNG
    verbose           : print config and result summaries when True
    """
    batches:           int   = 200
    inactive:          int   = 50
    particles:         int   = 20_000
    axial_half_height: float = 50.0
    xs_output_path:    str   = 'MC_sweep/full_results.csv'
    diagnostics_output_path: str = 'MC_sweep/diagnostics.csv'
    plot_output:       str   = 'plots/openmc_fluxes.png'
    verbose:           bool  = True
    cmfd_on: bool = False
    cmfd_mesh_dim: tuple[int, int, int] = (1, 1, 1)   # independent of entropy_mesh_dim
    cmfd_tally_begin: int = 50
    cmfd_solver_begin: int = 70
    keff_trigger_std: float | None = None
    trigger_max_batches: int | None = None
    generations_per_batch: int = 5
    statepoint_mid_fraction: float = 0.75      # also save one statepoint at this fraction of max batches
    entropy_mesh_dim: tuple[int, int, int] = (20, 20, 20)
    convergence_output_dir: str = 'MC_sweep/convergence'
    copy_statepoint: bool = True
    openmc_work_dir: str | None = None   # per-run folder for XML + statepoint output


class MCConfig(NamedTuple):
    """
    Complete problem specification for one OpenMC run.

    geometry : 'spherical' | 'cylindrical' | 'slab'
    regions  : ordered tuple of RegionSpec, centre → outside.
               A single-region tuple is treated as a homogeneous problem.
    energy_groups : EnergyGroupSpec
    settings : SolverSettings
    """
    geometry:      str
    regions:       tuple          # tuple[RegionSpec, ...]
    energy_groups: EnergyGroupSpec
    settings:      SolverSettings
    mesh_size:     int            # cm per spatial cell



@dataclass(frozen=True)
class ParamSpec:
    nom: float
    bounds:  tuple[float, float]   # (min, max)
    @property
    def lo(self): return self.bounds[0]
    @property
    def hi(self): return self.bounds[1]



@dataclass(frozen=True)
class SweepConfig:
    """
    Bundles an MCConfig with its LHS sweep definition.

    Fields
    ------
    base_cfg    : the nominal MCConfig (geometry, materials, solver settings)
    params      : ordered list of (ParamSpec, label) pairs to sweep.
                  Label must match the keyword expected by apply_sample().
                  Comment out any entry to fix that parameter at its nominal.
    n_samples   : number of LHS points
    lhs_seed    : random seed for reproducibility
    data_dir    : folder where CSV/PNG outputs are written
    """
    base_cfg  : MCConfig
    params    : tuple               # Tuple[Tuple[ParamSpec, str], ...]
    n_samples : int   = 256
    lhs_seed  : int   = 2
    data_dir  : str   = 'CR/training_data'

    @property
    def labels(self) -> list[str]:
        return [label for _, label in self.params]

    @property
    def bounds_array(self):
        import numpy as np
        return np.array([[p.lo, p.hi] for p, _ in self.params])

    @property
    def dataset_file(self) -> str:
        return f'{self.data_dir}/LHS_full_dataset.csv'

    @property
    def checkpoint_file(self) -> str:
        return f'{self.data_dir}/checkpoint.csv'

    def plot_path(self, run_index: int) -> str:
        return f'{self.data_dir}/flux_run_{run_index:04d}.png'
