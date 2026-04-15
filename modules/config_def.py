from typing import NamedTuple, Optional


# ══════════════════════════════════════════════════════════════════════════════
#  NAMED-TUPLE DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

class MatProperties(NamedTuple):
    """
    Composition / material knobs.  Set any field to None to leave it
    unspecified; predict_xs() will ignore None fields.

    enrichment         : U-235 atom %
    moderator_fraction : volume fraction of moderator in the lattice cell
    plutonium_fraction : Pu atom % (for MOX fuels)
    # ── Add more knobs here as the training data grows ────────────────────────
    # temperature      : float = None   # K
    # burnup           : float = None   # MWd/tHM
    # void_fraction    : float = None   # coolant void fraction
    """
    enrichment:          Optional[float] = None
    moderator_fraction:  Optional[float] = None
    plutonium_fraction:  Optional[float] = None
    CRabsorber_fraction: Optional[float] = None
    

class MaterialSpec(NamedTuple):
    """
    Declares one material zone and its row index in the xs_tensor.
    Does NOT hold XS values — those live in the tensor.

    region_ID  : human-readable name used as column-name prefix in the
                   training CSV, e.g. 'core', 'moderator', 'mix'
    region_index : which row in xs_tensor (0-based, ordered center → outside)
    """
    region_ID:  str
    region_index: int


class BoundarySpec(NamedTuple):
    """
    One named radial boundary. The tuple order is always centre → outside.

    name   : descriptive label, e.g. 'core_outer', 'reflector_outer'
    radius : outer edge of this zone in cm
    height : axial height in cm — only relevant for cylindrical geometry;
             ignored for slab / sphere.  Set None if not applicable.
    """
    name:   str
    radius: float
    height: Optional[float] = None


class BoundaryCondition(NamedTuple):
    """
    Named boundary condition for the outer surface r = R.

    Supported bc_type strings and their physical meaning
    (see One_Group_Diffusion_Equation.pdf §2.3):

      'reflective'      dφ/dr|_R = 0  ──  no net current out (mirror symmetry)
                        A=0, B=1, C=0

      'vacuum'          zero-flux Dirichlet  φ(R) = 0
                        A=1, B=0, C=0

      'dirichlet'       prescribed flux  φ(R) = phi_val
                        A=1, B=0, C=phi_val

      'albedo'          fraction α of outgoing neutrons reflected back
                        A=(1-α)/(4(1+α)),  B=D_val/2,  C=0
                        Requires: alpha, D_val

      'partial_current' prescribes the incoming partial current J_in
                        A=1/4,  B=D_val/2,  C=Jin
                        Requires: Jin, D_val

    Fields
    ------
    bc_type : str
    alpha   : float, albedo fraction  (for 'albedo')
    Jin     : float, incoming partial current  (for 'partial_current')
    phi_val : float, fixed flux value  (for 'dirichlet'); default 0.0
    D_val   : float, diffusion coeff at boundary used in albedo / partial-current
              formulas.  If None, the solver will read D from the outermost cell.
    """
    bc_type: str
    alpha:   Optional[float] = None
    Jin:     Optional[float] = None
    phi_val: float           = 0.0
    D_val:   Optional[float] = None


class GeometryConfig(NamedTuple):
    """
    Complete problem specification — geometry, regions, and solver settings.

    G              : number of energy groups (≥ 1)
    regions      : ordered tuple of MaterialSpec, from centre outward.
                     If exactly ONE material is given, the problem is treated
                     as homogeneous (single zone filling the whole domain).
    boundaries     : ordered tuple of BoundarySpec, same length as regions.
                     boundaries[-1].radius is the outer edge R of the domain.
    geometry       : 'slab' | 'cylindrical' | 'spherical'
    mat_properties : MatProperties knobs used as regression inputs
    bc             : BoundaryCondition at r = R
    mesh_size      : spatial cell width in cm (default 0.1)
    """
    G:              int
    regions:      tuple           # tuple[MaterialSpec, ...]
    boundaries:     tuple           # tuple[BoundarySpec, ...]
    geometry:       str             # 'slab' | 'cylindrical' | 'spherical'
    mat_properties: MatProperties
    bc:             BoundaryCondition
    mesh_size:      float 

