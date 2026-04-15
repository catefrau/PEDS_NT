# run_config.py
# definition of specific problems

from config_def import GeometryConfig, MaterialSpec, BoundarySpec, BoundaryCondition, MatProperties

# TODO create sets of p vectors with general names that can be changed in the loops

# TODO include here the general definition column names of the regression inputs

# for the geometry definition, choose between
# 'slab', 'cylindrical', 'spherical'

GEO_CYL = GeometryConfig(
    G = 2,
    regions = (
        MaterialSpec('control_rod',      region_index=0),
        MaterialSpec('core', region_index=1),
        MaterialSpec('moderator', region_index=2),
    ),
    boundaries = ( # The tuple order is always centre → outside.
        BoundarySpec(name='CR_outer',      radius=2.0),        
        BoundarySpec(name='core_outer',      radius=20.0),
        BoundarySpec(name='moderator_outer', radius=50.0),
    ),
    geometry = 'cylindrical',
    mat_properties = MatProperties(
        enrichment         = 4.0,   # atom % U-235
        moderator_fraction = 0.40,  
        CRabsorber_fraction = 1.,   # like the boron but maybe sth different
    ),
    bc = BoundaryCondition(
        bc_type = 'vacuum',          # zero-flux at outer surface
    ),
    mesh_size = 0.5,                 # cm per spatial cell
) 

GEO_HET = GeometryConfig(
    G = 2,
    regions = (
        MaterialSpec('core',      region_index=0),
        MaterialSpec('moderator', region_index=1),
    ),
    boundaries = ( # The tuple order is always centre → outside.
        BoundarySpec(name='core_outer',      radius=50.0),
        BoundarySpec(name='moderator_outer', radius=60.0),
    ),
    geometry = 'spherical',
    mat_properties = MatProperties(
        enrichment         = 4.0,   # atom % U-235
        moderator_fraction = 0.40,   # volume fraction (not used by current regression)
        plutonium_fraction = None,   # not a MOX case
    ),
    bc = BoundaryCondition(
        bc_type = 'vacuum',          # zero-flux at outer surface
    ),
    mesh_size = 0.5,                 # cm per spatial cell
)

# Homogeneous example (single-zone 'mix' — uncomment to use):
GEO_HOM = GeometryConfig(
     G = 2,
     regions  = (MaterialSpec('mix', region_index=0),),
     boundaries = (BoundarySpec(name='outer', radius=50.0),),
     geometry   = 'spherical',
     mat_properties = MatProperties(enrichment=4.0, 
               moderator_fraction = 0.40,), 
     bc         = BoundaryCondition(bc_type='vacuum'),
     mesh_size  = 0.5,
)

