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
        MaterialSpec('b4c_rod',      region_index=0),
        MaterialSpec('fuel_annulus', region_index=1),
        MaterialSpec('water', region_index=2),
    ),
    boundaries = ( # The tuple order is always centre → outside.
        BoundarySpec(name='CR_outer',      radius=8.),        
        BoundarySpec(name='core_outer',      radius=30.),
        BoundarySpec(name='moderator_outer', radius=50.),
    ),
    geometry = 'cylindrical',
    mat_properties = MatProperties(
        cr_fraction = 1.,   # like the boron but maybe sth different        
        enrichment         = 5.,   # atom % U-235
        f_mod = 0.5,  
    ),
    bc = BoundaryCondition(
        bc_type = 'vacuum',          # zero-flux at outer surface
    ),
    mesh_size = 1,                 # cm per spatial cell
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
        f_mod = 0.40,   # volume fraction (not used by current regression)
        plutonium_fraction = None,   # not a MOX case
    ),
    bc = BoundaryCondition(
        bc_type = 'vacuum',          # zero-flux at outer surface
    ),
    mesh_size = 0.5,                 # cm per spatial cell
)

# Homogeneous example :
GEO_HOM = GeometryConfig(
     G = 2,
     regions  = (MaterialSpec('mix', region_index=0),),
     boundaries = (BoundarySpec(name='outer', radius=50.0),),
     geometry   = 'spherical',
     mat_properties = MatProperties(enrichment=4., 
               f_mod = 0.50,), 
     bc         = BoundaryCondition(bc_type='vacuum'),
     mesh_size  = 2.,
)

GEO_STACY = GeometryConfig(
    G = 2,
    regions = (
        MaterialSpec('uranyl_fuel',      region_index=0),
        MaterialSpec('water_reflector', region_index=1),
    ),
    boundaries = ( # The tuple order is always centre → outside.
        BoundarySpec(name='core_outer',      radius=29.5),
        BoundarySpec(name='moderator_outer', radius=59.8),
    ),
    geometry = 'cylindrical',
    mat_properties = MatProperties(
    ),
    bc = BoundaryCondition(
        bc_type = 'vacuum',          # zero-flux at outer surface
    ),
    mesh_size = 0.5,                 # cm per spatial cell
)

