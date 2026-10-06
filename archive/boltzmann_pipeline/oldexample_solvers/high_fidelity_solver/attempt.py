from openbte import load_rta

rta_data = load_rta('Si_rta')

from openbte import RTA2DSym

mat = RTA2DSym(rta_data)

from openbte import Geometry

G = Geometry(0.1)

from openbte import rectangle

L        = 10 #nm

G.add_shape(rectangle(area = L*L))

#porosity = 0.2

#area = porosity*L*L

#G.add_hole(rectangle(area = area,x=0,y=0))

#G.set_boundary_region(selector = 'inner',region = 'Boundary')

G.set_periodicity(direction = 'x',region      = 'Periodic_x')

G.set_periodicity(direction = 'y',region      = 'Periodic_y')

G.save()

from openbte import get_mesh

mesh = get_mesh()

from openbte.objects import BoundaryConditions

boundary_conditions = BoundaryConditions(periodic={'Periodic_x': 1,'Periodic_y':0},diffuse='Boundary')

from openbte.objects import EffectiveThermalConductivity

effective_kappa = EffectiveThermalConductivity(normalization=-1,contact='Periodic_x')

from openbte import Fourier

fourier     = Fourier(mesh,mat.thermal_conductivity,boundary_conditions,\
                       effective_thermal_conductivity=effective_kappa)

from openbte import BTE_RTA

bte     = BTE_RTA(mesh,mat,boundary_conditions,fourier=fourier,\
          effective_thermal_conductivity=effective_kappa)

from openbte.objects import OpenBTEResults

results =  OpenBTEResults(mesh=mesh,material = mat,solvers={'bte':bte,'fourier':fourier})

results.show()