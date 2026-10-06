# High fidelity solver using the open-source package OpenBTE.
# OpenBTE, an efficient solver for the steady-state phonon BTE in multidimensional structures.
# This tool is interfaced to first-principles calculations, thus it unlocks the calculations of thermal-related properties with no fitting-parameters

import numpy as np
import os

# OpenBTE
from openbte import load_rta, RTA2DSym, Geometry, BTE_RTA, Fourier, rectangle, get_mesh
from openbte.objects import BoundaryConditions, OpenBTEResults, EffectiveThermalConductivity


def highfidelity_solver(pores, step_size, save_show_res = False):

    # cancel any previous geometry saved 

    # Create Material (load material properties from file, here silicon)
    rta_data = load_rta('Si_rta') 
    mat = RTA2DSym(data=rta_data)

    # Create Base Mesh
    mesh = Geometry(step_size) # prova anche 1
    L= 100 # mesh 
    mesh.add_shape(rectangle(area = L*L))

    # Pores
    pores_centers = convert_pores(pores)
    for pore in pores_centers:
        mesh.add_hole(rectangle(area = 100,x=pore[0],y=pore[1]))

    # Set Boundary conditions
    if pores.any():
        mesh.set_boundary_region(selector = 'inner',region = 'Boundary')

    mesh.set_periodicity(direction = 'x',region = 'Periodic_x')
    mesh.set_periodicity(direction = 'y',region = 'Periodic_y')

    
    mesh.save()

    mesh = get_mesh()

    

    # Boundary Conditions
    if pores.any():
        boundary_conditions = BoundaryConditions(periodic={'Periodic_x': 0,'Periodic_y':1}, diffuse='Boundary')
    
    else:
        boundary_conditions = BoundaryConditions(periodic={'Periodic_x': 0,'Periodic_y':1})
    # Effective Thermal Conductivity
    effective_kappa = EffectiveThermalConductivity(normalization=-1,contact='Periodic_y')

    #print(mat.thermal_conductivity)
   
    # Base Solver for standard Heat Conduction (first guess)
    fourier = Fourier(mesh,mat.thermal_conductivity,boundary_conditions, effective_thermal_conductivity=effective_kappa, verbose=False)
    
    # Boltzmann Transport EquationSolver
    bte = BTE_RTA(mesh ,mat ,boundary_conditions ,fourier=fourier, effective_thermal_conductivity=effective_kappa)
   

    results = OpenBTEResults(mesh=mesh,material = mat,solvers={'bte':bte})
    
    results.save()


    results = OpenBTEResults.load()

    
    results_bte = results[-2]['bte']
    #results_fourier = results[-2]['fourier']

    kappa_eff_BTE, temp_BTE, flux_BTE = results_bte.kappa_eff, results_bte.variables['Temperature_BTE']['data'], results_bte.variables['Flux_BTE']['data']
    #kappa_eff_Fourier, temp_Fourier, flux_Fourier = results_fourier.kappa_eff, results_fourier.variables['Temperature_Fourier'], results_fourier.variables['Flux_Fourier']

    # Possibly need to integrate the flux of Fourier 

    return kappa_eff_BTE, temp_BTE, flux_BTE


def convert_pores(pores):
    
    indices = np.argwhere(pores)

    indices = indices[:, [1, 0]]  

    # Calculate the centers for each pore
    pores_centers = (indices * (-20) +40).tolist()

    for p in pores_centers:
        p[0] = -p[0]
    
    return pores_centers


if  __name__ == "__main__":

    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    #pores = np.array([0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1])
    pores = np.zeros(25)
    pores = pores.reshape((5,5))

    results = highfidelity_solver(pores, step_size= 2.0, save_show_res=False)

    kappa, temp, flux = results

    print(flux.shape)

    def plot_temperature(Temperatures, base_conductivities, index=0):
            cmap = plt.cm.viridis
            norm = mcolors.Normalize(vmin=Temperatures[index].min(), vmax=Temperatures[index].max())
            
            threshold = np.min(base_conductivities[:3]) + 0.01
            masked_T = np.ma.masked_where(base_conductivities[index] < threshold, Temperatures[index])

            plt.figure(figsize=(6, 5))
            im = plt.imshow(masked_T, cmap=cmap, norm=norm, interpolation='nearest')
            plt.colorbar(im, label='Temperature')
            plt.contour(masked_T, levels=np.linspace(Temperatures[index].min(), Temperatures[index].max(), 25), 
                        colors='white', linewidths=0.5)
            
            plt.title(f'Heatmap of T (Index {index}) with Level Sets')
            plt.xlabel('x direction')
            plt.ylabel('y direction')
            plt.show()


    import pandas as pd


    """model_name = "PEDS_gauss"

    results = pd.read_csv(f"data/optimization/{model_name}/evolutionary_geometries.csv")
    result_new = pd.DataFrame(columns=["kappa_target", "geometries", "kappa_BTE"])

    kappa_BTE = []

    for geom, k in zip(results['geometries'], results['kappa_target']):

        # Convert to list of integers
        hof_array = geom.strip("\"").strip("[]")  # Remove quotes and brackets
        hof_array = np.array([int(x) for x in hof_array.split(", ")]) 

        pores = hof_array.reshape((5,5))
    
        results = highfidelity_solver(pores, save_show_res=False)

        kappa, _, _ = results

        print(k)

        result_new = result_new._append({"kappa_target":k, "geometries":geom, "kappa_BTE":kappa}, ignore_index=True)
    
    result_new.to_csv(f"data/optimization/{model_name}/evolutionary_geometries_withBTE.csv", index=False)
"""