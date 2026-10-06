import jax
import jax.numpy as jnp
import numpy as np
import os
from OpenBTE_highfid import highfidelity_solver
import re


# make it such that the results only store pores and kappas. With stores in a single 1D array and kappa a float

# Number of data points to generate
num_data_points = 1500
step_size = 2
perc = 0.5


results = {'pores': [], 'kappas': []}

save_dir = "./data/highfidelity"
os.makedirs(save_dir, exist_ok=True)
# Regex pattern to find files with the naming convention "high_fidelity_#.npz"
pattern = re.compile(f"high_fidelity_{step_size}_(\d+)\.npz")

# Find the file with the highest numerical value in its name
def get_highest_value_file(directory, pattern):
    files = [f for f in os.listdir(directory) if pattern.match(f)]
    if not files:
        return None, 0
    # Extract numerical values and find the max
    max_file = max(files, key=lambda x: int(pattern.search(x).group(1)))
    max_value = int(pattern.search(max_file).group(1))
    return os.path.join(directory, max_file), max_value

# Check for the latest high-fidelity file
filename, existing_count = get_highest_value_file(save_dir, pattern)

if filename:
    print(f"Loading existing data from {filename}...")
    with np.load(filename, allow_pickle=True) as data:
        existing_results = {key: data[key].tolist() for key in data.keys()}
    print(f"Existing count: {existing_count}")
else:
    print("No existing data found. Initializing empty results...")
    existing_results = {'pores': [], 'kappas': []}
    existing_count = 0
# Print results for verification
print(f"Initialized with {existing_count} existing observations.")
# Generate and save new data points
for i in range(len(existing_results['pores']), len(existing_results['pores']) + num_data_points):
   
    key = jax.random.PRNGKey(i + existing_count)
    
    # Generate pores as a 2D boolean array, then convert to int
    pores = (jax.random.uniform(key, (5, 5)) < perc).astype(int)


    # Run high fidelity solver with 2D `pores`
    kappa_bte, temp_bte, flux_bte = highfidelity_solver(pores, step_size, save_show_res=False)
    
    # Flatten pores, temp_bte, and flux_bte for storage
    pores_flat = pores.flatten()
    temp_bte_flat = temp_bte.flatten()
    flux_bte_flat = flux_bte.reshape(-1, 2)  # Keep as (N, 2) if needed

    # Append each result to the corresponding key in the dictionary
    results['pores'].append(pores_flat)
    results['kappas'].append(kappa_bte)
    

# Save results if any new points were generated
if any(results.values()):
    # Update existing results
    for key in results:
        existing_results[key].extend(results[key])
    
    # Save the data with the new filename
    filename = os.path.join(save_dir, f"high_fidelity_{step_size}_{existing_count+num_data_points}.npz")
    np.savez(filename, **{k: np.array(v, dtype=object) for k, v in existing_results.items()})

# Load and print the final saved data
with np.load(filename, allow_pickle=True) as data:
    print("\nFinal data saved in:", filename)
    for key in data.files:
        print(f"{key}: {np.shape(data[key][0])}, for {len(data[key])}")