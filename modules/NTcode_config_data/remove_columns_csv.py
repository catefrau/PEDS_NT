import pandas as pd

csv_path = "../FILES/26june.csv"

keep_cols = [
    "geometry",
    "G",
    "r0_b4c_rod_outer_radius",
    "r0_b4c_rod_material",
    "r0_b4c_rod_cr_fraction",
    "r1_fuel_annulus_outer_radius",
    "r1_fuel_annulus_material",
    "r1_fuel_annulus_enrichment",
    "r1_fuel_annulus_f_mod",
    "r2_water_outer_radius",
    "r2_water_material",
    "keff",
    "keff_std",
    "b4c_rod_diffusion-coefficient_g1",
    "b4c_rod_diffusion-coefficient_g2",
    "b4c_rod_absorption_g1",
    "b4c_rod_absorption_g2",
    "b4c_rod_nu-fission_g1",
    "b4c_rod_nu-fission_g2",
    "b4c_rod_scatter matrix_g1",
    "b4c_rod_scatter matrix_g2",
    "b4c_rod_scatter matrix_g3",
    "b4c_rod_scatter matrix_g4",
    "b4c_rod_chi_g1",
    "b4c_rod_chi_g2",
    "fuel_annulus_diffusion-coefficient_g1",
    "fuel_annulus_diffusion-coefficient_g2",
    "fuel_annulus_absorption_g1",
    "fuel_annulus_absorption_g2",
    "fuel_annulus_nu-fission_g1",
    "fuel_annulus_nu-fission_g2",
    "fuel_annulus_scatter matrix_g1",
    "fuel_annulus_scatter matrix_g2",
    "fuel_annulus_scatter matrix_g3",
    "fuel_annulus_scatter matrix_g4",
    "fuel_annulus_chi_g1",
    "fuel_annulus_chi_g2",
    "water_diffusion-coefficient_g1",
    "water_diffusion-coefficient_g2",
    "water_absorption_g1",
    "water_absorption_g2",
    "water_nu-fission_g1",
    "water_nu-fission_g2",
    "water_scatter matrix_g1",
    "water_scatter matrix_g2",
    "water_scatter matrix_g3",
    "water_scatter matrix_g4",
    "water_chi_g1",
    "water_chi_g2",
    "keff_bin"
]

df = pd.read_csv(csv_path)

# Keep only columns that actually exist in the file
existing_cols = [col for col in keep_cols if col in df.columns]
df = df[existing_cols]

# Save cleaned file
output_path = "../FILES/26june_trimmed.csv"
df.to_csv(output_path, index=False)

print("Saved cleaned file to:", output_path)
print("Remaining columns:", df.columns.tolist())
print("Shape:", df.shape)