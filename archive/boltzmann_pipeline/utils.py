import os


# File Sys Management
def create_folders(experiment_name, model_name):
    os.makedirs(f"experiments/{experiment_name}", exist_ok=True) # put it in the final validation folder
    os.makedirs(f"experiments/{experiment_name}/figures", exist_ok=True) # put it in the final validation folder
    os.makedirs(f"experiments/{experiment_name}/figures/peds_evolution/{model_name}", exist_ok=True)
    os.makedirs(f"experiments/{experiment_name}/curves", exist_ok=True)
    os.makedirs(f"experiments/{experiment_name}/results", exist_ok=True)
    os.makedirs(f"experiments/{experiment_name}/optimization", exist_ok=True)