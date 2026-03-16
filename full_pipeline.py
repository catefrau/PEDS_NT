import os
from mpi4py import MPI
from flax import nnx

from modules.data_ingestion import data_ingestion
from modules.choose_model import select_model
from modules.params_utils import initialize_or_restore_params
from modules.training import train_model
from utils import create_folders
from modules.run_optimization import optimize
from modules.validation import final_validation

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

from config_experiment import e1 as exp_config
from config_model import m1 as model_config


# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Initialize random key
rngs = nnx.Rngs(exp_config["seed"])

create_folders(exp_config['exp_name'], model_config['model_name']) # experiment name

model = select_model(
    rngs=rngs, 
    model_type=model_config["model"], 
    resolution=model_config["resolution"], 
    learn_residual=model_config["learn_residual"], 
    hidden_sizes=model_config["hidden_sizes"], 
    activation=model_config["activation"],
    solver=model_config["solver"],
    initialization=model_config['initialization'],
)

# Params initializing or restoring
model, checkpointer = initialize_or_restore_params(model, model_config["model_name"], base_dir= exp_config['exp_name'], rank=rank) # check or do it deeper

# Ingest data
dataset_train, dataset_test, dataset_valid_small, kappas_design_valid = data_ingestion(
    rank=rank,
    exp_name=exp_config['exp_name'],
    filename=exp_config["filename_data"], 
    train_size=exp_config["train_size"], 
    test_size=exp_config['test_size'],
    stratified=exp_config['stratified'],
    key=rngs
)

if exp_config['training']:
    if rank == 0:
        print(f"Training on {size} devices...")
    mse_train, mse_test, perc_error_test = train_model(
        exp_name=exp_config['exp_name'], 
        model_name=model_config["model_name"],
        dataset_train=dataset_train,
        dataset_test=dataset_test,
        model=model,
        learn_rate_min=exp_config["learn_rate_min"], 
        learn_rate_max=exp_config["learn_rate_max"], 
        schedule=exp_config["schedule"], 
        epochs=exp_config["epochs"], 
        batch_size=exp_config["batch_size"],
        checkpointer=checkpointer
        )

if exp_config['valid'] and rank == 0:
    print("Validation...")
    final_validation(exp_config["exp_name"], model, model_config['model_name'], dataset_valid_small, mse_train, mse_test, perc_error_test)

if rank == 0 and exp_config['optimization']:
    
    if exp_config['kappas'] is None:
        exp_config['kappas'] = kappas_design_valid.tolist()
    
    kappas_valid = exp_config['kappas']

    optimize(
        exp_name=exp_config['exp_name'],
        model_name=model_config["model_name"],
        model=model,
        opt=exp_config['opt'],
        kappas=kappas_valid,
        seed=rngs
    )


# ToDo:

# 1 Create a configuration file that is separate...
# 2 Adjust file system with experiment names
# 3 Create comparison between models automated
# 4 Adjust ingestion with dropout
# 5 Add uncertainty to each model (ideally put it inside each model like you did with generator+solver)
# 6 Same but with ensemble (store N models)


