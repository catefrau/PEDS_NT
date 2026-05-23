e1 = {
    "exp_name": "coding",
    "seed": 10,

    # Run, turn on/off different steps of the pipeline
    "training": True,
    "valid": False, # change the validation to try different validations
    "optimization": False,  # inverse design

    # Data
    "filename_data": "high_fidelity_2_20000.npz",
    "train_size": 100, # total data points
    "test_size": 100,
    "stratified": "all",

    # Training
    "epochs": 100,
    "batch_size": 100, # training samples per batch
    "learn_rate_max": 5e-3,
    "learn_rate_min": 5e-4,
    "schedule": "cosine-cycles",

    # Optimization
    "opt": "grad",
    #"kappas": None, # None means we will use the kappas from the validation set
    "kappas": [20.0],
}

e2 = {
    "exp_name": "coding_2",
    "seed": 10,
    # Run     ....                
}
