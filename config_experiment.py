e1 = {
    "exp_name": "coding",
    "seed": 10,

    # Run
    "training": True,
    "valid": True, # change the validation to try different validations
    "optimization": True,

    # Data
    "filename_data": "high_fidelity_2_20000.npz", # do we need this?
    "train_size": 200, # change to train, test, validation
    "test_size": 200,
    "stratified": "all",

    # Training
    "epochs": 100,
    "batch_size": 200, # training samples per batch
    "learn_rate_max": 5e-3,
    "learn_rate_min": 5e-4,
    "schedule": "cosine-cycles",

    # Optimization
    "opt": "grad",
    "kappas": [10.0, 20.0]
}

e2 = {
    "exp_name": "coding_2",
    "seed": 10,
    # Run     ....                
}
