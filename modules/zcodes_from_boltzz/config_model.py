m1 = {
    "model_name": "code_check",
    "model": "PEDS",
    "resolution": 5,
    "learn_residual": False,
    "hidden_sizes": [32, 32],
    "activation": "mixed", 
    "solver": "gauss",
    "initialization": "he",
}

m2 = {
    "model_name": "PEDS_gauss",
    "model": "PEDS",
    "resolution": 20,
    "learn_residual": False,
    "hidden_sizes": [32, 64, 128],
    "activation": "relu",
    "solver": "gauss",
    "initialization": "he",
}



