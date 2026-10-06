from models.mlp import mlp
from models.peds import PEDS

def select_model(rngs, model_type, **kwargs):

    if model_type == "PEDS":
        return PEDS(
            resolution=kwargs["resolution"], 
            learn_residual=kwargs["learn_residual"], 
            hidden_sizes=kwargs["hidden_sizes"], 
            activation=kwargs["activation"], 
            solver=kwargs["solver"],
            initialization=kwargs['initialization'],
        )
    elif model_type == "MLP":
        return mlp(
            layer_sizes=[25] + kwargs["hidden_sizes"] + [1],  # Assuming this maps correctly
            activation=kwargs["activation"],
            initialization=kwargs['initialization'], 
            rngs=rngs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")