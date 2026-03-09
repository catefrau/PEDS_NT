from flax import nnx
import jax.numpy as jnp
from models.mlp import mlp
from solvers.low_fidelity_solvers.lowfidsolver_class import lowfid
from solvers.low_fidelity_solvers.base_conductivity_grid_converter import conductivity_original_wrapper


# An easy PEDS Wrapper
class PEDS(nnx.Module):

    def __init__(self, resolution:int, 
                learn_residual: bool, hidden_sizes:list, activation:str,
                solver:str, initialization:str):
        super().__init__()

        # 100 nanometers / step_size nanometer
        self.resolution = resolution
        self.layer_sizes = [25] + hidden_sizes + [resolution**2]  # 25 input features (5x5 grid) and output is the conductivity grid (resolution x resolution)
        self.activation = activation
        self.learn_residual = learn_residual

        # Create model
        key = nnx.Rngs(42)

        self.generator = mlp(layer_sizes=self.layer_sizes, activation = activation, rngs=key, initialization=initialization) # 
        
        # Low Fidelity Solver
        self.lowfidsolver = lowfid(solver=solver, iterations=1000)
    
    def __call__(self, pores, training=False): # Here

        batch_size = pores.shape[0]

        pores = jnp.reshape(pores, (batch_size,25))

        #pores_new = 1 - jnp.reshape(pores, (batch_size, 25)) 

        # Process data through the generator (MLP)
        conductivity_generated = nnx.jit(self.generator, static_argnames=("training",))(pores, training)
        conductivities = None

        # Reshape the output
        conductivity_generated = jnp.reshape(conductivity_generated, (batch_size, self.resolution, self.resolution))

        # Rescale and Adjust
        if self.learn_residual: # If learn_residual is True, the model adds a base conductivity to the generator outpu
            conductivities = conductivity_original_wrapper(pores, self.resolution)
            conductivity_generated = conductivity_generated+conductivities 

        if self.activation =="relu":
            conductivity_generated = jnp.maximum(conductivity_generated,1e-16) # make it bigger if instability with numerical method

        
    
        kappa = self.lowfidsolver(conductivity_generated) 

        return kappa, conductivity_generated

