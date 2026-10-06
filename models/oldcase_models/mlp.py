from flax import nnx
import jax.numpy as jnp
from modules.training_utils import choose_activation


class mlp(nnx.Module):

    def __init__(self, layer_sizes: list, activation: str, initialization: str, rngs: nnx.Rngs):
        super().__init__()

        if initialization == "he":
            dense_init = nnx.initializers.kaiming_normal()
        elif initialization == "xavier":
            dense_init = nnx.initializers.xavier_normal()
        
        bias_init = nnx.initializers.constant(0.0)
            
        self.layers = [
            nnx.Linear(i, o, kernel_init=dense_init, bias_init=bias_init, rngs=rngs) 
            for i, o in zip(layer_sizes[:-1], layer_sizes[1:])
        ]
        
        """if final_init:
            self.layers.pop(-1)
            self.layers.append(nnx.Linear(layer_sizes[-2], layer_sizes[-1], kernel_init=nnx.initializers.constant(0.1), bias_init=bias_init, rngs=rngs))"""
        
        self.activations = choose_activation(activation, len(self.layers))

        #self.dropout = [nnx.Dropout(0.4, rngs=rngs) for _ in self.layers[:-1]]  # No dropout on final layer


    def __call__(self, x, training=False):
        for en, (layer, activation_fn) in enumerate(zip(self.layers, self.activations)):
            x = layer(x)
            x = activation_fn(x)  # Apply the activation function for this layer
    
        return x

