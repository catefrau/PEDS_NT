from flax import nnx
import jax.numpy as jnp
import jax

class cnn(nnx.Module):

    def __init__(self, rngs: nnx.Rngs):
        super().__init__()

        # Define the number of filters for each layer (encoder filters)
        self.filters = [1, 16, 32, 64]  # Example filter sizes

        # Create a list of Conv layers with Xavier initialization
        self.layers = [
            nnx.Conv(in_features=i, out_features=o, kernel_size=(3, 3), strides=(1, 1), padding="SAME",
                     kernel_init=nnx.initializers.xavier_uniform(), rngs=rngs)
            for i, o in zip(self.filters[:-1], self.filters[1:])
        ]

        self.dropout = [
            nnx.Dropout(rate=0.5, rngs=rngs)
            for _ in range(len(self.layers))
        ]


    @nnx.jit
    def __call__(self, x):
        x = jnp.expand_dims(x, 0)
        x = jnp.expand_dims(x, -1)
        print(x.shape)
        # Input (1, 20, 20)
        for en, (layer, dropout) in enumerate(zip(self.layers, self.dropout)):
            x = layer(x)
            x = nnx.relu(x)
            #x = dropout(x)
        
        x = jnp.mean(x, axis=-1)

        
        x = - jnp.reshape(x, (20, 20))

        #x =  jnp.minimum(150, x)
        #x = jnp.maximum(-50.0, x)
        
        return x
        # output (1, 20, 20)