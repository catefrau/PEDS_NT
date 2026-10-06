import jax
import jax.random as random
from flax import nnx
import jax.numpy as jnp

class BayesianLinear(nnx.Module):
    def __init__(self, in_features: int, out_features: int, rng):
        self.in_features = in_features
        self.out_features = out_features
        key_w, key_b = random.split(rng)

        # initialize
        self.w_mu = nnx.Param(jax.random.normal(key_w, (out_features, in_features)) * 0.1)
        self.w_logstd = nnx.Param(jnp.full((out_features, in_features), -3.0))
        self.b_mu = nnx.Param(jnp.zeros(out_features))
        self.b_logstd = nnx.Param(jnp.full(out_features, -3.0))

    def __call__(self, x, rng):
        key_w, key_b = random.split(rng)

        # sample
        w_std = jnp.exp(self.w_logstd)
        b_std = jnp.exp(self.b_logstd)
        w = self.w_mu + w_std * random.normal(key_w, self.w_mu.shape)
        b = self.b_mu + b_std * random.normal(key_b, self.b_mu.shape)

        return jnp.dot(x, w.T) + b

class BNN(nnx.Module):
    def __init__(self, layer_sizes: list, activation: Callable, rng):
        self.activation = activation
        keys = random.split(rng, len(layer_sizes) - 1)
        self.layers = [BayesianLinear(layer_sizes[i], layer_sizes[i+1], rng=keys[i]) for i in range(len(layer_sizes) - 1)]

    def __call__(self, x, rng):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, rng)
            x = self.activation(x)

        return self.layers[-1](x, rng)

rng = random.PRNGKey(0)
bnn = BNN([2, 16, 16, 1], jax.nn.relu, rng=rng)
x = jnp.array([[0.5, -1.2]])

output = bnn(x, rng=random.PRNGKey(1))
print("Output:", output)
