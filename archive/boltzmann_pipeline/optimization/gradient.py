import jax
import jax.numpy as jnp
import optax
from models.peds import PEDS

def gradient_opt(model, target, seed, neigh=True, batch_size=10, steps=100, lr=0.1):

    if neigh:
        def loss_fn(params, model, target):
            noise = jax.random.normal(jax.random.PRNGKey(0), params.shape) * 0.05
            perturbed_params = jnp.clip(params + noise, 0, 1)  # what are those???

            if isinstance(model, PEDS):
                k, _ = model(perturbed_params)
            else:
                k = model(perturbed_params)
        
            return jnp.mean(jnp.abs(k - target))
    
    else:
        pass

    seed = seed.unwrap() if hasattr(seed, "unwrap") else seed # Extract JAX key if it's an nnx RngStream

    params = jax.random.uniform(seed, (batch_size, 25))  # Continuous relaxation in [0,1]

    # Optimizer with momentum
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params) # initializes the optimizer state based on the initial parameters


    def step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params, model, target) # loss_fn computes the MSE between calculated and target kappa within a batch
        updates, opt_state = optimizer.update(grads, opt_state) # computes the parameter updates based on the gradients and the optimizer state 
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # Optimization loop
    for _ in range(steps):
        params, opt_state, loss = step(params, opt_state) #params shape is (batch_size, 25)
        if _ % 25 == 0:   # Print loss every 25 steps
            print(f"Step {_} losses mean: {loss}")
            #print(f"--- the params are: {params}")
            # print(f"--- the param shapes is: {params.shape}")  #(batch_size, 25)
            #print(f"--- this misterious opt_state is: {opt_state}")
    
    # Binarization step
    binary_params = (params > 0.5).astype(jnp.float32)

    if isinstance(model, PEDS):
            k, _ = model(params)  # Model output
            k_binarized, _ = model(binary_params)
    else:
        k = model(params)

    # Compute loss for each element in batch
    losses = jnp.abs(k - target)  # Loss for continuous params
    best_idx = jnp.argmin(losses)  # Index of best parameter set

    # Select best parameters
    best_params = params[best_idx]
    best_binary_params = binary_params[best_idx]
    best_k = k[best_idx]
    best_k_binarized = k_binarized[best_idx]

    print(f"Best Found params: {best_params} with kappa= {best_k}")
    print(f"size of this array: {best_binary_params.shape}")
    print(f"Binarized {best_binary_params} with kappa={best_k_binarized}")

    return best_binary_params, best_k_binarized # this is design, kappa_optimized in the run_optimization.py file, which is then saved in the csv file in the experiments folder.


"""
def gradient_opt(model, target, out, seed, steps=100, lr=0.1, runs=200, batch_size=200):
    
    seed = seed.unwrap() if hasattr(seed, "unwrap") else seed  # Extract JAX key if it's an nnx RngStream

    # Initializing optimizer
    optimizer = optax.adam(lr)
    
    # Generate random seed for batch initialization
    key = jax.random.PRNGKey(seed)
    params = jax.random.uniform(key, (batch_size, 25))  # Initialize batch of 200 individual parameter sets (shape: 200 x 25)
    opt_state = optimizer.init(params)

    # Define the loss function for an individual parameter set (this will be vectorized later)
    def loss_fn_single(param, target):
        if out == "peds":
            kappa, _ = model(param)  # Model output for a single individual
        else:
            kappa = model(param)

        # Compute loss for this individual
        loss = jnp.abs(kappa - target)
        return loss

    # Vectorize the loss function to apply it to the entire batch (200 individuals)
    loss_fn_batch = jax.vmap(loss_fn_single, in_axes=(0, None))  # Apply loss function to each individual in batch

    # Gradient step for the batch
    def step(params, opt_state, target):
        loss, grads = jax.value_and_grad(loss_fn_batch)(params, target)  # Compute gradient for all individuals in batch
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # Optimization loop
    for _ in range(steps):
        params, opt_state, loss = step(params, opt_state, target)

    # Binarization step for the whole batch
    binary_params = (params > 0.5).astype(jnp.float32)

    # Get model output for the continuous params and binarized params
    if out == "peds":
        kappa, _ = model(params)  # Model output for continuous params (batch)
        kappa_binarized, _ = model(binary_params)  # Model output for binarized params (batch)
    else:
        kappa = model(params)
        kappa_binarized = model(binary_params)

    # Compute final loss for the whole batch and select best one
    loss_continuous = jnp.abs(kappa - target)
    loss_binarized = jnp.abs(kappa_binarized - target)

    best_idx_cont = jnp.argmin(loss_continuous)  # Find best continuous parameter set
    best_idx_bin = jnp.argmin(loss_binarized)  # Find best binarized parameter set

    # Select best parameter sets
    best_params = params[best_idx_cont]
    best_binary_params = binary_params[best_idx_bin]
    best_k = kappa[best_idx_cont]
    best_k_binarized = kappa_binarized[best_idx_bin]

    print(f"Best Found params: {best_params} with kappa= {best_k}")
    print(f"Binarized {best_binary_params} with kappa={best_k_binarized}")

    return best_binary_params, best_k_binarized"""




