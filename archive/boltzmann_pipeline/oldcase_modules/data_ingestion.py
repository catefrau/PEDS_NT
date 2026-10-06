import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def data_ingestion(rank, exp_name, filename, train_size, test_size, stratified, key=42):
    full_data = jnp.load("data/highfidelity/" + filename, allow_pickle=True)
    key = key.unwrap() if hasattr(key, "unwrap") else key  # Handle JAX key from nnx

    # Load data
    pores = jnp.asarray(full_data['pores'], dtype=jnp.float32)
    kappas = jnp.asarray(full_data['kappas'], dtype=jnp.float32)
    indices = jrandom.permutation(key, len(full_data['pores']))  # Shuffle indices

    total_size = train_size + test_size
    design_valid_size = 40  # Fixed additional validation set

    # Identify small-kappa samples
    small_indices = jnp.where(kappas < 80)[0]
    shuffled_small_indices = jrandom.permutation(key, small_indices.shape[0])

    if small_indices.shape[0] < test_size + design_valid_size:
        raise ValueError("Not enough small kappa samples to fulfill valid/design sets.")

    valid_indices = small_indices[shuffled_small_indices[:test_size]]
    design_valid_indices = small_indices[shuffled_small_indices[test_size:test_size + design_valid_size]]

    # Assign train and test based on stratification
    if stratified == "all":
        train_indices = indices[:train_size]
        test_indices = indices[train_size:total_size]

    elif stratified == "small->small":
        if small_indices.shape[0] < total_size + design_valid_size:
            raise ValueError("Not enough small kappa samples for train/test/valid/design sets.")

        train_indices = small_indices[shuffled_small_indices[:train_size]]
        test_indices = small_indices[shuffled_small_indices[train_size:total_size]]

    # Construct datasets
    fidelity = jnp.ones_like(pores)
    dataset_train = [pores[train_indices], kappas[train_indices]]
    dataset_test = [pores[test_indices], kappas[test_indices]]
    dataset_valid_small = [pores[valid_indices], kappas[valid_indices]]
    kappas_design_valid = kappas[design_valid_indices]


    if rank == 0:

        # Plot prob distribution of kappa for train and test and save it to train_results


        # Assuming pores[train_indices] and pores[test_indices] are already defined
        train_data = np.array(kappas[train_indices])
        test_data = np.array(kappas[test_indices])

        # Create the plot
        plt.figure(figsize=(8, 6))

        # Plot the KDE for the training data
        sns.kdeplot(train_data, label="Train Data", color="blue", fill=True)

        # Plot the KDE for the test data
        sns.kdeplot(test_data, label=f"Test Data", color="red", fill=True)

        # Add labels and title
        plt.title("Distribution of kappas", fontsize=14)
        plt.xlabel("Pore Values", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.legend()

        # Show the plot
        plt.savefig(f"experiments/{exp_name}/figures/kappa_traintest.png")
        plt.close()

    return dataset_train, dataset_test, dataset_valid_small, kappas_design_valid