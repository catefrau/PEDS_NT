import os
import orbax.checkpoint as ocp
from flax import nnx
from pathlib import Path
from datetime import datetime
import shutil
import jax
import jax.numpy as jnp

def get_shapes(params):
    return jax.tree_util.tree_map(lambda x: x.shape if isinstance(x, jnp.ndarray) else None, params)

def filter_dropout(state):
    if 'generator' in state:
        generator_state = state['generator']
        if 'dropout' in generator_state:
            del generator_state['dropout']
    
    if 'dropout' in state:
        del state['dropout']
    return state

# Function for initializing or restoring model parameters
def initialize_or_restore_params(generator, model_name, rank, base_dir="weights"):
    """
    Initialize or restore model parameters based on the existence of a checkpoint.

    Parameters:
        generator: Callable
            The model generator function.
        model_name: str
            Name of the model for checkpointing.
        base_dir: str
            Base directory for storing weights.

    Returns:
        model: nnx.Model
            The initialized or restored model.
    """


    # Define paths
    base_dir = "experiments/" + base_dir + "/weights"
    ckpt_dir = Path(base_dir) / model_name
    
    # Ensure checkpoint directory exists
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Create the checkpointer
    checkpointer = ocp.StandardCheckpointer()

    # Generate the abstract model
    graphdef, abstract_state = nnx.split(generator)

    #abstract_state = filter_dropout(abstract_state)

    try:
        last_checkpoint = max(
            (d for d in ckpt_dir.iterdir() if d.is_dir()),  # Consider all directories
            key=lambda x: x.name.split('_')[-2] + '_' + x.name.split('_')[-1],  # Use the last two parts as timestamp
        )
        if rank ==0:
            print(f"Found last checkpoint: {last_checkpoint}")
        
        # Attempt to restore the state
        try:
            checkpoint_to_restore = os.path.abspath(last_checkpoint)
            state_restored = checkpointer.restore(checkpoint_to_restore , abstract_state)
            state_restored = filter_dropout(state_restored)
            
            if rank ==0:
                print(f"Successfully restored state from {last_checkpoint} and training curves")
        except Exception as e:
            if rank ==0:
                print(f"Restoration failed with error: {e}. Initializing new parameters.")
            state_restored = abstract_state  # Initialize new state

    except ValueError:
        # If no checkpoint is found
        if rank ==0:
            print("No checkpoints found. Initializing new parameters.")
        state_restored = abstract_state  # Initialize new state


    # Merge graph definition with restored or initialized state
    model = nnx.merge(graphdef, state_restored)

    return model, checkpointer

def save_params(exp_name, model_name, generator, checkpointer, epoch=None):

    _, state = nnx.split(generator)
    
    state = filter_dropout(state) # check if this one works
  
    #nnx.display(state)
    base_dir = os.path.abspath(f'experiments/{exp_name}/weights/{model_name}')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')  # e.g., "20241213_123456"
    if epoch is None:
        timestamp = f'final_' + timestamp
    else:
        timestamp = f'epoch{epoch}_' + timestamp
    
    new_checkpoint_dir = os.path.join(base_dir, timestamp)

    if os.path.exists(new_checkpoint_dir):
        shutil.rmtree(new_checkpoint_dir)  # Remove the directory and its contents

    #os.makedirs(new_checkpoint_dir, exist_ok=False)  # Ensure the directory does not already exist
    checkpointer.save(new_checkpoint_dir, state)

   
    
