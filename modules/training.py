import time
import sys
from mpi4py import MPI

import jax.numpy as jnp
import numpy as np

import optax
from flax import nnx
import jax

from modules.params_utils import save_params
from modules.training_utils import data_loader, print_generated, update_and_check_grads, clip_gradients, plot_update_learning_curves, plot_loss_curves, choose_schedule, accumulate_gradients, distribute_dataset, mpi_allreduce_gradients, update_curves
from models.peds import PEDS



def train_model(exp_name, model_name,
                dataset_train, dataset_test, 
                model,
                learn_rate_max, learn_rate_min, schedule,  epochs, batch_size,
                checkpointer): 

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # Current process ID
    size = comm.Get_size()  # Total number of processes

    n_past_epoch = update_curves(model_name)

    # Scheduler optimizer
    lr_schedule = choose_schedule(rank, schedule, learn_rate_min, learn_rate_max, epochs, exp_name, n_past_epoch)
    optimizer = nnx.Optimizer(model, optax.adam(lr_schedule))

    def train_step(batch_local, epoch, batch_n, rank):

        pores, kappas = batch_local
        
        if isinstance(model, PEDS):
            def loss_fn(model):
                kappa_pred, conductivity_res = model(pores, True) # takes pore geometry and outputs PEDS estimate
                if (epoch+1+n_past_epoch) in [1, 2, 4, 7, 11, 17, 26, 39, 58, 86, 130, 195, 293, 440, 660, 999] and batch_n == 0 and rank == 0:
                    print_generated(model, pores, conductivity_res, epoch+1+n_past_epoch, model_name, exp_name, kappa_pred, kappas) # change here
                residuals = kappa_pred - kappas
                residuals_weighted = residuals
                loss = jnp.sum(residuals_weighted**2)
                return loss
        else:
            def loss_fn(model):
                pores_reshaped = jnp.reshape(pores, (pores.shape[0], 25))
                kappa_pred = model(pores_reshaped, True)
                kappa_pred = jnp.squeeze(kappa_pred, -1)
                residuals = kappa_pred - kappas
                residuals_weighted = residuals
                loss = jnp.sum(residuals_weighted**2)
                return loss
            
        # Compute loss and gradients
        loss, grads = nnx.value_and_grad(loss_fn)(model) # nnx.value_and_grad is JAX's automatic differentiation engine CHECK HOW THIS PASSES THROUGH THE SOLVER

        return loss, grads

    # OCCHIO A QUESTO... COMM.ALL_REDUCE NON STA VENENDO USATO CORRETTAMENTE PENSO... OCCHIO ALLA VALIDATION CHE ORA SEMBRA ANDARE 
    def valid_step(dataset, batch_size, comm, valid_size):

        # 2000, 5, 5 ---> 10, 200, 5, 5 ne avanzano 2 dopo le prime 8 diversamente da
        # 8000, 5, 5 ---> 40, 200, 5, 5
        total_val_loss = 0.0
        total_error_perc  = 0.0

        for en, val_batch in enumerate(data_loader(*dataset, batch_size=batch_size)):
            val_pores, val_kappas = val_batch

            if isinstance(model, PEDS):
                kappa_val, _ = model(val_pores) # here
            else:
                val_pores_reshaped = jnp.reshape(val_pores, (val_pores.shape[0], 25))
                kappa_val = model(val_pores_reshaped)
                kappa_val = jnp.squeeze(kappa_val, -1)

            local_res = (kappa_val - val_kappas)**2
            local_error = jnp.abs(kappa_val - val_kappas)*100.0/jnp.abs(val_kappas) 

            # To be adapted to 
            total_val_loss += comm.allreduce(jnp.sum(local_res), op=MPI.SUM)
            total_error_perc += comm.allreduce(jnp.sum(local_error), op=MPI.SUM)
        
        total_val_loss = total_val_loss / valid_size
        total_error_perc = total_error_perc / valid_size
        
        return total_val_loss, total_error_perc

    if rank == 0:
        print(f"Past epochs: {n_past_epoch}")
        sys.stdout.flush()

    epoch_losses = np.zeros(epochs) 
    valid_losses = np.zeros(epochs)
    valid_perc_losses = np.zeros(epochs)
    epoch_times = np.zeros(epochs)

    # Shard training and validation datasets
    dataset_train_local = distribute_dataset(dataset_train, rank, size)  # splits the data so each rank (process) handles a portion
    dataset_test_local = distribute_dataset(dataset_test, rank, size)

    for epoch in range(epochs): # one update of weights (one pass of data through model)

        epoch_time = time.time()
        grads = None
        total_loss = 0.0 

        for en, batch_local in enumerate(data_loader(*dataset_train_local, batch_size=batch_size)): # data_loader yields mini-batches of (pores, kappas) pairs 

            # Compute loss and gradients locally (for this batch, epoch, and process)
            local_loss, local_grads = train_step(batch_local, epoch, en, rank)
            
            # Accumulate loss across ranks
            total_loss += comm.allreduce(local_loss, op=MPI.SUM)

            grads = accumulate_gradients(grads, mpi_allreduce_gradients(local_grads, comm))
        
        # Update of parameters! # should we impose it to be done when rank ==1?
        
        optimizer.update(grads) # where the weights and biases change

        if epoch % 50 == 0:
            jax.clear_caches()

        # Validation step
        avg_val_loss, total_loss_perc = valid_step(dataset_test_local, batch_size, comm, valid_size = dataset_test[0].shape[0])

        avg_loss = total_loss / dataset_train[0].shape[0]
        epoch_times[epoch] = time.time() - epoch_time
        epoch_losses[epoch] = avg_loss
        valid_losses[epoch] = avg_val_loss
        valid_perc_losses[epoch] = total_loss_perc

        if rank == 0 and (epoch+1)%25 == 0:

            print(f"Epoch {epoch+1+n_past_epoch}/{epochs+n_past_epoch}, Training Losses: [{avg_loss:.2f}] , Validation Losses: [{avg_val_loss:.2f}, {total_loss_perc:.2f}%], Epoch time: {time.time() - epoch_time:.2f}s")
            sys.stdout.flush() 

        if (epoch+1) % 500 == 0 and rank == 0:
            plot_update_learning_curves(exp_name, model_name, n_past_epoch, epoch, epoch_times, epoch_losses, valid_losses, valid_perc_losses, schedule, learn_rate_max, learn_rate_min)
            save_params(exp_name, model_name, model, checkpointer)


    if rank == 0:
        
        plot_update_learning_curves(exp_name, model_name, n_past_epoch, epoch, epoch_times, epoch_losses, valid_losses, valid_perc_losses, schedule, learn_rate_max, learn_rate_min)
        # visual evolution of the losses 
        plot_loss_curves(exp_name, model_name, epoch_losses, valid_losses, valid_perc_losses)
        save_params(exp_name, model_name, model, checkpointer)

        return avg_loss.item(), avg_val_loss.item(), total_loss_perc.item()
       
        
    








    