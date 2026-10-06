import os
import numpy as np
from models.mlp import mlp
from models.peds import PEDS
from flax import nnx
import jax
from modules.params_utils import initialize_or_restore_params

import random
import numpy as np
from deap import base, creator, tools


def prng_key_to_int(key):
    """Convert a JAX PRNGKey to an integer."""
    return int(jax.random.randint(key, (), 0, 2*10))  # Convert to a sa

def genetic_algorithm(model, target, seed, n=25, pop_size=100, generations=40, cxpb=0.5, mutpb=0.2, tournsize=3, indpb=0.05):
    """
    Runs a genetic algorithm to optimize a design given a model and a target value.
    
    Args:
        model: Callable model that takes a batched population and returns predictions.
        target: Target value the algorithm tries to match.
        n: Number of elements in each individual (default: 25).
        pop_size: Size of the population (default: 100).
        generations: Number of generations to evolve (default: 40).
        cxpb: Crossover probability (default: 0.5).
        mutpb: Mutation probability (default: 0.2).
        tournsize: Tournament selection size (default: 3).
        indpb: Probability of flipping each bit in mutation (default: 0.05).
        seed: Random seed for reproducibility (default: None).
    
    Returns:
        np.array: Best individual found by the algorithm.
    """
    seed = seed.unwrap() if hasattr(seed, "unwrap") else seed # Extract JAX key if it's an nnx RngStream

    seed = prng_key_to_int(seed)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        
    # Check if classes already exist before creating them
    if "Fitness" not in creator.__dict__:
        creator.create("Fitness", base.Fitness, weights=(-1.0,))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", np.ndarray, fitness=creator.Fitness)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    def cxTwoPointCopy(ind1, ind2):
        size = len(ind1)
        cxpoint1, cxpoint2 = sorted(random.sample(range(1, size), 2))
        ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()
        return ind1, ind2
    
    toolbox.register("mate", cxTwoPointCopy)
    toolbox.register("mutate", tools.mutFlipBit, indpb=indpb)
    toolbox.register("select", tools.selTournament, tournsize=tournsize)
    
    def evaluate_batch(population):
        """Evaluate the entire population as a batch using the model."""
        batch = np.array(population)  # Shape: (pop_size, n)
        if isinstance(model, PEDS):
            batch = batch.reshape((batch.shape[0], 5, 5))  # Shape: (batch_size, 1, 5, 5)
        if isinstance(model, PEDS):
            kappas, _ = model(batch)  # Shape: (batch_size, kappas)
        
        else:
            kappas = model(batch)

       
        # Compute fitness as the absolute difference from the target
        return [(np.abs(kappa - target),) for kappa in kappas]

    toolbox.register("evaluate", evaluate_batch)
    
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1, similar=np.array_equal)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    
    # Evaluate the initial population as a batch
    fitnesses = evaluate_batch(pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    # Evolution loop
    for g in range(generations):
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Evaluate only individuals that need fitness recalculation
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        if invalid_ind:
            fitnesses = evaluate_batch(invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
        
        pop[:] = offspring
        record = stats.compile(pop)
        hof.update(pop)
    
    #hof_array = geom.strip("\"").strip("[]")  # Remove quotes and brackets
    #hof_array = np.array([int(x) for x in hof_array.split(", ")]) 

    hof_list = hof[0].tolist()
    hof_array = np.array(hof_list)

    if isinstance(model, PEDS):
        hof_array_resh = hof_array.reshape((1,5,5))
        k_pred, _ = model(hof_array_resh)
    else:
        hof_array_resh = hof_array.reshape((1,25))
        k_pred = model(hof_array_resh)
    
    return hof_array, k_pred

