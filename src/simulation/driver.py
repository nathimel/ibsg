"""Functions for running simulations."""

import torch

from tqdm import tqdm
from omegaconf import DictConfig
from multiprocessing import Pool, cpu_count
from simulation.dynamics import dynamics_map
from game.game import Game


##############################################################################
# Helper functions for running experiments
##############################################################################


def run_simulations(config: DictConfig) -> list[Game]:
    """Run a simulation for multiple runs."""

    population_init_gammas = torch.linspace(
        config.simulation.dynamics.population_init_minalpha, 
        config.simulation.dynamics.population_init_maxalpha, 
        config.simulation.num_runs,
        )

    if config.simulation.multiprocessing:
        return run_simulations_multiprocessing(config, population_init_gammas)
    else:
        return [
            run_simulation(config, population_init_gammas[run]) for run in tqdm(range(config.simulation.num_runs))
        ]


def run_simulations_multiprocessing(config: DictConfig, population_init_gammas: torch.Tensor) -> list[Game]:
    """Use multiprocessing apply_async to run multiple runs at once."""
    num_processes = cpu_count()
    if config.simulation.num_processes is not None:
        num_processes = config.simulation.num_processes

    with Pool(num_processes) as p:
        async_results = [
            p.apply_async(
            run_simulation, 
            [config, population_init_gammas[run]], # args
            )
            for run in range(config.simulation.num_runs)
        ]
        p.close()
        p.join()
    return [async_result.get() for async_result in tqdm(async_results)]


def run_simulation(config: DictConfig, population_init_gamma: float) -> Game:
    """Run one run of a simulation and return the resulting game."""
    dynamics = dynamics_map[config.simulation.dynamics.name]
    dynamics = dynamics(
        Game.from_hydra(config), 
        **config.simulation.dynamics, 
        use_decoder = config.simulation.use_decoder, 
        population_init_gamma = population_init_gamma,
        )
    dynamics.run()
    return dynamics.game
