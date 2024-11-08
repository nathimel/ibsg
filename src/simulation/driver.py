"""Functions for running simulations."""

import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig
from multiprocessing import Pool, cpu_count
from simulation.dynamics import dynamics_map, Dynamics
from game.game import Game
from misc import util


##############################################################################
# Helper functions for running experiments
##############################################################################


def run_simulations(config: DictConfig) -> list[Game]:
    """Run a simulation for multiple runs."""

    if config.simulation.multiprocessing:
        return run_simulations_multiprocessing(config)
    else:
        return [run_simulation(config) for _ in tqdm(range(config.simulation.num_runs))]


def run_simulations_multiprocessing(
    config: DictConfig,
) -> list[Game]:
    """Use multiprocessing apply_async to run multiple runs at once."""
    num_processes = cpu_count()
    if config.simulation.num_processes is not None:
        num_processes = config.simulation.num_processes

    with Pool(num_processes) as p:
        async_results = [
            p.apply_async(
                run_simulation,
                [config],  # args
            )
            for _ in range(config.simulation.num_runs)
        ]
        p.close()
        p.join()
    return [async_result.get() for async_result in tqdm(async_results)]


def run_simulation(config: DictConfig) -> Game:
    """Run one run of a simulation and return the resulting game."""

    # Load up the IB optimal encoders and betas
    betas = np.load(util.get_bound_fn(config, "betas"))
    optimal_encoders = np.load(util.get_bound_fn(config, "encoders"))  # ordered by beta

    dynamics: Dynamics = dynamics_map[config.simulation.dynamics.name]
    dynamics = dynamics(
        Game.from_hydra(config),
        **config.simulation.dynamics,
        ib_optimal_encoders=optimal_encoders,
        ib_optimal_betas=betas,
    )
    dynamics.run()
    return dynamics.game
