"""Functions for running one trial of a simulation."""

from tqdm import tqdm
from omegaconf import DictConfig
from multiprocessing import Pool, cpu_count
from simulation.dynamics import dynamics_map
from game.game import Game


##############################################################################
# Helper functions for running experiments
##############################################################################


def run_trials(config: DictConfig) -> list[Game]:
    """Run a simulation for multiple trials."""
    if config.simulation.multiprocessing:
        return run_trials_multiprocessing(config)
    else:
        return [
            run_simulation(config) for _ in tqdm(range(config.simulation.num_trials))
        ]


def run_trials_multiprocessing(config: DictConfig) -> list[Game]:
    """Use multiprocessing apply_async to run multiple trials at once."""
    num_processes = cpu_count()
    if config.simulation.num_processes is not None:
        num_processes = config.simulation.num_processes

    with Pool(num_processes) as p:
        async_results = [
            p.apply_async(run_simulation, [config])
            for _ in range(config.simulation.num_trials)
        ]
        p.close()
        p.join()
    return [async_result.get() for async_result in tqdm(async_results)]


def run_simulation(config: DictConfig) -> Game:
    """Run one trial of a simulation and return the resulting game."""
    dynamics = dynamics_map[config.simulation.dynamics.name]
    dynamics = dynamics(
        Game.from_hydra(config), 
        **config.simulation.dynamics, 
        use_decoder = config.simulation.use_decoder,
        )
    dynamics.run()
    return dynamics.game
