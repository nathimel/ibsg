"""Main driver script for running an experiment."""

import hydra
from misc import util
from simulation.driver import run_simulations


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config):
    util.set_seed(config.seed)

    # setup and run experiment
    runs = run_simulations(config)

    # save trajectories from every run
    util.save_all_agents(
        config.filepaths.trajectory_encoders_save_fn, 
        config.filepaths.trajectory_decoders_save_fn,
        runs,
    )


if __name__ == "__main__":
    main()
