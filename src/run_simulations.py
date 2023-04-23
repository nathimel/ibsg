"""Main driver script for running an experiment."""

import hydra
from misc import util
from simulation.driver import run_trials
from simulation.data import trials_to_df


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config):
    util.set_seed(config.seed)

    # setup and run experiment
    trials = run_trials(config)

    df_points = trials_to_df(trials)
    util.save_points_df(fn=config.filepaths.simulation_points_save_fn, df=df_points)

if __name__ == "__main__":
    main()