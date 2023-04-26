"""Main driver script for running an experiment."""

import hydra
from misc import util
from simulation.driver import run_trials
from simulation.variants import get_hypothetical_variants


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config):
    util.set_seed(config.seed)

    # setup and run experiment
    trials = run_trials(config)

    df_points = util.points_to_df([g.ib_points[-1] for g in trials])
    util.save_points_df(fn=config.filepaths.simulation_points_save_fn, df=df_points)

    if config.simulation.trajectory:
        # save trajectories from every trial
        trajs_df = util.trajectories_df(trials)
        util.save_points_df(fn=config.filepaths.trajectory_points_save_fn, df=trajs_df)

    variant_points_df = get_hypothetical_variants(trials, 100)
    util.save_points_df(fn=config.filepaths.variant_points_save_fn, df=variant_points_df)

if __name__ == "__main__":
    main()