"""Main driver script for running an experiment."""

import hydra
from misc import util
from simulation.driver import run_simulations
from simulation.variants import get_hypothetical_variants


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config):
    util.set_seed(config.seed)

    # setup and run experiment
    runs = run_simulations(config)

    # (I[M:W], I[W:U], KL[M, M'], MSE)
    points_df = util.final_points_df(runs)
    util.save_points_df(fn=config.filepaths.simulation_points_save_fn, df=points_df)

    # save trajectories from every run
    if config.simulation.trajectory:
        trajs_df = util.trajectories_df(runs)
        util.save_points_df(fn=config.filepaths.trajectory_points_save_fn, df=trajs_df)

    # save rotations of final encoders
    if config.simulation.variants:
        variant_points_df = get_hypothetical_variants(runs, 100)
        util.save_points_df(fn=config.filepaths.variant_points_save_fn, df=variant_points_df)

    # save the final encoders to a csv
    util.save_final_encoders(config.filepaths.final_encoders_save_fn, runs)

if __name__ == "__main__":
    main()