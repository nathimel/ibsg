"""Main driver script for running an experiment."""

import hydra
from misc import util
from simulation.driver import run_simulations


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config):
    util.set_seed(config.seed)

    # setup and run experiment
    runs = run_simulations(config)

    # (I[M:W], I[W:U], KL[M, M'], MSE)
    util.save_points_df(
        fn=config.filepaths.simulation_points_save_fn, 
        df=util.final_points_df(runs),
    )
    # save the final round emergent encoders to a csv
    util.save_final_encoders(config.filepaths.final_encoders_save_fn, runs)    

    # save trajectories from every run
    if config.simulation.trajectory:
        trajs_df = util.trajectories_df(runs)
        util.save_points_df(
            fn=config.filepaths.trajectory_points_save_fn, 
            df=trajs_df,
        )


if __name__ == "__main__":
    main()
