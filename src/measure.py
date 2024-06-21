"""Measure the efficiency of emergent systems w.r.t. the optimal IB encoders."""

import hydra
import os

import numpy as np
import pandas as pd

from omegaconf import DictConfig


from analysis import helpers
from analysis.simulation_measuring import (
    get_optimal_encoders_eu,
    measure_encoders,
)
from game.game import Game
from misc import util, vis



@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    util.set_seed(config.seed)
    g = Game.from_hydra(config)

    # helper function to load the right files
    cwd = os.getcwd()
    fps = config.filepaths
    fullpath = lambda fn: os.path.join(cwd, fn)

    ##########################################################################
    # Load
    ##########################################################################

    trajectory_encoders: dict[str, np.ndarray] = np.load(
        fps.trajectory_encoders_save_fn
    )
    trajectory_decoders: dict[str, np.ndarray] = np.load(
        fps.trajectory_decoders_save_fn
    )    
    steps_recorded: dict[str, np.ndarray] = np.load(fps.steps_recorded_save_fn)

    # Optimal Data
    betas = np.load(util.get_bound_fn(config, "betas"))
    optimal_encoders = np.load(util.get_bound_fn(config, "encoders"))  # ordered by beta
    curve_fn = util.get_bound_fn(config, "ib")
    curve_data = pd.read_csv(curve_fn)

    ##########################################################################
    # Get all measurements
    ##########################################################################    
    measurement = measure_encoders(
        g,
        trajectory_encoders,
        trajectory_decoders,
        steps_recorded,
        optimal_encoders,
        betas,
        curve_data,
    )
    traj_data: pd.DataFrame = measurement.trajectory_dataframe
    fitted_optima: np.ndarray = measurement.fitted_encoders
    optima_eus: list[float] = get_optimal_encoders_eu(g, optimal_encoders)

    ##########################################################################
    # Write data
    ##########################################################################

    util.save_points_df(
        fullpath(fps.trajectory_points_save_fn), 
        traj_data,
    )

    # Overwrite curve data with EU measurements
    curve_data[f"eu_gamma={config.game.discriminative_need_gamma}"] = optima_eus
    print("overwriting curve data with optima's expected utility for gamma")
    util.save_points_df(curve_fn, curve_data)


    # Inspect a single epsilon-fitted plot
    if config.simulation.inspect_eps:
        # Write nearest optimal data for plotting later
        opt_data = helpers.alt_encoders_to_df(fitted_optima, g.meaning_dists, g.prior)
        util.save_points_df(
            fullpath(fps.nearest_optimal_points_save_fn),
            opt_data,
        )

        # select (just one for now) emergent encoder
        idx = -1
        curve_data["eps"] = measurement.final_F_deviation
        single_encoder_data = traj_data.iloc[[idx]]
        single_optimal_data = opt_data.iloc[[idx]]
        plot = vis.single_eps_heatmap_tradeoff_plot(
            curve_data, single_encoder_data, single_optimal_data
        )
        util.save_plot(
            fullpath(fps.single_eps_inspect_plot_fn).replace("idx", str(idx + 1)),
            plot,
        )


if __name__ == "__main__":
    main()
