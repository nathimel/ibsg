"""Collect all explored points, including those from simulation and the IB curve and get simple plot."""

import os
import hydra
import pandas as pd
from misc import util, vis


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config):
    util.set_seed(config.seed)

    # load datapaths
    cwd = os.getcwd()
    game_dir = cwd.replace(config.filepaths.simulation_subdir, "")
    curve_fn = os.path.join(game_dir, config.filepaths.curve_points_save_fn)
    fps = config.filepaths
    sim_fn = os.path.join(cwd, fps.simulation_points_save_fn)
    variant_fn = os.path.join(cwd, fps.variant_points_save_fn)

    plot_fn = os.path.join(cwd, fps.tradeoff_plot_fn)

    # load data
    curve_data = pd.read_csv(curve_fn)
    sim_data = pd.read_csv(sim_fn)
    variant_data = pd.read_csv(variant_fn)

    # get plot
    plot = vis.basic_tradeoff_plot(curve_data, sim_data, variant_data)

    util.save_plot(plot_fn, plot)


if __name__ == "__main__":
    main()