"""Collect all explored points, including those from simulation and the IB curve and get simple plot."""

import os
import hydra
import pandas as pd
from misc import util, vis


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config):
    util.set_seed(config.seed)

    # helper function to load the right files
    cwd = os.getcwd()
    fps = config.filepaths
    fullpath = lambda fn: os.path.join(cwd, fn)

    # load datapaths
    curve_fn = util.get_curve_fn(config)
    mse_curve_fn = util.get_curve_fn(config, "mse")


    sim_fn = fullpath(fps.simulation_points_save_fn)
    traj_fn = fullpath(fps.trajectory_points_save_fn)
    # variant_fn = fullpath(fps.variant_points_save_fn)

    comp_acc_plot_fn = fullpath(fps.complexity_accuracy_plot_fn)
    comp_dist_plot_fn = fullpath(fps.complexity_distortion_plot_fn)
    comp_mse_plot_fn = fullpath(fps.complexity_mse_plot_fn)

    # load data
    sim_data = pd.read_csv(sim_fn)
    # variant_data = pd.read_csv(variant_fn)
    curve_data = pd.read_csv(curve_fn)
    mse_curve_data = pd.read_csv(mse_curve_fn) # diff len than curve_data!

    traj_data = None
    if config.simulation.trajectory:
        traj_data = pd.read_csv(traj_fn)

    plot_args = [curve_data, sim_data]

    plot_kwargs = {
        "trajectory_data": traj_data, 
        "variant_data": None,
        }

    plot = vis.basic_tradeoff_plot(*plot_args, **plot_kwargs, y="accuracy")
    cc_plot = vis.basic_tradeoff_plot(*plot_args, **plot_kwargs, y="distortion")
    mse_plot = vis.basic_tradeoff_plot(mse_curve_data, sim_data, **plot_kwargs, y="mse")
    util.save_plot(comp_acc_plot_fn, plot)
    util.save_plot(comp_dist_plot_fn, cc_plot)
    util.save_plot(comp_mse_plot_fn, mse_plot)


if __name__ == "__main__":
    main()
