"""Collect all explored points, including those from simulation and the IB curve and get simple plot."""

import os
import hydra
import pandas as pd
from misc import util, vis


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config):
    util.set_seed(config.seed)

    # load datapaths
    curve_fn = util.get_curve_fn(config)
    ub_curve_fn = util.get_curve_fn(config, "ub")
    cwd = os.getcwd()
    fps = config.filepaths

    sim_fn = os.path.join(cwd, fps.simulation_points_save_fn)
    traj_fn = os.path.join(cwd, fps.trajectory_points_save_fn)
    # variant_fn = os.path.join(cwd, fps.variant_points_save_fn)

    plot_fn = os.path.join(cwd, fps.tradeoff_plot_fn)
    comm_cost_plot_fn = os.path.join(cwd, fps.comm_cost_plot_fn)
    ub_plot_fn = os.path.join(cwd, fps.ub_plot_fn)

    # load data
    sim_data = pd.read_csv(sim_fn)
    # variant_data = pd.read_csv(variant_fn)
    curve_data = pd.read_csv(curve_fn)
    ub_curve_data = pd.read_csv(ub_curve_fn) # diff len than curve_data!

    traj_data = None
    if config.simulation.trajectory:
        traj_data = pd.read_csv(traj_fn)

    plot_args = [curve_data, sim_data]

    plot_kwargs = {
        "trajectory_data": traj_data, 
        "variant_data": None,
        }

    # get (complexity, accuracy) plot
    plot = vis.basic_tradeoff_plot(*plot_args, **plot_kwargs, y="accuracy")
    cc_plot = vis.basic_tradeoff_plot(*plot_args, **plot_kwargs, y="distortion")
    ub_plot = vis.basic_tradeoff_plot(ub_curve_data, sim_data, **plot_kwargs, y="mse")
    util.save_plot(plot_fn, plot)
    util.save_plot(comm_cost_plot_fn, cc_plot)
    util.save_plot(ub_plot_fn, ub_plot)


if __name__ == "__main__":
    main()
