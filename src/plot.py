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
    curve_fn = util.get_bound_fn(config)
    mse_curve_fn = util.get_bound_fn(config, "mse")

    sim_fn = fullpath(fps.simulation_points_save_fn)
    traj_fn = fullpath(fps.trajectory_points_save_fn)
    variant_fn = fullpath(fps.variant_points_save_fn)
    nearest_optimal_points_fn = fullpath(fps.nearest_optimal_points_save_fn)

    encoders_fn = fullpath(fps.final_encoders_save_fn)
    nearest_optimal_fn = fullpath(fps.nearest_optimal_save_fn)

    comp_acc_plot_fn = fullpath(fps.complexity_accuracy_plot_fn)
    comp_dist_plot_fn = fullpath(fps.complexity_distortion_plot_fn)
    comp_mse_plot_fn = fullpath(fps.complexity_mse_plot_fn)
    efficiency_plot_fn = fullpath(fps.efficiency_plot_fn)

    faceted_lines_plot_fn = fullpath(fps.encoders_faceted_lines_plot_fn)
    faceted_tiles_plot_fn = fullpath(fps.encoders_faceted_tiles_plot_fn)
    faceted_lines_opt_plot_fn = fullpath(fps.nearest_optimal_faceted_lines_plot_fn)
    faceted_tiles_opt_plot_fn = fullpath(fps.nearest_optimal_faceted_tiles_plot_fn)

    tile_plots_dir = fullpath(fps.encoder_tile_plots_dir)
    line_plots_dir = fullpath(fps.encoder_line_plots_dir)
    util.ensure_dir(tile_plots_dir)
    util.ensure_dir(line_plots_dir)

    # load data
    sim_data = pd.read_csv(sim_fn)
    variant_data = None
    if config.simulation.variants:
        variant_data = pd.read_csv(variant_fn)
    curve_data = pd.read_csv(curve_fn)
    mse_curve_data = pd.read_csv(mse_curve_fn) # diff len than curve_data!
    nearest_optimal_points_data = pd.read_csv(nearest_optimal_points_fn)

    encoders_data = util.load_encoders_as_df(encoders_fn)
    nearest_optimal_data = util.load_encoders_as_df(nearest_optimal_fn)

    traj_data = None
    if config.simulation.trajectory:
        traj_data = pd.read_csv(traj_fn)

    plot_args = [curve_data, sim_data]

    plot_kwargs = {
        "trajectory_data": traj_data, 
        # "variant_data": variant_data,
        "nearest_optimal_encoders_data": nearest_optimal_points_data,
    }
    # breakpoint()
    # Trade-off plots
    ca_plot = vis.basic_tradeoff_plot(*plot_args, **plot_kwargs, y="accuracy")
    cd_plot = vis.basic_tradeoff_plot(*plot_args, **plot_kwargs, y="distortion")
    mse_plot = vis.basic_tradeoff_plot(mse_curve_data, sim_data, **plot_kwargs, y="mse")

    # Efficiency loss plot
    # TODO: get a histogram of efficiency loss for each emergent system
    efficiency_plot = vis.basic_efficiency_plot(sim_data)

    # Encoders, faceted by run
    faceted_lines_plot = vis.faceted_encoders(encoders_data, "line")
    faceted_tiles_plot = vis.faceted_encoders(encoders_data, "tile")
    faceted_lines_opt = vis.faceted_encoders(nearest_optimal_data, "line")
    faceted_tiles_opt = vis.faceted_encoders(nearest_optimal_data, "tile")

    tile_plots = vis.get_n_encoder_plots(encoders_data, "tile")
    line_plots = vis.get_n_encoder_plots(encoders_data, "line")
    opt_tiles = vis.get_n_encoder_plots(nearest_optimal_data, "tile")
    opt_lines = vis.get_n_encoder_plots(nearest_optimal_data, "line")

    # TODO: get side-by-side plots for each emergent and its nearest optimal

    util.save_plot(comp_acc_plot_fn, ca_plot)
    util.save_plot(comp_dist_plot_fn, cd_plot)
    util.save_plot(comp_mse_plot_fn, mse_plot)

    util.save_plot(efficiency_plot_fn, efficiency_plot)

    util.save_plot(faceted_lines_plot_fn, faceted_lines_plot)
    util.save_plot(faceted_tiles_plot_fn, faceted_tiles_plot)

    util.save_plot(faceted_lines_opt_plot_fn, faceted_lines_opt)
    util.save_plot(faceted_tiles_opt_plot_fn, faceted_tiles_opt)

    [util.save_plot(os.path.join(tile_plots_dir, f"run_{i+1}.png"), plot) for i, plot in enumerate(tile_plots)]
    [util.save_plot(os.path.join(line_plots_dir, f"run_{i+1}.png"), plot) for i, plot in enumerate(line_plots)]

    [util.save_plot(os.path.join(tile_plots_dir, f"near_opt_{i+1}.png"), plot) for i, plot in enumerate(opt_tiles)]
    [util.save_plot(os.path.join(line_plots_dir, f"near_opt_{i+1}.png"), plot) for i, plot in enumerate(opt_lines)]

if __name__ == "__main__":
    main()
