"""Collect all explored points, including those from simulation and the IB curve and get simple plot."""

import os
import hydra
import pandas as pd
from misc import util, vis


def generate_tradeoff_plots(
    ib_curve_data: pd.DataFrame,
    mse_curve_data: pd.DataFrame,
    point_data: pd.DataFrame,
    comp_acc_fn: str,
    comp_dist_fn: str,
    comp_mse_fn: str,
    trajectory_data: pd.DataFrame = None,
    nearest_opt_data: pd.DataFrame = None,
    variant_data: pd.DataFrame = None,
) -> None:
    """Generates and saves plots illustrating the complexity/accuracy trade-off."""
    # setup
    args = [
        ib_curve_data,
        point_data,
    ]
    kwargs = {
        "trajectory_data": trajectory_data,
        "variant_data": variant_data,
        "nearest_optimal_encoders_data": nearest_opt_data,
    }

    # plot
    ca_plot = vis.basic_tradeoff_plot(*args, **kwargs, y="accuracy")
    cd_plot = vis.basic_tradeoff_plot(*args, **kwargs, y="distortion")
    mse_plot = vis.basic_tradeoff_plot(mse_curve_data, point_data, **kwargs, y="mse")

    # save
    util.save_plot(comp_acc_fn, ca_plot)
    util.save_plot(comp_dist_fn, cd_plot)
    util.save_plot(comp_mse_fn, mse_plot)


def generate_encoder_plots(
    encoders_data: pd.DataFrame,
    faceted_lines_fn: str,
    faceted_tiles_fn: str,
    lines_dir: str,
    tiles_dir: str,
    individual_file_prefix: str,
):
    """Generates and saves plots of encoder distributions."""
    # ensure tile and line plots directories exist
    util.ensure_dir(tiles_dir)
    util.ensure_dir(lines_dir)

    # Encoders, faceted by run
    faceted_lines_plot = vis.faceted_encoders(encoders_data, "line")
    faceted_tiles_plot = vis.faceted_encoders(encoders_data, "tile")

    # Individual encoders
    tile_plots = vis.get_n_encoder_plots(encoders_data, "tile")
    line_plots = vis.get_n_encoder_plots(encoders_data, "line")

    # Save
    util.save_plot(faceted_lines_fn, faceted_lines_plot)
    util.save_plot(faceted_tiles_fn, faceted_tiles_plot)
    [
        util.save_plot(
            os.path.join(tiles_dir, f"{individual_file_prefix}_{i+1}.png"), plot
        )
        for i, plot in enumerate(tile_plots)
    ]
    [
        util.save_plot(
            os.path.join(lines_dir, f"{individual_file_prefix}_{i+1}.png"), plot
        )
        for i, plot in enumerate(line_plots)
    ]


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config):
    util.set_seed(config.seed)

    # helper function to load the right files
    cwd = os.getcwd()
    fps = config.filepaths
    fullpath = lambda fn: os.path.join(cwd, fn)

    sim_data = pd.read_csv(fullpath(fps.simulation_points_save_fn))

    ##########################################################################
    # Plot
    ##########################################################################

    curve_args = [
        pd.read_csv(util.get_bound_fn(config)),  # ib curve
        pd.read_csv(util.get_bound_fn(config, "mse")),  # mse curve, diff len!
    ]

    # Main simulation tradeoff
    generate_tradeoff_plots(
        *curve_args,
        sim_data,  # emergent points
        fullpath(fps.complexity_accuracy_plot_fn),
        fullpath(fps.complexity_distortion_plot_fn),
        fullpath(fps.complexity_mse_plot_fn),
        # kwargs
        trajectory_data=pd.read_csv(fullpath(fps.trajectory_points_save_fn))
        if config.simulation.trajectory
        else None,
        nearest_opt_data=pd.read_csv(fullpath(fps.nearest_optimal_points_save_fn)),
        variant_data=pd.read_csv(fullpath(fps.variant_points_save_fn))
        if config.simulation.variants
        else None,
    )

    # Tradeoff w sample-approximated encoders
    generate_tradeoff_plots(
        *curve_args,
        pd.read_csv(fullpath(fps.approximated_simulation_points_save_fn)),
        fullpath(fps.approximated_complexity_accuracy_plot_fn),
        fullpath(fps.approximated_complexity_distortion_plot_fn),
        fullpath(fps.approximated_complexity_mse_plot_fn),
        # comparison is exploratory, don't include kwargs yet
    )

    encoder_args = [
        fullpath(fps.encoder_line_plots_dir),
        fullpath(fps.encoder_tile_plots_dir),
    ]
    # Main simulation encoders
    generate_encoder_plots(
        util.load_encoders_as_df(fullpath(fps.final_encoders_save_fn)),
        fullpath(fps.encoders_faceted_lines_plot_fn),
        fullpath(fps.encoders_faceted_tiles_plot_fn),
        *encoder_args,
        "run",
    )
    # sample-approxd encoders
    generate_encoder_plots(
        util.load_encoders_as_df(fullpath(fps.approximated_encoders_save_fn)),
        fullpath(fps.approximated_encoders_faceted_lines_plot_fn),
        fullpath(fps.approximated_encoders_faceted_tiles_plot_fn),
        *encoder_args,
        "approxd_run",
    )
    # nearest IB-optimal encoders
    generate_encoder_plots(
        util.load_encoders_as_df(fullpath(fps.nearest_optimal_save_fn)),
        fullpath(fps.nearest_optimal_faceted_lines_plot_fn),
        fullpath(fps.nearest_optimal_faceted_tiles_plot_fn),
        *encoder_args,
        "nearest_opt",
    )

    # Efficiency loss plot
    # TODO: get a histogram of efficiency loss for each emergent system
    # TODO: also check opt efficiency for sanity, and then approx_data
    efficiency_plot = vis.basic_efficiency_plot(sim_data)
    util.save_plot(fullpath(fps.efficiency_plot_fn), efficiency_plot)


if __name__ == "__main__":
    main()
