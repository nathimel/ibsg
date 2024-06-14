"""Collect all explored points, including those from simulation and the IB curve and get simple plot."""

import os
import hydra
import pandas as pd
import numpy as np
from misc import util, vis
from game.game import Game


def generate_tradeoff_plots(
    ib_curve_data: pd.DataFrame,
    mse_curve_data: pd.DataFrame,
    comp_acc_fn: str,
    comp_dist_fn: str,
    comp_mse_fn: str,
    sim_data: pd.DataFrame = None,    
    trajectory_data: pd.DataFrame = None,
    nearest_opt_data: pd.DataFrame = None,
    variant_data: pd.DataFrame = None,
) -> None:
    """Generates and saves plots illustrating the complexity/accuracy trade-off."""
    # setup
    args = [
        ib_curve_data,
    ]
    kwargs = {
        "simulation_data": sim_data,
        "trajectory_data": trajectory_data,
        "variant_data": variant_data,
        "nearest_optimal_encoders_data": nearest_opt_data,
    }

    # plot

    if sim_data is not None:
        # with simulations
        del kwargs["simulation_data"]
        args.append(sim_data)
        ca_plot = vis.basic_tradeoff_plot(*args, **kwargs, y="accuracy")
        cd_plot = vis.basic_tradeoff_plot(*args, **kwargs, y="distortion")
        mse_plot = vis.basic_tradeoff_plot(mse_curve_data, sim_data, **kwargs, y="mse")
    else:
        # curve only
        ca_plot = vis.bound_only_plot(ib_curve_data, y="accuracy")
        cd_plot = vis.bound_only_plot(ib_curve_data, y="distortion")
        mse_plot = vis.bound_only_plot(mse_curve_data, y="mse")

    # save
    util.save_plot(comp_acc_fn, ca_plot)
    util.save_plot(comp_dist_fn, cd_plot)
    util.save_plot(comp_mse_fn, mse_plot)


def generate_encoder_plots(
    encoders: np.ndarray,
    prior: np.ndarray,
    faceted_lines_fn: str,
    faceted_tiles_fn: str,
    lines_dir: str,
    tiles_dir: str,
    individual_file_prefix: str,
    title_nums: np.ndarray,
    facet_runs: bool = False,
):
    """Generates and saves plots of encoder distributions."""
    # ensure tile and line plots directories exist
    util.ensure_dir(tiles_dir)
    util.ensure_dir(lines_dir)

    encoders_data = util.encoders_to_df(encoders, labels=title_nums, col=individual_file_prefix)

    # Encoders, faceted by run
    if facet_runs and len(encoders_data["run"].value_counts().to_dict()) - 1:  # > one run
        faceted_lines_plot = vis.faceted_encoders(encoders_data, "line")
        faceted_tiles_plot = vis.faceted_encoders(encoders_data, "tile")
        util.save_plot(faceted_lines_fn, faceted_lines_plot)
        util.save_plot(faceted_tiles_fn, faceted_tiles_plot)


    # NOTE: I'm commenting out encoder plots for now because I never need them and its slow to generate them
    # Individual encoders
    # tile_plots = vis.get_n_encoder_plots(encoders_data, "tile", 
    # item_key=individual_file_prefix, title_var="t",)

    # line_plots = vis.get_n_encoder_plots(encoders_data, "line", item_key=individual_file_prefix,) #REPLACED WITH BELOW

    # Centroid lineplot
    line_figs = vis.get_n_centroid_plots(
        encoders, 
        prior,
        item_key=individual_file_prefix,
        title_nums=title_nums,
        title_var="t",
    )


    # Save each
    # [
    #     util.save_plot(
    #         os.path.join(tiles_dir, f"{individual_file_prefix}_{i+1}.png"), plot
    #     )
    #     for i, plot in enumerate(tile_plots)
    # ]
    [
        util.save_fig(
            os.path.join(lines_dir, f"{individual_file_prefix}_{i+1}.png"), fig
        )
        for i, fig in enumerate(line_figs)
    ]


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config):
    util.set_seed(config.seed)

    # helper function to load the right files
    cwd = os.getcwd()
    fps = config.filepaths
    fullpath = lambda fn: os.path.join(cwd, fn)

    sim_fn = fullpath(fps.simulation_points_save_fn)
    sim_data = pd.read_csv(sim_fn) if os.path.exists(sim_fn) else None

    g = Game.from_hydra(config)

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
        fullpath(fps.complexity_accuracy_plot_fn),
        fullpath(fps.complexity_distortion_plot_fn),
        fullpath(fps.complexity_mse_plot_fn),
        # kwargs
        sim_data=sim_data, # emergent points        
        # trajectory_data=pd.read_csv(fullpath(fps.trajectory_points_save_fn))
        # if config.simulation.trajectory
        # else None,
        # nearest_opt_data=pd.read_csv(fullpath(fps.nearest_optimal_points_save_fn)),
        # variant_data=pd.read_csv(fullpath(fps.variant_points_save_fn))
        # if config.simulation.variants
        # else None,
    )

    # Tradeoff w sample-approximated encoders
    if config.simulation.approximate_encoders:
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
    encoders_fn = fullpath(fps.final_encoders_save_fn)
    title_nums = range(config.simulation.num_runs)
    if os.path.exists(encoders_fn):
        generate_encoder_plots(
            np.load(encoders_fn),
            g.prior,
            fullpath(fps.encoders_faceted_lines_plot_fn),
            fullpath(fps.encoders_faceted_tiles_plot_fn),
            *encoder_args,
            "run",
            title_nums,
        )
    # sample-approxd encoders
    approxd_encoders_fn = fullpath(fps.approximated_encoders_save_fn)
    if config.simulation.approximate_encoders and os.path.exists(approxd_encoders_fn):
        generate_encoder_plots(
            np.load(approxd_encoders_fn),
            g.prior,
            fullpath(fps.approximated_encoders_faceted_lines_plot_fn),
            fullpath(fps.approximated_encoders_faceted_tiles_plot_fn),
            *encoder_args,
            "approxd_run",
        )
    # nearest IB-optimal encoders
    nearopt_encoders_fn = fullpath(fps.nearest_optimal_save_fn)
    if os.path.exists(nearopt_encoders_fn):
        # pass
        # Can't work on the below until we sort out the epsilon fit
        generate_encoder_plots(
            np.load(nearopt_encoders_fn),
            g.prior,
            fullpath(fps.nearest_optimal_faceted_lines_plot_fn),
            fullpath(fps.nearest_optimal_faceted_tiles_plot_fn),
            *encoder_args,
            "nearest_opt",
            title_nums,
        )


if __name__ == "__main__":
    main()
