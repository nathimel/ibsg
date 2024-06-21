"""Measure the efficiency of emergent systems w.r.t. the optimal IB encoders."""

import hydra
import os

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from omegaconf import DictConfig
from ultk.effcomm.rate_distortion import ib_encoder_to_point, ib_optimal_decoder

from analysis import helpers
from game.game import Game
from misc import util, vis
from tqdm import tqdm


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    util.set_seed(config.seed)
    g = Game.from_hydra(config)

    # helper function to load the right files
    cwd = os.getcwd()
    fps = config.filepaths
    fullpath = lambda fn: os.path.join(cwd, fn)

    sim_fn = fullpath(fps.simulation_points_save_fn)

    ##########################################################################
    # Load
    ##########################################################################

    sim_data = pd.read_csv(
        fullpath(fps.simulation_points_save_fn)
    )  # we will update with gNID and eps
    emergent_encoders = np.load(fullpath(fps.final_encoders_save_fn))
    betas = np.load(util.get_bound_fn(config, "betas"))
    optimal_encoders = np.load(
        util.get_bound_fn(config, "encoders")
    )  # ordered by beta
    curve_fn = util.get_bound_fn(config, "ib")
    curve_data = pd.read_csv(curve_fn)

    # We will analyze, for each point in each trajectory, its dist to curve
    if config.simulation.trajectory:
        traj_fn = fullpath(fps.trajectory_points_save_fn)
        traj_data = pd.read_csv(traj_fn)

    ##########################################################################
    # Record expected utility of ib optimal encoders w.r.t the specific game
    ##########################################################################

    optima_eus = []
    for encoder in tqdm(optimal_encoders, desc="computing eu of optima"):
        bayesian_decoder = ib_optimal_decoder(encoder, g.prior, g.meaning_dists)
        optimal_team = g.meaning_dists @ encoder @ bayesian_decoder @ g.meaning_dists
        # Expected Utility, relative to discriminative_need
        eu_gamma = np.sum(g.prior * (optimal_team * g.utility))
        optima_eus.append(eu_gamma)

    ##########################################################################
    # Measure trajectory points' Euclidean distances to curve
    ##########################################################################

    # Measure Euclidean dist of each language to any optimal curve point
    traj_points = traj_data[["complexity", "accuracy"]].values
    curve_points = curve_data[["complexity", "accuracy"]].values
    distances = cdist(traj_points, curve_points) # shape `(traj_pts, curve_pts)`
    # For each traj_point, get the minimum dist to any curve_point
    min_distances = np.min(distances, axis=1)

    ##########################################################################
    # Measure efficiency of emergent encoders
    ##########################################################################
    
    # Compute min efficiency loss and gNID for each final emergent encoder
    
    # Measure efficiency loss 1/beta (F_emergent - F_optimal) to any F_optimal
    # F_[q] = em_complexity - em_acc
    # eps = 1/beta * ( F_[q] - F_[q*] )
    F_opt = curve_points[:,0] - betas * curve_points[:,1]
    fitted_betas = []
    fitted_epsilon = []    
    fitted_encoders = []
    for em in tqdm(emergent_encoders, desc="fitting final encoders"):
        comp, acc, _ = ib_encoder_to_point(g.prior, g.meaning_dists, em)
        F_em = comp - betas * acc

        # Do we have F_[q] >= F_[q*] for all q?
        F_em_deviation = (F_em - F_opt) / betas
        min_ind = np.argmin(F_em_deviation)

        beta_em = betas[min_ind]
        epsilon_em = np.min(F_em_deviation)
        fitted_opt = optimal_encoders[min_ind]

        fitted_betas.append(beta_em)
        fitted_epsilon.append(epsilon_em)
        fitted_encoders.append(fitted_opt)

        if epsilon_em < 0:
            breakpoint()
            raise Exception(f"A final encoder has negative efficiency loss: epsilon={epsilon_em}.")


    # Save the fitted optimal encoder to each emergent encoder
    util.save_ndarray(fullpath(fps.nearest_optimal_save_fn), np.array(fitted_encoders))

    ##########################################################################
    # Measure trajectory points' efficiency loss to any encoder (like with Euclidean distance to curve, above)
    ##########################################################################  
    
    # Repeat for all trajectory points, not just final
    min_eps = []
    min_beta = []
    for (traj_complexity, traj_accuracy) in tqdm(
        traj_points, 
        desc="finding minimum efficiency loss per trajectory point",
    ):
        F_traj = traj_complexity - betas * traj_accuracy
        F_traj_deviation = (F_traj - F_opt) / betas

        min_ind = np.argmin(F_traj_deviation)

        beta_traj = betas[min_ind]
        epsilon_traj = np.min(F_traj_deviation)

        min_eps.append(epsilon_traj)
        min_beta.append(beta_traj)

        if epsilon_traj < 0:
            raise Exception(f"A trajectory encoder has negative efficiency loss: epsilon={epsilon_traj}.")

    ##########################################################################
    # Write data
    ##########################################################################

    # Overwrite curve data with EU measurements
    gamma = config.game.discriminative_need_gamma
    curve_data[f"eu_gamma={gamma}"] = optima_eus
    print("overwriting curve data with optima's expected utility for gamma")
    util.save_points_df(curve_fn, curve_data)

    # Overwrite simulation data
    sim_data["min_beta"] = fitted_betas
    sim_data["min_epsilon"] = fitted_epsilon
    util.save_points_df(sim_fn, sim_data)

    # Write nearest optimal data for plotting later
    opt_data = helpers.alt_encoders_to_df(fitted_encoders, g.meaning_dists, g.prior)
    util.save_points_df(
        fullpath(fps.nearest_optimal_points_save_fn),
        opt_data,
    )

    # Add the min_distances column to trajectory points
    traj_data["min_distance_to_curve"] = min_distances
    # And likewise for epsilon, beta
    traj_data["min_epsilon"] = min_eps
    traj_data["min_beta"] = min_beta
    util.save_points_df(traj_fn, traj_data)

    # Inspect a single epsilon-fitted plot
    if config.simulation.inspect_eps:
        # select (just one for now) emergent encoder
        idx = config.simulation.inspect_eps_encoder - 1
        assert idx >= 0, "Use 1-indexing for fitted epsilon encoder inspection."
        assert idx <= config.simulation.num_runs, "Inspection index must be less than num_runs"
        # last_encoder_eps = fitted_epsilon[idx]
        curve_data["eps"] = F_em_deviation
        single_encoder_data = sim_data.iloc[[idx]]
        single_optimal_data = opt_data.iloc[[idx]]
        plot = vis.single_eps_heatmap_tradeoff_plot(
            curve_data, single_encoder_data, single_optimal_data
        )
        util.save_plot(
            fullpath(fps.single_eps_inspect_plot_fn).replace("idx", str(idx+1)), 
            plot,
        )

    # Approximate encoders via sampling
    if config.simulation.approximate_encoders:
        # approximate
        sampled_encoders = np.stack(
            [
                helpers.finite_sample_encoder(
                    enc, config.simulation.num_approximation_samples
                )
                for enc in emergent_encoders
            ]
        )
        # measure
        approximate_data = helpers.alt_encoders_to_df(
            sampled_encoders, g.meaning_dists, g.prior
        )
        # save
        util.save_tensor(fullpath(fps.approximated_encoders_save_fn), sampled_encoders)
        util.save_points_df(
            fn=config.filepaths.approximated_simulation_points_save_fn,
            df=approximate_data,
        )

    # Variants via rotations of emergent encoders
    if config.simulation.variants:
        util.save_points_df(
            fn=config.filepaths.variant_points_save_fn,
            df=helpers.alt_encoders_to_df(
                helpers.hypothetical_variants(
                    emergent_encoders,
                    config.simulation.num_variants,
                ),
                g.meaning_dists,
                g.prior,
            ),
        )


if __name__ == "__main__":
    main()
