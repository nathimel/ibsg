"""Measure the efficiency of emergent systems w.r.t. the optimal IB encoders via gNID."""

import hydra
import os
import torch

import pandas as pd
from scipy.spatial.distance import cdist

from omegaconf import DictConfig
from rdot.information import gNID

from analysis import efficiency
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

    sim_fn = fullpath(fps.simulation_points_save_fn)

    ##########################################################################
    # Load
    ##########################################################################

    sim_data = pd.read_csv(
        fullpath(fps.simulation_points_save_fn)
    )  # we will update with gNID and eps
    emergent_encoders = torch.load(fullpath(fps.final_encoders_save_fn))
    betas = torch.load(util.get_bound_fn(config, "betas"))
    optimal_encoders = torch.load(
        util.get_bound_fn(config, "encoders")
    )  # ordered by beta
    curve_data = pd.read_csv(util.get_bound_fn(config, "ib"))

    # We will analyze, for each point in each trajectory, its dist to curve
    if config.simulation.trajectory:
        traj_fn = fullpath(fps.trajectory_points_save_fn)
        traj_data = pd.read_csv(traj_fn)

    ##########################################################################
    # Measure efficiency of emergent encoders
    ##########################################################################

    # Compute matrix of pairwise gNID between emergent and optimal
    gNIDs = torch.tensor(
        [
            [gNID(em.float(), opt.float(), g.prior.float()) for opt in optimal_encoders]
            for em in emergent_encoders
        ]
    )

    # Compute, for emergent encoder, the minimum gNID to an IB optima
    min_gnids, min_gnid_indices = torch.min(gNIDs, dim=1)
    fitted_betas = betas[min_gnid_indices]

    # Save the most similar optimal encoder to each emergent encoder
    similar_encoders = optimal_encoders[min_gnid_indices]
    util.save_tensor(fullpath(fps.nearest_optimal_save_fn), similar_encoders)

    # Update simulation data to include efficiency measurements
    losses = torch.Tensor(
        [  # compute efficiency loss for each emergent encoder
            efficiency.efficiency_loss(
                emergent_encoder,
                similar_encoders[i],
                fitted_betas[i],
                g.meaning_dists,
                g.prior,
            )
            for i, emergent_encoder in enumerate(emergent_encoders)
        ]
    )

    ##########################################################################
    # Measure trajectory points' distances to curve
    ##########################################################################

    # Measure Euclidean dist of each language to any optimal curve point
    traj_points = traj_data[["complexity", "accuracy"]].values
    curve_points = curve_data[["complexity", "accuracy"]].values
    distances = cdist(traj_points, curve_points) # shape `[traj_pts, curve_pts]`
    # For each traj_point, get the minimum dist to any curve_point
    min_distances, _ = torch.min(torch.from_numpy(distances), dim=1)

    ##########################################################################
    # Write data
    ##########################################################################

    # Overwrite simulation data
    sim_data["gNID"] = min_gnids.tolist()
    sim_data["eps"] = losses.tolist()
    sim_data["beta"] = fitted_betas.tolist()
    util.save_points_df(sim_fn, sim_data)

    # Write nearest optimal data for plotting later
    opt_data = efficiency.alt_encoders_to_df(similar_encoders, g.meaning_dists, g.prior)
    util.save_points_df(
        fullpath(fps.nearest_optimal_points_save_fn),
        opt_data,
    )

    # Add the min_distances column to trajectory points
    traj_data["min_distance_to_curve"] = min_distances
    util.save_points_df(traj_fn, traj_data)

    # Inspect a single gnid plot
    if config.simulation.inspect_gnid:
        # select (just one for now) emergent encoder
        idx = config.simulation.inspect_gnid_encoder - 1
        assert idx >= 0, "Use 1-indexing for gnid encoder inspection."
        assert idx <= len(gNIDs), "Inspection index must be less than num_runs"
        last_encoder_gnids = gNIDs[idx]
        curve_data["gNID"] = last_encoder_gnids.tolist()
        single_encoder_data = sim_data.iloc[[idx]]
        single_optimal_data = opt_data.iloc[[idx]]
        plot = vis.single_gnid_heatmap_tradeoff_plot(
            curve_data, single_encoder_data, single_optimal_data
        )
        util.save_plot(f"gnid_encoder_{idx+1}.png", plot)

    # Approximate encoders via sampling
    if config.simulation.approximate_encoders:
        # approximate
        sampled_encoders = torch.stack(
            [
                efficiency.finite_sample_encoder(
                    enc, config.simulation.num_approximation_samples
                )
                for enc in emergent_encoders
            ]
        )
        # measure
        approximate_data = efficiency.alt_encoders_to_df(
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
            df=efficiency.alt_encoders_to_df(
                efficiency.hypothetical_variants(
                    emergent_encoders,
                    config.simulation.num_variants,
                ),
                g.meaning_dists,
                g.prior,
            ),
        )


if __name__ == "__main__":
    main()
