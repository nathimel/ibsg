"""Measure the efficiency of emergent systems w.r.t. the optimal IB encoders via gNID."""

import hydra
import os
import torch

import pandas as pd

from altk.effcomm.util import gNID
from altk.effcomm.information import ib_encoder_to_point
from game.game import Game
from omegaconf import DictConfig
from misc import util


def efficiency_loss(emergent, optimal, beta, meaning_dists, prior) -> float:
    """Compute the efficiency loss of a semantic system:
    
        eps = 1/beta * ( F_[q] - F_[q*] )
    
    where F is the IB objective, q is an emergent encoder, q* is its most similar optimal counterpart, and beta is the value of beta input to the IB method yielding the optimal encoder. The IB objective is given by

       F_[q(w|m)] = I[M:W] − beta * I(W;U)
    
    i.e., a Lagrangian to minimize complexity and maximize accuracy. See Zaslavsky et. al. 2018, "Near-Optimal Trade-Offs" for details.
    """
    # return is complexity, accuracy, comm_cost
    em_complexity, em_acc, _ =  ib_encoder_to_point(meaning_dists, prior, emergent)
    opt_complexity, opt_acc, _ = ib_encoder_to_point(meaning_dists, prior, optimal)
    
    em_value = em_complexity - beta * em_acc
    opt_value = opt_complexity - beta * opt_acc
    loss = 1 / beta * (em_value - opt_value) # value is to be minimized, so emergent larger
    return loss

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    util.set_seed(config.seed)
    g = Game.from_hydra(config)

    # helper function to load the right files
    cwd = os.getcwd()
    fps = config.filepaths
    fullpath = lambda fn: os.path.join(cwd, fn)

    sim_fn = fullpath(fps.simulation_points_save_fn)
    emergent_encoders_fn = fullpath(fps.final_encoders_save_fn)
    similar_encoders_fn = fullpath(fps.similar_encoders_save_fn)    
    betas_fn = fullpath(fps.betas_save_fn)    
    encoders_fn = util.get_bound_fn(config, "encoders")

    sim_data = pd.read_csv(sim_fn) # will update gNID and eps
    emergent_encoders = torch.load(emergent_encoders_fn)
    betas = torch.load(betas_fn)
    optimal_encoders = torch.load(encoders_fn) # ordered by beta

    # Compute matrix of pairwise gNID between emergent and optimal
    gNIDs = torch.tensor([
        [ gNID(em, opt, g.prior) for em in emergent_encoders] for opt in optimal_encoders ])
    
    # Compute, for emergent encoder, the minimum gNID to an IB optima
    min_gnid_indices = [torch.argmin(gnids) for gnids in gNIDs]
    min_gnids = gNIDs[min_gnid_indices]
    fitted_betas = betas[min_gnid_indices]

    # Save the most similar optimal encoder to each emergent encoder
    similar_encoders = optimal_encoders[min_gnid_indices]
    util.save_encoders(similar_encoders_fn, similar_encoders)

    # Update simulation data to include efficiency measurements
    losses = [ # compute efficiency loss for each emergent encoder
        efficiency_loss(
            emergent_encoder, 
            similar_encoders[i], 
            fitted_betas[i], 
            g.meaning_dists, 
            g.prior,
        ) 
        for i, emergent_encoder in enumerate(emergent_encoders)
    ]

    # Overwrite simulation data
    sim_data["gNID"] = min_gnids
    sim_data["eps"] = losses
    sim_data["beta"] = fitted_betas
    sim_data.to_csv(sim_fn)

if __name__ == "__main__":
    main()    