"""Generate a baseline by randomly permuting the emergent encoders. This script is basically a copy of measure.py but targets permutations of the emergent encoders instead of the encoders."""


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

def permute_encoder(encoder: np.ndarray) -> np.ndarray:
    """Randomly permute the indices of the meanings M in a (M,W) stochastic matrix."""
    return encoder[np.random.permutation(encoder.shape[0])]

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

    # Permute encoders
    trajectory_encoders_permuted = np.array(permute_encoder(enc) for enc in trajectory_encoders)

    ##########################################################################
    # Get all measurements
    ##########################################################################  

    optima_eus: list[float] = get_optimal_encoders_eu(g, optimal_encoders)
    measurement = measure_encoders(
        g,
        trajectory_encoders_permuted,
        trajectory_decoders,
        steps_recorded,
        optimal_encoders,
        betas,
        curve_data,
        optima_eus,
    )
    traj_data: pd.DataFrame = measurement.trajectory_dataframe
    fitted_optima: np.ndarray = measurement.fitted_encoders

    ##########################################################################
    # Write data
    ##########################################################################

    util.save_points_df(
        fullpath(fps.permuted_trajectory_points_save_fn), 
        traj_data,
    )

    # Save fitted optima for a trajectory to npz binary
    np.savez_compressed(
        fps.permuted_nearest_optimal_save_fn,
        **{f"run_{run_i}": encs for run_i, encs in enumerate(fitted_optima)},
    )
    print(f"Saved all fitted optima across rounds and runs to {os.path.join(os.getcwd(), fps.permuted_nearest_optimal_save_fn)}")
