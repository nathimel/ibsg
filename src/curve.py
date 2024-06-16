"""Script to compute the IB curve."""
import hydra
import os

import numpy as np

from omegaconf import DictConfig, OmegaConf
from analysis.ib import get_bottleneck, get_rd_curve
from game.game import Game
from misc import util


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    util.set_seed(config.seed)
    g = Game.from_hydra(config)

    ##########################################################################
    # Set up datapaths
    ##########################################################################

    curve_fn = util.get_bound_fn(config)
    mse_curve_fn = util.get_bound_fn(config, "mse")
    encoders_fn = util.get_bound_fn(config, "encoders")
    betas_save_fn = util.get_bound_fn(config, "betas")
    metadata_fn = util.get_bound_fn(config, "metadata")

    ##########################################################################
    # Estimate IB bound
    ##########################################################################

    if config.game.overwrite_curves or not os.path.isfile(curve_fn):
        print("computing ib curve...")

        ib_results = get_bottleneck(config)

        encoders, ib_points, betas = zip(*[
            (item.qxhat_x,
            (item.rate, item.accuracy, item.distortion, item.beta),
            item.beta)
            for item in ib_results
        ])

        util.save_points_df(
            curve_fn,
            util.points_to_df(
                ib_points,
                columns=[
                    "complexity", 
                    "accuracy", 
                    "distortion",
                    "beta",
                ]
            ),
        )
        util.save_ndarray(encoders_fn, np.stack(encoders))
        util.save_ndarray(betas_save_fn, np.array(betas))

    else:
        print("data found, skipping ib curve estimation.")

    ##########################################################################
    # Estimate MSE bound for comparison
    ##########################################################################

    if config.game.overwrite_curves or not os.path.isfile(mse_curve_fn):
        print("computing mse curve...")
        mse_points = get_rd_curve(config)

        util.save_points_df(
            mse_curve_fn, util.points_to_df(mse_points, columns=["complexity", "mse"])
        )
    else:
        print("data found, skipping mse curve estimation")


    ##########################################################################
    # Save metadata
    ##########################################################################

    if config.game.overwrite_curves or not os.path.isfile(metadata_fn):
        print("saving curve metadata...")

        curve_metadata = config.game
        OmegaConf.save(curve_metadata, metadata_fn)

        print(f"Saved a hydra config as curve metadata to {metadata_fn}")
    else: 
        print("data found, skipping curve metadata save.")


if __name__ == "__main__":
    main()
