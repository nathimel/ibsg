"""Script to compute the IB curve."""
import hydra
import torch
import os

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

        bottleneck_result = get_bottleneck(config)
        ib_points = bottleneck_result["coordinates"]
        encoders = bottleneck_result["encoders"]
        betas = bottleneck_result["betas"]

        ib_points = [(tup[0], tup[1], tup[2], beta) for tup, beta in list(zip(ib_points, betas))]

        util.save_points_df(
            curve_fn,
            util.points_to_df(
                ib_points, columns=[
                    "complexity", 
                    "accuracy", 
                    "distortion", 
                    "beta",
                ]
            ),
        )
        util.save_tensor(encoders_fn, torch.stack(encoders))
        util.save_tensor(betas_save_fn, torch.tensor(betas))

    else:
        print("data found, skipping ib curve estimation.")

    ##########################################################################
    # Estimate MSE bound for comparison
    ##########################################################################

    if config.game.overwrite_curves or not os.path.isfile(mse_curve_fn):
        print("computing mse curve...")
        mse_points = get_rd_curve(g.prior, g.dist_mat)

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
