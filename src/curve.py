"""Script to compute the IB curve."""
import hydra
import os
from altk.effcomm.information import get_bottleneck, get_rd_curve
from omegaconf import DictConfig
from game.game import Game
from misc import util
from multiprocessing import cpu_count


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    util.set_seed(config.seed)

    curve_fn = util.get_curve_fn(config)
    mse_curve_fn = util.get_curve_fn(config, "mse")

    g = Game.from_hydra(config)
    # breakpoint()

    if config.game.overwrite_curves or not os.path.isfile(curve_fn):
        print("computing ib curve...")

        bottleneck = get_bottleneck(
            prior=g.prior,
            meaning_dists=g.meaning_dists,
            maxbeta=g.maxbeta,
            minbeta=g.minbeta,
            numbeta=g.numbeta,
            processes=g.num_processes,
        )
        ib_points = list(zip(*bottleneck))

        util.save_points_df(curve_fn, util.points_to_df(ib_points, columns=["complexity", "accuracy", "distortion"]))
    
    else:
        print("data found, skipping ib curve estimation.")
    
    if config.game.overwrite_curves or not os.path.isfile(mse_curve_fn):
        print("computing mse curve...")
        mse_points = get_rd_curve(g.prior, g.dist_mat)

        util.save_points_df(mse_curve_fn, util.points_to_df(mse_points, columns=["complexity", "mse"]))
    else:
        print("data found, skipping mse curve estimation")

if __name__ == "__main__":
    main()
