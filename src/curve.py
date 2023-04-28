"""Script to compute the IB curve."""
import hydra
import torch
from altk.effcomm.information import get_bottleneck, get_rd_curve
from omegaconf import DictConfig
from game.game import Game
from misc import util
from multiprocessing import cpu_count


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    util.set_seed(config.seed)

    curve_fn = util.get_curve_fn(config)
    ub_curve_fn = util.get_curve_fn(config, "ub")

    g = Game.from_hydra(config)
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

    print("computing ub curve...")
    ub_points = get_rd_curve(g.prior, g.dist_mat)

    util.save_points_df(ub_curve_fn, util.points_to_df(ub_points, columns=["complexity", "mse"]))

if __name__ == "__main__":
    main()
