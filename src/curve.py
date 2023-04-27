"""Script to compute the IB curve."""
import hydra
import torch
from altk.effcomm.information import get_ib_curve
from omegaconf import DictConfig
from game.game import Game
from misc import util
from multiprocessing import cpu_count


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    util.set_seed(config.seed)

    # save one curve for multiple analyses
    curve_fn = util.get_curve_fn(config)
    ub_curve_fn = util.get_curve_fn(config, "ub")

    # curve_points = get_ib_curve_(config)["coordinates"]
    g = Game.from_hydra(config)
    curve_points = torch.tensor(get_ib_curve(
        g.prior, 
        g.meaning_dists,
        # g.maxbeta,
        5,
        10 ** g.minbeta,
        g.numbeta,
        # processes=g.num_processes,
        processes=cpu_count(),
        )).flip([0,1])
    
    from altk.effcomm.information import get_rd_curve
    ub_curve_points = torch.tensor(get_rd_curve(g.prior, g.dist_mat))

    util.save_points_df(curve_fn, util.points_to_df(curve_points))

    util.save_points_df(ub_curve_fn, util.points_to_df(ub_curve_points, columns=["complexity", "mse"]))

if __name__ == "__main__":
    main()
