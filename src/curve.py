"""Script to compute the IB curve."""
import hydra
import os
import torch
from omegaconf import DictConfig
from altk.effcomm.information import get_ib_curve
from game.game import Game
from misc import util


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    util.set_seed(config.seed)

    # save one curve for multiple analyses
    game_dir = os.getcwd().replace(config.filepaths.simulation_subdir, "")
    curve_fn = os.path.join(game_dir, config.filepaths.curve_points_save_fn)

    evol_game = Game.from_hydra(config)
    curve_points = get_ib_curve(evol_game.prior, evol_game.meaning_dists)

    util.save_ib_curve(curve_fn, curve_points)

if __name__ == "__main__":
    main()