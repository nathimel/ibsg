"""Minimal script to check that basic construction of the game and analysis works. Useful when run before any real steps of the experiment."""

import hydra
from omegaconf import DictConfig

from misc import util
from game.game import Game


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    util.set_seed(config.seed)

    g = Game.from_hydra(config)


if __name__ == "__main__":
    main()
