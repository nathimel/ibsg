"""Simple experimental setup things. So far just creates universe, prior files."""

import hydra
import os
import torch

from omegaconf import DictConfig

from misc import util, tools

from game.game import Game

# TODO: more intuitive would be to simply run hydra, and then launch a warning/guard function at the start of each script that ensures that a prior and universe have been specified

# TODO: Idea: just load the universe, prior from existing files, if not default. the folder name i guess will be the filename under data/prior/__.csv or data/universe/__.csv

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    util.set_seed(config.seed)

    g = Game.from_hydra(config)


if __name__ == "__main__":
    main()
