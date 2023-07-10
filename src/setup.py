"""Simple experimental setup things. So far just creates the prior file."""

import hydra
import os

from omegaconf import DictConfig

from misc import util, tools


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    util.set_seed(config.seed)

    prior_init_alpha = config.game.prior_init_alpha
    fp = util.get_prior_fn(config)

    # Initialize prior
    prior = None
    if isinstance(prior_init_alpha, int):
        prior = tools.random_stochastic_matrix((config.game.num_states, ), 10 ** prior_init_alpha) # this will have to be generalized to ndim

    if not os.path.exists(fp):
        util.save_tensor(fp, prior)
    else:
        print(f"Found an existing prior file at {fp}, skipping creation.")


if __name__ == "__main__":
    main()
