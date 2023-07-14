import os
import torch

import pandas as pd

from multiprocessing import cpu_count

from altk.language.semantics import Referent, Universe
from game.perception import generate_dist_matrix, generate_sim_matrix
from game.meaning import build_universe

from misc.tools import normalize_rows, random_stochastic_matrix
from misc.util import get_prior_fn, get_universe_fn

class Game:
    """The basic object that contains all of the relevant parameters for the agents, environment, payoffs, etc., that are shared or at least initialized across simulations."""
    def __init__(
        self, 
        universe: Universe,
        num_signals: int, 
        prior: torch.Tensor, 
        distance: str, 
        discr_need_gamma: float,
        meaning_dist_gamma: float,
        **kwargs,
        ) -> None:
        """Construct an evolutionary game.

        Args:
            num_states: the number of states in the environment.

            num_signals: the number of signals available to agents.

            prior: the prior distribution over states in the environment.

            distance: the kind of distance measure to use as input to the similarity-based utility and meaning distributions.

            discr_need_gamma: a float controlling the uniform-ness of the payoff / utility / fitness function, representing discriminative need. Higher discr -> all or nothing reward.
            
            meaning_dist_temp: a float controlling the uniform-ness of the meaning distributions P(U|M), which represent perceptual uncertainty. higher temp -> full certainty.
        """
        # specify distance matrix
        dist_mat = generate_dist_matrix(universe, distance)

        # construct utility function
        utility = generate_sim_matrix(universe, discr_need_gamma, dist_mat)

        # construct perceptually uncertain meaning distributions
        # TODO: maybe create an ib_model object?
        meaning_dists = normalize_rows(generate_sim_matrix(universe, meaning_dist_gamma, dist_mat))

        # Constant
        self.universe = universe
        self.num_states = len(universe)
        self.num_signals = num_signals
        self.prior = prior  # N.B.: we assume p(states) = p(meanings)
        self.dist_mat = dist_mat
        self.utility = utility
        self.meaning_dists = meaning_dists

        # updated by dynamics
        self.points = [] # list of (complexity, accuracy, comm_cost, MSE) points
        self.ib_encoders = []

        self.__dict__.update(**kwargs)

    @classmethod
    def from_hydra(cls, config, *args, **kwargs):
        """Automatically construct a sim-max game from a hydra config."""

        # default to local number of cpus for multiprocessing of simulations
        if config.game.num_processes is None:
            config.game.num_processes = cpu_count()

        # Load prior and universe
        universe = None
        prior = None

        # Initialize Universe, default is a list of integers
        if isinstance(config.game.universe, str):
            referents_df = pd.read_csv(get_universe_fn(config))
        elif isinstance(config.game.universe, int):
            referents_df = pd.DataFrame(list(range(config.game.universe)), columns=["name"])
        else:
            raise ValueError(f"The value of config.game.universe must be the number of natural number states (int) or the name of a file located at data/universe (str). Received type: {type(universe)}.")
        # Set Prior
        if isinstance(config.game.prior, str):
            prior_df = pd.read_csv(get_prior_fn(config))
        else:
            prior_df = pd.DataFrame(list(range(len(referents_df))), columns=["name"])
            prior_df["probability"] = random_stochastic_matrix((len(referents_df), ), beta=10 ** config.game.prior).tolist()

        universe = build_universe(referents_df, prior_df)
        prior = torch.from_numpy(universe.prior_numpy())
        if not torch.isclose(prior.float().sum(), torch.tensor([1.]).float()):
            raise Exception(f"Prior does not sum to 1.0. (sum={prior.sum()})")
        
        # temp
        meaning_dist_gamma = config.game.meaning_dist_gamma
        if isinstance(meaning_dist_gamma, str):
            meaning_dist_gamma = torch.nan # 


        game = cls(
            # config.game.num_states,
            universe,
            config.game.num_signals,
            prior,
            config.game.distance,
            10 ** config.game.discriminative_need_gamma, # input to softmax
            # 10 ** config.game.meaning_dist_gamma, # input to softmax
            10 ** meaning_dist_gamma,
            maxbeta = config.game.maxbeta, # we want about 1.0 - 2.0
            minbeta = 10 ** config.game.minbeta,
            numbeta = config.game.numbeta,
            num_processes = config.game.num_processes,
        )

        if isinstance(config.game.meaning_dist_gamma, str):
            # temporary: replace meaning dists
            game.meaning_dists = torch.load("/Users/nathanielimel/uci/projects/ibsg/data/meaning_dists/wcs_model.pt") # double
            game.meaning_dists = game.meaning_dists.float()
        
        return game
