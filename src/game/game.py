import os
import warnings

import numpy as np
import pandas as pd

from multiprocessing import cpu_count

from ultk.language.semantics import Universe
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
        prior: np.ndarray,
        distance: str,
        meaning_dist_pi: float,
        discr_need_gamma: float,
        **kwargs,
    ) -> None:
        """Construct an evolutionary game.

        Args:
            universe: a collection of world referents, corresponding to states of nature in the signaling game and the source random variable in IB.

            num_signals: the number of signals available to agents / support of the bottleneck variable in IB.

            prior: the prior distribution over states in the environment / cognitive source in the IB framework.

            distance: the kind of distance measure to use as input to the similarity-based utility and meaning distributions.

            meaning_dist_pi: a float controlling the uniform-ness of the meaning distributions P(U|M), which represent perceptual uncertainty. higher pi -> full certainty.

            discr_need_gamma: a float controlling the uniform-ness of the payoff / utility / fitness function, representing discriminative need. Higher discr -> all or nothing reward.
        """
        # specify distance matrix
        dist_mat = generate_dist_matrix(universe, distance)

        # construct utility function
        utility = generate_sim_matrix(universe, discr_need_gamma, dist_mat)

        # construct perceptually uncertain meaning distributions
        # TODO: maybe create a separate ib_model object?
        meaning_dists = normalize_rows(
            generate_sim_matrix(universe, meaning_dist_pi, dist_mat)
        )

        # Constant
        self.universe = universe
        self.num_states = len(universe)
        self.num_signals = num_signals
        self.prior = prior  # N.B.: we assume p(states) = p(meanings)
        self.dist_mat = dist_mat
        self.utility = utility
        self.meaning_dists = meaning_dists

        # updated by dynamics
        self.points: list[
            tuple[float]
        ] = (
            []
        )  # list of (complexity, accuracy, comm_cost, MSE, EU_gamma, KL_eb, min_gNID, gnid_beta) points
        self.ib_encoders: list[np.ndarray] = []
        self.steps_recorded = []  # for bookkeeping above

        self.__dict__.update(**kwargs)

    @classmethod
    def from_hydra(cls, config, *args, **kwargs):
        """Automatically construct a sim-max game from a hydra config."""

        # Warn the user against inappropriate comparisons
        if (
            config.game.meaning_dist_pi
            != config.simulation.dynamics.imprecise_imitation_alpha
        ):
            warnings.warn(
                "The values of {config.game.meaning_dist_width, config.simulation.dynamics.imprecise_imitation_alpha} should be equal. It is unlikely that you want to run analyses where they vary."
            )

        # Load prior and universe
        universe = None
        prior = None

        # Initialize Universe, default is a list of integers
        if isinstance(config.game.universe, str):
            # breakpoint()
            fn = get_universe_fn(config, *args, **kwargs)
            if not os.path.exists(fn):
                breakpoint()
            referents_df = pd.read_csv(get_universe_fn(config, *args, **kwargs))
        elif isinstance(config.game.universe, int):
            referents_df = pd.DataFrame(
                list(range(1, config.game.universe + 1)), columns=["name"]
            )
        else:
            raise ValueError(
                f"The value of config.game.universe must be the number of natural number states (int) or the name of a file located at data/universe (str). Received type: {type(universe)}."
            )
        # Set Prior
        if isinstance(config.game.prior, str):
            prior_df = pd.read_csv(get_prior_fn(config, *args, **kwargs))
        else:
            prior_df = referents_df.copy()[["name"]]
            prior_df["probability"] = random_stochastic_matrix(
                (len(referents_df),), beta=config.game.prior
            ).tolist()

        universe = build_universe(referents_df, prior_df)
        prior = universe.prior_numpy()
        if not np.isclose(prior.sum(), np.array([1.0])):
            raise Exception(f"Prior does not sum to 1.0. (sum={prior.sum()})")

        game = cls(
            universe,
            config.game.num_signals,
            prior,
            config.game.distance,
            config.game.meaning_dist_pi,  # input to softmax
            config.game.discriminative_need_gamma,  # input to softmax
            dev_betas=config.game.dev_betas,
        )

        return game
