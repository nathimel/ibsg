import numpy as np

from game.perception import generate_dist_matrix, generate_sim_matrix

from misc.tools import normalize_rows, random_stochastic_matrix

# TODO: You're going to want to do replicator dynamics, and potentially Nowak and Krakauer exact replication, so it's worth separating the dynamics out from the game.

class Game:
    """The basic object that contains all of the relevant parameters for the agents, environment, payoffs, etc., that are shared or at least initialized across simulations."""
    def __init__(
        self, 
        num_states: int, 
        num_signals: int, 
        prior_init_temp: float, 
        distance: str, 
        discr: float,
        **kwargs,
        ) -> None:
        """Construct an evolutionary game.

        Args:
            num_states: the number of states in the environment.

            num_signals: the number of signals available to agents.

            population_size: the number of agents (P,Q) pairs if simulating finite population evolution.

            prior_type: {'uniform', 'dirac', 'random'} the kind of prior to initialize.

            distance: the kind of distance measure to use as input to the similarity-based utility and meaning distributions.

            discr: a float corresponding to gamma in the softmax-based distribution, representing discriminative need in the game.

            init_temperature: a float controlling the uniform-ness of the seed population of agents (composed of a Sender P and Receiver Q)
        """
        # define a meaning space with some 'similarity' structure
        universe = [i for i in range(num_states)]

        # specify prior and distortion matrix for all trials
        prior = random_stochastic_matrix((num_states, ), temperature = prior_init_temp)
        dist_mat = generate_dist_matrix(universe, distance)

        # construct utility function
        utility = generate_sim_matrix(universe, discr, dist_mat)

        # construct perceptually uncertain meaning distributions
        meaning_dists = normalize_rows(utility)

        # Constant
        self.num_states = num_states
        self.num_signals = num_signals
        self.prior = prior
        self.dist_mat = dist_mat
        self.utility = utility
        self.meaning_dists = meaning_dists

        # updated by dynamics
        self.ib_points = [] # (complexity, accuracy)

    @classmethod
    def from_hydra(cls, config):
        """Automatically construct a evolutionary game from a hydra config."""
        return cls(
            config.game.num_states,
            config.game.num_signals,
            config.game.prior_init_temp,
            config.game.distance,
            config.game.discriminative_need,
        )
