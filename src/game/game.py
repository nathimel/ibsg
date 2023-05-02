import torch
from multiprocessing import cpu_count

from game.perception import generate_dist_matrix, generate_sim_matrix

from misc.tools import normalize_rows, random_stochastic_matrix

class Game:
    """The basic object that contains all of the relevant parameters for the agents, environment, payoffs, etc., that are shared or at least initialized across simulations."""
    def __init__(
        self, 
        num_states: int, 
        num_signals: int, 
        prior_init_alpha: float, 
        distance: str, 
        discr_need_gamma: float,
        meaning_dist_gamma: float,
        **kwargs,
        ) -> None:
        """Construct an evolutionary game.

        Args:
            num_states: the number of states in the environment.

            num_signals: the number of signals available to agents.

            population_size: the number of agents (P,Q) pairs if simulating finite population evolution.

            prior_type: {'uniform', 'dirac', 'random'} the kind of prior to initialize.

            distance: the kind of distance measure to use as input to the similarity-based utility and meaning distributions.

            discr_need_gamma: a float controlling the uniform-ness of the payoff / utility / fitness function, representing discriminative need. Higher discr -> all or nothing reward.
            
            meaning_dist_temp: a float controlling the uniform-ness of the meaning distributions P(U|M), which represent perceptual uncertainty. higher temp -> full certainty.
        """
        # define a meaning space with some 'similarity' structure
        universe = [i for i in range(num_states)]

        # specify prior and distance matrix for all trials
        prior = random_stochastic_matrix((num_states, ), beta = prior_init_alpha) # N.B.: we assume p(states) = p(meanings)
        dist_mat = generate_dist_matrix(universe, distance)

        # construct utility function
        utility = generate_sim_matrix(universe, discr_need_gamma, dist_mat)

        # construct perceptually uncertain meaning distributions
        meaning_dists = normalize_rows(generate_sim_matrix(universe, meaning_dist_gamma, dist_mat))

        # Constant
        self.universe = universe
        self.num_states = num_states
        self.num_signals = num_signals
        self.prior = prior
        self.dist_mat = dist_mat
        self.utility = utility
        self.meaning_dists = meaning_dists

        # updated by dynamics
        self.points = [] # list of (complexity, accuracy, comm_cost, MSE) points
        self.ib_encoders = []

        self.__dict__.update(**kwargs)

    @classmethod
    def from_hydra(cls, config):
        """Automatically construct a evolutionary game from a hydra config."""

        if config.game.num_processes is None:
            config.game.num_processes = cpu_count()

        return cls(
            config.game.num_states,
            config.game.num_signals,
            10 ** config.game.prior_init_alpha,
            config.game.distance,
            10 ** config.game.discriminative_need_gamma, # input to softmax
            10 ** config.game.meaning_dist_gamma, # input to softmax
            maxbeta = config.game.maxbeta, # we want about 1.0 - 2.0
            minbeta = 10 ** config.game.minbeta,
            numbeta = config.game.numbeta,
            num_processes = config.game.num_processes,
        )
