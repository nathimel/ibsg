import torch

from game.perception import generate_dist_matrix, generate_sim_matrix

from misc.tools import normalize_rows, random_stochastic_matrix

class Game:
    """The basic object that contains all of the relevant parameters for the agents, environment, payoffs, etc., that are shared or at least initialized across simulations."""
    def __init__(
        self, 
        num_states: int, 
        num_signals: int, 
        prior_init_beta: float, 
        distance: str, 
        discr: float,
        meaning_dist_beta: float,
        **kwargs,
        ) -> None:
        """Construct an evolutionary game.

        Args:
            num_states: the number of states in the environment.

            num_signals: the number of signals available to agents.

            population_size: the number of agents (P,Q) pairs if simulating finite population evolution.

            prior_type: {'uniform', 'dirac', 'random'} the kind of prior to initialize.

            distance: the kind of distance measure to use as input to the similarity-based utility and meaning distributions.

            discr: a float controlling the uniform-ness of the payoff / utility / fitness function, representing discriminative need. Higher discr -> all or nothing reward.
            
            meaning_dist_temp: a float controlling the uniform-ness of the meaning distributions P(U|M), which represent perceptual uncertainty. higher temp -> full certainty.
        """
        # define a meaning space with some 'similarity' structure
        universe = [i for i in range(num_states)]

        # specify prior and distortion matrix for all trials
        prior = random_stochastic_matrix((num_states, ), beta = prior_init_beta)
        dist_mat = generate_dist_matrix(universe, distance)

        # construct utility function
        utility = generate_sim_matrix(universe, discr, dist_mat)

        # construct perceptually uncertain meaning distributions
        meaning_dists = normalize_rows(generate_sim_matrix(universe, meaning_dist_beta, dist_mat))

        # Constant
        self.num_states = num_states
        self.num_signals = num_signals
        self.prior = prior
        self.dist_mat = dist_mat
        self.utility = utility
        self.meaning_dists = meaning_dists

        # updated by dynamics
        self.ib_points = [] # (complexity, accuracy)

        self.__dict__.update(**kwargs)

    @classmethod
    def from_hydra(cls, config):
        """Automatically construct a evolutionary game from a hydra config."""
        return cls(
            config.game.num_states,
            config.game.num_signals,
            config.game.prior_init_beta,
            config.game.distance,
            config.game.discriminative_need,
            config.game.meaning_dist_beta,
            beta_start = config.game.beta_start,
            beta_stop = config.game.beta_stop,
            steps = config.game.steps,
        )
