import numpy as np

from game.agent import Sender, Receiver, SymmetricAgent
from game.graph import generate_adjacency_matrix
from game.languages import State, Signal, StateSpace, SignalMeaning, SignalingLanguage
from game.perception import generate_dist_matrix, generate_sim_matrix
from game.prior import generate_prior_over_states

from misc.tools import normalize_rows

# TODO: You're going to want to do replicator dynamics, and potentially Nowak and Krakauer exact replication, so it's worth separating the dynamics out from the game.

class Game:
    """The basic object that contains all of the relevant parameters for the agents, environment, payoffs, etc., that are shared or at least initialized across simulations."""
    def __init__(
        self, 
        num_states: int, 
        num_signals: int, 
        num_agents: int,
        prior_type: str, 
        distance: str, 
        discr: float,
        ) -> None:
        """Construct an evolutionary game.

        Args:
            num_states: the number of states in the environment.

            num_signals: the number of signals available to agents.

            prior_type: {'uniform', 'dirac', 'random'} the kind of prior to initialize.

            distance: the kind of distance measure to use as input to the similarity-based utility and meaning distributions.

            discr: a float corresponding to gamma in the softmax-based distribution, representing discriminative need in the game.
        """
        # dummy names for signals, states
        state_names = [i for i in range(num_states)]
        signal_names = [f"signal_{i}" for i in range(num_signals)]

        # Construct the universe of states, and language defined over it
        universe = StateSpace([State(name=str(name)) for name in state_names])

        # All meanings are dummy placeholders at this stage, but they can be substantive once agents are given a weight matrix.
        dummy_meaning = SignalMeaning(states=[], universe=universe)
        signals = [Signal(form=name, meaning=dummy_meaning) for name in signal_names]

        # Create a seed language to initialize agents.
        seed_language = SignalingLanguage(signals=signals)

        # create n many agents, on a graph
        sender = Sender(seed_language, name="sender")
        receiver = Receiver(seed_language, name="receiver")
        population = [SymmetricAgent(sender, receiver, id=i) for i in range(num_agents)]

        # define the adjacency matrix for the environment of interacting agents
        adj_mat = generate_adjacency_matrix(num_agents)

        ######################################################################
        # Set attributes
        ###################################################################### 

        # specify prior and distortion matrix for all trials
        prior = generate_prior_over_states(num_states, prior_type)
        dist_mat = generate_dist_matrix(universe, distance)

        # construct utility function
        utility = generate_sim_matrix(universe, discr, dist_mat)

        # construct perceptually uncertain meaning distributions
        meaning_dists = normalize_rows(utility)

        # Constant
        self.states = universe.referents
        self.signals = signals
        self.adj_mat = adj_mat
        self.prior = prior
        self.dist_mat = dist_mat        
        self.utility = utility        
        self.meaning_dists = meaning_dists

        # updated by dynamics
        self.population = population
        self.ib_points = [] # (complexity, accuracy)

    @classmethod
    def from_hydra(cls, config):
        """Automatically construct a evolutionary game from a hydra config."""
        return cls(
            config.game.num_states,
            config.game.num_signals,
            config.game.num_agents,
            config.game.prior,
            config.game.distance,
            config.game.discriminative_need,
        )
    
