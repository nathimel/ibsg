import numpy as np

from game.agent import Sender, Receiver, SymmetricAgent
from game.graph import generate_adjacency_matrix
from game.languages import State, Signal, StateSpace, SignalMeaning, SignalingLanguage
from game.perception import generate_dist_matrix, generate_sim_matrix
from game.prior import generate_prior_over_states

from misc.tools import normalize_rows


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

        # create n many agents, fully connected
        sender = Sender(seed_language, name="sender")
        receiver = Receiver(seed_language, name="receiver")
        agents = [SymmetricAgent(sender, receiver, id=i) for i in range(num_agents)]

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

        self.states = universe.referents
        self.signals = signals
        self.agents = agents
        self.adj_mat = adj_mat
        self.prior = prior
        self.dist_mat = dist_mat        
        self.utility = utility        
        self.meaning_dists = meaning_dists

    @classmethod
    def from_hydra(cls, config):
        """Automatically construct a evolutionary game from a hydra config."""
        return cls(
            config.game.num_states,
            config.game.num_words,
            config.game.num_agents,
            config.game.prior,
            config.game.distance,
            config.game.discriminative_need,
        )
    
    def simulate_interactions(self) -> None:
        """Simulate a single round of the game wherein agents interact with each other and collect payoffs."""
        raise NotImplementedError

    def moran_evolve(self) -> None:
        """Simulate evolution in the finite population by running the Moran process, at each iteration randomly replacing an individual with an (randomly selected proportional to fitness) agent's offspring."""
        raise NotImplementedError
    

    def interact(self, agent_1: SymmetricAgent, agent_2: SymmetricAgent) -> None:
        """Simulate the interaction between two agents by recording their expected payoff in repeated communicative exchanges."""
        raise NotImplementedError
    
    def choose_offspring(self) -> SymmetricAgent:
        """Choose an individual to asexually reproduce, and return their identical copy (child)."""
        raise NotImplementedError

    def choose_decedent(self) -> SymmetricAgent:
        """Choose an individual to be removed from the population."""
        raise NotImplementedError
