from game.agents import Sender, Receiver
from game.languages import State, Signal, StateSpace, SignalMeaning, SignalingLanguage
from game.perception import generate_dist_matrix, generate_sim_matrix, generate_meaning_distributions
from game.prior import generate_prior_over_states

class Game:
    """The basic object that contains all of the relevant parameters for the agents, environment, payoffs, etc., that are shared or at least initialized across simulations."""
    def __init__(
        self, 
        num_states: int, 
        num_signals: int, 
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
        sender = Sender(seed_language, name="sender")
        receiver = Receiver(seed_language, name="receiver")

        ######################################################################
        # Set attributes
        ###################################################################### 

        # specify prior and distortion matrix for all trials
        prior = generate_prior_over_states(num_states, prior_type)
        dist_mat = generate_dist_matrix(universe, distance)

        # construct utility function
        utility = generate_sim_matrix(universe, discr, dist_mat)

        # construct perceptually uncertain meaning distributions
        meaning_dists = generate_meaning_distributions(utility)

        self.states = universe.referents
        self.signals = signals
        self.sender = sender
        self.receiver = receiver
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
            config.game.prior,
            config.game.distance,
            config.game.discriminative_need,
        )