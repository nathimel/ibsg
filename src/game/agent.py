"""Senders and Receivers for signaling games."""

import numpy as np
from altk.effcomm.agent import Speaker, Listener
from altk.language.semantics import Meaning
from game.languages import Signal, SignalMeaning, SignalingLanguage
from typing import Any

class Sender(Speaker):
    """A Sender agent in a signaling game chooses a signal given an observed state of nature, according to P(signal | state)."""

    def __init__(
        self,
        language: SignalingLanguage,
        weights=None,
        name: str = None,
    ):
        super().__init__(language, name=name)
        self.shape = (len(self.language.universe), len(self.language))

        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.random.rand(*self.shape)


class Receiver(Listener):
    """A Receiver agent in a signaling game chooses an action=state given a signal they received, according to P(state | signal)."""

    def __init__(self, language: SignalingLanguage, weights=None, name: str = None):
        super().__init__(language, name=name)
        self.shape = (len(self.language), len(self.language.universe))

        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.random.rand(*self.shape)
    
# TODO: perhaps refactor altk CommunicativeAgent to be compatible with more this more general inherited class?
class SymmetricAgent:
    """A symmetric communicative agent has both speaker and listener modules."""
    def __init__(self, sender: Sender, receiver: Receiver, **kwargs) -> None:

        self.sender = sender
        self.receiver = receiver
        self.accumulated_rewards = [] # track the payoff (fitness) at each round

        self.__dict__.update(**kwargs)

        if sender.language != receiver.language:
            raise Exception("Language not shared between sender and receiver language")
        