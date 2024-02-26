import copy
import warnings

import numpy as np

from analysis.ib import ib_encoder_to_measurements
from ultk.effcomm.rate_distortion import rows_zero_to_uniform, bayes
from game.game import Game
from game.perception import generate_confusion_matrix
from game.graph import generate_adjacency_matrix
from misc.tools import random_stochastic_matrix, normalize_rows

from tqdm import tqdm


##############################################################################
# Base classes
##############################################################################


class Dynamics:
    def __init__(self, game: Game, **kwargs) -> None:
        self.game = game

        self.max_its = kwargs["max_its"]
        self.threshold = kwargs["threshold"]
        self.confusion_gamma = kwargs["imprecise_imitation_gamma"]

        self.confusion = generate_confusion_matrix(self.game.universe, self.confusion_gamma, self.game.dist_mat)

        pt_args = [
            self.game.meaning_dists,
            self.game.prior,
            self.game.dist_mat,
            self.confusion,
        ]

        self.get_point = lambda encoder, _: ib_encoder_to_measurements(
            *pt_args,
            encoder=encoder,
        )

        if kwargs["use_decoder"]:
            self.get_point = lambda encoder, decoder: ib_encoder_to_measurements(
                *pt_args,
                encoder=encoder,
                decoder=decoder,
            )

    def run(self):
        raise NotImplementedError

    def evolution_step(self):
        """The step of evolution that varies between different models."""
        raise NotImplementedError


class FinitePopulationDynamics(Dynamics):
    def __init__(self, game: Game, **kwargs) -> None:
        """
        Args:
            max_its: int maximum number of steps to run evolution

            threshold: a float controlling convergence of evolution

            init_gamma: a float controlling the sharpness of the seed population (composed of a Sender P and Receiver Q) agents' distributions
        """
        super().__init__(game, **kwargs)
        self.n = kwargs["population_size"]
        self.population_init_gamma = kwargs["population_init_gamma"]

        # create a population of n many (P,Q) agents
        self.Ps = random_stochastic_matrix(
            (self.n, self.game.num_states, self.game.num_signals),
            self.population_init_gamma,
        )
        self.Qs = random_stochastic_matrix(
            (self.n, self.game.num_signals, self.game.num_states),
            self.population_init_gamma,
        )

        # define the adjacency matrix for the environment of interacting agents
        self.adj_mat = generate_adjacency_matrix(self.n, kwargs["graph"])
        # since stochastic edge weights not supported yet just cast to int
        self.adj_mat = np.array(self.adj_mat, dtype=int)

    def population_mean_weights(self) -> tuple[float]:
        """Return the average agent (Sender, Receiver) weights."""
        return (
            normalize_rows(np.mean(self.Ps, axis=0)),
            normalize_rows(np.mean(self.Qs, axis=0)),
        )

    def measure_fitness(self) -> np.ndarray:
        """Measure the fitness of communicating individuals in the population.

        Returns:
            a 1D array of floats (normalized to [0,1]) of shape `num_agents` such that fitnesses[i] corresponds to the ith individual.
        """
        payoffs = np.zeros(self.n)  # 1D, since fitness is symmetric

        # iterate over every adjacent pair in the graph
        for i in range(self.n):
            for j in range(self.n):
                # TODO: generalize to stochastic behavior for real-valued edge weights
                if i != j and not isinstance(self.adj_mat[i, j].item(), int):
                    raise Exception(
                        f"Stochastic edge weights not yet supported. Edge weight for i={i}, j={j} was {self.adj_mat[i,j].item()}."
                    )
                if not self.adj_mat[i, j]:
                    continue
                # accumulate payoff for interaction symmetrically
                payoff = self.fitness(
                    p=self.Ps[i],
                    q=self.Qs[i],
                    p_=self.Ps[j],
                    q_=self.Qs[j],
                )
                payoffs[i] += payoff
                payoffs[j] += payoff

        return payoffs / payoffs.sum()

    def fitness(
        self,
        p: np.ndarray,
        q: np.ndarray,
        p_: np.ndarray,
        q_: np.ndarray,
    ) -> float:
        """Compute pairwise fitness as F[L, L'] = F[(P, Q'), (P', Q)] = 1/2(f(P,Q')) + 1/2(f(P', Q))

        where f(X,Y) = sum( diag(prior) @ C @ X @ Y @ C * Utility )

        where X is a sender, Y is a receiver, and C is a symmetric confusion matrix, to compare to IB meaning distributions.
        """
        f = lambda X, Y: np.sum(
            np.diag(self.game.prior)
            @ self.confusion
            @ X
            @ Y
            @ self.confusion
            * self.game.utility
        )
        return (f(p, q_) + f(p_, q)) / 2.0

    def run(self):
        """Main loop to simulate evolution and track data."""
        mean_p, mean_q = self.population_mean_weights()

        i = 0
        converged = False
        progress_bar = tqdm(total=self.max_its)
        while not converged:
            i += 1
            progress_bar.update(1)

            mean_p_prev = copy.deepcopy(mean_p)
            mean_q_prev = copy.deepcopy(mean_q)

            # track data
            self.game.points.append(self.get_point(mean_p, mean_q))
            self.game.ib_encoders.append(mean_p)

            self.evolution_step()

            # update population fitnesses and mean behavior
            mean_p, mean_q = self.population_mean_weights()

            # Check for convergence
            if (
                np.abs(mean_p - mean_p_prev).sum() < self.threshold
                and np.abs(mean_q - mean_q_prev).sum() < self.threshold
            ) or (i == self.max_its):
                converged = True

        progress_bar.close()


##############################################################################
# Nowak and Krakauer
##############################################################################


def mutate(parent_behavior: np.ndarray, num_samples: int):
    eye = np.eye(parent_behavior.shape[-1])
    sample_indices = np.stack([
        np.random.choice(len(sub_p), size=num_samples, replace=True, p=sub_p)
        for sub_p in parent_behavior
    ])
    samples = eye[sample_indices]
    return samples.mean(axis=-2)


class NowakKrakauerDynamics(FinitePopulationDynamics):
    def __init__(self, game: Game, **kwargs) -> None:
        super().__init__(game, **kwargs)
        self.num_samples = kwargs["num_samples"]

    def evolution_step(self):
        """Children learn the language of their parents by sampling their responses to objects (Nowak and Krakauer, 1999)."""

        fitnesses = self.measure_fitness()

        new_population = np.random.choice(fitnesses, self.n, replace=True)
        self.Ps = mutate(self.Ps[new_population], self.num_samples)
        self.Qs = mutate(self.Qs[new_population], self.num_samples)


##############################################################################
# Frequency-Dependent Moran Process
##############################################################################


class MoranProcess(FinitePopulationDynamics):
    def __init__(self, game: Game, **kwargs) -> None:
        super().__init__(game, **kwargs)

    def evolution_step(self):
        """Simulate evolution in the finite population by running the frequency dependent Moran process, at each iteration randomly replacing an individual with an (randomly selected proportional to fitness) agent's offspring."""
        fitnesses = self.measure_fitness()

        # i = np.random.choice(fitnesses, 1)
        i = np.random.choice(np.ones(self.n), 1, p=fitnesses) # birth
        j = np.random.choice(np.ones(self.n), 1) # death

        # replace the random deceased with fitness-sampled offspring
        self.Ps[j] = self.Ps[i]
        self.Qs[j] = self.Qs[i]
        # N.B.: no update to adj_mat necessary


##############################################################################
# Discrete Time Replicator Dynamics (with perceptual uncertainty in meanings)
##############################################################################


class ReplicatorDynamics(Dynamics):
    """Discrete Time Replicator Dynamics, with perceptual uncertainty in meaning distributions (see Franke and Correia, 2018 on imprecise imitation)."""

    def __init__(self, game: Game, **kwargs) -> None:
        super().__init__(game, **kwargs)
        self.init_gamma = 10 ** kwargs["population_init_gamma"]

        self.P = random_stochastic_matrix(
            (self.game.num_states, self.game.num_signals), self.init_gamma
        )  # Sender 'population frequencies'
        self.Q = random_stochastic_matrix(
            (self.game.num_signals, self.game.num_states), self.init_gamma
        )  # Receiver 'population frequencies'

    def run(self):
        its = 0
        converged = False
        progress_bar = tqdm(total=self.max_its)
        while not converged:
            its += 1
            progress_bar.update(1)

            P_prev = copy.deepcopy(self.P)
            Q_prev = copy.deepcopy(self.Q)

            self.game.points.append(self.get_point(P_prev, Q_prev))

            self.game.ib_encoders.append(rows_zero_to_uniform(normalize_rows(P_prev)))

            self.evolution_step()  # N.B.: fitness requires population update

            # Check for convergence
            if (
                np.abs(self.P - P_prev).sum() < self.threshold
                and np.abs(self.Q - Q_prev).sum() < self.threshold
            ) or (its == self.max_its):
                converged = True

        # breakpoint()

        progress_bar.close()

    def evolution_step(self):
        """Simulate evolution of strategies in a near-infinite population of agents x using a discrete-time version of the replicator equation:

            x_i' = x_i * ( f_i(x) - sum_j f_j(x_j) )

        Changes in agent type (pure strategies) depend only on their frequency and their fitness.

        Note that the only forms (RDD and ICI) of the replicator-dynamics we apply assume both
            - (i) evolution of behavioral strategies at the level of choice points (word meanings), instead of evolution of entire contingency plans (i.e., lexicons) and
            - (ii) two infinite, well mixed populations of senders and receivers.

        Update steps in the two population replicator dynamics for signaling is given by:

        freq(sender)' = freq(sender) * fitness_relative_to_receiver(sender)

        freq(receiver)' = freq(receiver) * fitness_relative_to_prior_and_sender(receiver)
        """
        raise NotImplementedError

    def warn_if_all_zero(self) -> None:
        """Check if a Sender's encoder or Receiver's decoder is all zeros, and warn appropriately.

        N.B.: This means the entire universe is ineffable. Since this feature of the model appears to reflect a logical (albeit remote) possibility, our analysis proceeds as usual (rather than, e.g., assigning a uniform distribution to every row).
        """
        if np.any(self.P.sum() == 0):
            warnings.warn("Dynamics yielded an encoder with all zeros.")

        if np.any(self.Q.sum() == 0):
            warnings.warn("Dynamics yielded a decoder with all zeros.")


class ReplicatorDiffusionDynamics(ReplicatorDynamics):
    """The 'replicator-diffusion dynamic' introduced in Correia (2013). Unlike Franke and Correia (2018), There is no direct connection between agent-level behavior and the imprecision/noise in this dynamic. It is also more mathematically simple, and closer to the replicator-mutator dynamic."""

    def __init__(self, game: Game, **kwargs) -> None:
        super().__init__(game, **kwargs)

    def evolution_step(self):
        """Updates to sender and receiver are given on page 6 of https://github.com/josepedrocorreia/vagueness-games/blob/master/paper/version_01/paper.pdf. The original implementation is found at https://github.com/josepedrocorreia/vagueness-games/blob/master/vagueness-games.py#L285."""
        P = self.P  # `[states, signals]`
        Q = self.Q  # `[signals, states]`
        U = self.game.utility  # `[states, states]`
        C = self.confusion  # `[states, states]`, compare self.game.meaning_dists
        p = self.game.prior  # `[states,]`

        P *= (Q @ U).T
        P = C @ P
        P = normalize_rows(P)

        # The RDD is formulated in Correia (2013) such that the Receiver update is weighted by the _joint_ distribution Sender(word, state). 
        Q *= p * (U @ P).T        

        # In Franke and Correia, the plain RD (without diffusion/noise) is such that Receiver update is weighted by Sender(state|word), via Bayes rule.
        # Q *= (U @ bayes(P, p))

        Q = Q @ C  # C symmetric, and if C = M, we thus assume m(u) = u(m).
        Q = normalize_rows(Q)

        self.P = copy.deepcopy(P)
        self.Q = copy.deepcopy(Q)

        self.warn_if_all_zero()


class ImpreciseConditionalImitation(ReplicatorDynamics):
    """The dynamics described in Franke and Correia (2018), which has an explicit derivation from the continuous time replicator dynamic in terms of individual agents' imprecise imitation of signaling behavior. In the limiting case where there is no noise/imprecision, the dynamics is equivalent to the replicator diffusion dynamics."""

    def __init__(self, game: Game, **kwargs) -> None:
        super().__init__(game, **kwargs)

    def evolution_step(self):
        """Updates to sender and receiver are given on page 11 (Section 3.2, volume page 1047) and derivation is given on page 24 (Section A.1.1, volume page 1060) of https://www.journals.uchicago.edu/doi/full/10.1093/bjps/axx002. The original implementation is found at https://github.com/josepedrocorreia/vagueness-games/blob/master/newCode/vagueness-games.py#L291."""

        sender = self.P  # `[states, signals]`
        receiver = self.Q  # `[signals, states]`
        utility = self.game.utility  # `[states, states]`
        confusion = (
            self.confusion
        )  # `[states, states]`, compare self.game.meaning_dists
        prior = self.game.prior  # `[states,]`

        # --------- Sender update ---------

        # prob state a obtained (col) given state o was observed (row)
        # `[states, states]`
        observation_noise = normalize_rows(prior * confusion)

        # prob signal w (col) given actual state a obtained (row)
        # `[states, signals]`
        sigma = confusion @ sender #TODO: ask Michael why not observation_noise @ sender

        # P_o(w|m_o) prob a teacher sends signal w given imitator sees state o
        # `[states, signals]`
        observed_sender = observation_noise @ sigma #TODO: ask Michael why not consusion @ sigma

        # prob receiver chooses state r (column) given signal w (row)
        # `[signals, states]`
        rho = receiver @ confusion

        # Expected utility for sender:
        # sum all observation_noise[m_o, m_a] * rho[w, m_r] * utility[m_a, m_r]
        sender_eu = np.einsum("oa,wr,ar->ow", observation_noise, rho, utility)

        # Update sender according to replicator dynamics
        sender = normalize_rows(observed_sender * sender_eu)

        # --------- Receiver update ---------

        # prob state o was observed (col) given signal w received (row)
        # `[signals, states]`
        observed_receiver = rho @ confusion

        # The probability of an actual state given random sender produced w
        # `[signals, states]`
        sigma_inverse = bayes(sender, prior) # n.b.: bayesian optimal decoder

        # Expected utility for receiver:
        # sum all  sigma_inverse[w,a] * C[i, r] * U[a,r]
        receiver_eu = np.einsum("wa,ia,ar->wi", sigma_inverse, confusion, utility)

        # Update receiver according to replicator dynamics
        receiver = normalize_rows(observed_receiver * receiver_eu)

        # update instance

        self.P = copy.deepcopy(sender)
        self.Q = copy.deepcopy(receiver)

        self.warn_if_all_zero()


dynamics_map = {
    "moran_process": MoranProcess,
    "nowak_krakauer": NowakKrakauerDynamics,
    "replicator_diffusion": ReplicatorDiffusionDynamics,
    "imprecise_conditional_imitation": ImpreciseConditionalImitation,
}
