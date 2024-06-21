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
        self.confusion_alpha = kwargs["imprecise_imitation_alpha"]

        self.ib_optimal_encoders = kwargs["ib_optimal_encoders"]
        self.ib_optimal_betas = kwargs["ib_optimal_betas"]

        self.confusion = generate_confusion_matrix(self.game.universe, self.confusion_alpha, self.game.dist_mat)

        # TODO: move all measurement to measure.py, and only do model checkpointing.
        pt_args = [
            self.game.meaning_dists,
            self.game.prior,
            self.game.dist_mat,
            self.game.utility,
            self.confusion,
            self.ib_optimal_encoders,
            self.ib_optimal_betas,
        ]

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
        super().__init__(game, **kwargs)
        self.n = kwargs["population_size"]

        self.population_init_tau = kwargs["population_init_tau"]

        # create a population of n many (P,Q) agents
        self.Ps = random_stochastic_matrix(
            (self.n, self.game.num_states, self.game.num_signals),
            self.population_init_tau,
        )
        self.Qs = random_stochastic_matrix(
            (self.n, self.game.num_signals, self.game.num_states),
            self.population_init_tau,
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
        self.init_tau = None
        if kwargs["population_init_tau"] is not None:
            self.init_tau = kwargs["population_init_tau"]

        self.P = random_stochastic_matrix(
            (self.game.num_states, self.game.num_signals), self.init_tau
        )  # Sender 'population frequencies'
        self.Q = random_stochastic_matrix(
            (self.game.num_signals, self.game.num_states), self.init_tau
        )  # Receiver 'population frequencies'

        # Record first 100 and logspaced values to max_its
        self.steps_to_record = list(range(100)) + [int(x) for x in np.logspace(start=np.log10(100), stop=np.log10(self.max_its), num=99, endpoint=False)] + [self.max_its-1]

    def run(self):
        its = 0
        converged = False
        progress_bar = tqdm(total=self.max_its)
        while not converged:
            its += 1
            progress_bar.update(1)

            P_prev = copy.deepcopy(self.P)
            Q_prev = copy.deepcopy(self.Q)

            point = self.get_point(P_prev, Q_prev)

            # self.game.points.append(self.get_point(P_prev, Q_prev))

            self.evolution_step()  # N.B.: fitness requires population update

            # Check for convergence
            if (
                np.abs(self.P - P_prev).sum() < self.threshold
                and np.abs(self.Q - Q_prev).sum() < self.threshold
            ) or (its == self.max_its):
                converged = True

            # start recording the convergence info
            P_delta = np.abs(self.P - P_prev).sum()
            Q_delta = np.abs(self.Q - Q_prev).sum()
            point = point + (P_delta, Q_delta)

            # We record the first 200 steps and then 100 evenly spaced steps between the current and 20k
            # if (its < 200) or (its % 2000 == 0) or converged:
            # logspaced
            if its in self.steps_to_record or converged:
                # Record data from before evolution step + convergence check
                self.game.ib_encoders.append(rows_zero_to_uniform(normalize_rows(P_prev)))
                self.game.points.append(point)
                self.game.steps_recorded.append(its)

                # won't this lose info about the iteration number? How to work around this without causing index errors?

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

        # --------- Simulate communicative interaction to be imitated ---------

        # probability that an agent observes state s_o given actual s_a
        observation_noise = confusion # `[states, states]`
        interpretation_noise = confusion # `[states, states]`

        # probability that a random sender sends signal w in actual s_a
        sigma = observation_noise @ sender # `[states, signals]`

        # probability that s_r is realized by a random receiver in response to signal w
        rho = receiver @ interpretation_noise # `[signals, states]`

        # --------- Simulate observation/imitation by agents ---------

        # probability that actual state is s_a if random sender produced w
        sigma_inverse = bayes(sigma, prior) # `[signals, states]`

        # probability that s_a is actual if s_o is observed by an agent
        obs_inverse = bayes(confusion, prior) # `[states, states]`

        # probability that a random 'learner' observes s_o and observes a random 'teacher' send signal w
        # marginalize s_a over shape `[s_o, s_a, w]` to get `[s_o, w]`
        sender_imitation = np.sum(
            obs_inverse[:, :, None] * sigma[None, :, :,],
            axis=1,
        ) # `[states, signals]`

        # probability that a random learner will observe a random teacher choose interpretation s_o given signal w
        # marginalize s_r over shape `[w, s_r, s_o]` to get `[w, s_o]`
        receiver_imitation = np.sum(
            interpretation_noise[None, :, :,] * rho[:, :, None],
            axis=1,
        )

        # --------- Expected utilities ---------    

        # EU for sender: function of w, s_o, and current receiver

        # first marginalize s_r over `[s_a, w, s_r]`
        prob_w_given_s_a = np.sum(
            # `[1, w, s_r]` * `[s_a, 1, s_r]`
            rho[None, :, :,] * utility[:, None, :,],
            axis=-1,
        ) # to get shape `[s_a, w]`

        # then marginalize s_a over resulting `[s_o, s_a, w]`
        prob_w_given_s_o = np.sum(
            # `[s_o, s_a, 1]` * `[1, s_a, w]`
            obs_inverse[:, :, None] * prob_w_given_s_a[None, :, :,],
            axis=1,
        ) # to get shape `[s_o, w]`

        eu_sender = prob_w_given_s_o # `[states, signals]`

        # ------------------------------------------

        # EU for receiver: function of s_i, w, and current sender

        # first marginalize s_r over `[s_i, s_a, s_r]`
        prob_s_a_given_s_i = np.sum(
            # `[s_i, 1, s_r]` * `[1, s_a, s_r]` # TODO: order of s_i, s_a?
            interpretation_noise[:, None, :,] * utility[None, :, :,],
            axis=-1,
        ) # to get shape `[s_i, s_a]`

        # then marginalize s_a over `[w, s_i, s_a]`
        prob_w_given_s_i = np.sum(
            # `[w, 1, s_a]` * `[1, s_i, s_a]`
            sigma_inverse[:, None, :,] * prob_s_a_given_s_i[None, :, :,],
            axis=-1,
        ) # to get shape `[w, s_i]`

        eu_receiver = prob_w_given_s_i # `[signals, states]`

        # --------- Discrete-time replicator dynamics updates ---------

        sender_new = normalize_rows(sender_imitation * eu_sender)
        receiver_new = normalize_rows(receiver_imitation * eu_receiver)

        # update instance
        self.P = copy.deepcopy(sender_new)
        self.Q = copy.deepcopy(receiver_new)
        self.warn_if_all_zero()


dynamics_map = {
    "moran_process": MoranProcess,
    "nowak_krakauer": NowakKrakauerDynamics,
    "replicator_diffusion": ReplicatorDiffusionDynamics,
    "imprecise_conditional_imitation": ImpreciseConditionalImitation,
}
