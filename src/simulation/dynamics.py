import copy

import torch

from altk.effcomm.information import ib_encoder_decoder_to_point
from game.game import Game
from game.graph import generate_adjacency_matrix
from misc.tools import random_stochastic_matrix

from tqdm import tqdm

##############################################################################
# Base classes
##############################################################################


class Dynamics:
    def __init__(self, game: Game, **kwargs) -> None:
        self.game = game
        self.ib_point = lambda encoder, decoder: ib_encoder_decoder_to_point(encoder, decoder, self.game.meaning_dists, self.game.prior)

    def run(self):
        raise NotImplementedError


class FinitePopulationDynamics(Dynamics):
    def __init__(self, game: Game, **kwargs) -> None:
        """
        Args:
            max_its: int maximum number of steps to run evolution

            threshold: a float controlling convergence of evolution

            init_beta: a float controlling the sharpness of the seed population (composed of a Sender P and Receiver Q) agents' distributions        
        """
        super().__init__(game, **kwargs)
        self.max_its = kwargs["max_its"]
        self.threshold = kwargs["threshold"]
        self.n = kwargs["population_size"]

        # create a population of n many (P,Q) agents
        self.Ps = random_stochastic_matrix((self.n, self.game.num_states, self.game.num_signals), kwargs["population_init_beta"])
        self.Qs = random_stochastic_matrix((self.n, self.game.num_states, self.game.num_states), kwargs["population_init_beta"])

        # define the adjacency matrix for the environment of interacting agents
        self.adj_mat = generate_adjacency_matrix(self.n)

    def population_mean_weights(self) -> tuple[float]:
        """Return the average agent (Sender, Receiver) weights."""
        return torch.mean(self.Ps, dim=0), torch.mean(self.Qs, dim=0)

    def measure_fitness(self) -> torch.Tensor:
        """Measure the fitness of communicating individuals in the population.
        
        Returns:
            a 1D array of floats (normalized to [0,1]) of shape `num_agents` such that fitnesses[i] corresponds to the ith individual.
        """
        payoffs = torch.zeros(self.n) # 1D, since fitness is symmetric

        # iterate over every adjacent pair in the graph
        for i in range(self.n): 
            for j in range(self.n):
                # TODO: generalize to stochastic behavior for real-valued edge weights
                if not self.adj_mat[i,j]:
                    continue
                # accumulate payoff for interaction symmetrically
                payoff = self.fitness(
                    p = self.Ps[i],
                    q = self.Qs[i],
                    p_ = self.Ps[j],
                    q_ = self.Qs[j],
                )
                payoffs[i] += payoff
                payoffs[j] += payoff
                
        return payoffs / payoffs.sum()
    
    def fitness(
        self, 
        p: torch.Tensor,
        q: torch.Tensor,
        p_: torch.Tensor,
        q_: torch.Tensor,
    ) -> float:
        """Compute pairwise fitness as F[L, L'] = F[(P, Q'), (P', Q)] = 1/2(f(P,Q')) + 1/2(f(P', Q))

        where f(X,Y) = sum( Prior @ Meanings @ X @ Y * Utility )
        """
        f = lambda X,Y: torch.sum(torch.diag(self.game.prior) @ self.game.meaning_dists @ X @ Y * self.game.utility)
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
            self.game.ib_points.append(self.ib_point(mean_p, mean_q))

            self.evolution_step()

            # update population fitnesses and mean behavior
            mean_p, mean_q = self.population_mean_weights()

            # Check for convergence
            if (torch.abs(mean_p - mean_p_prev).sum() < self.threshold
            and torch.abs(mean_q - mean_q_prev).sum() < self.threshold
            ) or (i == self.max_its):
                converged = True

        progress_bar.close()
        torch.set_printoptions(sci_mode=False)
        print(mean_p)

    def evolution_step(self):
        """The step of evolution that varies between different stochastic processes modeling evolution.
        """
        raise NotImplementedError


##############################################################################
# Nowak and Krakauer
##############################################################################


def mutate(p, num_samples):
    eye = torch.eye(p.shape[-1])
    sample_indices = torch.stack([torch.multinomial(sub_p, num_samples, replacement=True) for sub_p in p])
    samples = eye[sample_indices]
    return samples.mean(axis=-2)


class NowakKrakauerDynamics(FinitePopulationDynamics):
    def __init__(self, game: Game, **kwargs) -> None:
        super().__init__(game, **kwargs)
        self.num_samples = kwargs["num_samples"]

    def evolution_step(self):
        """Children learn the language of their parents by sampling their responses to objects (Nowak and Krakauer, 1999)."""

        fitnesses = self.measure_fitness()

        new_population = torch.multinomial(fitnesses, self.n, replacement=True)
        self.Ps = mutate(self.Ps[new_population], self.num_samples)
        self.Qs = mutate(self.Qs[new_population], self.num_samples)


##############################################################################
# Frequency-Dependent Moran Process
##############################################################################


class MoranProcess(FinitePopulationDynamics):
    def __init__(self, game: Game, **kwargs) -> None:
        super().__init__(game, **kwargs)

    def evolution_step(self):
        """Simulate evolution in the finite population by running the Moran process, at each iteration randomly replacing an individual with an (randomly selected proportional to fitness) agent's offspring."""            
        fitnesses = self.measure_fitness()

        i = torch.multinomial(fitnesses, 1) # birth
        j = torch.multinomial(torch.ones_like(fitnesses), 1) # death

        # replace the random deceased with fitness-sampled offspring
        self.Ps[j] = copy.deepcopy(self.Ps[i])
        self.Qs[j] = copy.deepcopy(self.Qs[j])
        # N.B.: no update to adj_mat necessary

class NoisyDiscreteTimeReplicatorDynamics(Dynamics):
    def __init__(self, game: Game, **kwargs) -> None:
        super().__init__(game, **kwargs)
        self.max_its = kwargs["max_its"]
        self.threshold = kwargs["threshold"]

    def run(self):
        """Simulate evolution of strategies in a near-infinite population of agents x using a discrete-time version of the replicator equation:

            x_i' = x_i * ( f_i(x) - sum_j f_j(x_j) )

        Changes in agent type (pure strategies) depend only on their frequency and their fitness.
        """
        raise NotImplementedError

dynamics_map = {
    "moran_process": MoranProcess,
    "nowak_krakauer": NowakKrakauerDynamics,
    "replicator_dynamics": NoisyDiscreteTimeReplicatorDynamics,
}