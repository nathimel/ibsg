import copy

import numpy as np

from altk.effcomm.information import ib_encoder_decoder_to_point
from game.game import Game
from game.graph import generate_adjacency_matrix
from misc.tools import random_stochastic_matrix

from tqdm import tqdm

class Dynamics:
    def __init__(self, game: Game, **kwargs) -> None:
        self.game = game

    def run(self):
        raise NotImplementedError
    
def population_mean_weights(population: list[tuple[np.ndarray]]) -> tuple[float]:
    """Given a list of the population of symmetric agents (Sender, Receiver), return the average agent (Sender, Receiver) weights."""
    # vectorize if |states| = |signals|
    if population[0][1].shape[0] == population[0][0].shape[1]:
        return np.mean(population, axis=0)

    Ps, Qs = [], []
    for p, q in population:
        Ps.append(p)
        Qs.append(q)
    return np.mean(Ps, axis=0), np.mean(Qs, axis=0)


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
    
class MoranProcess(Dynamics):
    def __init__(self, game: Game, **kwargs) -> None:
        super().__init__(game, **kwargs)
        self.max_its = kwargs["max_its"]
        self.threshold = kwargs["threshold"]
        self.n = kwargs["population_size"]

        # create a population of n many (P,Q) agents
        Ps = random_stochastic_matrix((self.n, self.game.num_states, self.game.num_signals), kwargs["population_init_temp"])
        Qs = random_stochastic_matrix((self.n, self.game.num_states, self.game.num_states), kwargs["population_init_temp"])
        self.population = list(zip(*(Ps, Qs)))

        # define the adjacency matrix for the environment of interacting agents
        self.adj_mat = generate_adjacency_matrix(self.n)

    def run(self) -> None:
        """Simulate evolution in the finite population by running the Moran process, at each iteration randomly replacing an individual with an (randomly selected proportional to fitness) agent's offspring."""
        ib_point = lambda pop_mean: ib_encoder_decoder_to_point(*pop_mean, self.game.meaning_dists, self.game.prior)

        fitnesses = self.measure_fitness()
        pop_mean = population_mean_weights(self.population)

        i = 0
        converged = False
        progress_bar = tqdm(total=self.max_its)
        while not converged:
            i += 1
            progress_bar.update(1)

            pop_mean_prev = copy.deepcopy(pop_mean)

            # track data
            self.game.ib_points.append(ib_point(pop_mean))

            i = np.random.choice(self.n, p=fitnesses) # birth
            j = np.random.choice(self.n) # death

            # replace the random deceased with fitness-sampled offspring
            self.population[j] = copy.deepcopy(self.population[i])
            # N.B.: no update to adj_mat necessary

            # update population fitnesses and mean behavior
            fitnesses = self.measure_fitness()
            pop_mean = population_mean_weights(self.population)

            # Check for convergence
            if (np.abs(pop_mean[0] - pop_mean_prev[0]).sum() < self.threshold
            and np.abs(pop_mean[1] - pop_mean_prev[1]).sum() < self.threshold) or (i == self.max_its):
                converged = True

        progress_bar.close()
        np.set_printoptions(suppress=True)
        print(pop_mean[0])

    
    def measure_fitness(self) -> np.ndarray:
        """Measure the fitness of communicating individuals in the population.
        
        Returns:
            a 1D array of floats (normalized to [0,1]) of shape `num_agents` such that fitnesses[i] corresponds to the ith individual.
        """
        payoffs = np.zeros(self.n) # 1D, since fitness is symmetric

        # iterate over every adjacent pair in the graph
        for i in range(self.n): 
            for j in range(self.n):
                # TODO: generalize to stochastic behavior for real-valued edge weights
                if not self.adj_mat[i,j]:
                    continue
                agent_i = self.population[i]
                agent_j = self.population[j]

                # accumulate payoff for interaction symmetrically
                payoff = self.fitness(agent_i, agent_j)
                payoffs[i] += payoff
                payoffs[j] += payoff
                
        return (payoffs / payoffs.sum()).astype(float)
    
    def fitness(self, agent_a: tuple[np.ndarray], agent_b: tuple[np.ndarray]) -> float:
        """Compute pairwise fitness as F[L, L'] = F[(P, Q'), (P', Q)] = 1/2(f(P,Q')) + 1/2(f(P', Q))

        where f(X,Y) = sum( Prior @ Meanings @ X @ Y * Utility )
        """
        P, Q = agent_a
        P_, Q_ = agent_b

        f = lambda X,Y: np.sum(np.diag(self.game.prior) @ self.game.meaning_dists @ X @ Y * self.game.utility)

        return (f(P, Q) + f(P_, Q_)) / 2.0


dynamics_map = {
    "moran_process": MoranProcess,
    "replicator_dynamics": NoisyDiscreteTimeReplicatorDynamics,
}