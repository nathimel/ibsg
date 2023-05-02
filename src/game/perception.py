"""Functions for computing similarity distributions and utility functions for signaling games."""

import torch

# distance measures
def hamming_dist(t: int, u: int) -> float:
    # indicator
    return t == u


def abs_dist(t: int, u: int) -> float:
    return abs(t - u)


def squared_dist(t: int, u: int) -> float:
    return (t - u) ** 2


distance_measures = {
    "abs_dist": abs_dist,
    "squared_dist": squared_dist,
    "hamming_dist": hamming_dist,
}

def generate_dist_matrix(
    universe: list[int],
    distance: str = "squared_dist",
) -> torch.Tensor:
    """Given a universe, compute the distance for every pair of points in the universe.

    Args:
        universe: a list of ints representing the universe of states (referents), and objects bear Euclidean distance relations

        distance: a string corresponding to the name of a pairwise distance function on states, one of {'abs_dist', 'squared_dist'}
    
    Returns:

        an array of shape `(|universe|, |universe|)` representing pairwise distances
    """
    return torch.Tensor(
        [
                [
                    distance_measures[distance](t, u)
                    for u in universe
                ]
            for t in universe
        ]
    )


def generate_sim_matrix(universe: list[int], gamma: float, dist_mat: torch.Tensor) -> torch.Tensor:
    """Given a universe, compute a similarity score for every pair of points in the universe.

    NB: this is a wrapper function that generates the similarity matrix using the data contained in each State.

    Args:
        universe: a list of ints representing the universe of states (referents), and objects bear Euclidean distance relations

        similarity: a string corresponding to the name of a pairwise similarity function on states
    """
    return torch.stack(
        [
            exp(
                target=t,
                objects=universe,
                gamma=gamma,
                dist_mat=dist_mat,
            )
            for t in universe
        ]
    )


##############################################################################
# SIMILARITY / UTILITY functions
##############################################################################
# N.B.: we use **kwargs so that sim_func() can have the same API

def exp(
    target: int,
    objects: torch.Tensor,
    gamma: float,
    dist_mat: torch.Tensor,
) -> torch.Tensor:
    """The (unnormalized) exponential function sim(x,y) = exp(-gamma * d(x,y)).

    Args:
        target: value of state

        objects: set of points with measurable similarity values

        gamma: perceptual discriminability parameter

        distance: a string corresponding to the name of a pairwise distance function on states, one of {'abs_dist', 'squared_dist'}

    Returns:
        a similarity matrix representing pairwise inverse distance between states
    """
    exp_term = lambda t, u: -gamma * dist_mat[t, u]
    return torch.exp(torch.stack([exp_term(target, u) for u in objects]))


def sim_utility(x: int, y: int, sim_mat: torch.Tensor) -> float:
    return sim_mat[x, y]
