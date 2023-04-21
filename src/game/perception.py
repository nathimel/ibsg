"""Functions for computing similarity distributions and utility functions for signaling games."""

import numpy as np
from game.languages import StateSpace, State

# distance measures
def abs_dist(t: int, u: int) -> float:
    return np.abs(t - u)


def squared_dist(t: int, u: int) -> float:
    return (t - u) ** 2


distance_measures = {
    "abs_dist": abs_dist,
    "squared_dist": squared_dist,
}

def generate_dist_matrix(
    universe: StateSpace,
    distance: str = "squared_dist",
) -> np.ndarray:
    """Given a universe, compute the distance for every pair of points in the universe.

    Args:
        universe: a StateSpace such that objects bear Euclidean distance relations

        distance: a string corresponding to the name of a pairwise distance function on states, one of {'abs_dist', 'squared_dist'}
    
    Returns:

        an array of shape `(|universe|, |universe|)` representing pairwise distances
    """
    return np.array(
        [
            np.array(
                [
                    distance_measures[distance](t.data, u.data)
                    for u in universe.referents
                ]
            )
            for t in universe.referents
        ]
    )


def generate_sim_matrix(universe: StateSpace, gamma: float, dist_mat: np.ndarray) -> np.ndarray:
    """Given a universe, compute a similarity score for every pair of points in the universe.

    NB: this is a wrapper function that generates the similarity matrix using the data contained in each State.

    Args:
        universe: a StateSpace such that objects bear Euclidean distance relations

        similarity: a string corresponding to the name of a pairwise similarity function on states
    """

    return np.array(
        [
            exp(
                target=t.data,
                objects=[u.data for u in universe.referents],
                gamma=gamma,
                dist_mat=dist_mat,
            )
            for t in universe.referents
        ]
    )


##############################################################################
# SIMILARITY / UTILITY functions
##############################################################################
# N.B.: we use **kwargs so that sim_func() can have the same API

def exp(
    target: int,
    objects: np.ndarray,
    gamma: float,
    dist_mat: np.ndarray,
) -> np.ndarray:
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
    return np.exp(np.array([exp_term(target, u) for u in objects]))


def sim_utility(x: State, y: State, sim_mat: np.ndarray) -> float:
    return sim_mat[int(x.data), int(y.data)]
