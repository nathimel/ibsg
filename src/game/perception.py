"""Functions for computing similarity distributions and utility functions for signaling games."""

import numpy as np
from ultk.language.semantics import Universe, Referent
from misc.tools import normalize_rows

# distance measures
def hamming_dist(t: np.ndarray, u: np.ndarray) -> float:
    # indicator
    return 1 - np.array_equal(t, u)


def abs_dist(t: np.ndarray, u: np.ndarray) -> float:
    return np.linalg.norm(t - u)


def squared_dist(t: np.ndarray, u: np.ndarray) -> float:
    return abs_dist(t, u) ** 2


distance_measures = {
    "abs_dist": abs_dist,
    "squared_dist": squared_dist,
    "hamming_dist": hamming_dist,
}


def referent_to_ndarray(referent: Referent):
    return np.array(referent.point, dtype=float)


def generate_confusion_matrix(
    universe: Universe,
    gamma: float, 
    dist_mat: np.ndarray,
) -> np.ndarray:
    """Given a universe, confusion gamma (NOTE: will be exponentiated by 10), and distance matrix, generate a conditional probability distribution over points in the universe given points in the universe."""
    return normalize_rows(
            generate_sim_matrix(
                universe, 10 ** gamma, dist_mat
            )
        )

def generate_dist_matrix(
    universe: Universe,
    distance: str = "squared_dist",
) -> np.ndarray:
    """Given a universe, compute the distance for every pair of points in the universe.

    Args:
        universe: a list of ints representing the universe of states (referents), and objects bear Euclidean distance relations

        distance: a string corresponding to the name of a pairwise distance function on states, one of {'abs_dist', 'squared_dist'}

    Returns:

        an array of shape `(|universe|, |universe|)` representing pairwise distances
    """
    return np.array(
        [
            [
                distance_measures[distance](
                    referent_to_ndarray(t),
                    referent_to_ndarray(u),
                )
                for u in universe.referents
            ]
            for t in universe.referents
        ]
    )


def generate_sim_matrix(
    universe: Universe, gamma: float, dist_mat: np.ndarray
) -> np.ndarray:
    """Given a universe, compute a similarity score for every pair of points in the universe.

    NB: this is a wrapper function that generates the similarity matrix using the data contained in each State.

    Args:
        universe: a list of ints representing the universe of states (referents), and objects bear Euclidean distance relations

        similarity: a string corresponding to the name of a pairwise similarity function on states
    """
    return np.stack(
        [
            exp(
                target_index=idx,
                referents=universe.referents,
                gamma=gamma,
                dist_mat=dist_mat,
            )
            for idx in range(len(universe))
        ]
    )


##############################################################################
# SIMILARITY / UTILITY functions
##############################################################################
# N.B.: we use **kwargs so that sim_func() can have the same API


def exp(
    target_index: int,
    referents: list[Referent],
    gamma: float,
    dist_mat: np.ndarray,
) -> np.ndarray:
    """The (unnormalized) exponential function sim(x,y) = exp(-gamma * d(x,y)).

    Args:
        target: index for a Referent in the distance matrix

        objects: list of Referents with measurable similarity values

        gamma: perceptual discriminability parameter

        distance: a string corresponding to the name of a pairwise distance function on states, one of {'abs_dist', 'squared_dist'}

    Returns:
        a similarity matrix representing pairwise inverse distance between states
    """
    return np.exp(
        np.stack(
            [-gamma * dist_mat[target_index, idx] for idx in range(len(referents))]
        )
    )


def sim_utility(x: int, y: int, sim_mat: np.ndarray) -> float:
    return sim_mat[x, y]
