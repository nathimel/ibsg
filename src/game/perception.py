"""Functions for computing similarity distributions and utility functions for signaling games."""

import torch
from altk.language.semantics import Universe, Referent


# distance measures
def hamming_dist(t: torch.Tensor, u: torch.Tensor) -> float:
    # indicator
    return 1 - torch.equal(t, u)


def abs_dist(t: torch.Tensor, u: torch.Tensor) -> float:
    return torch.linalg.vector_norm(t - u)


def squared_dist(t: torch.Tensor, u: torch.Tensor) -> float:
    return abs_dist(t, u) ** 2


distance_measures = {
    "abs_dist": abs_dist,
    "squared_dist": squared_dist,
    "hamming_dist": hamming_dist,
}


def referent_to_tensor(referent: Referent):
    return torch.tensor(referent.point, dtype=float)


def generate_dist_matrix(
    universe: Universe,
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
                distance_measures[distance](
                    referent_to_tensor(t),
                    referent_to_tensor(u),
                )
                for u in universe.referents
            ]
            for t in universe.referents
        ]
    )


def generate_sim_matrix(
    universe: Universe, gamma: float, dist_mat: torch.Tensor
) -> torch.Tensor:
    """Given a universe, compute a similarity score for every pair of points in the universe.

    NB: this is a wrapper function that generates the similarity matrix using the data contained in each State.

    Args:
        universe: a list of ints representing the universe of states (referents), and objects bear Euclidean distance relations

        similarity: a string corresponding to the name of a pairwise similarity function on states
    """
    return torch.stack(
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
    dist_mat: torch.Tensor,
) -> torch.Tensor:
    """The (unnormalized) exponential function sim(x,y) = exp(-gamma * d(x,y)).

    Args:
        target: index for a Referent in the distance matrix

        objects: list of Referents with measurable similarity values

        gamma: perceptual discriminability parameter

        distance: a string corresponding to the name of a pairwise distance function on states, one of {'abs_dist', 'squared_dist'}

    Returns:
        a similarity matrix representing pairwise inverse distance between states
    """
    return torch.exp(
        torch.stack(
            [-gamma * dist_mat[target_index, idx] for idx in range(len(referents))]
        )
    )


def sim_utility(x: int, y: int, sim_mat: torch.Tensor) -> float:
    return sim_mat[x, y]
