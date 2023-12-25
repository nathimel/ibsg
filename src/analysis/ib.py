import torch

import numpy as np

from ultk.effcomm.rate_distortion import (
    rows_zero_to_uniform, 
    ib_optimal_decoder, 
    ib_encoder_to_point,
)
from rdot import ba, probability, distortions

from misc.tools import normalize_rows

from omegaconf import DictConfig
from game.game import Game


def ib_encoder_to_measurements(
    meaning_dists: torch.Tensor,
    prior: torch.Tensor,
    dist_mat: torch.Tensor,
    confusion: torch.Tensor,
    encoder: torch.Tensor,
    decoder: torch.Tensor = None,
) -> tuple[float]:
    """Return (complexity, accuracy, distortion, mean squared error) point.

    Args:
        meaning_dists: array of shape `(|meanings|, |meanings|)` representing the distribution over world states given meanings.

        prior: array of shape `|meanings|` representing the cognitive source

        dist_mat: array of shape `(|meanings|, |meanings|)` representing pairwise distance/error for computing MSE.

        confusion: array of shape `(|meanings|, |meanings|)` representing the distribution over world states given meanings.

        encoder: array of shape `(|meanings|, |words|)` representing P(W | M)

        decoder: array of shape `(|words|, |meanings|)` representing P(M | W). If is None, and the Bayesian optimal decoder will be inferred.
    """
    # NOTE: altk requires numpy
    meaning_dists = np.array(meaning_dists)
    prior = np.array(prior)
    dist_mat = np.array(dist_mat)
    confusion = np.array(confusion)
    encoder = np.array(encoder)

    # NOTE: Here is where we rectify ineffable meanings, by replacing rows of all zeros with uniform distributions.
    # Another option would simple be to drop them.
    encoder = rows_zero_to_uniform(normalize_rows(encoder))

    # NOTE: while ib_encoder_to_point does this step, we still need the decoder for the MSE step.
    if decoder is not None:
        decoder = np.array(decoder)
    else:
        decoder = ib_optimal_decoder(encoder, prior, meaning_dists)


    complexity, accuracy, distortion = ib_encoder_to_point(
        meaning_dists,
        prior,
        encoder,
        decoder,
    )

    if complexity < 0:
        if np.isclose(complexity, 0.0, atol=1e-5):
            complexity = 0.0
        else:
            raise Exception

    if accuracy < 0:
        if np.isclose(accuracy, 0.0, atol=1e-5):
            accuracy = 0.0
        else:
            breakpoint()
            raise Exception

    if distortion < 0:
        if np.isclose(distortion, 0.0, atol=1e-5):
            distortion = 0.0
        else:
            raise Exception

    # NOTE: use meaning dists, not confusions!
    # system = confusion @ encoder @ decoder @ confusion
    system = meaning_dists @ encoder @ decoder @ meaning_dists

    # rectify ineffability again
    system = rows_zero_to_uniform(normalize_rows(system))
    mse = distortions.expected_distortion(prior, system, dist_mat)

    return (complexity, accuracy, distortion, mse)


##############################################################################
# IB CURVE ESTIMATION
##############################################################################

# We hard code the list of betas because through trial and error this set of values best sweeps out the curve we're interested in.
betas = np.concatenate(
    [
        # these betas were hand selected to try and evenly flesh out the curve for meaning_dist=-1 as much as possible
        np.linspace(start=0, stop=0.3, num=333),
        
        # very sparse region
        np.linspace(start=0.7, stop=0.77, num=500),
        np.linspace(start=0.69, stop=0.72, num=500),

        np.linspace(start=0.3, stop=0.9, num=333),
        np.linspace(start=0.9, stop=4, num=334),
        np.linspace(start=4, stop=2**7, num=500),
    ]
)
# unique
betas = list(sorted(set(betas.tolist())))


def get_bottleneck(config: DictConfig) -> ba.IBResult:
    """Compute the `(complexity, accuracy, comm_cost)` values and optimal encoders corresponding to an Information Bottleneck theoretical bound.

    Args:
        config: A Hydra DictConfig the config file for the experiment.

    Returns:
        a list of `ba.IBResult` namedtuples
    """
    g = Game.from_hydra(config)
    pxy = probability.joint(g.meaning_dists, g.prior)
    results = ba.IBOptimizer(
        pxy=pxy,
        betas=betas,
    ).get_results()
    return [item for item in results if item is not None]

##############################################################################
# RD CURVE ESTIMATION
##############################################################################

def get_rd_curve(config: DictConfig) -> list[tuple[float]]:
    """Compute the `(complexity, comm_cost)` values corresponding to a Rate-Distortion (with MSE distortion) theoretical bound.

    Args:
        config: A Hydra DictConfig the config file for the experiment.

    Returns:
        a list of `(complexity, comm_cost)` coordinates

    """
    g = Game.from_hydra(config)
    results = ba.RateDistortionOptimizer(
        g.prior,
        g.dist_mat,
        betas,
    ).get_results()
    return [(item.rate, item.distortion) for item in results if item is not None]
