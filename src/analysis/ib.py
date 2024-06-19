

import numpy as np

from ultk.effcomm.rate_distortion import (
    rows_zero_to_uniform, 
    ib_optimal_decoder, 
    ib_encoder_to_point,
    get_ib_bound,
)
from rdot import optimizers, probability, distortions, information

from misc.tools import normalize_rows

from omegaconf import DictConfig
from game.game import Game


def ib_encoder_to_measurements(
    meaning_dists: np.ndarray,
    prior: np.ndarray,
    dist_mat: np.ndarray,
    util_mat: np.ndarray,
    confusion: np.ndarray,
    ib_optimal_encoders: np.ndarray,
    ib_optimal_betas: np.ndarray,
    encoder: np.ndarray,
    decoder: np.ndarray,
) -> tuple[float]:
    """Return (complexity, accuracy, distortion, mse, eu_gamma, kl_eb, min_gnid) point.

    Args:
        meaning_dists: array of shape `(|meanings|, |meanings|)` representing the distribution over world states given meanings.

        prior: array of shape `|meanings|` representing the cognitive source

        dist_mat: array of shape `(|meanings|, |meanings|)` representing pairwise distance/error for computing MSE.

        util_mat: array of shape `dist_mat.shape` representing the pairwise payoff matrix for each actual and finally reconstructed state. This utility is an exponential function of dist_mat, paramterized by `discriminative_need_gamma` in the game.

        confusion: array of shape `(|meanings|, |meanings|)` representing the distribution over world states given meanings.

        ib_optimal_encoders: array of shape `(num_curve_points, |meanings|, |words|)` representing the IB optima that sweep out the curve

        encoder: array of shape `(|meanings|, |words|)` representing P(W | M)

        decoder: array of shape `(|words|, |meanings|)` representing P(M | W). We will always infer the Bayesian optimal decoder for IB Accuracy measurements.
    """
    # NOTE: Here is where we rectify ineffable meanings, by replacing rows of all zeros with uniform distributions.
    # Another option would simple be to drop them.
    # encoder = rows_zero_to_uniform(normalize_rows(encoder))

    bayesian_decoder = ib_optimal_decoder(encoder, prior, meaning_dists)

    complexity, accuracy, distortion = ib_encoder_to_point(
        prior=prior,
        meaning_dists=meaning_dists,
        encoder=encoder,
        decoder=bayesian_decoder,
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
            raise Exception

    if distortion < 0:
        if np.isclose(distortion, 0.0, atol=1e-5):
            distortion = 0.0
        else:
            raise Exception

    # NOTE: the analysis doesn't make sense unless confusion probabilities equal meaning distributions.
    system = meaning_dists @ encoder @ decoder @ meaning_dists

    # rectify ineffability again
    system = rows_zero_to_uniform(normalize_rows(system))
    mse = distortions.expected_distortion(prior, system, dist_mat)

    # Expected Utility, relative to discriminative_need
    eu_gamma = np.sum(prior * (system * util_mat))

    # Expected KL between emergent receiver and the Bayesian optimal inverse of the Sender.
    # D[ R(\hat{x}_o | w) || S_bayes(\hat{x}_o | w) ]
    kl_vec = information.kl_divergence(
        p=decoder, # shape `(words, meanings)`
        q=bayesian_decoder, # `(words, meanings)`
        axis=1, # take entropy of meanings, i.e. sum over 2nd axis
        base=2,
    )

    # Take expectation over p(w)
    pw = probability.marginalize(encoder, meaning_dists @ prior)
    kl_eb = np.sum(pw * kl_vec)

    # TODO: find the min gnid distance to the curve
    # Need to load up the entire curve sigh
    # TODO: I think I should just refactor everything, and measure after simulations instead of during
    gnids_to_curve = [information.gNID(encoder, opt_enc, prior) for opt_enc in ib_optimal_encoders]
    min_index = np.argmin(gnids_to_curve)
    min_gnid = gnids_to_curve[min_index]
    gnid_beta = ib_optimal_betas[min_index]

    return (
        complexity, 
        accuracy, 
        distortion, 
        mse,
        eu_gamma,
        kl_eb,
        min_gnid,
        gnid_beta,
    )


##############################################################################
# IB CURVE ESTIMATION
##############################################################################


from .betas import betas # this is hard-codey

def get_bottleneck(config: DictConfig) -> optimizers.IBResult:
    """Compute the `(complexity, accuracy, comm_cost)` values and optimal encoders corresponding to an Information Bottleneck theoretical bound.

    Args:
        config: A Hydra DictConfig the config file for the experiment.

    Returns:
        a list of `ba.IBResult` namedtuples
    """
    g = Game.from_hydra(config)
    results = get_ib_bound(
        g.prior,
        g.meaning_dists,
        betas=betas,
        ensure_monotonicity=False,
    )
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
    results = optimizers.RateDistortionOptimizer(
        g.prior,
        g.dist_mat,
        betas,
        ensure_monotonicity=True,
    ).get_results()
    return [(item.rate, item.distortion) for item in results if item is not None]
