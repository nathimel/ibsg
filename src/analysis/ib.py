import os
import torch
import numpy as np

from altk.effcomm.util import rows_zero_to_uniform, bayes
from altk.effcomm import information
from altk.effcomm.information import (
    expected_distortion,
    ib_encoder_to_point,
    get_rd_curve,
)

from misc.tools import normalize_rows

from omegaconf import DictConfig
from game.game import Game
from tqdm import tqdm
from multiprocessing import cpu_count, Pool


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
    # encoder = np.array(encoder)
    # NOTE: Here is where we rectify ineffable meanings, by replacing rows of all zeros with uniform distributions.
    encoder = rows_zero_to_uniform(normalize_rows(encoder))
    if decoder is not None:
        decoder = np.array(decoder)
    else:
        decoder = bayes(encoder, prior)
    
    complexity, accuracy, distortion = ib_encoder_to_point(
        meaning_dists,
        prior,
        encoder,
        decoder,
    )

    if complexity < 0:
        if np.isclose(complexity, 0., atol=1e-5):
            complexity = 0.
        else:
            raise Exception
    
    if accuracy < 0:
        if np.isclose(accuracy, 0., atol=1e-5):
            accuracy = 0.
        else:
            raise Exception
    
    if distortion < 0:
        if np.isclose(distortion, 0., atol=1e-5):
            distortion = 0.
        else:
            raise Exception

    # NOTE: use meaning dists, not confusions!
    # system = confusion @ encoder @ decoder @ confusion
    system = meaning_dists @ encoder @ decoder @ meaning_dists

    # rectify ineffability again
    system = rows_zero_to_uniform(normalize_rows(system))
    mse = expected_distortion(prior, system, dist_mat)

    return (complexity, accuracy, distortion, mse)


# IB CURVE ESTIMATION

def get_bottleneck(config: DictConfig) -> list[tuple]:
    """Compute the `(complexity, accuracy, comm_cost)` values and optimal encoders corresponding to an Information Bottleneck theoretical bound. 
    
    The config specifies whether to use the embo package, which is faster and filters non-monotonicity, or a homebuilt version, which can be useful for  sanity checks with minimal overhead.

    Args:
        config: A Hydra DictConfig the config file for the experiment.

    Returns:
        bottleneck: A dict of the form
            {
            "encoders": a list of encoders of shape `(num_meanings, num_signals)`,\n
            "coordinates": a list of `(complexity, accuracy, comm_cost)` coordinates,\n
            "betas": a list of the beta values used for IB method.
            }
    """
    g = Game.from_hydra(config)
    func = config.game.ib_bound_function
    if func == "embo":
        bottleneck = information.get_bottleneck(
            prior=g.prior,
            meaning_dists=g.meaning_dists,
            maxbeta=g.maxbeta,
            minbeta=g.minbeta,
            numbeta=g.numbeta,
            processes=g.num_processes,
        )
    
    elif func == "homebuilt":
        bottleneck = get_ib_curve_(config)

    else:
        raise ValueError(f"IB bound functions include 'embo', 'homebuilt', but received {func}.")

    # interestingly, we sometimes need to normalize encoders from IB method
    bottleneck["encoders"] = [normalize_rows(torch.tensor(enc)) for enc in bottleneck["encoders"]]

    # convert from numpy to list
    if isinstance(bottleneck["betas"], np.ndarray):
        bottleneck["betas"] = bottleneck["betas"].tolist()

    return bottleneck

def ensure_monotonic_bound(curve_points: torch.Tensor):
    """Fix any randomness in curve leading to nonmonotonicity."""
    # We can fix random noise that tends to occur at the high complexity region by ensuring monotonicity: that as we increase beta, accuracy must increase. If accuracy does not increase, drop these values. I'll implement this if embo ever gives trouble and I want to move to homebuilt.
    raise NotImplementedError


def ib_blahut_arimoto(
    num_words: int,
    beta: float,
    p_M: torch.Tensor,
    p_U_given_M: torch.Tensor,
    max_its: int = 100,
    eps: float = 1e-5,
    ignore_converge: bool = False,    
    init_temperature: float = 1,
) -> torch.Tensor:
    """Compute the optimal IB encoder, using the IB-method.

    Args:
        num_words: size of the target support (vocabulary size)

        beta: related to the slope of the IB curve, corresponds to softmax temperature.

        p_M: prior probability distribution over source variables, P(m) (i.e. the cognitive source)

        p_U_given_M: the Bayes' optimal decoder P(U|M) (i.e. the listener meaning)

        max_its: the number of iterations to run IB method

        eps: accuracy required by the algorithm: the algorithm stops if there is no change in distortion value of more than 'eps' between consequtive iterations

        ignore_converge: whether to run the optimization until `max_it`, ignoring the stopping criterion specified by `eps`.        

        init_temperature: specifies the entropy of the encoder's initialization distribution

    Returns:

        lnq: the optimal encoder for gamma.
    """
    # convert to logspace for ease
    lnp_M = torch.log(p_M)
    lnp_U_given_M = torch.log(p_U_given_M)

    # add dummy dimensions to make things easy
    U_dim, M_dim, W_dim = -3, -2, -1
    lnp_M = lnp_M[None, :, None]  # shape 1M1
    lnp_U_given_M = lnp_U_given_M[:, :, None]  # shape UM1

    # q(w|m) is an M x W matrix
    lnq = ((1 / init_temperature) * torch.randn(1, lnp_M.shape[M_dim], num_words)).log_softmax(
        W_dim
    )  # shape 1MW

    it = 0
    d = 2 * eps
    converged = False
    while not converged:
        it += 1
        d_prev = d

        # start by getting q(m,w) = p(m) q(w|m)
        lnq_joint = lnp_M + lnq

        # get q0(w) = \sum_m q(m, w)
        lnq0 = lnq_joint.logsumexp(M_dim, keepdim=True)  # shape 11W

        # to get the KL divergence,
        # first need p(m | w) =  q(m, w) / q0(w)
        lnq_inv = lnq_joint - lnq0  # shape 1MW

        # now need q(u|w) = \sum_m p(m | w) p(u | m)
        lnquw = (lnq_inv + lnp_U_given_M).logsumexp(M_dim, keepdim=True) # shape U1W

        # now need \sum_u p(u|m) ln q(u|w); use torch.xlogy for 0*log0 case
        d = -(lnp_U_given_M.exp() * lnquw).sum(U_dim, keepdim=True)  # shape 1MW

        # finally get the encoder
        lnq = (lnq0 - beta * d).log_softmax(W_dim)  # shape 1MW

        # convergence check
        if ignore_converge:
            converged = it == max_its
        else:
            converged = it == max_its or (d - d_prev).abs().sum() < eps

    return lnq.squeeze(U_dim).exp()  # remove dummy U dimension, convert from logspace

def get_ib_curve_(config: DictConfig):
    """Reverse deterministic annealing (Zaslavsky and Tishby, 2019).
    
    Args: 
        config: A Hydra DictConfig the config file for the experiment.

    Returns: a dict of containing the IB optimal encoders and their coordinates, of the form

        {
            "encoders": a list of encoders of shape `(num_meanings, num_signals)`,\n
            "coordinates": a list of `(complexity, accuracy, comm_cost)` coordinates,\n
            "betas": a list of the beta values used for IB method.
        }

    """
    # load params
    breakpoint()

    evol_game = Game.from_hydra(config, cwd = os.getcwd())
    prior = evol_game.prior
    meaning_dists = evol_game.meaning_dists
    max_signals = evol_game.num_signals    

    encoders = []
    coordinates = []

    betas = torch.linspace(config.game.maxbeta, config.game.minbeta, config.game.numbeta)
    # curve can get sparse in the high-interest regions, beta 1.02-1.1

    # Multiprocessing
    if len(prior) > 100:
        num_processes = cpu_count()
        with Pool(num_processes) as p:
            async_results = [
                p.apply_async(
                ib_blahut_arimoto, 
                args=[max_signals, beta, prior, meaning_dists],
                )
                for beta in betas
            ]
            p.close()
            p.join()
        encoders = [async_result.get() for async_result in tqdm(async_results)]
        coordinates = [ib_encoder_to_point(meaning_dists, prior, normalize_rows(encoder)) for encoder in encoders]

    else:
        for beta in tqdm(betas):
            encoder = ib_blahut_arimoto(max_signals, beta, prior, meaning_dists)
            coordinates.append(ib_encoder_to_point(meaning_dists, prior, encoder))
            encoders.append(encoder)

    return {
        "encoders": encoders,
        "coordinates": coordinates,
        "betas": betas, # return all betas for now
    }