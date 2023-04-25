"""Script to compute the IB curve."""
import hydra
import os
import torch
from altk.effcomm.information import ib_encoder_to_point
from altk.effcomm.information import get_ib_curve
from omegaconf import DictConfig
from game.game import Game
from misc import util
from tqdm import tqdm
from multiprocessing import cpu_count, Pool

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
    """Compute the optimal IB encoder, using the IB-method. Implementation belongs to Futrell.

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
    """Reverse deterministic annealing (Zaslavsky and Tishby, 2019)"""
    # load params
    evol_game = Game.from_hydra(config)
    prior = evol_game.prior
    meaning_dists = evol_game.meaning_dists
    max_signals = evol_game.num_signals    

    encoders = []
    coordinates = []

    betas = torch.logspace(config.game.beta_start, config.game.beta_stop, config.game.steps)

    # Multiprocessing
    if len(prior) > 25:
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
        coordinates = [ib_encoder_to_point(encoder, meaning_dists, prior) for encoder in encoders]

    else:
        for beta in tqdm(betas):
            encoder = ib_blahut_arimoto(max_signals, beta, prior, meaning_dists)
            coordinates.append(ib_encoder_to_point(encoder, meaning_dists, prior))
            encoders.append(encoder)

    return {
        "encoders": encoders,
        "coordinates": coordinates,
    }


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    util.set_seed(config.seed)

    # save one curve for multiple analyses
    game_dir = os.getcwd().replace(config.filepaths.simulation_subdir, "")
    curve_fn = os.path.join(game_dir, config.filepaths.curve_points_save_fn)

    # curve_points = get_ib_curve(config)["coordinates"]
    g = Game.from_hydra(config)
    curve_points = torch.tensor(get_ib_curve(g.prior, g.meaning_dists)).flip([0,1])

    util.save_points_df(curve_fn, util.points_to_df(curve_points))

if __name__ == "__main__":
    main()
