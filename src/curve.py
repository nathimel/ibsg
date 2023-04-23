"""Script to compute the IB curve."""
import hydra
import os
import torch
from omegaconf import DictConfig
from altk.effcomm.information import ib_encoder_to_point
from game.game import Game
from misc import util

def ib_method(
    px: torch.Tensor, 
    py_x: torch.Tensor, 
    beta: float, 
    init: torch.Tensor, 
    max_its: int = 20,
    ) -> torch.Tensor:
    """IB method (a Blahut Arimoto iteration) introduced by Tishby (1999). Implementation credit to https://github.com/mahowak/deictic_adverbs/blob/ba589a27ab04d8cccda1ca5c9057ad7b506aaedd/run_ib_new.py#L69
    
    Args: 
        px: tensor of shape `|X|` representing prior over the universe.

        py_x: tensor of shape `(|X|, |Y|)` representing the meaning distribution over world states.

        beta: float in [0, \infty] controlling trade-off between complexity and accuracy

        init: tensor of shape `(|X|, |Z|)` representing the initial encoder to begin BA iteration.

        max_its: int setting the maximum number of iterations.

    Returns:
        pz_x: tensor of shape `(|X|, |Z|)` representing the optimal encoder mapping meanings to words.
    """
    # initialize encoder q(z|x)
    qz_x = init
    py_x = py_x[:, None, :] # shape (X, 1,  Y)
    px = px[:, None] # shape (X, 1)

    # Blahut-Arimoto iteration to find the minimizing q(z|x)
    for _ in range(max_its):
        qxz = px * qz_x # Joint q(x,z), shape (X, Z)
        qz = qxz.sum(dim=0, keepdim=True) # Marginal q(z), shape (1, Z)
        qy_z = ((qxz / qz)[:, :, None] * py_x).sum(dim=0, keepdim=True) #  decoder q(y|z), shape (1, Z, Y)

        # expected distortion over Y
        # use xlogy for 0*log0 case
        d = ( 
            torch.xlogy(py_x, py_x) 
            - torch.xlogy(py_x, qy_z) # -D_KL[p(y|x) || q(y|z)]
        ).sum(axis=-1) # shape (X, Z)

        # encoder q(z|x) = 1/Z q(z) exp(-beta * d)
        qz_x = torch.log_softmax(qz - beta*d, dim=-1)

    return qz_x


def ib_blahut_arimoto(
    beta: float,
    pM: torch.Tensor,
    pU_given_M: torch.Tensor,
    init: torch.Tensor,
    num_steps: int = 10,
    init_temperature: float = 1,
) -> torch.Tensor:
    """Compute the optimal IB encoder, using the IB-method.

    Args:
        beta: related to the slope of the IB curve, corresponds to softmax temperature.

        pM: prior probability distribution over source variables, P(m) (i.e. the cognitive source)

        pU_given_M: the Bayes' optimal decoder P(U|M) (i.e. the listener meaning)

        init: initial encoder

        num_steps: the number of iterations to run IB method

        init_temperature: specifies the entropy of the encoder's initialization distribution

    Returns:

        q: the optimal encoder for beta.
    """
    lnp_M = torch.log(pM)
    lnp_U_given_M = torch.log(pU_given_M)
    W = len(pM) # max vocab size

    # add dummy dimensions to make things easy
    U_dim, M_dim, W_dim = -3, -2, -1
    lnp_M = lnp_M[None, :, None]  # shape 1M1
    lnp_U_given_M = lnp_U_given_M[:, :, None]  # shape UM1

    # q(w|m) is an M x W matrix
    # lnq = ((1 / init_temperature) * torch.randn(1, lnp_M.shape[M_dim], W)).log_softmax(
    #     W_dim
    # )  # shape 1MW
    lnq = init.exp()

    for i in range(num_steps):
        # start by getting q(m,w) = p(m) q(w|m)
        lnq_joint = lnp_M + lnq

        # get q0(w) = \sum_m q(m, w)
        lnq0 = lnq_joint.logsumexp(M_dim, keepdim=True)  # shape 11W

        # to get the KL divergence,
        # first need p(m | w) =  q(m, w) / q0(w)
        lnq_inv = lnq_joint - lnq0  # shape 1MW

        # now need q(u|w) = \sum_m p(m | w) p(u | m)
        lnquw = (lnq_inv + lnp_U_given_M).logsumexp(M_dim, keepdim=True)  # shape U1W

        # now need \sum_u p(u|m) ln q(u|w); use torch.xlogy for 0*log0 case
        d = -(lnp_U_given_M.exp() * lnquw).sum(U_dim, keepdim=True)  # shape 1MW

        # finally get the encoder
        lnq = (lnq0 - beta * d).log_softmax(W_dim)  # shape 1MW

    return (lnq.squeeze(U_dim)).exp() # remove dummy U dimension


def get_ib_curve(game: Game):
    """Reverse deterministic annealing (Zaslavsky and Tishby, 2019)"""
    init = torch.eye(game.num_states) # max number of words = states

    encoders = []
    coordinates = []

    for beta in torch.logspace(game.beta_start, game.beta_stop, game.steps):

        encoder = ib_blahut_arimoto(beta, game.prior, game.meaning_dists, init = init)
        coordinates.append(ib_encoder_to_point(encoder, game.meaning_dists, game.prior))
        encoders.append(encoder)
        init = encoder

    return {
        "encoders": torch.stack(encoders),
        "coordinates": torch.tensor(coordinates),
    }


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    util.set_seed(config.seed)

    # save one curve for multiple analyses
    game_dir = os.getcwd().replace(config.filepaths.simulation_subdir, "")
    curve_fn = os.path.join(game_dir, config.filepaths.curve_points_save_fn)

    curve_points = get_ib_curve(Game.from_hydra(config))["coordinates"]

    util.save_ib_curve(curve_fn, curve_points)

if __name__ == "__main__":
    main()