"""Functions for quantifying the efficiency of emergent systems, and different 'variants' of them, w.r.t IB bounds."""

import random
import torch

import pandas as pd

from ultk.effcomm.rate_distortion import ib_encoder_to_point
from misc import util


def efficiency_loss(emergent, optimal, beta, meaning_dists, prior) -> float:
    """Compute the efficiency loss of a semantic system:

        eps = 1/beta * ( F_[q] - F_[q*] )

    where F is the IB objective, q is an emergent encoder, q* is its most similar optimal counterpart, and beta is the value of beta input to the IB method yielding the optimal encoder. The IB objective is given by

       F_[q(w|m)] = I[M:W] - I(W;U) (beta is constant for both q, q*)

    i.e., a Lagrangian to minimize complexity and maximize accuracy. See Zaslavsky et. al. 2018, "Near-Optimal Trade-Offs", and SI Section 5, for details.
    """
    # interestingly, optima rows do not always sum to 1

    # Need to convert to numpy for ultk, rdot
    meaning_dists = meaning_dists.numpy()
    prior = prior.numpy()
    emergent = emergent.numpy()
    optimal = optimal.numpy()

    # return is complexity, accuracy, comm_cost
    em_complexity, em_acc, _ = ib_encoder_to_point(meaning_dists, prior, emergent)
    opt_complexity, opt_acc, _ = ib_encoder_to_point(meaning_dists, prior, optimal)

    em_value = em_complexity - em_acc
    opt_value = opt_complexity - opt_acc

    loss = (
        1 / beta * (em_value - opt_value)
    )  # value is to be minimized, so emergent larger
    return loss


def alt_encoders_to_df(
    encoders: torch.Tensor, meaning_dists: torch.Tensor, prior: torch.Tensor
) -> pd.DataFrame:
    """Convert a tensor of alternative encoders (to the original emergent ones) to a dataframe."""
    encoders = encoders.numpy()
    meaning_dists = meaning_dists.numpy()
    prior = prior.numpy()
    return util.points_to_df(
        points=[
            (
                *ib_encoder_to_point(
                    meaning_dists, prior, encoders[i]
                ),  # comp, acc, distortion
                None,  # mse
                i,  # run
            )
            for i in range(len(encoders))
        ],
        columns=["complexity", "accuracy", "distortion", "mse", "run"],
    )


def finite_sample_encoder(encoder: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Construct a corrupted version of an emergent encoder via finite sampling.

    Args:
        encoder: the encoder to reconstruct

        num_samples: the number of samples to use for reconstruction. With enough samples the original encoder is reconstructed.

    Returns:
        the approximated encoder via finite sampling
    """
    sampled_encoder = []
    for row in encoder:
        # sample
        indices = random.choices(
            population=range(len(row)),
            weights=row,
            k=num_samples,
        )
        # count
        sampled_row = torch.zeros_like(row)
        for idx in indices:
            sampled_row[idx] += 1
        # normalize
        sampled_row /= sampled_row.sum()
        sampled_encoder.append(sampled_row)

    return torch.stack(sampled_encoder)


def hypothetical_variants(encoders: torch.Tensor, num: int) -> torch.Tensor:
    """For each emergent system, generate `num` hypothetical variants by permuting the signals that the system assigns to states."""
    variants_per_system = int(num / len(encoders))

    permuted_encoders = []
    for encoder in encoders:
        seen = set()
        while len(seen) < variants_per_system:
            # permute columns of speaker weights
            permuted = encoder[:, torch.randperm(encoder.shape[1])]
            seen.add(tuple(permuted.flatten()))

        for permuted_weights in seen:
            permuted_encoders.append(
                torch.tensor(permuted_weights).reshape(encoder.shape)
            )

    return torch.stack(permuted_encoders)
