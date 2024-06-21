"""Some additional helper functions for analyzing systems."""

import random
import numpy as np

import pandas as pd

from ultk.effcomm.rate_distortion import ib_encoder_to_point
from misc import util


def alt_encoders_to_df(
    encoders: np.ndarray, meaning_dists: np.ndarray, prior: np.ndarray
) -> pd.DataFrame:
    """Convert an array of alternative encoders (to the original emergent ones) to a dataframe.

    Args:
        encoders: an array of shape `(num_beta, num_meanings, num_words)`

        meaning_dists: the meaning distributions p(y|x) that characterize the domain

        prior: the prior over meanigns p(x) that characterizes the domain

    Returns:
        a pd.DataFrame with columns ["complexity", "accuracy", "distortion", "mse", "run"]
    """
    return util.points_to_df(
        points=[
            (
                *ib_encoder_to_point(
                    prior, meaning_dists, encoders[i]
                ),  # comp, acc, distortion
                None,  # mse
                i,  # run
            )
            for i in range(len(encoders))
        ],
        columns=["complexity", "accuracy", "distortion", "mse", "run"],
    )


def finite_sample_encoder(encoder: np.ndarray, num_samples: int) -> np.ndarray:
    """Construct a corrupted version of an emergent encoder via finite sampling.

    Args:
        encoder: the encoder to treat as the 'ground truth' from which a new encoder will be constructed via finite samples.

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
        sampled_row = np.zeros_like(row)
        for idx in indices:
            sampled_row[idx] += 1
        # normalize
        sampled_row /= sampled_row.sum()
        sampled_encoder.append(sampled_row)

    return np.stack(sampled_encoder)
