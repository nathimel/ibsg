"""Obtain approximate versions of emergent encoders, treating each emergent encoder as the 'ground truth' and then estimating its behavior under finite samples."""


import hydra
import os
import torch
import random

from altk.effcomm.information import ib_encoder_to_point
from omegaconf import DictConfig

from game.game import Game
from misc import util


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


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config: DictConfig):
    util.set_seed(config.seed)
    g = Game.from_hydra(config)

    # helper function to load the right files
    cwd = os.getcwd()
    fps = config.filepaths
    fullpath = lambda fn: os.path.join(cwd, fn)

    # load original
    emergent_encoders = torch.load(fullpath(fps.final_encoders_save_fn))

    # approximate
    sampled_encoders = torch.stack(
        [
            finite_sample_encoder(enc, config.simulation.num_approximation_samples)
            for enc in emergent_encoders
        ]
    )

    # measure
    approximate_data = util.points_to_df(
        points=[
            (
                *ib_encoder_to_point(
                    g.meaning_dists, g.prior, sampled_encoders[i]
                ),  # comp, acc, distortion
                None,  # mse
                i,  # run
            )
            for i in range(len(emergent_encoders))
        ],
        columns=["complexity", "accuracy", "distortion", "mse", "run"],
    )

    util.save_tensor(fullpath(fps.approximated_encoders_save_fn), sampled_encoders)
    util.save_points_df(
        fn=config.filepaths.approximated_simulation_points_save_fn, df=approximate_data
    )


if __name__ == "__main__":
    main()
