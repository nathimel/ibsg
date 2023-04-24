import torch
import pandas as pd
from game.game import Game

from altk.effcomm.information import ib_encoder_to_point

from misc.util import points_to_df

def get_hypothetical_variants(games: list[Game], num: int) -> pd.DataFrame:
    """For each emergent system from a SignalingGame, generate `num` hypothetical variants by permuting the signals that the system assigns to states."""
    variants_per_system = int(num / len(games))

    points = []
    for game in games:
        encoder = game.ib_encoders[-1] # unfortunate clash in notation P, Q
        seen = set()
        while len(seen) < variants_per_system:
            # permute columns of speaker weights
            permuted = encoder[:,torch.randperm(encoder.shape[1])]
            seen.add(tuple(permuted.flatten()))

        for permuted_weights in seen:
            encoder_ = torch.tensor(permuted_weights).reshape(encoder.shape)
            points.append(ib_encoder_to_point(encoder_, game.meaning_dists, game.prior))

    return points_to_df(points)