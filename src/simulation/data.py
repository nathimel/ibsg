import numpy as np
import pandas as pd

from game.game import Game
from misc.util import points_to_df

##############################################################################
# Functions for extracting / measuring data from evolutionary games
##############################################################################

def mean_trajectory(trials: list[Game]) -> pd.DataFrame:
    """Compute the mean IB trajectory of a game across trials."""

    # extrapolate values since replicator dynamic converges early
    lengths = np.array([len(trial.ib_points) for trial in trials])
    max_length = lengths.max()
    if np.all(lengths == max_length):
        # no need to extrapolate
        points = np.array([np.array(trial.ib_points) for trial in trials])

    else:
        # pad each array with its final value
        extrapolated = []
        for trial in trials:
            points = np.array(trial.ib_points)
            extra = points[-1] * np.ones((max_length, 2))  # 2D array of points
            extra[: len(points)] = points
            extrapolated.append(extra)
        points = np.array(extrapolated)

    mean_traj = np.mean(points, axis=0)
    points = np.squeeze(mean_traj)
    points_df = points_to_df(points)
    points_df["round"] = pd.to_numeric(points_df.index)  # continuous scale
    return points_df


def trajectory_points_to_df(trajectory_points: list[tuple[float]]) -> pd.DataFrame:
    """Get a dataframe of each IB point that obtains after one iteration of a dynamics."""
    points_df = points_to_df(trajectory_points)
    points_df["round"] = pd.to_numeric(points_df.index)  # continuous scale
    return points_df


def trials_to_df(games: list[Game]) -> list[tuple[float]]:
    """Compute the pareto points for a list of resulting simulation languages, based on the distributions of their final / converged senders, receiver.

    Args:
        trials: a list of Games after convergence

    Returns:
        a pandas DataFrame of IB points
    """
    return points_to_df([g.ib_points[-1] for g in games])