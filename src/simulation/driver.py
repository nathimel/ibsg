"""Functions for running one trial of a simulation."""

import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Any

from game.languages import SignalingLanguage
from misc.util import points_to_df
from simulation.dynamics import dynamics_map

from multiprocessing import Pool, cpu_count

from game.game import Game


##############################################################################
# Helper functions for running experiments
##############################################################################


def run_trials(
    *args,
    **kwargs,
) -> list[Game]:
    """Run a simulation for multiple trials."""
    if kwargs["multiprocessing"] == True:
        return run_trials_multiprocessing(*args, **kwargs)
    else:
        return [
            run_simulation(*args, **kwargs) for _ in tqdm(range(kwargs["num_trials"]))
        ]


def run_trials_multiprocessing(
    *args,
    **kwargs,
) -> list[SignalingGame]:
    """Use multiprocessing apply_async to run multiple trials at once."""
    num_processes = cpu_count()
    if kwargs["num_processes"] is not None:
        num_processes = kwargs["num_processes"]

    with Pool(num_processes) as p:
        async_results = [
            p.apply_async(run_simulation, args=args, kwds=kwargs)
            for _ in range(kwargs["num_trials"])
        ]
        p.close()
        p.join()
    return [async_result.get() for async_result in tqdm(async_results)]


def run_simulation(
    *args,
    **kwargs,
) -> SignalingGame:
    """Run one trial of a simulation and return the resulting game."""
    game_params = game_parameters(**kwargs)
    dynamics = dynamics_map[kwargs["dynamics"]](SignalingGame(**game_params), **kwargs)
    dynamics.run()
    return dynamics.game


##############################################################################
# Functions for measuring signaling games
##############################################################################


def mean_trajectory(trials: list[SignalingGame]) -> pd.DataFrame:
    """Compute the mean (rate, distortion) trajectory of a game across trials."""

    # extrapolate values since replicator dynamic converges early
    lengths = np.array([len(trial.data["points"]) for trial in trials])
    max_length = lengths.max()
    if np.all(lengths == max_length):
        # no need to extrapolate
        points = np.array([np.array(trial.data["points"]) for trial in trials])

    else:
        # pad each array with its final value
        extrapolated = []
        for trial in trials:
            points = np.array(trial.data["points"])
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
    """Get a dataframe of each (rate, distortion) point that obtains after one iteration of an optimization procedure, e.g. interactions in RL, updates in a replicator dynamic, or iterations of Blahut-Arimoto."""
    points_df = points_to_df(trajectory_points)
    points_df["round"] = pd.to_numeric(points_df.index)  # continuous scale
    return points_df


def trials_to_df(signaling_games: list[SignalingGame]) -> list[tuple[float]]:
    """Compute the pareto points for a list of resulting simulation languages, based on the distributions of their senders, receiver.

    Args:
        trials: a list of SignalingGames after convergence

        trajectory: whether for each trial to return a DataFrame of final round points or a DataFrame of all rounds points (i.e., the game trajectory).

    Returns:
        df: a pandas DataFrame of (rate, distortion) points
    """
    return points_to_df([sg.data["points"][-1] for sg in signaling_games])


def games_to_languages(games: list[SignalingGame]) -> list[tuple[SignalingLanguage]]:
    """For each game, extract the sender and receiver's language (signal-state mapping)."""
    languages = [
        (agent.to_language() for agent in [g.sender, g.receiver]) for g in games
    ]
    return languages
