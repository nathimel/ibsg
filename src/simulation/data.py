from game.game import Game
from misc.util import points_to_df

##############################################################################
# Functions for extracting / measuring data from evolutionary games
##############################################################################


def trials_to_df(games: list[Game]) -> list[tuple[float]]:
    """Compute the pareto points for a list of resulting simulation languages, based on the distributions of their final / converged senders, receiver.

    Args:
        trials: a list of Games after convergence

    Returns:
        a pandas DataFrame of IB points
    """
    return points_to_df([g.ib_points[-1] for g in games])