import os
import torch
import pandas as pd
from plotnine import ggplot
from omegaconf import DictConfig
from game.game import Game

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Random
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def set_seed(seed: int) -> None:
    """Sets random seeds."""
    torch.manual_seed(seed)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def points_to_df(points: torch.Tensor, columns = ["complexity", "accuracy"]) -> pd.DataFrame:
    """Convert a Tensor of points to a dataframe with rate, distortion as columns."""
    return pd.DataFrame(
        data=points,
        columns=columns,
    )

def save_points_df(fn: str, df: pd.DataFrame) -> None:
    """Save a dataframe of (complexity, accuracy) points (and potentially other dims) to a CSV."""
    df.to_csv(fn, index=False)
    print(f"Saved {len(df)} language points to {os.path.join(os.getcwd(), fn)}")

def trajectories_df(trials: list[Game]) -> pd.DataFrame:
    """Collect the (complexity, accuracy, distortion, mse) trajectories of every game across trials and store in one dataframe."""
    # build a df for each and concatenate
    df = pd.concat([
        pd.DataFrame(
            # label trials
            data=torch.hstack((
                # label rounds
                torch.hstack((
                    torch.Tensor(trial.points), # (num_rounds, 4)
                    torch.arange(len(trial.points))[:, None]
                )),
                torch.ones(len(trial.ib_points))[:, None] * trial_num+1),
            ),
            columns=["complexity", "accuracy", "distortion", "mse", "round", "trial"]) for trial_num, trial in enumerate(trials)
        ])
    return df


def get_curve_fn(config: DictConfig, curve_type: str = "ib", curve_dir: str = None) -> str:
    """Get the full path of the IB curve, relative to hydra interpolations."""
    if curve_dir is None:
        curve_dir = os.getcwd().replace(config.filepaths.leaf_subdir, config.filepaths.curve_subdir)

    if curve_type == "ib":
        curve_fn = os.path.join(curve_dir, config.filepaths.curve_points_save_fn)
    elif curve_type == "mse":
        curve_fn = os.path.join(curve_dir, config.filepaths.mse_curve_points_save_fn)
    else:
        raise ValueError()
    return curve_fn


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def save_plot(fn: str, plot: ggplot, width=10, height=10, dpi=300) -> None:
    """Save a plot with some default settings."""
    plot.save(fn, width=10, height=10, dpi=300)
    print(f"Saved a plot to {fn}")