import os
import torch
import pandas as pd
from plotnine import ggplot
from omegaconf import DictConfig

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Random
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def set_seed(seed: int) -> None:
    """Sets random seeds."""
    torch.manual_seed(seed)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def points_to_df(points: torch.Tensor) -> pd.DataFrame:
    """Convert a Tensor of points to a dataframe with rate, distortion as columns."""
    return pd.DataFrame(
        data=points,
        columns=["complexity", "accuracy"],
    )

def save_points_df(fn: str, df: pd.DataFrame) -> None:
    """Save a dataframe of (complexity, accuracy) points to a CSV."""
    df.to_csv(fn, index=False)
    print(f"Saved {len(df)} language points to {os.path.join(os.getcwd(), fn)}")


def get_curve_fn(config: DictConfig) -> str:
    """Get the full path of the IB curve, relative to hydra interpolations."""
    curve_dir = os.getcwd().replace(config.filepaths.simulation_subdir, "").replace(config.filepaths.common_sweep_subdir, "")
    curve_fn = os.path.join(curve_dir, config.filepaths.curve_points_save_fn)
    return curve_fn


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def save_plot(fn: str, plot: ggplot, width=10, height=10, dpi=300) -> None:
    """Save a plot with some default settings."""
    plot.save(fn, width=10, height=10, dpi=300)
    print(f"Saved a plot to {fn}")