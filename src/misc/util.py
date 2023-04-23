import os
import torch
import pandas as pd
from plotnine import ggplot

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Random
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def set_seed(seed: int) -> None:
    """Sets random seeds."""
    torch.manual_seed(seed)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def points_to_df(points: list[tuple[float]]) -> pd.DataFrame:
    """Convert a list of points to a dataframe with rate, distortion as columns."""
    return pd.DataFrame(
        data=points,
        columns=["complexity", "accuracy"],
    )

def save_points_df(fn: str, df: pd.DataFrame) -> None:
    """Save a dataframe of (complexity, accuracy) points to a CSV."""
    df.to_csv(fn, index=False)
    print(f"Saved {len(df)} language points to {os.path.join(os.getcwd(), fn)}")


def save_ib_curve(fn: str, curve) -> None:
    """Save a dataframe of (accuracy, complexity) points to a CSV."""
    df = pd.DataFrame(data=curve, columns=["complexity", "accuracy"])
    save_points_df(fn, df)
    # df.to_csv(fn, index=False)
    # print(f"Saved {len(df)} language points to {fn}")

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def save_plot(fn: str, plot: ggplot, width=10, height=10, dpi=300) -> None:
    """Save a plot with some default settings."""
    plot.save(fn, width=10, height=10, dpi=300)
    print(f"Saved a plot to {fn}")