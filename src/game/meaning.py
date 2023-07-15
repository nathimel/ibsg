"""Minimal module for loading / constructing meaning space and prior. """
import torch
import pandas as pd
from altk.language.semantics import Referent, Universe

def load_universe(
    universe_fn: str,
    prior_fn: str,
) -> Universe:
    """Construct a universe from csv files."""
    return build_universe(
        pd.read_csv(universe_fn), 
        pd.read_csv(prior_fn) if prior_fn is isinstance(prior_fn, str) else None, # ignore if not a filename (e.g. a temp param)
    )

def build_universe(
    referents_df: pd.DataFrame,
    prior_df: pd.DataFrame = None,
    features: list[str] = None,
) -> Universe:
    """Construct a universe from dataframes encoding referents and a prior over them.
    
    Args:
        referents_df: DataFrame of referents

        prior_df: DataFrame encoding prior / communicative need distribution over referents

        features: list of column-names of `referents_df` to encode as a feature vector (stored in a tuple named `point`) associated with each referent. For example, if ['x', 'y', 'z'] is passed, then a possible value of referent.point could be (0.2, -8, 10). Default is None, and all columns of `referents_df` except 'name' will be considered features.
    """
    assert (referents_df["name"] == prior_df["name"]).all()

    referents = tuple(
        Referent(
            record["name"], 
            properties={"point": tuple(record[f] for f in features) if features is not None else record["name"]}
        ) for record in referents_df.to_dict("records")
    )
    prior = dict(zip(prior_df["name"], prior_df["probability"]))

    return Universe(referents, prior)

# Meaning spaces

def nball_points(
    n: int, 
    num_samples: int,
    radius: float = 1., 
    center: float = 0.,
    ) -> torch.Tensor:
    """Uniformly sample random points contained within an n-sphere. 

    For example, if `n=3`, `num_samples=100` generate 100 points in R^3 approximating a solid ball of size centered at the origin with radius 1. NOTE: not recommended for high dimensions, as the curse of dimensionality implies  sparsity.

    Args:
        n: dimension of the sphere to sample from. For example, 

            - 1: line interval
            - 2: disk (bounded by circle)
            - 3: ball (bounded by sphere)
            - ...

        num_samples: the number of points to draw

        radius: the radius of the n-sphere. Unity by default.

        center: the center of the ball.

    Returns:
        points: a tensor of shape `[num_samples, n]`        
    """
    


def ncube_points(
    n: int,
    num_samples: int,
    length: float = 1.,
    center: float = 0.,
) -> torch.Tensor:
    """Uniformly sample random points in n-dimensional space (resulting in a cube).

    For example, if `n=1`, `num_samples=10`, we just return `torch.rand(10) - 0.5`.

    Args:
        n: dimension of the cube to sample from. For example, 

            - 1: line interval
            - 2: plane (bounded by square)
            - 3: euclidean space (bounded by cube)
            - ... (hyperplanes)

        num_samples: the number of points to draw

        radius: the radius of the n-sphere. Unity by default.

        center: the center of the ball.

    Returns:
        points: a tensor of shape `[num_samples, n]`
    """
    # random samples from N(0,1)
    return (torch.randn(num_samples, n) + center) * length
