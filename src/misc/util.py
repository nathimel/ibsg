import os
import pickle

import numpy as np
import pandas as pd
from plotnine import ggplot
from omegaconf import DictConfig


# To silence 'SettingWithCopyWarning'
pd.options.mode.chained_assignment = None  # default='warn'

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Random
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def set_seed(seed: int) -> None:
    """Sets random seeds."""
    np.random.seed(seed)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def write_pickle(fn: str, data):
    with open(fn, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(fn: str):
    with open(fn, "rb") as f:
        data = pickle.load(f)
    return data


def points_to_df(
    points: np.ndarray, columns=["complexity", "accuracy"]
) -> pd.DataFrame:
    """Convert a Tensor of points to a dataframe with rate, distortion as columns.

    Obviously, this function adds nothing except a wrapper to unifying code / signal what's happening at various stages in a consistent way.
    """
    return pd.DataFrame(
        data=points,
        columns=columns,
    )


def save_points_df(fn: str, df: pd.DataFrame) -> None:
    """Save a dataframe of (complexity, accuracy) points (and potentially other dims) to a CSV."""
    df.to_csv(fn, index=False)
    print(f"Saved {len(df)} language points to {os.path.join(os.getcwd(), fn)}")


def save_encoders_df(fn: str, df: pd.DataFrame) -> None:
    if "run" in df.columns:
        num = df["run"].max()
    elif "round" in df.columns:
        num = df["round"].max()
    else:
        breakpoint()
        raise Exception
    df.to_csv(fn, index=False)
    print(f"Saved {num} encoders to {os.path.join(os.getcwd(), fn)}")


def save_final_encoders(fn: str, runs: list) -> None:
    """Save the encoders from the last round of each game to a file.

    Args:
        fn: the file to save the final encoders to

        runs: a list of Game objects
    """
    np.save(fn, np.stack([np.array(g.ib_encoders[-1]) for g in runs]))
    print(f"Saved {len(runs)} encoders to {os.path.join(os.getcwd(), fn)}")


def save_ndarray(fn: str, arr: np.ndarray) -> None:
    np.save(fn, arr)
    print(
        f"Saved ndarray of shape {arr.shape if arr is not None else None} to {os.path.join(os.getcwd(), fn)}"
    )


def load_encoders_as_df(fn: str) -> pd.DataFrame:
    """Load encoders saved in a .pt file, and convert from torch.tensor to pd.DataFrame."""
    return encoders_to_df(np.load(fn))


points_columns = [
    "complexity",
    "accuracy",
    "distortion",
    "mse",
]
efficiency_columns = [
    "gNID",
    "eps",
    "beta",
]


def final_points_df(runs: list) -> pd.DataFrame:
    """Collect the (complexity, accuracy, ...) points for the final round of every game across runs and store in one dataframe.

    Args:
        runs: a list of Game objects
    """
    return points_to_df(
        [
            (
                *g.points[-1],  # comp, acc, dist, mse,
                None,  # gNID computed later
                None,  # eps (efficiency loss)
                None,  # beta
                i,  # run number
            )
            for i, g in enumerate(runs)
        ],
        columns=points_columns + efficiency_columns + ["run"],
    )


def trajectories_df(runs: list) -> pd.DataFrame:
    """Collect the (complexity, accuracy, distortion, mse) trajectories of every game across runs and store in one dataframe.

    Args:
        runs: a list of Game objects
    """
    # build a df for each and concatenate
    return pd.concat(
        [
            pd.DataFrame(
                # label runs
                data=np.hstack(
                    (
                        # label rounds
                        np.hstack(
                            (
                                np.array(run.points),  # (num_rounds, 4)
                                np.arange(len(run.points))[:, None],
                            )
                        ),
                        np.ones(len(run.points))[:, None] * run_num + 1,
                    ),
                ),
                columns=points_columns + ["round", "run"],
            )
            for run_num, run in enumerate(runs)
        ]
    )


encoder_columns = [
    "word",
    "meaning",
    "naming probability \n",
    "p",
]


def encoders_to_df(encoders: np.ndarray, col: str = "run") -> pd.DataFrame:
    """Get a dataframe with columns ['meanings', 'words', 'p', 'naming probability \n'].

    Args:

        encoders: tensor of shape `(runs, meanings, words)`

        col: {"run", "round"} whether `encoders` is a list of final encoders across runs, or intermediate encoders across game rounds.
    """
    num_meanings, num_words = encoders[0].shape

    meanings = np.array([[i] * num_words for i in range(num_meanings)]).flatten()
    words = np.array(list(range(num_words)) * num_meanings)
    ones = np.ones_like(encoders[0]).flatten()
    return pd.concat(
        [
            pd.DataFrame(
                np.stack(
                    [
                        ones * i,  # run
                        words,
                        meanings,
                        encoder.flatten(),  # 'naming probability \n' is alias for 'p'
                        encoder.flatten(),  # p = p(word | meaning)
                    ]
                ).T,
                columns=[col] + encoder_columns,
            )
            for i, encoder in enumerate(encoders)
        ]
    )


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# File handling
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path)
        print(f"Created directory {path}.")


def get_bound_fn(
    config: DictConfig, bound_type: str = "ib", curve_dir: str = None
) -> str:
    """Get the full path of a theoretically optimal bound curve, list of optimal encoders, or list of saved tradeoff parameters relative to hydra interpolations."""
    if curve_dir is None:
        curve_dir = os.getcwd().replace(
            config.filepaths.leaf_subdir, config.filepaths.curve_subdir
        )

    if bound_type == "ib":
        fn = os.path.join(curve_dir, config.filepaths.curve_points_save_fn)
    elif bound_type == "mse":
        fn = os.path.join(curve_dir, config.filepaths.mse_curve_points_save_fn)
    elif bound_type == "encoders":
        fn = os.path.join(curve_dir, config.filepaths.optimal_encoders_save_fn)
    elif bound_type == "betas":
        fn = os.path.join(curve_dir, config.filepaths.betas_save_fn)
    elif bound_type == "metadata":
        fn = os.path.join(curve_dir, config.filepaths.curve_metadata_save_fn)
    else:
        raise ValueError()
    return fn


# TODO: Look into hydra.utils.get_original_cwd!

def get_root(config: DictConfig, cwd=None) -> str:
    """Get the full path of the root of the repo, relative to hydra interpolations."""
    return os.path.abspath(
        os.path.join(str(cwd).replace(config.filepaths.leaf_subdir, ""), os.pardir)
    )  # use join, not dirname (which requires path to exist, among other things)


def get_prior_fn(config: DictConfig, *args, cwd=None, **kwargs) -> str:
    """Get the full path of the prior need distribution over meanings, relative to hydra interpolations."""
    if cwd is None:
        cwd = os.getcwd()
    root = get_root(config, cwd)
    fp = os.path.join(root, config.filepaths.prior_fn)

    return fp


def get_universe_fn(config: DictConfig, *args, cwd=None, **kwargs) -> str:
    """Get the full path of the Universe (collection of referents/ world states) determining the size and structure of the semantic space, relative to hydra interpolations."""
    if cwd is None:
        cwd = os.getcwd()
    root = get_root(config, cwd)
    fp = os.path.join(root, config.filepaths.universe_fn)

    return fp


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def save_plot(fn: str, plot: ggplot, width=10, height=10, dpi=300) -> None:
    """Save a plot with some default settings."""
    plot.save(fn, width=width, height=height, dpi=dpi, verbose=False)
    print(f"Saved a plot to {fn}")
