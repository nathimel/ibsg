"""Utility functions for plotting."""
import plotnine as pn
import pandas as pd
import numpy as np

from misc.util import encoder_columns


def numeric_col_to_categorical(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Change a float valued column (e.g. run or round) to Categorical for visualization."""
    # adjust values to Categorical where appropriate
    df[col] = df[col].astype(int)
    df = df.assign(**{col: pd.Categorical(df[col])})
    return df


def basic_tradeoff_plot(
    curve_data: pd.DataFrame,
    simulation_data: pd.DataFrame,
    variant_data: pd.DataFrame = None,
    trajectory_data: pd.DataFrame = None,
    nearest_optimal_encoders_data: pd.DataFrame = None,
    y: str = "accuracy",
) -> pn.ggplot:
    """Get a basic plotnine point plot of languages in a complexity vs accuracy 2D plot.

    Args:
        y: {'accuracy', 'distortion', 'mse'}
    """
    plot = bound_only_plot(curve_data, y=y)

    if variant_data is not None:
        plot = plot + pn.geom_point(  # hypothetical langs bottom layer
            variant_data,
            color="gray",
            shape="o",
            size=2,
            alpha=0.6,
        )

    sim_data = simulation_data.copy()
    sim_data["run"] = sim_data["run"].astype(int) + 1
    sim_data = numeric_col_to_categorical(sim_data, "run")

    plot = plot + pn.geom_point(  # simulation langs
        data=sim_data,
        mapping=pn.aes(color="run"),
        shape="o",
        size=4,
        # width=0.1,
        # height=0.1,
    )

    if nearest_optimal_encoders_data is not None:
        nearopt_data = nearest_optimal_encoders_data.copy()
        nearopt_data["run"] = nearopt_data["run"].astype(int) + 1
        nearopt_data = numeric_col_to_categorical(nearopt_data, "run")
        plot = plot + pn.geom_point(
            data=nearopt_data,
            mapping=pn.aes(color="run"),
            shape="+",
            size=5,
        )

    if trajectory_data is not None:
        traj_data = trajectory_data.copy()
        traj_data = numeric_col_to_categorical(traj_data, "run")
        plot = plot + pn.geom_line(
            data=traj_data,
            mapping=pn.aes(
                color="run",
                # color="round",
            ),  # categorical
            # shape='d',
            size=1,
        )

    return plot


def bound_only_plot(
    curve_data: pd.DataFrame,
    y: str = "accuracy",
) -> pn.ggplot:
    if y == "accuracy":
        ystr = "Accuracy $I[W:U]$ bits"
    elif y == "distortion":
        ystr = "Distortion $\mathbb{E}[D_{KL}[ M || \hat{M} ]]$"
    elif y == "mse":
        ystr = "Distortion $\mathbb{E}[(u - \hat{u})^2]$"

    data = curve_data.copy()
    plot = (
        # Set data and the axes
        pn.ggplot(data=data, mapping=pn.aes(x="complexity", y=y))
        + pn.xlab("Complexity $I[M:W]$ bits")
        + pn.ylab(ystr)
    )
    # plot = plot + pn.geom_line()
    plot = plot + pn.geom_point()
    return plot


def single_eps_heatmap_tradeoff_plot(
    eps_curve_data: pd.DataFrame,
    sim_data: pd.DataFrame,  # a single point
    opt_data: pd.DataFrame,  # same
) -> pn.ggplot:
    data = eps_curve_data.copy()
    plot = (
        pn.ggplot(data=data, mapping=pn.aes(x="complexity", y="accuracy"))
        + pn.geom_point(size=3, mapping=pn.aes(color="eps"))
        + pn.scale_color_continuous("inferno", trans = "log10")
        + pn.geom_point(
            data=sim_data,
            fill="red",
            size=5,
        )
        + pn.geom_point(
            data=opt_data,
            color="limegreen",
            shape="X",
            size=5,
        )
    )
    return plot


def basic_efficiency_plot(
    simulation_data: pd.DataFrame,
) -> pn.ggplot:
    """Get a basic histogram/density plot of the distribution of languages over efficiency loss."""
    data = simulation_data.copy()
    plot = (
        pn.ggplot(data=data, mapping=pn.aes(x="eps"))
        + pn.geom_histogram()
        # + pn.geom_density()
        + pn.xlab("Efficiency loss")
        + pn.ylab("count")
    )
    return plot


# Encoders


def get_n_encoder_plots(
    df: pd.DataFrame,
    plot_type: str,
    all_items: bool = True,
    item_key: str = "run", # can also pass "round"
    title_var: str = None,
    n: int = 8,
) -> list[pn.ggplot]:
    """Return a list of plots, one for each encoder corresponding to each run our round. If `all_items` is False, get `n` plots, which is 8 by default.

    Args:
        plot_type: {"tile", "line"}

        item_key: {"run", "round"}

        title_var: defaults to `item_key`
    """
    data = df.copy()
    data[item_key] = data[item_key].astype(int) + 1
    items = data[item_key].unique()
    if not all_items:
        items = items[:n]
    if title_var is None:
        title_var = item_key

    # TODO: label title $S^{t}(w|x_o)$
    return [
        (
            plots[plot_type](data[data[item_key] == item], item_key=item_key) 
            + pn.ggtitle(f"{title_var} {item}")
            # + pn.ggtitle(f"$S^{{{title_var}={item}}}(w | x_o)$")
        )
        for item in items
    ]

# TODO: create a plot similar to plot_type="line", but 
# - the color corresponds to the centroid.??
# - and the cmap is for ordinal/continuous data
def get_n_centroid_plots(
    encoders: np.ndarray,
    prior: np.ndarray,
    all_items: bool = True,
    item_key: str = "run",     
    title_nums: np.ndarray = None,
    title_var: str = None,
    n: int = 8,
) -> list:
    """Return a list of plots, one for each encoder corresponding to each run our round. If `all_items` is False, get `n` plots, which is 8 by default.

        Args:
            plot_type: {"tile", "line"}

            item_key: {"run", "round"}

            title_var: defaults to `item_key`
    """
    if title_nums is None:
        title_nums = np.arange(len(encoders))
    if title_var is None:
        title_var = item_key        
    # TODO: replace enumerate with title_nums, an arg    
    return [get_centroid_lineplot(enc, prior, title=f"{title_var}={idx}") for idx, enc in zip(title_nums, encoders)]


def faceted_encoders(df: pd.DataFrame, plot_type: str) -> pn.ggplot:
    """Return a plot of different encoder subplots, faceted by run.

    Args:
        plot_type: {"tile", "line"}
    """
    data = df.copy()
    data["run"] = data["run"].astype(int) + 1
    return (
        plots[plot_type](data)
        + pn.facet_grid("run ~ .")
        + pn.theme(
            axis_text_y=pn.element_blank(),
            axis_text_x=pn.element_blank(),
        )
    )


# Heatmaps
def basic_encoder_tile(df: pd.DataFrame, item_key = "run") -> pn.ggplot:
    """Return a single tile (heatmap) plot for an encoder."""
    df = format_encoder_df(df, ["word", "meaning"], item_key=item_key)
    return (
        pn.ggplot(df, pn.aes(**dict(zip(["x", "y", "fill"], encoder_columns[:3]))))
        + pn.geom_tile()
        + pn.scale_fill_cmap(cmap_name = 'inferno', limits=[0, 1])
    )


# Lines
def basic_encoder_lineplot(df: pd.DataFrame, **kwargs) -> pn.ggplot:
    "Return a single line plot for an encoder."
    df = format_encoder_df(df, ["word"], **kwargs)  # meanings must be numeric!
    return (
        pn.ggplot(df, pn.aes(x="meaning", y="p"))
        + pn.geom_line(
            mapping=pn.aes(
                color="word",
            ),
            size=1,
        )
        + pn.ylim([0, 1])
    )

# Better version of above with lines colored by centroid.
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
def get_centroid_lineplot(qW_M: np.ndarray, pM: np.ndarray, title: str = "") -> Figure:
    """Return a single plot for an encoder, with lines colored by the centroid meaning for the word."""
    # Choose your colormap
    colormap = plt.get_cmap('copper')

    # Generate 100 equally spaced values between 0 and 1
    values = np.linspace(0, 1, 100)

    # Get the corresponding colors from the colormap
    colors = colormap(values)

    # Get bayesian q(m|w)
    qWM = qW_M * pM[:, None]
    qW = qW_M.T @ pM
    qM_W = np.where(qW > 1e-16, qWM / qW, 1 / qWM.shape[0]).T

    # try just argmax color first
    # TODO: consult noga for true weighted average centroid color!
    word_colors = [
        colors[np.argmax(qM_W[word_idx])] for word_idx in range(len(qM_W))
    ]

    x = list(range(100))
    # Plot each line with its corresponding color
    fig, ax = plt.subplots()
    for word_idx in range(len(qW_M.T)):
        ax.plot(
            x,
            qW_M.T[word_idx],
            color=word_colors[word_idx], 
            linewidth=1,
        )

    # if title: 
    #     ax.set_title(label=title)
    
    ax.set_ylim(0,1)

    ax.set_yticks(
        ticks=[0,1],
    )
    ax.set_xticks(
        ticks=[0,50,100],
    )
    ax.set_xlabel(r"$X_o$")
    # ax.set_xlabel("meaning")
    # ax.set_ylabel(r"$S^{t}(w | x_o)$")
    # ax.set_ylabel(r"$S_{\beta}(w | x_o)$")
    ax.set_ylabel(r"$S(w | m_o)$")

    # ax.set_ylabel("naming probability")
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(0, 100))
    sm.set_array([])

    # fig.colorbar(
    #     sm, 
    #     # label='centroid $m$ for $w$', 
    #     label='color coding for $w$',
    #     ticks=[0,50,100],
    #     location='bottom',
    # )

    return fig


def format_encoder_df(
    df: pd.DataFrame, numeric_to_categorical: list[str],
    item_key: str = "run",
) -> pd.DataFrame:
    # create new dataframe labeled by 1-indexed runs
    data = df.copy()
    data[item_key] = data[item_key].astype(int)
    for col in numeric_to_categorical:
        data = numeric_col_to_categorical(data, col)
    return data


plots = {
    "tile": basic_encoder_tile,
    "line": basic_encoder_lineplot,
}
