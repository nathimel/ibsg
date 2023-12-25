"""Utility functions for plotting."""
import plotnine as pn
import pandas as pd

from misc.util import encoder_columns


def numeric_col_to_categorical(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Change a float valued column (e.g. run or round) to Categorical for visualization."""
    # adjust values to Categorical where appropriate
    # df[col] = df[col].astype(int).astype(str)
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


def single_gnid_heatmap_tradeoff_plot(
    gnid_curve_data: pd.DataFrame,
    sim_data: pd.DataFrame,  # a single point
    opt_data: pd.DataFrame,  # same
) -> pn.ggplot:
    data = gnid_curve_data.copy()
    plot = (
        pn.ggplot(data=data, mapping=pn.aes(x="complexity", y="accuracy"))
        + pn.geom_point(size=3, mapping=pn.aes(color="gNID"))
        + pn.scale_color_continuous("inferno", limits=(0, 1))
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
        # + pn.geom_histogram()
        + pn.geom_density()
        + pn.xlab("Efficiency loss")
        + pn.ylab("count")
    )
    return plot


# Encoders


def get_n_encoder_plots(
    df: pd.DataFrame,
    plot_type: str,
    all_runs: bool = True,
    n: int = 8,
) -> list[pn.ggplot]:
    """Return a list of plots, one for each encoder corresponding to each run. If `all_runs` is False, get `n` plots, which is 8 by default.

    Args:
        plot_type: {"tile", "line"}
    """
    data = df.copy()
    data["run"] = data["run"].astype(int) + 1
    runs = data["run"].unique()
    if not all_runs:
        runs = runs[:n]
    return [
        (plots[plot_type](data[data["run"] == run]) + pn.ggtitle(f"run {run}"))
        for run in runs
    ]


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
def basic_encoder_tile(df: pd.DataFrame) -> pn.ggplot:
    """Return a single tile (heatmap) plot for an encoder."""
    df = format_encoder_df(df, ["word", "meaning"])
    return (
        pn.ggplot(df, pn.aes(**dict(zip(["x", "y", "fill"], encoder_columns[:3]))))
        + pn.geom_tile()
        + pn.scale_fill_cmap(cmap_name = 'inferno', limits=[0, 1])
    )


# Lines
def basic_encoder_lineplot(df: pd.DataFrame) -> pn.ggplot:
    "Return a single line plot for an encoder."
    df = format_encoder_df(df, ["word"])  # meanings must be numeric!
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


def format_encoder_df(
    df: pd.DataFrame, numeric_to_categorical: list[str]
) -> pd.DataFrame:
    # create new dataframe labeled by 1-indexed runs
    data = df.copy()
    data["run"] = data["run"].astype(int)
    for col in numeric_to_categorical:
        data = numeric_col_to_categorical(data, col)
    return data


plots = {
    "tile": basic_encoder_tile,
    "line": basic_encoder_lineplot,
}
