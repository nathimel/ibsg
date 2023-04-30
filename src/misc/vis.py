import plotnine as pn
import pandas as pd

from misc.util import encoder_columns

# Plotting util functions

def numeric_col_to_categorical(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Change a float valued column (e.g. trial or round) to Categorical for visualization."""
    # adjust values to Categorical where appropriate
    # df[col] = df[col].astype(int).astype(str)
    df[col] = df[col].astype(int)    
    df = df.assign(**{col: pd.Categorical(df[col])})
    return df


def basic_tradeoff_plot(
    curve_data: pd.DataFrame,
    sim_data: pd.DataFrame,
    variant_data: pd.DataFrame = None,
    trajectory_data: pd.DataFrame = None,
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

    sim_data["trial"] = sim_data["trial"].astype(int) + 1
    sim_data = numeric_col_to_categorical(sim_data, "trial")

    plot = plot + pn.geom_point(  # simulation langs
        data=sim_data,
        mapping=pn.aes(color="trial"),
        shape="o",
        size=4,
    )

    if trajectory_data is not None:
        trajectory_data = numeric_col_to_categorical(trajectory_data, "trial")
        plot = plot + pn.geom_line(
            data=trajectory_data,
            mapping=pn.aes(
                # x="complexity",
                # y="accuracy",
                color="trial",
                # color="round",
                ), # categorical
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

    plot = (
        # Set data and the axes
        pn.ggplot(data=curve_data, mapping=pn.aes(x="complexity", y=y))
        + pn.xlab("Complexity $I[M:W]$ bits")
        + pn.ylab(ystr)
    )
    plot = plot + pn.geom_line()
    # plot = plot + pn.geom_point()
    return plot


# encoders stuff

def faceted_encoders(df: pd.DataFrame) -> pn.ggplot:
    """Return a plot of different encoder heatmap subplots, faceted by trial."""
    return (
        basic_encoder_heatmap(df)
        + pn.facet_grid("trial ~ .")
        + pn.theme(
            axis_text_y=pn.element_blank(),
            axis_text_x=pn.element_blank(),
        )
    )

def get_n_heatmaps(df: pd.DataFrame, all_trials: bool = True, n: int = 8,) -> list[pn.ggplot]:
    """Return a list of heatmaps, one for each encoder corresponding to each trial. If `all_trials` is False, get `n` heatmaps, which is 8 by default."""
    trials = df["trial"].unique()
    if not all_trials:
        trials = trials[:8]
    return [
        (
            basic_encoder_heatmap(df[df["trial"] == trial])
            + pn.ggtitle(f"Trial {trial}")
        )
        for trial in trials
    ]


def basic_encoder_heatmap(df: pd.DataFrame) -> pn.ggplot:
    """Return a single heatmap plot for an encoder."""
    df["trial"] = df["trial"].astype(int) + 1
    df = numeric_col_to_categorical(df, "meanings")
    df = numeric_col_to_categorical(df, "words")
    return (
        pn.ggplot(df, pn.aes(**dict(zip(["x", "y", "fill"], encoder_columns[:3]))))
        + pn.geom_tile()
        + pn.scale_fill_cmap(limits=[0,1])
    )
