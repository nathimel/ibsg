import plotnine as pn
import pandas as pd

# Plotting util functions

def numeric_col_to_categorical(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Change a float valued column (e.g. trial or round) to Categorical for visualization."""
    # adjust values to Categorical where appropriate
    df[col] = df[col].astype(int).astype(str)
    df = df.assign(trial=pd.Categorical(df[col]))
    return df


# Canonical plotting functions




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

    plot = plot + pn.geom_point(  # simulation langs
        sim_data,
        color="blue",
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
        + pn.scale_color_cmap("cividis")
    )
    plot = plot + pn.geom_line()
    # plot = plot + pn.geom_point()
    return plot