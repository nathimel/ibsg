import plotnine as pn
import pandas as pd


def basic_tradeoff_plot(
    curve_data: pd.DataFrame,
    sim_data: pd.DataFrame,
    sampled_data: pd.DataFrame = None,
) -> pn.ggplot:
    """Get a basic plotnine point plot of languages in a complexity vs accuracy 2D plot."""
    plot = (
        # Set data and the axes
        pn.ggplot(data=curve_data, mapping=pn.aes(x="complexity", y="accuracy"))
        + pn.xlab("Complexity $I[M:W]$ bits")
        + pn.ylab("Accuracy $I[W:U]$ bits")
        + pn.scale_color_cmap("cividis")
    )
    if sampled_data is not None:
        plot = plot + pn.geom_point(  # hypothetical langs bottom layer
            sampled_data,
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
    # plot = plot + pn.geom_line(size=2)  # curve last
    plot = plot + pn.geom_point()
    return plot

def bound_only_plot(
    curve_data: pd.DataFrame,
) -> pn.ggplot:
    plot = (
        # Set data and the axes
        pn.ggplot(data=curve_data, mapping=pn.aes(x="complexity", y="accuracy"))
        + pn.xlab("Complexity $I[M:W]$ bits")
        + pn.ylab("Accuracy $I[W:U]$ bits")
        + pn.scale_color_cmap("cividis")
    )
    plot = plot + pn.geom_line()
    return plot