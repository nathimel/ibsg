import plotnine as pn
import pandas as pd


def basic_tradeoff_plot(
    pareto_data: pd.DataFrame,
    sim_data: pd.DataFrame,
    sampled_data: pd.DataFrame = None,
) -> pn.ggplot:
    """Get a basic plotnine point plot of languages in a complexity vs comm_cost 2D plot."""
    plot = (
        # Set data and the axes
        pn.ggplot(data=pareto_data, mapping=pn.aes(x="rate", y="distortion"))
        + pn.xlab("Complexity $I(S;\hat{S})$")
        + pn.ylab("Communicative Cost $D[S, \hat{S}]$")
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
    plot = plot + pn.geom_line(size=2)  # pareto frontier last
    return plot