import matplotlib.pyplot as plt
import pandas as pd


def pie_plot_hospital(df: pd.DataFrame) -> plt.Figure:
    """Generate a pie chart to visualize the distribution of samples across
    different centers in a hospital.

    Args:
        df: input DataFrame containing the sample data.

    Returns:
        The generated matplotlib Figure object representing the pie chart.
    """
    vals = df["Sample center"].value_counts()

    colours = ["#0d1a26", "#204060", "#336699", "#538cc6", "#8cb3d9", "#c6d9ec"]

    fig, ax = plt.subplots()
    # theme = plt.get_cmap("gnuplot2")
    # ax.set_prop_cycle("color", [theme(1.0 * i / len(vals)) for i in range(len(vals))])

    labels = []
    for center, count in zip(vals.index, vals):
        labels.append(
            center
            + "\n"
            + str(count)
            + " ("
            + str(round(count / vals.sum() * 100))
            + "%)"
        )

    ax.pie(vals, labels=labels, startangle=90, colors=colours)

    centre_circle = plt.Circle((0, 0), 0.65, fc="white")
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    ax.axis("equal")
    plt.tight_layout()

    return fig
