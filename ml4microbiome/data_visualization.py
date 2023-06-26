import textwrap

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")


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


def _countplot(
    df: pd.DataFrame,
    y: str,
    hue: str,
    ax: plt.Axes,
    title: str,
    legend_labels: list[str],
    yticklabels: list[str] | None = None,
) -> None:
    """Create a countplot using seaborn to visualize the count of data points
    in each category.

    Args:
        df: DataFrame containing the data.
        y: column name of the categorical variable to be plotted on the y-axis.
        hue: column name of the categorical variable used for grouping and coloring.
        ax: matplotlib Axes object to draw the plot onto.
        title: title of the plot.
        legend_labels: labels for the legend.
        yticklabels: labels for the y-axis tick marks. If None, it uses the unique
            values of the 'y' column with line breaks for long labels.
    """
    sns.countplot(data=df, y=y, hue=hue, ax=ax)
    ax.legend(labels=legend_labels)
    if yticklabels:
        ax.set_yticklabels(yticklabels)
    else:
        ax.set_yticklabels(
            [
                "\n".join(textwrap.wrap(status.capitalize(), 20))
                for status in df[y].unique()
            ]
        )
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_title(title)


def _violinplot(
    df: pd.DataFrame, x: str, y: str, ax: plt.Axes, title: str, yticklabels: list[str]
) -> None:
    """Create a violin plot using seaborn to visualize the distribution of data
    points in each category.

    Args:
        df: DataFrame containing the data.
        x: column name of the numerical variable to be plotted on the x-axis.
        y: column name of the categorical variable to be plotted on the y-axis.
        ax: matplotlib Axes object to draw the plot onto.
        title: title of the plot.
        yticklabels: labels for the y-axis tick marks.
    """
    sns.violinplot(data=df, x=x, y=y, inner="points", ax=ax)
    ax.set_yticklabels(yticklabels)
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_title(title)


def plot_demographics(df: pd.DataFrame) -> plt.Figure:
    """Generate a figure with multiple subplots to visualize demographic
    information.

    Args:
        df: DataFrame containing the data.

    Returns:
        The generated matplotlib Figure object.
    """
    fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(20, 20))

    # Diagnosis
    diagnosis = df["Diagnosis"].value_counts()
    colors = ["#0d1a26", "#204060", "#336699"]
    labels = []
    for center, count in zip(diagnosis.index, diagnosis):
        labels.append(
            center.capitalize()
            + "\n"
            + str(count)
            + " ("
            + str(round(count / diagnosis.sum() * 100))
            + "%)"
        )
    axes[0, 0].pie(diagnosis, labels=labels, startangle=90, colors=colors)
    centre_circle = plt.Circle((0, 0), 0.65, fc="white")
    axes[0, 0].add_artist(centre_circle)
    axes[0, 0].set_title("Diagnosis")

    # Age
    _violinplot(
        df=df,
        x="Age",
        y="Group",
        ax=axes[0, 1],
        title="Age",
        yticklabels=["Control", "Schizophrenia"],
    )

    # Gender
    _countplot(
        df=df,
        y="Gender (1:male, 2:female)",
        hue="Group",
        ax=axes[0, 2],
        title="Gender",
        legend_labels=["Control", "Schizophrenia"],
        yticklabels=["Male", "Female"],
    )

    # Marital status
    _countplot(
        df=df,
        y="Marital status",
        hue="Group",
        ax=axes[1, 0],
        title="Marital status",
        legend_labels=["Control", "Schizophrenia"],
    )

    # Dwelling condition
    _countplot(
        df=df,
        y="Dwelling condition",
        hue="Group",
        ax=axes[1, 1],
        title="Dwelling condition",
        legend_labels=["Control", "Schizophrenia"],
    )

    # Education level
    _countplot(
        df=df,
        y="Education level",
        hue="Group",
        ax=axes[1, 2],
        title="Education level",
        legend_labels=["Control", "Schizophrenia"],
    )

    # Sample center
    _countplot(
        df=df,
        y="Sample center",
        hue="Group",
        ax=axes[2, 0],
        title="Sample center",
        legend_labels=["Control", "Schizophrenia"],
    )

    # BMI
    _violinplot(
        df=df,
        x="BMI",
        y="Group",
        ax=axes[2, 1],
        title="BMI",
        yticklabels=["Control", "Schizophrenia"],
    )

    # Smoking
    _countplot(
        df=df,
        y="Smoking",
        hue="Group",
        ax=axes[2, 2],
        title="Smoking",
        legend_labels=["Control", "Schizophrenia"],
    )

    plt.tight_layout()

    return fig
