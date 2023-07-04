import random
import textwrap

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE

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
    xlabel: str,
    ylabel: str,
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
        xlabel: label for the x-axis.
        ylabel: label for the y-axis.
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
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    sns.move_legend(ax, "lower right")


def _violinplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    ax: plt.Axes,
    title: str,
    xlabel: str,
    ylabel: str,
    yticklabels: list[str],
) -> None:
    """Create a violin plot using seaborn to visualize the distribution of data
    points in each category.

    Args:
        df: DataFrame containing the data.
        x: column name of the numerical variable to be plotted on the x-axis.
        y: column name of the categorical variable to be plotted on the y-axis.
        ax: matplotlib Axes object to draw the plot onto.
        title: title of the plot.
        xlabel: label for the x-axis.
        ylabel: label for the y-axis.
        yticklabels: labels for the y-axis tick marks.
    """
    sns.violinplot(data=df, x=x, y=y, ax=ax, inner=None)
    sns.stripplot(data=df, x=x, y=y, ax=ax, color="black")
    ax.set_yticklabels(yticklabels)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
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
        xlabel="Age (years)",
        ylabel="",
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
        xlabel="count",
        ylabel="",
        yticklabels=["Male", "Female"],
    )

    # Marital status
    _countplot(
        df=df,
        y="Marital status",
        hue="Group",
        ax=axes[1, 0],
        title="Marital status",
        xlabel="count",
        ylabel="",
        legend_labels=["Control", "Schizophrenia"],
    )

    # Dwelling condition
    _countplot(
        df=df,
        y="Dwelling condition",
        hue="Group",
        ax=axes[1, 1],
        title="Dwelling condition",
        xlabel="count",
        ylabel="",
        legend_labels=["Control", "Schizophrenia"],
    )

    # Education level
    _countplot(
        df=df,
        y="Education level",
        hue="Group",
        ax=axes[1, 2],
        title="Education level",
        xlabel="count",
        ylabel="",
        legend_labels=["Control", "Schizophrenia"],
    )

    # Sample center
    _countplot(
        df=df,
        y="Sample center",
        hue="Group",
        ax=axes[2, 0],
        title="Sample center",
        xlabel="count",
        ylabel="",
        legend_labels=["Control", "Schizophrenia"],
    )
    # Add horizontal lines
    axes[2, 0].axhline(y=0.7, color="gray", linestyle="--")
    axes[2, 0].axhline(y=1.6, color="gray", linestyle="--")
    axes[2, 0].axhline(y=2.5, color="gray", linestyle="--")
    axes[2, 0].axhline(y=3.5, color="gray", linestyle="--")
    axes[2, 0].axhline(y=4.5, color="gray", linestyle="--")

    # BMI
    _violinplot(
        df=df,
        x="BMI",
        y="Group",
        ax=axes[2, 1],
        title="BMI",
        xlabel="BMI",
        ylabel="",
        yticklabels=["Control", "Schizophrenia"],
    )

    # Smoking
    _countplot(
        df=df,
        y="Smoking",
        hue="Group",
        ax=axes[2, 2],
        title="Smoking",
        xlabel="count",
        ylabel="",
        legend_labels=["Control", "Schizophrenia"],
    )

    plt.tight_layout()

    return fig


def plot_blood_based_biomarkers(df: pd.DataFrame) -> plt.Figure:
    """Generate a figure with multiple subplots to visualize the distribution
    of blood-based biomarkers.

    Args:
        df: DataFrame containing the data.

    Returns:
        The generated matplotlib Figure object.
    """
    fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(20, 20))

    biomarker_columns = [
        "Tryptophane(μM)",
        "Glutamic acid(μM)",
        "Tyrosine(μM)",
        "Phenylalanine(μM)",
        "Dopamine(ng/ml)",
        "Gamma-aminobutyric acid  (GABA)(ng/L)",
        "Serotonin(ng/ml)",
        "Kynurenine (KYN)(nmol/L)",
        "Kynurenic acid (KYNA)(nmol/L)",
    ]

    row_index = 0
    col_index = 0
    for column in biomarker_columns:
        _violinplot(
            df=df,
            x=column,
            y="Group",
            ax=axes[row_index, col_index],
            title=column[: column.index("(")].strip(),
            xlabel=column[column.index("(") :].strip(),
            ylabel="",
            yticklabels=["Control", "Schizophrenia"],
        )

        col_index += 1
        if col_index == 3:
            row_index += 1
            col_index = 0

    plt.tight_layout()

    return fig


def plot_genus(
    df: pd.DataFrame, controls: list[str], schizophrenia: list[str]
) -> plt.Figure:
    """Generate a bar plot to compare the relative abundance of different
    genera between control and schizophrenia groups.

    Args:
        df: DataFrame containing the genus data.
        controls: list of column names representing the control group in the DataFrame.
        schizophrenia: list of column names representing the schizophrenia group in the DataFrame.

    Returns:
        The generated matplotlib Figure object.
    """
    random.seed(92)
    colors = [[random.random() for _ in range(3)] for _ in range(len(df))]

    fig, (ax_1, ax_2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))

    for ax, group in zip([ax_1, ax_2], [controls, schizophrenia]):
        df_group = df[group]
        df_group.T.plot(
            kind="bar",
            stacked=True,
            width=1,
            edgecolor=None,
            color=colors,
            ax=ax,
            legend=False,
        )
        ax.set_xticks(ticks=range(len(df_group.columns)), labels=[])
        ax.set_xlim(-0.5, len(df_group.columns) - 0.5)
        ax.set_ylim(0, 100)

    ax_1.set_ylabel("Relative abundance (%)")
    ax_1.set_xlabel("Control")
    ax_2.set_xlabel("Schizophrenia")

    n_taxa = 25
    handles, labels = ax_2.get_legend_handles_labels()
    ax_2.legend(
        handles[:n_taxa],
        [i.split("g__")[1].replace("_", ", ") for i in labels[:n_taxa]],
        bbox_to_anchor=(1.05, 1),
        title=f"Top {n_taxa} most abundant genera",
        title_fontsize=14,
    )

    plt.tight_layout()

    return fig
