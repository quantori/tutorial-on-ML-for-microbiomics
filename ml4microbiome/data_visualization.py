"""Module containing utility functions to visualize data."""

import random
import textwrap
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE

sns.set_theme(style="whitegrid")


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
    yticklabels: list[str] | None = None,
    plot_dps: bool = True,
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
        plot_dps: plot stripplot overtop violin plot.
    """
    sns.violinplot(data=df, x=x, y=y, ax=ax, inner=None)
    if plot_dps:
        sns.stripplot(data=df, x=x, y=y, ax=ax, color="black")
    if yticklabels:
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


def plot_tsne(df: pd.DataFrame, hue: pd.Series, random_state: int) -> plt.Figure:
    """Generate a scatter plot using t-SNE dimensionality reduction for CLR
    transformed genus data.

    Args:
        df: DataFrame containing the genus data.
        hue: Series specifying the grouping variable for coloring the data points.
        random_state: seed used by the random number generator for reproducibility.

    Returns:
        The generated matplotlib Figure object.
    """
    projection = TSNE(random_state=random_state).fit_transform(df.values)
    fig = plt.figure(figsize=(10, 10))
    sns.scatterplot(x=projection[:, 0], y=projection[:, 1], hue=hue)

    legend_labels = ["Control", "Schizophrenia"]
    legend_handles = plt.gca().get_legend_handles_labels()[0]
    plt.legend(legend_handles, legend_labels, loc="lower right")

    return fig


def plot_data_type(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots()

    _violinplot(
        df=df,
        x="Data type",
        y="AUROC",
        ax=ax,
        title="",
        xlabel="",
        ylabel="AUROC",
        yticklabels=[],
        plot_dps=False,
    )

    ax.tick_params(axis="x", labelrotation=90)
    ax.set_ylim(0, 1)

    return fig


def plot_tax_level(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots()

    _violinplot(
        df=df,
        x="Taxonomic level",
        y="AUROC",
        ax=ax,
        title="",
        xlabel="",
        ylabel="AUROC",
        yticklabels=[],
        plot_dps=False,
    )

    ax.tick_params(axis="x", labelrotation=90)
    ax.set_ylim(0, 1)

    return fig


def plot_ml_alg(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots()

    _violinplot(
        df=df,
        x="ML algorithm",
        y="AUROC",
        ax=ax,
        title="",
        xlabel="",
        ylabel="AUROC",
        yticklabels=[],
        plot_dps=False,
    )

    ax.tick_params(axis="x", labelrotation=90)
    ax.set_ylim(0, 1)

    return fig


def get_data(
    col_name: str,
    count_samples: defaultdict,
    count_errs: defaultdict,
    df_metadata: pd.DataFrame,
) -> tuple[list[str], list[float]]:
    """For a given demographic factor, retrieves the error rate (%) per group.
    This is later used for plotting.

    Args:
        col_name: name of demographic factor.
        count_samples: for each sample, the number of times that the prediction was wrong 
        	across all test reps.
        count_errs: for each sample, the number of predictions made across all test reps.
        df_metadata: metadata mapping sample to demographic factors.

    Returns:
        x: names of groups within a demographic factor.
        y: for each group, the percentage of samples that were incorrectly predicted.

    Raises:
        ValueError: if an invalid taxonomic level is provided.
    """
    if count_samples[0] < count_errs[0]:
        raise ValueError(
            "It appears that count_samples and count_errs were provided in the reverse order."
        )

    df1 = (
        pd.DataFrame(count_errs.items())
        .sort_values(by=[0])
        .set_index(df_metadata[col_name].index)
        .rename(columns={1: "num errors"})
    )
    df2 = (
        pd.DataFrame(count_samples.items())
        .sort_values(by=[0])
        .set_index(df_metadata[col_name].index)
        .rename(columns={1: "num samples"})
    )
    df1 = pd.concat([df2, df1], axis=1)
    df1 = pd.concat([df_metadata[col_name], df1], axis=1)

    groups = []
    err_perc = []
    for i in df1[col_name].unique():
        errs = df1.groupby(col_name).sum().loc[i]["num errors"]
        samples = df1.groupby(col_name).sum().loc[i]["num samples"]
        groups.append(i)
        err_perc.append(errs / samples * 100)

    return groups, err_perc
