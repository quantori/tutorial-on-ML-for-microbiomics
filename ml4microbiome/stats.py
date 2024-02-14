"""Module containing utility functions to perform statistical analyses."""

import numpy as np
import pandas as pd
from scikit_posthocs import posthoc_dunn
from scipy.stats import kruskal
from scipy.stats._stats_py import KruskalResult


def kruskal_dunn(
    df: pd.DataFrame, param: str, metric: str
) -> tuple[KruskalResult, pd.DataFrame]:
    """Apply Kruskal-Wallis test to determine whether samples from groups
    specified by 'param' originate from the same distribution of 'metric'
    scores.

    Args:
        df: DataFrame containing the data.
        param: column name of the categorical variable / groups.
        metric: column name of the numerical variable whose distribution will be queried.

    Returns:
        kw: Kruskal-Wallis test results.
        dunn: Dunn test results.
    """
    df.drop(df.columns.difference([param, metric]), axis=1)
    kw = kruskal(*[group[metric].values for name, group in df.groupby(param)])
    dunn = posthoc_dunn(df, val_col=metric, group_col=param, p_adjust="bonferroni")

    return kw, dunn


def MAD(x: np.ndarray) -> int:
    """Calculates median absolute deviation of a numpy array.

    Args:
        x: variable elements.

    Returns:
        An int representing median absolute deviation.
    """
    med = np.median(x)
    x = abs(x - med)
    mad = np.median(x)

    return mad


def convert_df_for_stats(results_auroc: pd.DataFrame) -> pd.DataFrame:
    """Converts format of results_auroc data frame into a different format that
    can be input to stats modules.

    Args:
        results_auroc: for each experimental condition, has AUROC column with list of AUROC scores
        for all replicates.

    Returns:
        out: for each experimental condition and replicate, a unique row denoting data type and AUROC.
    """
    out = pd.DataFrame([], [])
    for i in range(len(results_auroc["Data type"])):
        aurocs = results_auroc.iloc[i]["AUROC"]
        for auroc in aurocs:
            out = pd.concat(
                [
                    out,
                    pd.DataFrame(
                        [
                            [
                                results_auroc.iloc[i]["Data type"],
                                results_auroc.iloc[i]["Taxonomic level"],
                                results_auroc.iloc[i]["ML algorithm"],
                                auroc,
                            ]
                        ]
                    ),
                ]
            )
    out = out.rename(
        columns={0: "Data type", 1: "Taxonomic level", 2: "ML algorithm", 3: "AUROC"}
    )

    return out
