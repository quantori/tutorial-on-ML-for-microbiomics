"""Module containing utility functions to perform statistical analyses."""
import pandas as pd
from scipy.stats import kruskal
from scipy.stats._stats_py import KruskalResult
from scikit_posthocs import posthoc_dunn

def kruskal_dunn(
    df: pd.DataFrame,
    param: str,
    metric: str
) -> tuple[KruskalResult, pd.DataFrame]:
    """Apply Kruskal-Wallis test to determine whether samples from groups specified by 'param' originate from the same distribution of 'metric' scores.   
    
    Args:
    df: DataFrame containing the data.
    param: column name of the categorical variable / groups.
    metric: column name of the numerical variable whose distribution will be queried.
    
    Returns:
    kw: Kruskal-Wallis test results.
    dunn: Dunn test results.
    """
    data = [df[df[param]==i][metric] for i in df[param].unique()]
    kw = kruskal(*data)
    dunn = posthoc_dunn(data, p_adjust="bonferroni")
    
    return kw, dunn
    