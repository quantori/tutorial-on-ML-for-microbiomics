"""Module containing utility functions to perform statistical analyses."""
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal
from scipy.stats._stats_py import KruskalResult
from scikit_posthocs import posthoc_dunn
from statsmodels.sandbox.stats.multicomp import multipletests

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
    df.drop(df.columns.difference([param, metric]), axis=1)
    kw = kruskal(*[group[metric].values for name, group in df.groupby(param)])
    dunn = posthoc_dunn(df, val_col=metric, group_col=param, p_adjust="bonferroni")  
      
    return kw, dunn
    
def get_chi_inputs(
    col_name: str, 
    count_errs: defaultdict, 
    count_samples: defaultdict,
    df_metadata: pd.DataFrame
) -> (pd.DataFrame):
    """For a given demographic factor, prepare a table for each group that shows the number of correct and incorrect predictions.
    
    Args:
        col_name: name of demographic factor
        count_errs: for each sample, the number of predictions made across all nested CV test reps
        count_samples: for each sample, the number of times that the prediction was wrong across all nested CV test reps
        df_metadata: metadata mapping sample to demographic factors
        
    Returns:
        pd.DataFrame:
    """
    df1 = pd.DataFrame(count_errs.items()).sort_values(by=[0]).set_index(df_metadata[col_name].index).rename(columns={1:"num errors"})
    df2 = pd.DataFrame(count_samples.items()).sort_values(by=[0]).set_index(df_metadata[col_name].index).rename(columns={1:"num samples"})
    df1 = pd.concat([df2, df1], axis=1)
    df1 = pd.concat([df_metadata[col_name], df1], axis=1)

    table = []
    groups = []
    for i in df1[col_name].unique():
        errs = df1.groupby(col_name).sum().loc[i]["num errors"]
        samples = df1.groupby(col_name).sum().loc[i]["num samples"]
        table.append([(samples-errs), errs])
        groups.append(i)
    
    return pd.DataFrame(table, index=groups, columns=['Correct', 'Incorrect'])    

def get_asterisks_for_pval(p_val):
    """Receives the p-value and returns asterisks string."""
    if p_val > 0.05:
        p_text = "ns"  # above threshold => not significant
    elif p_val < 1e-4:  
        p_text = '****'
    elif p_val < 1e-3:
        p_text = '***'
    elif p_val < 1e-2:
        p_text = '**'
    else:
        p_text = '*'
    
    return p_text

def chisq_and_posthoc_corrected(df: pd.DataFrame):
    """Receives a dataframe and performs chi2 test and then post hoc.
    Prints the p-values and corrected p-values (after FDR correction)
    
    With thanks to https://neuhofmo.github.io/chi-square-and-post-hoc-in-python/
    """
    # start by running chi2 test on the matrix
    chi2, p, dof, ex = chi2_contingency(df, correction=True)
    print(f"Chi2 result of the contingency table: {chi2}, p-value: {p}")
    
    if p < 0.05: 
    
        # post-hoc
        all_combinations = list(combinations(df.index, 2))  # gathering all combinations for post-hoc chi2
        p_vals = []
        print("Significance results:")
        for comb in all_combinations:
            new_df = df[(df.index == comb[0]) | (df.index == comb[1])]
            chi2, p, dof, ex = chi2_contingency(new_df, correction=True)
            p_vals.append(p)
            # print(f"For {comb}: {p}")  # uncorrected
    
        # checking significance
        # correction for multiple testing
        reject_list, corrected_p_vals = multipletests(p_vals, method='fdr_bh')[:2]
        
        for p_val, corr_p_val, reject, comb in zip(p_vals, corrected_p_vals, reject_list, all_combinations):
            print(f"{comb}: p_value: {p_val:5f}; corrected: {corr_p_val:5f} ({get_asterisks_for_pval(p_val)}) reject H0: {reject}")

def MAD(x: np.ndarray) -> int:
    """Calculates median absolute deviation of a numpy array.
    
    Args:
        x: variable elements.
    
    Returns:
        An int representing median absolute deviation. 
    """
    med = np.median(x)
    x   = abs(x-med)
    MAD = np.median(x)
    
    return MAD
                
def convert_df_for_stats(results_auroc: pd.DataFrame) -> pd.DataFrame:
    """Converts format of results_auroc data frame into a different format that can be input to stats modules. 
    
    Args:
        results_auroc: for each experimental condition, has AUROC column with list of AUROC scores for all replicates
    
    Returns:
        out: for each experimental condition and replicate, a unique row denoting data type and AUROC.
    """
    out = pd.DataFrame([], [])
    for i in range(len(results_auroc['Data type'])):
        aurocs = results_auroc.iloc[i]["AUROC"]
        for auroc in aurocs:
            out = pd.concat([out, pd.DataFrame([[results_auroc.iloc[i]["Data type"], 
                                               results_auroc.iloc[i]["Taxonomic level"],
                                               results_auroc.iloc[i]["ML algorithm"],
                                               auroc
                                               ]]
                                              )
                            ]
                           )
    out = out.rename(columns={0:"Data type", 1:"Taxonomic level", 2:"ML algorithm", 3:"AUROC"})                
       
    return out   