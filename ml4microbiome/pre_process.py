"""Module containing utility functions to preprocess data."""
import copy
from collections import defaultdict
from typing import Literal

import numpy as np
import pandas as pd
from skbio.stats.composition import clr, multiplicative_replacement
from sklearn.preprocessing import StandardScaler


def get_tax_level(
    df: pd.DataFrame,
    tax_level: Literal[
        "kingdom",
        "phylum",
        "class",
        "order",
        "family",
        "genus",
        "species",
    ],
) -> pd.DataFrame:
    """From a metaphlan output file of inferred taxa, select only entries at a
    specific taxonomic level.

    Args:
        df: metaphlan output (header and index set).
        tax_level: taxonomic level to return.

    Returns:
         The subset of the input DataFrame containing rows corresponding to the specified
         taxonomic level.

    Raises:
        ValueError: if an invalid taxonomic level is provided.
    """
    # Make sure that a valid taxonomic level was entered
    possibilities = [
        "kingdom",
        "phylum",
        "class",
        "order",
        "family",
        "genus",
        "species",
        "strain"
    ]
    
    tax_level = tax_level.lower()
    if tax_level not in possibilities:
        raise ValueError(
            "Invalid taxonomic level provided. The taxonomic level must be one "
            f"of the following options: {possibilities.keys()}"
        )

    idx = possibilities.index(tax_level)
    
    if tax_level != "strain" and tax_level != "species":
        return df[
        (df.index.str.contains(possibilities[idx][0] +"__")) 
        & ~(df.index.str.contains(possibilities[idx+1][0]+"__"))
        ]
    if tax_level == "species":
        return df[
        (df.index.str.contains(possibilities[idx][0] +"__")) 
        & ~(df.index.str.contains("t__"))
        ]
    if tax_level == "strain":
        return df[
        (df.index.str.contains("t__")) 
        ]

def clr_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Perform CLR transformation on a metaphlan inferred taxa file.

    Args:
        df: metaphlan output in DataFrame format (header and index set).

    Returns:
        CLR transformed metaphlan output as a DataFrame.
    """
    row_names = df.index
    col_names = df.columns

    x = np.array(np.array(df))
    x = clr(multiplicative_replacement(x))

    df = pd.DataFrame(x)
    df.index = row_names
    df.columns = col_names

    return df


def merge_metadata_microbiome(
    data_dict: defaultdict[str, defaultdict[str, pd.DataFrame]], tax_levels: list[str]
) -> defaultdict[str, defaultdict[str, pd.DataFrame]]:
    """Merge metadata and microbiome data at specified taxonomic levels and add
    them to the defaultdict.

    Args:
        data_dict: default dict where keys are data type (metadata_only, microbiome_only,
            metadata_microbiome) and taxonomic level (all, species, genus, etc).
        tax_levels: list of taxonomic levels.

    Returns:
        The updated nested defaultdict with merged metadata and microbiome data.
    """
    data_dict_copy = copy.deepcopy(data_dict)

    for tax_level in tax_levels:
        data_dict_copy["metadata_microbiome"][tax_level] = pd.merge(
            data_dict["metadata_only"]["all"],
            data_dict["microbiome_only"][tax_level],
            left_index=True,
            right_index=True,
        )

    return data_dict_copy


def scale_features(
    X_train: pd.DataFrame, X_test: pd.DataFrame, cols_to_scale: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Standardize features by removing the mean and scaling to unit variance.

    Args:
        X_train: numerical metadata features, training data.
        X_test: numerical metadata features, test data.
        cols_to_scale: names of feature columns to scale.

    Returns:
        A tuple containing feature scaled numerical metadata for training and test data.
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train[cols_to_scale]))
    X_train_scaled.columns = cols_to_scale
    X_train_scaled.index = X_train.index
    X_test_scaled = pd.DataFrame(scaler.transform(X_test[cols_to_scale]))
    X_test_scaled.columns = cols_to_scale
    X_test_scaled.index = X_test.index
    X_train = pd.merge(
        X_train_scaled,
        X_train.drop(columns=cols_to_scale),
        left_index=True,
        right_index=True,
    )
    X_test = pd.merge(
        X_test_scaled,
        X_test.drop(columns=cols_to_scale),
        left_index=True,
        right_index=True,
    )

    return X_train, X_test









    