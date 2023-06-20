import copy
from collections import defaultdict
from typing import Literal

import numpy as np
import pandas as pd
from skbio.stats.composition import clr, multiplicative_replacement


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
    ]
    tax_level = tax_level.lower()
    if tax_level not in possibilities:
        raise ValueError(
            "Invalid taxonomic level provided. The taxonomic level must be one "
            f"of the following options: {possibilities}"
        )

    idx = possibilities.index(tax_level)
    if tax_level != "species":
        one_lower = possibilities[idx + 1]
        return df[
            (df.index.str.contains(tax_level[0] + "__") == True)
            & (df.index.str.contains(one_lower[0] + "__") == False)
        ]
    else:
        return df[df.index.str.contains(tax_level[0] + "__")]


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
