import numpy as np
import pandas as pd
import sys

from skbio.stats.composition import clr, multiplicative_replacement
from sklearn.preprocessing import StandardScaler

def get_tax_level(df, tax_level):
    """
    From a Metaphlan output file of inferred taxa, select only entries at a specific taxonomic level
    
    Arguments:
        df (pd.DataFrame) -- metaphlan output (header and index set)
        tax_level (str) -- taxonomic level to return, options are "kingdom", "phylum", "class", "order", "family", "genus", "species"
        
    Returns:
        df (pd.DataFrame) -- for a single taxonomic level
    """    
    # Make sure that a valid taxonomic level was entered
    possibilities = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
    tax_level = tax_level.lower()
    if tax_level not in possibilities:
        print("Invalid taxonomic level provided, please enter a taxonomic level from the following options:", [i for i in possibilities.keys()])
                
    idx = possibilities.index(tax_level) 
    if tax_level != "species":
        one_lower = possibilities[idx+1]
        return df[(df.index.str.contains(tax_level[0]+"__")==True) & (df.index.str.contains(one_lower[0]+"__")==False)]
    else:
        return df[df.index.str.contains(tax_level[0]+"__")]

def clr_transform(df):
    """
    Perform CLR transform out metaphlan inferred taxa file
    
    Arguments:
        df (pd.DataFrame) -- metaphlan output in df format (header and index set)
    
    Returns:
        df (pd.DataFrame) -- CLR transformed metaphlan output 
    """
    
    row_names = df.index
    col_names = df.columns
    
    x = np.array(np.array(df))
    x = clr(multiplicative_replacement(x))
    
    df = pd.DataFrame(x)
    
    df.index = row_names
    df.columns = col_names
    
    return df

def merge_metadata_microbiome(data_dict, tax_levels):
    """
    Create pandas df that has both metadata and microbiome data
    
    Arguments:
        data_dict (defaultdict of defaultdict) -- keys are data type (metadata_only, microbiome_only, metadata_microbiome) and taxonomic level (all, species, genus, etc)
        tax_levels
        
    Returns:
    
    """
    
    for tax_level in tax_levels:
        data_dict["metadata_microbiome"][tax_level] = pd.merge(data_dict["metadata_only"]["all"], \
                                                       data_dict["microbiome_only"][tax_level], left_index=True,\
                                                       right_index=True)
    return data_dict


