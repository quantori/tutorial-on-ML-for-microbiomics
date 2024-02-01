"""Module containing utility functions to perform error analyses."""
from collections import defaultdict

import pandas as pd

def error_counts(
    expt_condition: defaultdict
    ) -> (defaultdict, defaultdict):
    """From a metaphlan output file of inferred taxa, select only entries at a
        specific taxonomic level.
        
        Args:    
            expt_condition: nested CV experimental condition results (e.g., outer_results["microbiome_only"]["species]["lightGBM"])
            
        Returns:
            count_samples: for each sample, the number of times that the prediction was wrong across all nested CV test reps
            count_errs: for each sample, the number of predictions made across all nested CV test reps
    """
    count_samples = defaultdict(int)
    count_errs = defaultdict(int)
    for i in range(len(expt_condition["errors"])):
        for cm in expt_condition["errors"][i]:
            if cm == "fp" or cm == "fn":
                for sample in expt_condition["errors"][i][cm]:
                    count_errs[sample]+=1
            for sample in expt_condition["errors"][i][cm]:
                count_samples[sample] += 1
    for i in count_samples:
        if i not in count_errs:
            count_errs[i] = 0
            
    return count_samples, count_errs
