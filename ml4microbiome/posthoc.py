from typing import Any

import numpy as np
import pandas as pd
import shap

def shap_(
    model: Any, 
    X_train: pd.DataFrame, 
    X_test: pd.DataFrame, 
    ) -> shap._explanation.Explanation:
    """
    Calculate SHAP values.
    
    Args:
    model: trained/tuned model.
    X_test: training feature dataset.
    X_test: test feature dataset.
    """
    max_evals = max(((2 * X_train.shape[1]) + 1), 500)
    explainer = shap.Explainer(model.predict, X_train.to_numpy(), max_evals=max_evals)
    shap_values = explainer(X_test.to_numpy())
    
    return shap_values

def errors(
	yhat: np.ndarray,
	y_test: pd.DataFrame
	) -> dict[str, int]:
	"""
	Record sample index of true positives, true negatives, false positives, and false negatives.
	
	Args:
	yhat: model predictions.
	y_test: test target labels.
	"""
	
	cm_idx = {'tp':[], 'tn':[], 'fp':[], 'fn':[]} 
	for i, idx in enumerate(y_test.index):
	    if y_test[idx] == yhat[i] == 1:
	        cm_idx['tp'].append(idx)
	    elif y_test[idx] == yhat[i] == 0:
	        cm_idx['tn'].append(idx)
	    elif y_test[idx] == 1 and yhat[i] == 0:
	        cm_idx['fn'].append(idx)
	    elif y_test[idx] == 0 and yhat[i] == 1:
	        cm_idx['fp'].append(idx)
	
	return cm_idx