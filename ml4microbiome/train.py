"""Module containing utility functions for model training and testing."""
from collections import Counter, defaultdict
import random
from statistics import mode
from typing import Any, Literal

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import median_abs_deviation
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV

def train_model(
    alg: Literal['random_forest', 'lightGBM', 'logistic_regression_L1'],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
    hyper_parameters: dict[str, Any],
) -> Any:
    """Train a model using the specified algorithm and hyperparameters.

    Args:
        alg: algorithm to use for training.
            - "random_forest": Random Forest classifier.
            - "lightGBM": LightGBM classifier.
            - "logistic_regression_L1": Logistic Regression with L1 regularization.
        X_train: training feature dataset.
        y_train: training target labels.
        random_state: seed for reproducibility.
        hyper_parameters: hyperparameters for the specified algorithm. The dictionary
            should contain the hyperparameters as key-value pairs.

    Returns:
        The trained model.

    Raises:
        ValueError: if an invalid algorithm name is provided.
    """
    if alg == "random_forest":
        model = RandomForestClassifier(random_state=random_state, **hyper_parameters)
    elif alg == "lightGBM":
        model = LGBMClassifier(random_state=random_state, **hyper_parameters)
    elif alg == "logistic_regression_L1":
        model = LogisticRegression(random_state=random_state, **hyper_parameters)
    else:
        raise ValueError("Invalid algorithm")

    model.fit(X_train.to_numpy(), y_train.to_numpy())

    return model


def tune_model(
    alg: Literal["random_forest", "lightGBM", "logistic_regression_L1"],
    param_distributions: dict[str, Any],
    scoring: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
    n_jobs: int = -1,
) -> tuple[Any, dict[str, Any]]:
    """Tune the hyperparameters of a model using randomized search.

    Args:
        alg: algorithm to use for training.
            - "random_forest": Random Forest classifier.
            - "lightGBM": LightGBM classifier.
            - "logistic_regression_L1": Logistic Regression with L1 regularization.
        param_distributions: hyperparameter distributions for the specified algorithm.
            The dictionary should contain the hyperparameters as key-value pairs,
            where the values define the parameter distributions.
        scoring: scoring metric to optimize during tuning.
        X_train: training feature dataset.
        y_train: training target labels.
        random_state: random seed for reproducibility.
        n_jobs: number of jobs to run in parallel during tuning.
            -1 means using all processors.

    Returns:
        A tuple containing the best estimator and the best hyperparameters found during tuning.

    Raises:
        ValueError: if an invalid algorithm name is provided.
    """
    if alg == "random_forest":
        model = RandomForestClassifier(random_state=random_state)
    elif alg == "lightGBM":
        model = LGBMClassifier(random_state=random_state)
    elif alg == "logistic_regression_L1":
        model = LogisticRegression(random_state=random_state)
    else:
        raise ValueError("Invalid algorithm")

    search = RandomizedSearchCV(
        model,
        param_distributions,
        scoring=scoring,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    search.fit(X_train.to_numpy(), y_train.to_numpy())

    return search.best_estimator_, search.best_params_


def test_model(yhat: np.ndarray, y_test: pd.Series) -> tuple[float, float, np.ndarray]:
    """Evaluate a model's performance on the test set.

    Args:
        yhat: test target labels.
        y_test: test target labels.

    Returns:
        The Area Under the Receiver Operating Characteristic Curve (AUROC)
        score of the model on the test set.
    """
    auc = roc_auc_score(y_test.to_numpy(), yhat)
    f1 = f1_score(y_test.to_numpy(), yhat)
    cm = confusion_matrix(y_test.to_numpy(), yhat)
    
    return auc, f1, cm

def baseline1(y_train: pd.Series, X_test: pd.DataFrame) -> list:
    """Train a baseline model that always predicts the majority class (binary) in the training data.
    
    Args:
        y_train: training target labels. 
        X_test: test feature dataset.
        
    Returns:
    	A list of len(y_train) in which all elements are the majority class.
    """
    return [mode(y_train)]*len(X_test)

def baseline2(y_train: pd.Series, X_test: pd.DataFrame, random_state: int) -> list:
    """Train a baseline model that predicts binary classes based on their frequency in the training data.
    
    Args:
        y_train: training target labels. 
        X_test: test feature dataset.
        random_state:  seed for reproducibility.
        
    Returns:
    	A list of len(y_train) in which all elements are one of two classes, in proportion to their frequency within the training data.
    """
    a = [0]*Counter(y_train)[0]
    b = [1]*Counter(y_train)[1]
    random.seed(random_state)
    return random.sample(a+b, len(X_test))
    
def baseline345(
	alg: Literal["random_forest", "lightGBM", "logistic_regression_L1"],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
    param_distributions: dict[str, Any],
    X_test: pd.DataFrame
) -> Any:
    """Train a baseline random forest model using y_train values that have been randomly shuffled.
    
    Args:
        alg: algorithm to use for training.
            - "random_forest": Random Forest classifier.
            - "lightGBM": LightGBM classifier.
            - "logistic_regression_L1": Logistic Regression with L1 regularization.
    	X_train: training feature dataset.
        y_train: training target labels. 
        random_state:  seed for reproducibility.
        X_test: test feature dataset.
        
    Returns:
    	The trained model.
    """
    random.seed(random_state)
    shuffled_y_train = pd.Series(random.sample(list(y_train), y_train.shape[0]))
    model, params = tune_model(
        alg,
        param_distributions[alg],
        "roc_auc",
        X_train,
        shuffled_y_train,
        random_state
    )

    return model.predict(X_test.to_numpy())
    
def save_results(	
	results: defaultdict, 
	save_name: Literal['metadata_only', 'microbiome_only', 'metadata_microbiome', 
					   'baseline1', 'baseline2', 'baseline_random_forest', 
					   'baseline_lightGBM', 'baseline_logistic_regression_L1'], 
	tax_level: Literal['species', 'genus', 'family', 'all'], 
	alg: Literal['random_forest', 'lightGBM', 'logistic_regression_L1'], 
	train_index: list, 
	test_index: list, 
	random_state: int, 
	auc: list, 
	f1: list, 
	cm: list, 
	yhat: list, 
	model: list 
) -> defaultdict:
	"""
	Save results of k-fold CV experimental condition within the results defaultdict.
	
	Args:
		results: k-fold CV experimental results. For each replicate per data_type, taxonomic_level, and alg, you can access train_index, test_index, model_init_seed, AUROC, F1, CM, yhat, model.
		save_name: name of data type.
		tax_level: name of taxonomic level.
		alg: name of learning algorithm.
		train_index: list of array of samples / sample indices that were split into the training set.
		test_index: list of array of samples / sample indices that were split into the test set.
		random_state: list of seeds used for reproducibility. 
		auc: list of AUROC scores for experimental replicates.
		f1: list of F1 scores for experimental replicates.
		cm: list of array of confusion matrix scores for experimental replicates.
		yhat: list of array of predictions made by each replicate's trained model.
		model: list of arrays of models.
		
	Returns: 
		results: updated results defaultdict.
	"""
	results[save_name][tax_level][alg]["train_index"].append(train_index)
	results[save_name][tax_level][alg]["test_index"].append(test_index)
	results[save_name][tax_level][alg]["model_init_seed"].append(random_state)
	results[save_name][tax_level][alg]["AUROC"].append(auc)
	results[save_name][tax_level][alg]["F1"].append(f1)
	results[save_name][tax_level][alg]["CM"].append(cm)
	results[save_name][tax_level][alg]["yhat"].append(yhat)
	results[save_name][tax_level][alg]["model"].append(model)
	
	return results

def results_overview_table(results: defaultdict) -> (pd.DataFrame, pd.DataFrame):
	"""
	Generate pandas dataframes that summarize k-fold CV experiments; one for easy viewing by readers and one for easy calculation in downstream analysis. 
	
	Args:
		results: k-fold CV experimental results. For each replicate per data_type, taxonomic_level, and alg, you can access train_index, test_index, model_init_seed, AUROC, F1, CM, yhat, model.		
	Returns:
		results_overview: summary of results (e.g., median AUROC, MAD AUROC, etc) for each experimental condition (data_type, taxonomic_level, alg).
		results_auroc: data frame where, for each experimental condition (data_type, taxonomic_level, alg), there is a column with list of all AUROC scores obtained.
	"""
	results_overview = []
	results_auroc = []
	for data_type_key, data_type_value in results.items():
	    for tax_level_key, tax_level_value in data_type_value.items():
	        for alg_key, alg_value in tax_level_value.items():
	            out1 = [data_type_key, tax_level_key, alg_key]
	            for i in ["AUROC", "F1"]:
	                out1.append(np.median(results[data_type_key][tax_level_key][alg_key][i]))
	                out1.append(median_abs_deviation(results[data_type_key][tax_level_key][alg_key][i]))         
	            tn, fp, fn, tp = [], [], [], []
	            for i in range(len(results[data_type_key][tax_level_key][alg_key]["CM"])):
	                tn.append(results[data_type_key][tax_level_key][alg_key]["CM"][i][0][0])
	                fp.append(results[data_type_key][tax_level_key][alg_key]["CM"][i][1][0])
	                fn.append(results[data_type_key][tax_level_key][alg_key]["CM"][i][0][1])
	                tp.append(results[data_type_key][tax_level_key][alg_key]["CM"][i][1][1])
	            for i in [tn, fp, fn, tp]:
	                out1.append(np.median(i))
	                out1.append(median_abs_deviation(i))
	            results_overview.append(out1)
	
	            out2 = [data_type_key, tax_level_key, alg_key, results[data_type_key][tax_level_key][alg_key]["AUROC"]] 
	            results_auroc.append(out2)
	            
	results_overview = pd.DataFrame(results_overview)
	results_overview.columns =['Data type', 'Taxonomic level', 'ML algorithm', 'Median AUROC', "MAD AUROC", 'F1', 'MAD F1',
	                          "TN", "MAD TN", "FP", "MAD FP", "FN", "MAD FN", "TP", "MAD TP"]
	results_overview=results_overview.sort_values(by=['Data type', 'Taxonomic level', 'ML algorithm'], ascending=False)
	results_auroc = pd.DataFrame(results_auroc)
	results_auroc.columns =['Data type', 'Taxonomic level', 'ML algorithm', 'AUROC']
	results_auroc=results_auroc.sort_values(by=['Data type', 'Taxonomic level', 'ML algorithm'], ascending=False)
	
	return results_overview, results_auroc