"""Module containing utility functions for model training and testing."""
from typing import Any, Literal

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV


def train_model(
    alg: Literal["random_forest", "lightGBM", "logistic_regression_L1"],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
    hyper_parameters: dict[str, Any],
) -> Any:
    """Train a model using the specified algorithm and hyperparameters.

    Args:
        alg: algorithm to use for training.
            - "random_forest": Random Forest classifier
            - "lightGBM": LightGBM classifier
            - "logistic_regression_L1": Logistic Regression with L1 regularization
        X_train: training feature dataset.
        y_train: training target labels.
        random_state: random seed for reproducibility.
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
            - "random_forest": Random Forest classifier
            - "lightGBM": LightGBM classifier
            - "logistic_regression_L1": Logistic Regression with L1 regularization
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


def test_model(model: Any, yhat: np.ndarray, y_test: pd.Series) -> tuple[float, float, np.ndarray]:
    """Evaluate a model's performance on the test set.

    Args:
        model: trained/tuned model.
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

