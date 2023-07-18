"""This script performs a nested cross-validation. It trains and evaluates
models and collects the results.

The script follows the following steps:
1. Imports necessary libraries and modules.
2. Defines the taxonomic levels, data types, columns containing continuous data,
learning algorithms, and other parameters.
3. Loads the features and target labels from pickle files.
4. Performs the experiment for each combination of data type, taxonomic level,
and algorithm:
    a. Splits the data into training and testing sets.
    b. Performs median imputation and feature scaling on the data.
    c. Splits the training set into training and validation sets to find the best
    hyperparameters.
    d. Trains a model using the best hyperparameters found in the inner loop.
    e. Evaluates the model's performance on the outer test set and records the results.
5. Saves the inner and outer results to pickle files.
"""
import datetime
from collections import defaultdict

import dill
import numpy as np
from sklearn.model_selection import train_test_split

from ml4microbiome import train

tax_levels = ["all", "species", "genus", "family"]
data_types = ["metadata_only", "microbiome_only", "metadata_microbiome"]
metadata_continuous_cols = [
    "Age",
    "Height(cm)",
    "Weight(kg)",
    "Pulse(c. p. m)",
    "Breathe(c. p. m)",
    "Systolic pressure(mmHg)",
    "Diastolic pressure(mmHg)",
    "Tryptophane(μM)",
    "Glutamic acid(μM)",
    "Tyrosine(μM)",
    "Phenylalanine(μM)",
    "Dopamine(ng/ml)",
    "Gamma-aminobutyric acid  (GABA)(ng/L)",
    "Serotonin(ng/ml)",
    "Kynurenine (KYN)(nmol/L)",
    "Kynurenic acid (KYNA)(nmol/L)",
]
learning_algs = ["random_forest", "lightGBM", "logistic_regression_L1"]
reps = 100
test_size = 0.2
params_distributions = {
    "random_forest": {
        "n_estimators": np.arange(50, 550, 50),
        "max_features": ["log2", "sqrt", None],
        "min_samples_split": np.arange(2, 10, 2),
        "min_samples_leaf": np.arange(2, 12, 2),
        "bootstrap": [True, False],
        "max_depth": np.arange(5, 11),
    },
    "lightGBM": {
        "learning_rate": np.linspace(0.01, 0.5, 50),
        "objective": ["binary"],
        "boosting_type": ["gbdt"],
        "num_leaves": np.arange(25, 200, 25),
        "max_depth": np.arange(1, 10),
        "subsample": np.arange(0.5, 1.0, 0.1),
        "subsample_freq": np.arange(2, 10, 2),
        "colsample_bytree": np.arange(0.5, 1.0, 0.1),
        "min_child_weight": np.arange(25, 100, 25),
        "reg_alpha": np.arange(25, 100, 25),
        "max_bin": np.arange(155, 355, 50),
        "min_child_samples": np.arange(10, 50, 10),
    },
    "logistic_regression_L1": {
        "max_iter": [10000],
        "penalty": ["l1"],
        "C": np.linspace(0.1, 100, 50),
        "solver": ["liblinear", "saga"],
    },
}

outer_results = defaultdict(
    lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
)
inner_results = defaultdict(
    lambda: defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )
)

with open("./data_dict.pickle", "rb") as data_dict_file:
    data_dict = dill.load(data_dict_file)

with open("./y_encoded.pickle", "rb") as y_encoded_file:
    y_encoded = dill.load(y_encoded_file)

for data_type in data_types:
    for tax_level in tax_levels:
        for alg in learning_algs:
            # For metadata_only, there are no taxonomic levels so only train once
            if data_type == "metadata_only" and tax_level != "all":
                continue

            print(f"Data type: {data_type}")
            print(f"Taxonomic level: {tax_level}")
            print(f"Algorithm: {alg}")

            for outer_iter_no in range(reps):
                start_time = datetime.datetime.now()
                # Define outer loop training and test sets
                (
                    outer_X_train,
                    outer_X_test,
                    outer_y_train,
                    outer_y_test,
                ) = train_test_split(
                    data_dict[data_type][tax_level],
                    y_encoded,
                    test_size=test_size,
                    stratify=y_encoded,
                    random_state=outer_iter_no,
                )

                # Median imputation based on outer cv set
                # To avoid data leakage:
                #     Calculate median values on the TRAINING set
                #     Use TRAINING set medians to impute TRAINING and TEST set missing values
                outer_X_train = outer_X_train.fillna(outer_X_train.median())
                outer_X_test = outer_X_test.fillna(outer_X_train.median())

                # Feature scaling based on outer cv set
                # (Probably not actually that helpful here, but good for demonstration purposes)
                # To avoid data leakage:
                #    Calculate scaling stats on the TRAINING set alone
                #    Use TRAINING set stats to scale TRAINING and TEST set values
                if data_type != "microbiome_only":
                    outer_X_train, outer_X_test = train.scale_features(
                        outer_X_train, outer_X_test, metadata_continuous_cols
                    )

                # To do: add feature selection step

                # Store AUROCs for inner test set
                # And the best hyper-parameters
                for inner_iter_no in range(reps):
                    # Define inner loop training and test sets
                    (
                        inner_X_train,
                        inner_X_test,
                        inner_y_train,
                        inner_y_test,
                    ) = train_test_split(
                        outer_X_train,
                        outer_y_train,
                        test_size=test_size,
                        stratify=outer_y_train,
                        random_state=inner_iter_no,
                    )

                    best_model, best_params = train.tune_model(
                        alg,
                        params_distributions[alg],
                        "roc_auc",
                        inner_X_train,
                        inner_y_train,
                        inner_iter_no,
                    )
                    auc = train.test_model(best_model, inner_X_test, inner_y_test)
                    inner_results[data_type][tax_level][alg][outer_iter_no][
                        "AUROC"
                    ].append(auc)
                    inner_results[data_type][tax_level][alg][outer_iter_no][
                        "best_params"
                    ].append(best_params)

                # Back to outer training-test split
                # For a given data_type, tax_level, and alg, identify the best model stored in inner_results
                aurocs_np = np.array(
                    inner_results[data_type][tax_level][alg][outer_iter_no]["AUROC"]
                )
                median_auroc = np.median(aurocs_np)
                max_auroc_index = aurocs_np.argmax()
                params = inner_results[data_type][tax_level][alg][outer_iter_no][
                    "best_params"
                ][max_auroc_index]

                outer_results[data_type][tax_level][alg][
                    "inner_median_valid_AUROC"
                ].append(median_auroc)
                outer_results[data_type][tax_level][alg]["params"].append(params)

                # Train a model with the best parameters
                model = train.train_model(
                    alg, outer_X_train, outer_y_train, outer_iter_no, params
                )

                # Evaluate and record its performance on the outer test set in outer_results
                auc = train.test_model(model, outer_X_test, outer_y_test)
                outer_results[data_type][tax_level][alg]["AUROC"].append(auc)

                end_time = datetime.datetime.now()
                outer_results[data_type][tax_level][alg]["elapsed_time"].append(
                    end_time - start_time
                )

with open("./inner_results.pickle", "wb") as inner_results_file:
    dill.dump(inner_results, inner_results_file)

with open("./outer_results.pickle", "wb") as outer_results_file:
    dill.dump(outer_results, outer_results_file)
