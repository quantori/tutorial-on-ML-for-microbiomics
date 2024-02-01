"""This script performs a repeated k-fold cross-validation. It trains and evaluates
models and collects the results.

The script follows the following steps:
1. Imports necessary libraries and modules.
2. Defines the taxonomic levels, data types, columns containing continuous data,
learning algorithms, and other parameters.
3. Loads the features and target labels from pickle files.
4. Performs the experiment for each combination of data type, taxonomic level,
and algorithm:
    a. Splits the data into folds sets.
    b. Performs median imputation and feature scaling on the data.
    d. Performs training and hyperparameter tuning.
    e. Evaluates the model's performance and records the results.
5. Saves the results to a pickle file.
"""

