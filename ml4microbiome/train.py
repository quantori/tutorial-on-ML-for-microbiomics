import pandas as pd
from sklearn.preprocessing import StandardScaler

def scale_features(X_train, X_test, cols_to_scale):
    """
    Standardize features by removing the mean and scaling to unit variance.
    
    Arguments:
        X_train (pd.DataFrame) -- numerical metadata features, training data
        X_test (pd.DataFrame) -- numerical metadata features, test data
        cols_to_scale (list) -- names of feature columns to scale
        
    Returns:
        X_train (pd.DataFrame) -- feature scaled numerical metadata, training data
        X_test (pd.DataFrame) -- feature scaled numerical metadata, test data
    """    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train[cols_to_scale]))
    X_train_scaled.columns = cols_to_scale
    X_train_scaled.index = X_train.index
    X_test_scaled = pd.DataFrame(scaler.transform(X_test[cols_to_scale]))
    X_test_scaled.columns = cols_to_scale
    X_test_scaled.index = X_test.index
    X_train = pd.merge(X_train_scaled, X_train.drop(columns=cols_to_scale), left_index=True, right_index=True)
    X_test = pd.merge(X_test_scaled, X_test.drop(columns=cols_to_scale), left_index=True, right_index=True)
                
    return X_train, X_test