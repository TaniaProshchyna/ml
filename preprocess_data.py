import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class IQR_Capper(object):
    """
    A transformer to cap outliers using the IQR method (factor * IQR).
    """
    def __init__(self, factor=1.5):
        self.factor = factor
        self.lower_bounds = {}
        self.upper_bounds = {}

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        for col in X.columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            self.lower_bounds[col] = Q1 - (self.factor * IQR)
            self.upper_bounds[col] = Q3 + (self.factor * IQR)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_transformed = X.copy()
        for col in X.columns:
            if col in self.lower_bounds and col in self.upper_bounds:
                X_transformed[col] = np.clip(X_transformed[col], self.lower_bounds[col], self.upper_bounds[col])
        return X_transformed

def preprocess_bank_data(train_df, val_df):
    """
    Preprocesses the bank marketing dataset for machine learning.

    Args:
        train_df (pd.DataFrame): Training DataFrame.
        val_df (pd.DataFrame): Validation DataFrame.

    Returns:
        tuple: A tuple containing:
            - train_inputs (pd.DataFrame): Processed training features.
            - train_targets (pd.Series): Training target variable.
            - val_inputs (pd.DataFrame): Processed validation features.
            - val_targets (pd.Series): Validation target variable.
            - preprocessor (ColumnTransformer): Fitted preprocessor pipeline.
    """
    train_df_processed = train_df.copy()
    val_df_processed = val_df.copy()

    # --- Feature Engineering for 'pdays' ---
    # 1. Create binary variable 'had_previous_contact'
    train_df_processed['had_previous_contact'] = (train_df_processed['pdays'] != 999).astype(int)
    val_df_processed['had_previous_contact'] = (val_df_processed['pdays'] != 999).astype(int)

    # 2. Process 999 in 'pdays': change to median of no-999 values
    # Calculate median only using train dataset to avoid data leakage
    pdays_median_for_imputation = train_df_processed[train_df_processed['pdays'] != 999]['pdays'].median()
    train_df_processed['pdays'] = train_df_processed['pdays'].replace({999: pdays_median_for_imputation})
    val_df_processed['pdays'] = val_df_processed['pdays'].replace({999: pdays_median_for_imputation})
    # ---------------------------------------

    # Delete 'duration' to avoid data leakage
    input_cols = list(train_df_processed.drop(columns=['y', 'duration']).columns)
    target_col = 'y'

    train_inputs = train_df_processed[input_cols].copy()
    train_targets = train_df_processed[target_col].copy()
    val_inputs = val_df_processed[input_cols].copy()
    val_targets = val_df_processed[target_col].copy()

    # Define columns for encoding
    numeric_cols = train_inputs.select_dtypes('number').columns.tolist()
    ordinal_cols = ['education']
    onehot_cols = [col for col in train_inputs.select_dtypes('object').columns.tolist() if col not in ordinal_cols]

    # Define order for 'education'
    education_order = [
        'basic.4y', 'basic.6y', 'basic.9y', 'high.school',
        'professional.course', 'university.degree', 'unknown'
    ]

    # Create transformers
    numeric_transformer = Pipeline(steps=[
         ('capper', IQR_Capper(factor=1.5)), # Winsorization Added
         ('poly', PolynomialFeatures(degree=2, include_bias=False)),
         ('scaler', StandardScaler())
    ])

    ordinal_transformer = Pipeline(steps=[
        ('ordinal', OrdinalEncoder(categories=[education_order], handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    # Combine transformers ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('ord', ordinal_transformer, ordinal_cols),
            ('cat', categorical_transformer, onehot_cols)
        ],
        remainder='passthrough'
    )

    # Fit the preprocessor on training data
    preprocessor.fit(train_inputs)

    return train_inputs, train_targets, val_inputs, val_targets, preprocessor