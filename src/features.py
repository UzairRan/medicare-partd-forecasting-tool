
"""
Feature engineering for drug spending forecasting.
"""
import numpy as np

def create_features(df):
    """Add lag, rolling, and log features."""
    df = df.sort_values(['brnd_name', 'year']).copy()
    
    # Target and log transform
    target_col = 'tot_spndng'
    target_col_log = f'{target_col}_log'
    df[target_col_log] = np.log1p(df[target_col])

    # Lag features
    df['spend_lag1'] = df.groupby('brnd_name')[target_col_log].shift(1)
    df['spend_lag2'] = df.groupby('brnd_name')[target_col_log].shift(2)

    # Rolling mean
    df['spend_roll_mean'] = df.groupby('brnd_name')['spend_lag1'].transform(
        lambda x: x.rolling(2, min_periods=1).mean()
    )

    # YoY change
    df['yoy_change'] = df['spend_lag1'] - df['spend_lag2']

    # Log-transform dynamic features
    dynamic_features = [
        'tot_dsg_unts', 'tot_clms', 'tot_benes',
        'avg_spnd_per_clm', 'avg_spnd_per_bene', 'outlier_flag'
    ]
    for col in dynamic_features:
        df[f"{col}_log"] = np.log1p(df[col])

    # Define feature columns
    feature_columns = [
        'spend_lag1', 'spend_lag2', 'spend_roll_mean', 'yoy_change', 'year', 'tot_mftr'
    ] + [f"{col}_log" for col in dynamic_features] + [
        'brnd_name', 'gnrc_name', 'mftr_name'
    ]

    categorical_cols = ['brnd_name', 'gnrc_name', 'mftr_name']

    # Return exactly what the tests expect
    return df, feature_columns, categorical_cols, target_col_log
