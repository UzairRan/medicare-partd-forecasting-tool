# src/validate.py
"""
Input validation for drug spending forecasting pipeline.
"""

import pandas as pd


def validate_input_data(df):
    """
    Validate input DataFrame for prediction.
    """
    required_columns = [
        'brnd_name', 'year', 'tot_spndng', 'tot_dsg_unts',
        'tot_clms', 'tot_benes', 'avg_spnd_per_clm', 'avg_spnd_per_bene'
    ]

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if df.empty:
        raise ValueError("Input DataFrame is empty")

    if df['brnd_name'].isnull().any():
        raise ValueError("brnd_name contains NaN values")

    if df['year'].isnull().any():
        raise ValueError("year column contains NaN values")

    # Check for negative values where not allowed
    numeric_positive = ['tot_spndng', 'tot_dsg_unts', 'tot_clms', 'tot_benes']
    for col in numeric_positive:
        if (df[col] < 0).any():
            raise ValueError(f"Column '{col}' contains negative values")

    print("✅ Input validation passed.")
    return True  