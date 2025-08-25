# src/model.py
import joblib
import pandas as pd
import numpy as np

class DrugSpendingPredictor:
    def __init__(self, model, feature_cols, cat_cols, target_log_col):
        self.model = model
        self.feature_cols = feature_cols
        self.cat_cols = cat_cols
        self.target_log_col = target_log_col

    def predict(self, df):
        """Predict total spending on new data."""
        df = df.sort_values(['brnd_name', 'year']).copy()
        target_col = "tot_spndng"
        target_log_col = f"{target_col}_log"

        # Add log-transformed target
        df[target_log_col] = np.log1p(df[target_col])

        # Lag features
        df['spend_lag1'] = df.groupby('brnd_name')[target_log_col].shift(1)
        df['spend_lag2'] = df.groupby('brnd_name')[target_log_col].shift(2)

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
            log_col = f"{col}_log"
            if log_col not in df.columns:
                df[log_col] = np.log1p(df[col])

        # Prepare feature matrix
        X = df[self.feature_cols].copy()
        for col in self.cat_cols:
            X[col] = X[col].astype('category')

        # Predict in log space and reverse transform
        log_pred = self.model.predict(X)
        return np.expm1(log_pred) 