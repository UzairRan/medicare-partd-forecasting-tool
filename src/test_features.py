# src/test_features.py
"""
Unit and integration tests for feature engineering.
"""

import unittest
import pandas as pd 
import numpy as np
from src.features import create_features  # Assumes you have this function
from src.validate import validate_input_data

class TestFeatureEngineering(unittest.TestCase):

    def setUp(self):
        # Create sample data
        self.df = pd.DataFrame({
            'brnd_name': ['DrugA', 'DrugA', 'DrugB', 'DrugB'],
            'gnrc_name': ['A', 'A', 'B', 'B'],
            'mftr_name': ['M1', 'M1', 'M2', 'M2'],
            'tot_mftr': [1, 1, 2, 2],
            'year': [2021, 2022, 2021, 2022],
            'tot_spndng': [1000, 1200, 500, 600],
            'tot_dsg_unts': [100, 110, 50, 55],
            'tot_clms': [90, 95, 45, 48],
            'tot_benes': [80, 85, 40, 42],
            'avg_spnd_per_clm': [11.1, 12.6, 11.1, 12.5],
            'avg_spnd_per_bene': [12.5, 14.1, 12.5, 14.3],
            'outlier_flag': [0, 0, 0, 1]
        })

    def test_create_features_runs(self):
        """Test that feature engineering function runs without error."""
        df, feature_cols, cat_cols, target_log = create_features(self.df)
        self.assertIn('spend_lag1', df.columns)
        self.assertIn('spend_roll_mean', df.columns)
        self.assertGreater(len(df), 0)

    def test_lag_feature(self):
        """Test that lag feature is correctly computed."""
        df, _, _, _ = create_features(self.df)
        drug_a = df[df['brnd_name'] == 'DrugA'].sort_values('year')
        self.assertFalse(np.isnan(drug_a.iloc[1]['spend_lag1']))
        self.assertTrue(np.isnan(drug_a.iloc[0]['spend_lag1']))

    def test_validation_passes(self):
        """Test that valid input passes validation."""
        result = validate_input_data(self.df)
        self.assertTrue(result)

    def test_validation_fails_on_missing_col(self):
        """Test validation fails when required column is missing."""
        df_bad = self.df.drop(columns=['tot_spndng'])
        with self.assertRaises(ValueError):
            validate_input_data(df_bad)

if __name__ == '__main__':
    unittest.main()

