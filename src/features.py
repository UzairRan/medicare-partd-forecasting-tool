
def create_features(df):
    """Add lag, rolling, and log features."""
    import numpy as np
    df = df.sort_values(['brnd_name', 'year']).copy()
    df['tot_spndng_log'] = np.log1p(df['tot_spndng'])

    df['spend_lag1'] = df.groupby('brnd_name')['tot_spndng_log'].shift(1)
    df['spend_lag2'] = df.groupby('brnd_name')['tot_spndng_log'].shift(2)
    df['spend_roll_mean'] = df.groupby('brnd_name')['spend_lag1'].transform(
        lambda x: x.rolling(2, min_periods=1).mean()
    )
    df['yoy_change'] = df['spend_lag1'] - df['spend_lag2']

    dynamic_features = [
        'tot_dsg_unts', 'tot_clms', 'tot_benes',
        'avg_spnd_per_clm', 'avg_spnd_per_bene', 'outlier_flag'
    ]
    for col in dynamic_features:
        df[f"{col}_log"] = np.log1p(df[col])

    return df
