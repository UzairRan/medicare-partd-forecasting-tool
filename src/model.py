
def train_and_save_model(df, feature_columns, categorical_cols, target_col_log):
    """Train LightGBM model and save with metadata."""
    from sklearn.model_selection import train_test_split
    import lightgbm as lgb
    import joblib

    train = df[df['year'] < 2023]
    val = df[df['year'] == 2023]

    X_train = train[feature_columns + categorical_cols]
    y_train = train[target_col_log]
    X_val = val[feature_columns + categorical_cols]
    y_val = val[target_col_log]

    for col in categorical_cols:
        X_train[col] = X_train[col].astype('category')
        X_val[col] = X_val[col].astype('category')

    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=categorical_cols)
    val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=categorical_cols)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'verbose': -1
    }

    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(10)]
    )

    joblib.dump(model, '../models/final_model.pkl')
    print("✅ Model trained and saved.")
    return model
