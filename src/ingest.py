
def load_data(file_path):
    """Load raw CMS data."""
    import pandas as pd
    df = pd.read_csv(file_path)
    print(f"✅ Data loaded: {df.shape}")
    return df
