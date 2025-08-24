
def clean_data(df):
    """Clean and standardize column names and values."""
    df.columns = (df.columns
                  .str.strip()
                  .str.lower()
                  .str.replace(' ', '_', regex=False)
                  .str.replace('-', '_', regex=False))
    df['mftr_name'] = (df['mftr_name']
                       .str.strip()
                       .str.lower()
                       .str.replace(r'\s+', ' ', regex=True))
    return df

def reshape_to_long(df):
    """Reshape from wide to long format."""
    import pandas as pd
    id_vars = ['brnd_name', 'gnrc_name', 'tot_mftr', 'mftr_name']
    value_vars = [col for col in df.columns if any(str(yr) in col for yr in range(2019, 2024))]

    df_long = pd.wide_to_long(
        df,
        stubnames=[
            'tot_spndng', 'tot_dsg_unts', 'tot_clms', 'tot_benes',
            'avg_spnd_per_dsg_unt_wghtd', 'avg_spnd_per_clm', 'avg_spnd_per_bene', 'outlier_flag'
        ],
        i=id_vars,
        j='year',
        sep='_',
        suffix=r'\d+'
    ).reset_index()
    return df_long
