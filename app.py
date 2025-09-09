import sys
import os

# Fix Python path to find src modules
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

# Import the model class before loading .pkl
from src.model import DrugSpendingPredictor
import joblib

# Import other libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1. Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Medicare Part D Drug Spending Forecast",
    layout="wide"
)

st.title("Medicare Part D Drug Spending Forecast")
st.markdown("AI-powered forecasting system for 2024 drug spending trends")

# -------------------------------
# 2. Load Data & Model
# -------------------------------
@st.cache_data
def load_data():
    df_long = pd.read_csv("data/processed/df_long.csv")
    forecast = pd.read_csv("data/processed/full_drug_forecasts_2024.csv")  # Full list
    return df_long, forecast

@st.cache_data
def load_model():
    model_path = os.path.join("models", "drug_spending_predictor.pkl")
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}")
        st.stop()
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Load data and model
df_long, forecast_df = load_data()
predictor = load_model()

# -------------------------------
# 3. Sidebar Filters
# -------------------------------
st.sidebar.header("Filter Options")

# Drug selection
drug_list = sorted(df_long['brnd_name'].dropna().unique())
selected_drug = st.sidebar.selectbox("Select Drug", ["All"] + drug_list)

# Manufacturer filter
manufacturer_list = sorted(df_long['mftr_name'].dropna().unique())
selected_manufacturer = st.sidebar.selectbox("Filter by Manufacturer", ["All"] + manufacturer_list)

# Year range
year_range = st.sidebar.slider("Year Range", 2019, 2023, (2019, 2023))

# View mode
view_mode = st.sidebar.radio(
    "View Mode",
    ["Total Spending", "Per-Unit Cost", "Total Claims", "CAGR & Outliers", "High-Volume Drugs"]
)

# -------------------------------
# 4. Apply Filters
# -------------------------------
df_filtered = df_long.copy()

if selected_drug != "All":
    df_filtered = df_filtered[df_filtered['brnd_name'] == selected_drug]

if selected_manufacturer != "All":
    df_filtered = df_filtered[df_filtered['mftr_name'] == selected_manufacturer]

df_filtered = df_filtered[df_filtered['year'].between(year_range[0], year_range[1])]

# Filter 2023 data for charts
df_2023_filtered = df_filtered[df_filtered['year'] == 2023]

# -------------------------------
# 5. Dashboard Tabs
# -------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Forecast Explorer",
    "Top Cost Drivers",
    "CAGR & Outliers",
    "High-Volume Drugs",
    "Explainability"
])

# -------------------------------
# Helper Function: Map view_mode to column and labels
# -------------------------------
def get_view_data(df, view_mode):
    """Return the appropriate column and label for given view mode."""
    if view_mode == "Total Spending":
        col = 'tot_spndng'
        ylabel = "Total Spending ($)"
        title_suffix = "Spending Trend"
    elif view_mode == "Per-Unit Cost":
        col = 'avg_spnd_per_dsg_unt_wghtd'
        ylabel = "Avg Spend per Unit ($)"
        title_suffix = "Per-Unit Cost Trend"
    elif view_mode == "Total Claims":
        col = 'tot_clms'
        ylabel = "Total Claims"
        title_suffix = "Claim Volume Trend"
    elif view_mode == "CAGR & Outliers":
        col = 'cagr_avg_spnd_per_dsg_unt_19_23'
        ylabel = "CAGR (%)"
        title_suffix = "CAGR Trend (2019–2023)"
    elif view_mode == "High-Volume Drugs":
        col = 'tot_clms'
        ylabel = "Total Claims"
        title_suffix = "Claim Volume Trend"
    else:
        col = 'tot_spndng'
        ylabel = "Total Spending ($)"
        title_suffix = "Spending Trend"
    return col, ylabel, title_suffix

# Tab 1: Forecast Explorer
with tab1:
    st.subheader("Drug Spending Forecast Explorer")

    if selected_drug == "All":
        st.info("Select a drug to view its forecast.")
    else:
        drug_data = df_long[df_long['brnd_name'] == selected_drug].sort_values('year')
        forecast_row = forecast_df[forecast_df['brnd_name'] == selected_drug]
        if forecast_row.empty:
            st.warning(f"⚠️ No 2024 forecast available for *{selected_drug}*")
        else:
            forecast_val = forecast_row['forecast_2024_total_spending'].iloc[0]

            col, ylabel, title = get_view_data(drug_data, view_mode)
            y_data = drug_data[col]

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(drug_data['year'], y_data, marker='o', label='Historical', color='blue')
            ax.axhline(y=forecast_val, color='red', linestyle='--', label='2024 Forecast')
            ax.set_xlabel("Year")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{selected_drug} {title}")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

# Tab 2: Top Cost Drivers
with tab2:
    st.subheader("Top 10 Costliest Drugs (2023 vs 2024)")

    if df_2023_filtered.empty:
        st.warning("No data available for selected filters.")
    else:
        col, ylabel, _ = get_view_data(df_2023_filtered, view_mode)

        # Filter by drug if selected
        if selected_drug != "All":
            df_2023_filtered = df_2023_filtered[df_2023_filtered['brnd_name'] == selected_drug]

        top_2023 = df_2023_filtered.groupby('brnd_name')[col].sum().nlargest(10).reset_index()
        top_2023 = top_2023.rename(columns={col: 'spend_2023'})
        top_2023 = top_2023.merge(forecast_df[['brnd_name', 'forecast_2024_total_spending']], on='brnd_name', how='left')
        top_2023['change_pct'] = ((top_2023['forecast_2024_total_spending'] - top_2023['spend_2023']) / top_2023['spend_2023']) * 100

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(top_2023))
        width = 0.4
        ax.bar(x - width/2, top_2023['spend_2023'], width, label='2023', color='skyblue')
        ax.bar(x + width/2, top_2023['forecast_2024_total_spending'], width, label='2024 Forecast', color='salmon')
        ax.set_xlabel("Drug")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Top 10 {view_mode} (2023 vs 2024)")
        ax.set_xticks(x)
        ax.set_xticklabels(top_2023['brnd_name'], rotation=45, ha='right')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)

        st.dataframe(top_2023)

# Tab 3: CAGR & Outliers
with tab3:
    st.subheader("Fastest-Growing & Outlier Drugs")

    if df_2023_filtered.empty:
        st.warning("No data available for selected filters.")
    else:
        col, ylabel, title_suffix = get_view_data(df_2023_filtered, view_mode)

        # Always show CAGR chart when mode is CAGR
        if view_mode == "CAGR & Outliers":
            cagr_drugs = df_2023_filtered.sort_values('cagr_avg_spnd_per_dsg_unt_19_23', ascending=False).head(10)
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(cagr_drugs['brnd_name'], cagr_drugs['cagr_avg_spnd_per_dsg_unt_19_23'], color='teal')
            ax.set_xlabel("CAGR (%)")
            ax.set_title("Top 10 Drugs by CAGR (2019–2023)")
            ax.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig)
        else:
            # Show other metrics
            pass

        # Always show outlier table regardless of view_mode
        outliers = df_2023_filtered[df_2023_filtered['chg_avg_spnd_per_dsg_unt_22_23'] > 50]
        if not outliers.empty:
            st.markdown("### 🔴 Drugs with >50% YoY Price Spike (2022 → 2023)")
            st.dataframe(outliers[['brnd_name', 'year', 'avg_spnd_per_dsg_unt_wghtd', 'chg_avg_spnd_per_dsg_unt_22_23']])

# Tab 4: High-Volume Drugs
with tab4:
    st.subheader("Most Prescribed Drugs (2023)")

    if df_2023_filtered.empty:
        st.warning("No data available for selected filters.")
    else:
        col, ylabel, _ = get_view_data(df_2023_filtered, view_mode)
        
        high_volume = df_2023_filtered.groupby('brnd_name')[col].sum().nlargest(10).reset_index()
        if high_volume.empty:
            st.warning("No high-volume drugs found.")
        else:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(data=high_volume, y='brnd_name', x=col, ax=ax, palette='viridis')
            ax.set_xlabel(ylabel)
            ax.set_title(f"Top 10 Most {view_mode.replace(' ', '')} Drugs in 2023")
            plt.tight_layout()
            st.pyplot(fig)

# Tab 5: Model Explainability
with tab5:
    st.subheader("Model Explainability")

    if selected_drug == "All":
        st.info("Select a drug to view explanation.")
    else:
        st.markdown(f"**Insight for {selected_drug}:**")
        st.markdown("""
        - **High forecast?** Likely due to rising per-unit cost or increasing number of claims.
        - **Low forecast?** Could be due to declining usage or price stabilization.
        - **Key drivers:** Lagged spending, claim trends, and historical growth.
        """)
        st.markdown("💡 This insight is based on model behavior — SHAP integration coming soon.")

# -------------------------------
# 6. Export Data
# -------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("Export Data")

# Use st.download_button for actual download
if st.sidebar.button("Download Forecast CSV"):
    # Save to temp file
    forecast_df.to_csv("full_drugs_forecast_2024_export.csv", index=False)
    # Force download
    with open("full_drugs_forecast_2024_export.csv", "r") as f:
        st.sidebar.download_button(
            label="Click to Download",
            data=f,
            file_name="medicare_drug_forecasts_2024.csv",
            mime="text/csv"
        )  