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

# View mode
view_mode = st.sidebar.radio(
    "View Mode",
    ["Total Spending", "Per-Unit Cost", "Total Claims", "CAGR & Outliers", "High-Volume Drugs"]
)

# -------------------------------
# 4. Apply Filters
# -------------------------------
df_filtered = df_long.copy()

# Apply drug filter
if selected_drug != "All":
    df_filtered = df_filtered[df_filtered['brnd_name'] == selected_drug]

# Apply manufacturer filter
if selected_manufacturer != "All":
    df_filtered = df_filtered[df_filtered['mftr_name'] == selected_manufacturer]

# Always include all years (2019–2023) — we're removing year range filter
df_filtered = df_filtered[df_filtered['year'].between(2019, 2023)]

# Filter 2023 data for charts (used in Top Cost Drivers, CAGR, High-Volume tabs)
df_2023_filtered = df_filtered[df_filtered['year'] == 2023].copy()

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
        # Get historical data for the selected drug across all years
        drug_data = df_long[df_long['brnd_name'] == selected_drug].sort_values('year')
        
        # Get forecast data
        forecast_row = forecast_df[forecast_df['brnd_name'] == selected_drug]
        if forecast_row.empty:
            st.warning(f"⚠️ No 2024 forecast available for *{selected_drug}*")
        else:
            forecast_val = forecast_row['forecast_2024_total_spending'].iloc[0]

            col, ylabel, title = get_view_data(drug_data, view_mode)
            y_data = drug_data[col]

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(drug_data['year'], y_data, marker='o', label='Historical', color='blue', linewidth=2, markersize=6)
            ax.axhline(y=forecast_val, color='red', linestyle='--', linewidth=2, label='2024 Forecast')
            ax.set_xlabel("Year", fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(f"{selected_drug} {title}", fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', which='major', labelsize=10)
            st.pyplot(fig)

# Tab 2: Top Cost Drivers
with tab2:
    st.subheader("Top 10 Costliest Drugs (2023 vs 2024)")

    if df_2023_filtered.empty:
        st.warning("No data available for selected filters.")
    else:
        col, ylabel, _ = get_view_data(df_2023_filtered, view_mode)

        # Always define top_2023 even if no drug is selected
        top_2023 = df_2023_filtered.groupby('brnd_name')[col].sum().nlargest(10).reset_index()
        top_2023 = top_2023.rename(columns={col: 'spend_2023'})
        
        # Merge with forecast data
        top_2023 = top_2023.merge(forecast_df[['brnd_name', 'forecast_2024_total_spending']], on='brnd_name', how='left')
        top_2023['change_pct'] = ((top_2023['forecast_2024_total_spending'] - top_2023['spend_2023']) / top_2023['spend_2023']) * 100

        # Only show chart when a specific drug is selected
        if selected_drug != "All":
            # Filter data for charting
            filtered_for_chart = top_2023[top_2023['brnd_name'] == selected_drug]
            if not filtered_for_chart.empty:
                fig, ax = plt.subplots(figsize=(10, 6))
                x = np.arange(len(filtered_for_chart))
                width = 0.4
                
                bars1 = ax.bar(x - width/2, filtered_for_chart['spend_2023'], width, label='2023', color='skyblue', edgecolor='black', linewidth=0.5)
                bars2 = ax.bar(x + width/2, filtered_for_chart['forecast_2024_total_spending'], width, label='2024 Forecast', color='salmon', edgecolor='black', linewidth=0.5)
                    
                ax.set_xlabel("Drug", fontsize=12)
                ax.set_ylabel(ylabel, fontsize=12)
                ax.set_title(f"{selected_drug} in {view_mode} (2023 vs 2024)", fontsize=14, fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(filtered_for_chart['brnd_name'], rotation=45, ha='right', fontsize=10)
                ax.legend(fontsize=10)
                plt.tight_layout()
                st.pyplot(fig)

        # Always show full top 10 table
        st.dataframe(top_2023)

# Tab 3: CAGR & Outliers
with tab3:
    st.subheader("Fastest-Growing & Outlier Drugs")

    if df_2023_filtered.empty:
        st.warning("No data available for selected filters.")
    else:
        col, ylabel, title_suffix = get_view_data(df_2023_filtered, view_mode)

        # Show CAGR chart only in CAGR mode and when a drug is selected
        if view_mode == "CAGR & Outliers" and selected_drug != "All":
            cagr_drugs = df_2023_filtered.sort_values('cagr_avg_spnd_per_dsg_unt_19_23', ascending=False).head(10)
            filtered_cagr = cagr_drugs[cagr_drugs['brnd_name'] == selected_drug] if selected_drug != "All" else cagr_drugs
            
            if not filtered_cagr.empty:
                fig, ax = plt.subplots(figsize=(8, 5))
                bars = ax.barh(filtered_cagr['brnd_name'], filtered_cagr['cagr_avg_spnd_per_dsg_unt_19_23'], color='teal', edgecolor='black', linewidth=0.5)
                ax.set_xlabel("CAGR (%)", fontsize=12)
                ax.set_title(f"Top Drugs by CAGR (2019–2023)", fontsize=14, fontweight='bold')
                ax.invert_yaxis()
                ax.tick_params(axis='both', which='major', labelsize=10)
                
                # Add value labels
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    ax.text(bar.get_x() + width + 0.1, bar.get_y() + bar.get_height()/2, 
                           f'{width:.1f}%', va='center', ha='left', fontsize=9)
                plt.tight_layout()
                st.pyplot(fig)

        # Always show outlier table regardless of drug selection
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
        
        # Only show chart when a specific drug is selected
        if selected_drug != "All":
            drug_in_high_volume = high_volume[high_volume['brnd_name'] == selected_drug]
            if not drug_in_high_volume.empty:
                fig, ax = plt.subplots(figsize=(7, 5))
                y_pos = np.arange(len(drug_in_high_volume))
                bars = ax.barh(y_pos, drug_in_high_volume[col], color='darkgreen', edgecolor='black', linewidth=0.5)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(drug_in_high_volume['brnd_name'], fontsize=10)
                ax.set_xlabel(ylabel, fontsize=12)
                ax.set_title(f"Prescription Volume for {selected_drug}", fontsize=13, fontweight='bold')
                ax.tick_params(axis='x', labelsize=10)
                
                # Add value labels
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    ax.text(bar.get_x() + width + 0.1, bar.get_y() + bar.get_height()/2, 
                           f'{width:,.0f}', va='center', ha='left', fontsize=9)
                
                plt.tight_layout()
                st.pyplot(fig)

        # Always show full top 10 table
        st.dataframe(high_volume)

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

# Prepare filtered data for download
export_df = forecast_df.copy()

if selected_drug != "All":
    export_df = export_df[export_df['brnd_name'] == selected_drug]

# Convert to CSV
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df_to_csv(export_df)

# Use download_button directly
st.sidebar.download_button(
    label="Download Forecast CSV",
    data=csv,
    file_name=f"medicare_forecast_2024_{'all_drugs' if selected_drug == 'All' else selected_drug.lower().replace(' ', '_')}.csv",
    mime="text/csv"
) 