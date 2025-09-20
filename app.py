import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Fix Python path to find src modules
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

# Import the model class before loading .pkl
try:
    from src.model import DrugSpendingPredictor
except ImportError:
    st.error("Error: The 'src/model.py' file or DrugSpendingPredictor class was not found. Please ensure your project structure is correct.")
    st.stop()

# -------------------------------
# 1. Page Configuration & Translations
# -------------------------------
st.set_page_config(
    page_title="Medicare Part D Drug Spending Forecast",
    layout="wide"
)

# Translations dictionary
translations = {
    "en": {
        "title": "Medicare Part D Drug Spending Forecast",
        "intro": "AI-powered forecasting system for 2024 drug spending trends",
        "language_select": "Select Language",
        "sidebar_header": "Filter Options",
        "drug_select": "Select Drug(s)",
        "mftr_filter": "Filter by Manufacturer",
        "view_mode": "View Mode",
        "total_spending": "Total Spending",
        "per_unit_cost": "Per-Unit Cost",
        "total_claims": "Total Claims",
        "total_beneficiaries": "Total Beneficiaries",
        "forecast_explorer_tab": "Forecast Explorer",
        "top_cost_drivers_tab": "Top Cost Drivers",
        "cagr_tab": "CAGR & Outliers",
        "high_volume_tab": "High-Volume Drugs",
        "explainability_tab": "Explainability",
        "no_data_warning": "No data available for the selected drugs and manufacturer combination.",
        "select_drug_info": "Select one or more drugs to view their forecast.",
        "forecast_start_vline": "Forecast Start",
        "drug_spending_forecast_explorer": "Drug Spending Forecast Explorer",
        "top_ten_title": "Top 10 ",
        "top_ten_title_suffix_spending": " (2023 vs 2024)",
        "top_ten_title_other": " (2023)",
        "cagr_title": "Fastest-Growing & Outlier Drugs",
        "cagr_subtitle": "Top 10 Fastest-Growing Drugs by CAGR (2019-2023)",
        "compare_title": "Comparison of Selected Drugs by ",
        "explain_title": "Model Explainability",
        "select_drug_explain": "Select a drug to view explanation.",
        "insight_title": "Insight for ",
        "high_forecast_bullet": "High forecast? Likely due to rising per-unit cost or increasing number of claims.",
        "low_forecast_bullet": "Low forecast? Could be due to declining usage or price stabilization.",
        "key_drivers_bullet": "Key drivers: Lagged spending, claim trends, and historical growth.",
        "note_explain": "💡 Note: These insights are based on a predefined set of rules and do not dynamically reflect individual drug characteristics or model changes.",
        "export_data_header": "Export Data",
        "download_button": "Download Filtered Forecast CSV",
        "drug_column": "Drug",
        "manufacturer_column": "Manufacturer",
        "value_2023": "2023 Value",
        "forecast_2024": "2024 Forecast",
        "cagr_percent": "CAGR (%)",
        "drug_name": "Drug Name",
        "year": "Year",
        "total_spending_label": "Total Spending ($)",
        "avg_spend_per_unit_label": "Avg Spend per Unit ($)",
        "total_claims_label": "Total Claims",
        "total_beneficiaries_label": "Total Beneficiaries",
        "spending_trend": "Spending Trend",
        "per_unit_cost_trend": "Per-Unit Cost Trend",
        "claim_volume_trend": "Claim Volume Trend",
        "beneficiaries_trend": "Beneficiaries Trend",
    },
    "ar": {
        "title": "توقعات الإنفاق على الأدوية في Medicare Part D",
        "intro": "نظام تنبؤات مدعوم بالذكاء الاصطناعي لاتجاهات الإنفاق على الأدوية لعام 2024",
        "language_select": "اختر اللغة",
        "sidebar_header": "خيارات التصفية",
        "drug_select": "اختر دواء (أدوية)",
        "mftr_filter": "التصفية حسب الشركة المصنعة",
        "view_mode": "وضع العرض",
        "total_spending": "إجمالي الإنفاق",
        "per_unit_cost": "التكلفة لكل وحدة",
        "total_claims": "إجمالي المطالبات",
        "total_beneficiaries": "إجمالي المستفيدين",
        "forecast_explorer_tab": "مستكشف التوقعات",
        "top_cost_drivers_tab": "أهم محركات التكلفة",
        "cagr_tab": "معدل النمو السنوي المركب (CAGR) والقيم الشاذة",
        "high_volume_tab": "الأدوية عالية الحجم",
        "explainability_tab": "قابلية الشرح",
        "no_data_warning": "لا توجد بيانات متاحة لمجموعة الأدوية والشركة المصنعة المحددة.",
        "select_drug_info": "اختر دواء واحدًا أو أكثر لعرض توقعاتهم.",
        "forecast_start_vline": "بدء التوقع",
        "drug_spending_forecast_explorer": "مستكشف توقعات الإنفاق على الأدوية",
        "top_ten_title": "أعلى 10 ",
        "top_ten_title_suffix_spending": " (2023 مقابل 2024)",
        "top_ten_title_other": " (2023)",
        "cagr_title": "الأدوية الأسرع نموًا والقيم الشاذة",
        "cagr_subtitle": "أعلى 10 أدوية أسرع نموًا حسب CAGR (2019-2023)",
        "compare_title": "مقارنة الأدوية المختارة حسب ",
        "explain_title": "قابلية شرح النموذج",
        "select_drug_explain": "اختر دواءً لعرض الشرح.",
        "insight_title": "رؤية لـ ",
        "high_forecast_bullet": "توقع مرتفع؟ على الأرجح بسبب ارتفاع تكلفة الوحدة أو زيادة عدد المطالبات.",
        "low_forecast_bullet": "توقع منخفض؟ قد يكون بسبب انخفاض الاستخدام أو استقرار الأسعار.",
        "key_drivers_bullet": "العوامل الرئيسية: الإنفاق المتأخر، اتجاهات المطالبات، والنمو التاريخي.",
        "note_explain": "💡 ملاحظة: هذه الرؤى تستند إلى مجموعة قواعد محددة مسبقًا ولا تعكس بشكل ديناميكي خصائص الدواء الفردية أو تغيرات النموذج.",
        "export_data_header": "تصدير البيانات",
        "download_button": "تحميل ملف CSV للتوقعات المفلترة",
        "drug_column": "الدواء",
        "manufacturer_column": "الشركة المصنعة",
        "value_2023": "قيمة 2023",
        "forecast_2024": "توقع 2024",
        "cagr_percent": "معدل النمو السنوي (%)",
        "drug_name": "اسم الدواء",
        "year": "السنة",
        "total_spending_label": "إجمالي الإنفاق ($)",
        "avg_spend_per_unit_label": "متوسط الإنفاق لكل وحدة ($)",
        "total_claims_label": "إجمالي المطالبات",
        "total_beneficiaries_label": "إجمالي المستفيدين",
        "spending_trend": "اتجاه الإنفاق",
        "per_unit_cost_trend": "اتجاه التكلفة لكل وحدة",
        "claim_volume_trend": "اتجاه حجم المطالبات",
        "beneficiaries_trend": "اتجاه المستفيدين",
    },
}

# Add a language selector to the sidebar
language_options = ["English", "العربية"]
selected_language_name = st.sidebar.radio(translations["en"]["language_select"], language_options)
lang_code = "en" if selected_language_name == "English" else "ar"

# Store the current language in session state
if 'lang' not in st.session_state or st.session_state.lang != lang_code:
    st.session_state.lang = lang_code
    st.rerun()

t = translations[st.session_state.lang]

# Apply custom CSS for RTL support for Arabic
if st.session_state.lang == "ar":
    st.markdown(
        """
        <style>
            html, body {
                direction: rtl;
                text-align: right;
            }
            .st-emotion-cache-1cypcdb { /* Streamlit's main block */
                text-align: right;
            }
            .st-emotion-cache-163j0a5 { /* Streamlit's container for most elements */
                direction: rtl;
                text-align: right;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

st.title(t["title"])
st.markdown(t["intro"])

# -------------------------------
# 2. Load Data & Model
# -------------------------------
@st.cache_data
def load_data():
    try:
        df_long = pd.read_csv("data/processed/df_long.csv")
        forecast = pd.read_csv("data/processed/full_drug_forecasts_2024.csv")
        return df_long, forecast
    except FileNotFoundError as e:
        st.error(f"Required data file not found: {e.filename}. Please ensure 'df_long.csv' and 'full_drug_forecasts_2024.csv' are in the 'data/processed/' folder.")
        st.stop()

@st.cache_data
def load_model():
    model_path = os.path.join("models", "drug_spending_predictor.pkl")
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}. Please ensure the model file is in the 'models/' folder.")
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Load data and model
df_long, forecast_df = load_data()
predictor = load_model()

# Data Cleaning and Preprocessing
df_long['mftr_name'] = df_long['mftr_name'].str.lower().str.strip()
df_long['brnd_name'] = df_long['brnd_name'].str.lower().str.strip()
forecast_df['brnd_name'] = forecast_df['brnd_name'].str.lower().str.strip()

# Create a mapping from brand name to manufacturer for joining
brand_to_mftr = df_long[['brnd_name', 'mftr_name']].drop_duplicates().set_index('brnd_name')
forecast_df = forecast_df.merge(brand_to_mftr, on='brnd_name', how='left')

# -------------------------------
# 3. Sidebar Filters
# -------------------------------
st.sidebar.header(t["sidebar_header"])

# Drug selection (changed to a multi-select for side-by-side comparison)
drug_list = sorted(df_long['brnd_name'].dropna().unique())
selected_drugs = st.sidebar.multiselect(t["drug_select"], drug_list, default=[])

# Manufacturer filter
manufacturer_list = sorted(df_long['mftr_name'].dropna().unique())
selected_manufacturer = st.sidebar.selectbox(t["mftr_filter"], ["All"] + manufacturer_list)

# View mode
view_mode = st.sidebar.radio(
    t["view_mode"],
    [t["total_spending"], t["per_unit_cost"], t["total_claims"], t["total_beneficiaries"]]
)

# -------------------------------
# 4. Helper Function: Map view_mode to column and labels
# -------------------------------
def get_view_data(view_mode_key):
    """Return the appropriate column and label for a given view mode."""
    if view_mode_key == t["total_spending"]:
        col = 'tot_spndng'
        ylabel = t["total_spending_label"]
        title_suffix = t["spending_trend"]
    elif view_mode_key == t["per_unit_cost"]:
        col = 'avg_spnd_per_dsg_unt_wghtd'
        ylabel = t["avg_spend_per_unit_label"]
        title_suffix = t["per_unit_cost_trend"]
    elif view_mode_key == t["total_claims"]:
        col = 'tot_clms'
        ylabel = t["total_claims_label"]
        title_suffix = t["claim_volume_trend"]
    elif view_mode_key == t["total_beneficiaries"]:
        col = 'tot_benes'
        ylabel = t["total_beneficiaries_label"]
        title_suffix = t["beneficiaries_trend"]
    else:
        # Default to Total Spending
        col = 'tot_spndng'
        ylabel = t["total_spending_label"]
        title_suffix = t["spending_trend"]
    return col, ylabel, title_suffix

# -------------------------------
# 5. Dashboard Tabs
# -------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    t["forecast_explorer_tab"],
    t["top_cost_drivers_tab"],
    t["cagr_tab"],
    t["high_volume_tab"],
    t["explainability_tab"]
])


# Tab 1: Forecast Explorer
with tab1:
    st.subheader(t["drug_spending_forecast_explorer"])
    col, ylabel, title = get_view_data(view_mode)

    # Filter the main DataFrame based on user selections and manufacturer
    df_filtered = df_long[df_long['brnd_name'].isin(selected_drugs)].copy()
    if selected_manufacturer != "All":
        df_filtered = df_filtered[df_filtered['mftr_name'] == selected_manufacturer]

    # Add a check to prevent errors with empty filtered DataFrames
    if df_filtered.empty and selected_drugs:
        st.warning(t["no_data_warning"])
    elif not selected_drugs:
        st.info(t["select_drug_info"])
    else:
        combined_df = pd.DataFrame()
        
        for drug in selected_drugs:
            # Historical data for the selected drug
            drug_data = df_filtered[df_filtered['brnd_name'] == drug].sort_values('year').copy()
            drug_data['is_forecast'] = False
            
            # Append to the combined DataFrame
            combined_df = pd.concat([combined_df, drug_data])

        # Add forecast points ONLY IF the view mode is 'Total Spending'
        if view_mode == t["total_spending"]:
            forecast_data_with_mftr = forecast_df[forecast_df['brnd_name'].isin(selected_drugs)].copy()
            if selected_manufacturer != "All":
                forecast_data_with_mftr = forecast_data_with_mftr[forecast_data_with_mftr['mftr_name'] == selected_manufacturer]

            for _, row in forecast_data_with_mftr.iterrows():
                if 'mftr_name' in row:  # Add an explicit check to ensure the key exists
                    forecast_point = pd.DataFrame({
                        'brnd_name': [row['brnd_name']],
                        'year': [2024],
                        'tot_spndng': [row['forecast_2024_total_spending']],
                        'is_forecast': [True],
                        'mftr_name': [row['mftr_name']]
                    })
                    combined_df = pd.concat([combined_df, forecast_point])

        # Create the line chart with Plotly Express
        if not combined_df.empty:
            fig = px.line(
                combined_df,
                x='year',
                y=col,
                color='brnd_name',
                markers=True,
                line_dash='is_forecast' if view_mode == t["total_spending"] else None,
                labels={
                    'year': t["year"],
                    col: ylabel,
                    'brnd_name': t["drug_column"]
                },
                title=f"{t['drug_column']} {title}",
                template="plotly_white"
            )

            # Add a vertical line for the forecast if applicable
            if view_mode == t["total_spending"]:
                fig.add_vline(x=2023.5, line_width=1, line_dash="dash", line_color="gray", annotation_text=t["forecast_start_vline"], annotation_position="bottom right")

            fig.update_layout(
                title_font_size=20,
                legend_title_text=t["drug_name"]
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("---") 



# Tab 2: Top Cost Drivers
with tab2:
    st.subheader(f"{t['top_ten_title']}{view_mode}")
    col, ylabel, _ = get_view_data(view_mode)

    df_2023 = df_long[df_long['year'] == 2023].copy()
    
    if selected_manufacturer != "All":
        df_2023 = df_2023[df_2023['mftr_name'] == selected_manufacturer]

    if df_2023.empty:
        st.warning(t["no_data_warning"])
    else:
        top_drugs = df_2023.groupby(['brnd_name', 'mftr_name'])[col].sum().nlargest(10).reset_index()
        top_drugs = top_drugs.rename(columns={col: f'value_2023'})
        top_drugs['label'] = top_drugs['brnd_name'] + ' (' + top_drugs['mftr_name'] + ')'

        if view_mode == t["total_spending"]:
            forecast_grouped = forecast_df.groupby('brnd_name')['forecast_2024_total_spending'].sum().reset_index()
            top_drugs = top_drugs.merge(forecast_grouped, on='brnd_name', how='left')
        
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=top_drugs['label'],
                y=top_drugs['value_2023'],
                name=f'2023 {view_mode}',
                marker_color='skyblue'
            ))
            fig.add_trace(go.Bar(
                x=top_drugs['label'],
                y=top_drugs['forecast_2024_total_spending'],
                name=f'2024 {t["forecast_2024"]}',
                marker_color='salmon'
            ))

            fig.update_layout(
                barmode='group',
                title=f"{t['top_ten_title']}{view_mode}{t['top_ten_title_suffix_spending']}",
                xaxis_title=f"{t['drug_column']} ({t['manufacturer_column']})",
                yaxis_title=ylabel,
                template='plotly_white'
            )
        else:
            fig = px.bar(
                top_drugs,
                x='value_2023',
                y='label',
                orientation='h',
                labels={
                    'value_2023': ylabel,
                    'label': f"{t['drug_column']} ({t['manufacturer_column']})"
                },
                title=f"{t['top_ten_title']}{view_mode}{t['top_ten_title_other']}",
                template='plotly_white'
            )
            fig.update_layout(yaxis={'categoryorder':'total descending'})

        st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")

# Tab 3: CAGR & Outliers
with tab3:
    st.subheader(t["cagr_title"])
    
    df_filtered_outliers = df_long.copy()
    if selected_manufacturer != "All":
        df_filtered_outliers = df_filtered_outliers[df_filtered_outliers['mftr_name'] == selected_manufacturer]

    if df_filtered_outliers.empty:
        st.warning(t["no_data_warning"])
    else:
        cagr_df = df_filtered_outliers.groupby('brnd_name').agg(
            cagr_avg_spnd_per_dsg_unt_19_23=('cagr_avg_spnd_per_dsg_unt_19_23', 'first')
        ).dropna().nlargest(10, 'cagr_avg_spnd_per_dsg_unt_19_23').reset_index()

        if not cagr_df.empty:
            st.markdown(f"### {t['cagr_subtitle']}")
            fig = px.bar(
                cagr_df,
                x='cagr_avg_spnd_per_dsg_unt_19_23',
                y='brnd_name',
                orientation='h',
                labels={'cagr_avg_spnd_per_dsg_unt_19_23': t["cagr_percent"], 'brnd_name': t["drug_column"]},
                title=t['cagr_title'],
                template='plotly_white'
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("---")


# Tab 4: High-Volume Drugs
with tab4:
    col, ylabel, title = get_view_data(view_mode)
    st.subheader(f"{t['compare_title']}{view_mode}")

    if not selected_drugs:
        st.info(t["select_drug_info"])
    else:
        df_2023 = df_long[df_long['year'] == 2023].copy()
        
        if selected_manufacturer != "All":
            df_2023 = df_2023[df_2023['mftr_name'] == selected_manufacturer]

        comparison_df = df_2023[df_2023['brnd_name'].isin(selected_drugs)].copy()

        if comparison_df.empty:
            st.warning(t["no_data_warning"])
        else:
            comparison_df = comparison_df.groupby(['brnd_name', 'mftr_name'])[col].sum().reset_index()
            comparison_df = comparison_df.rename(columns={col: 'value'})

            fig = px.bar(
                comparison_df,
                x='value',
                y='brnd_name',
                color='mftr_name',
                orientation='h',
                labels={
                    'value': ylabel,
                    'brnd_name': t["drug_name"],
                    'mftr_name': t["manufacturer_column"]
                },
                title=f"{t['compare_title']}{view_mode}{t['top_ten_title_other']}",
                template="plotly_white"
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("---")


# Tab 5: Model Explainability
with tab5:
    st.subheader(t["explain_title"])

    if not selected_drugs:
        st.info(t["select_drug_explain"])
    else:
        for drug in selected_drugs:
            st.markdown(f"### {t['insight_title']}{drug}:")
            st.markdown(f"""
            - {t['high_forecast_bullet']}
            - {t['low_forecast_bullet']}
            - {t['key_drivers_bullet']}
            """)
            st.markdown(f"💡 {t['note_explain']}")
            st.markdown("---")



# -------------------------------
# 6. Export Data
# -------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown(f"### {t['export_data_header']}")

export_df = forecast_df.copy()

if selected_drugs:
    export_df = export_df[export_df['brnd_name'].isin(selected_drugs)]
if selected_manufacturer != "All":
    export_df = export_df[export_df['mftr_name'] == selected_manufacturer]

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df_to_csv(export_df)

st.sidebar.download_button(
    label=t["download_button"],
    data=csv,
    file_name=f"medicare_forecast_2024_filtered.csv",
    mime="text/csv"
) 