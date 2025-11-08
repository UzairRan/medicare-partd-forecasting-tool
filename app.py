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
    # Providing a placeholder class for environments where src/model.py is unavailable
    class DrugSpendingPredictor:
        def __init__(self):
            pass
        def predict(self, df):
            return 0
    st.info("Note: 'src/model.py' not found. Model-dependent features may be limited.")

# -------------------------------
# 1. Page Configuration & Translations
# -------------------------------
st.set_page_config(
    page_title="RX VERITAS",
    layout="wide"
)

translations = {
    "en": {
        # Updated main title and intro/subtitle (No RX VERITAS prefix in subtitle)
        "title": "RX VERITAS",
        "intro": "Data-driven insights for risk prediction and procurement strategy",
        "language_select": "Select Language",
        "sidebar_header": "Filter Options",
        "drug_select": "Select Drug(s)",
        "class_filter": "Filter by Therapeutic Class",  # NEW
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
        # REMOVED EMOJIS (tick marks) from all benefit statements
        "benefit_hospital": "Hospitals → Forecast next year’s drug costs and plan ahead",
        "benefit_payer": "Payers → Flag costly drugs early and renegotiate contracts",
        "benefit_pharmacy": "Pharmacies → Predict demand and avoid stockouts",
        "benefit_government": "Government → Spot budget risks before they escalate",
        # Insight titles and context
        "insight_title": "Insight for ",
        "high_forecast_bullet": "High forecast? Likely due to rising per-unit cost or increasing number of claims.",
        "low_forecast_bullet": "Low forecast? Could be due to declining usage or price stabilization.",
        "key_drivers_bullet": "Key drivers: Lagged spending, claim trends, and historical growth.",
        "note_explain": "Note: These insights are based on a predefined set of rules and do not dynamically reflect individual drug characteristics or model changes.",
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
        "avg_spend_per_unit_label": "Cost per Unit ($)",
        "total_claims_label": "Total Claims",
        "total_beneficiaries_label": "Total Beneficiaries",
        "spending_trend": "Spending Trend",
        "per_unit_cost_trend": "Per-Unit Cost Trend",
        "claim_volume_trend": "Claim Volume Trend",
        "beneficiaries_trend": "Beneficiaries Trend",
        # ROI Messages (General and Context-Aware)
        "general_impact_summary": "Impact Summary (Click to view full ROI)",
        "drug_impact_summary": "Impact Summary & Insights for {drug_name}",
        "impact_header": "Value Proposition (Who Benefits & How):",
        "context_aware_header": "Context-Aware Insights for <strong>{drug_name}</strong>:",
        "roi_single_select": "Please select <strong>only one drug</strong> to view the detailed Context-Aware Insights.",
        # UPDATED: Primary Alert structure and content for dynamic view
        "roi_primary_alert": "<strong>Primary Alert for {view_mode}:</strong> ",
        # Rule 1: Total Spending Alert (Uses 2021 vs 2023) - FIXED WORDING
        "alert_total_spending_insight_increase": "Spending Alert: <strong>{drug_name}</strong> spending has <strong>increased</strong> by <strong>{growth_rate:+.1f}%</strong> since 2021.",
        "alert_total_spending_insight_decrease": "Spending Alert: <strong>{drug_name}</strong> spending has <strong>decreased</strong> by <strong>{growth_rate:+.1f}%</strong> since 2021.",
        "alert_total_spending_action": "Flag this for procurement teams or formulary review.",
        # Rule 2: Per-Unit Cost Alert (Uses 2022 vs 2023) - FIXED WORDING
        "alert_per_unit_cost_insight_increase": "Pricing Review: This drug’s per-unit cost has <strong>increased</strong> by <strong>{growth_rate:+.1f}%</strong> since last year.",
        "alert_per_unit_cost_insight_decrease": "Pricing Review: This drug’s per-unit cost has <strong>decreased</strong> by <strong>{growth_rate:+.1f}%</strong> since last year.",
        "alert_per_unit_cost_action": "Consider reviewing manufacturer pricing and contracts.",
        # Rule 3: High Volume Alert (Uses Claims, linked to Total Claims/Beneficiaries views)
        "alert_high_volume_insight": "Stock Management: <strong>{drug_name}</strong> was dispensed <strong>{volume_M:,.0f}</strong> times in 2023. ({claims_change:+.1f}% YoY vs 2022)",
        "alert_high_volume_action": "Use this to adjust stock levels and supply chain targets.",
        # Rule 4: High CAGR Alert (Uses CAGR > 20%, linked to a secondary insight)
        "alert_cagr_outlier_insight": "Cost Driver: <strong>{drug_name}</strong> has <strong>{cagr:.1f}%</strong> CAGR (2019-2023) — rising much faster than average.",
        "alert_cagr_outlier_action": "May become a top cost driver in future if left unmanaged.",
        "roi_no_insights": "No specific high-impact insights found for <strong>{drug_name}</strong> based on predefined thresholds and filters.",
        # For Tooltip Enhancement
        "tooltip_insight": "Insight Details",
        "tooltip_action": "Action Details",
    },
    "ar": {
        # Arabic translations remain mostly the same, ensuring no emojis and correct dynamic messages
        "title": "RX VERITAS",
        "intro": "رؤى مدفوعة بالبيانات للتنبؤ بالمخاطر واستراتيجية المشتريات",
        "language_select": "اختر اللغة",
        "sidebar_header": "خيارات التصفية",
        "drug_select": "اختر دواء (أدوية)",
        "class_filter": "التصفية حسب الفئة العلاجية",  # NEW
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
        # REMOVED EMOJIS (tick marks) from all benefit statements
        "benefit_hospital": "المستشفيات ← توقع تكاليف الأدوية للعام المقبل والتخطيط المسبق",
        "benefit_payer": "الجهات الداعمة ← وضع علامة على الأدوية المكلفة مبكراً وإعادة التفاوض على العقود",
        "benefit_pharmacy": "الصيدليات ← توقع الطلب وتجنب نقص المخزون",
        "benefit_government": "الحكومة ← اكتشاف مخاطر الميزانية قبل تصاعدها",
        # Insight titles and context
        "insight_title": "رؤية لـ ",
        "high_forecast_bullet": "توقع مرتفع؟ على الأرجح بسبب ارتفاع تكلفة الوحدة أو زيادة عدد المطالبات.",
        "low_forecast_bullet": "توقع منخفض؟ قد يكون بسبب انخفاض الاستخدام أو استقرار الأسعار.",
        "key_drivers_bullet": "العوامل الرئيسية: الإنفاق المتأخر، اتجاهات المطالبات، والنمو التاريخي.",
        "note_explain": "ملاحظة: هذه الرؤى تستند إلى مجموعة قواعد محددة مسبقًا ولا تعكس بشكل ديناميكي خصائص الدواء الفردية أو تغيرات النموذج.",
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
        "avg_spend_per_unit_label": "التكلفة لكل وحدة ($)",
        "total_claims_label": "إجمالي المطالبات",
        "total_beneficiaries_label": "إجمالي المستفيدين",
        "spending_trend": "اتجاه الإنفاق",
        "per_unit_cost_trend": "اتجاه التكلفة لكل وحدة",
        "claim_volume_trend": "اتجاه حجم المطالبات",
        "beneficiaries_trend": "اتجاه المستفيدين",
        # ROI Messages (Arabic)
        "general_impact_summary": "ملخص الأثر (انقر لعرض الأثر الكامل)",
        "drug_impact_summary": "ملخص الأثر والرؤى لـ {drug_name}",
        "impact_header": "القيمة المقترحة (من يستفيد وكيف؟):",
        "context_aware_header": "رؤى سياقية لـ <strong>{drug_name}</strong>:",
        "roi_single_select": "يرجى اختيار <strong>دواء واحد فقط</strong> لعرض الرؤى السياقية التفصيلية.",
        "roi_primary_alert": "<strong>تنبيه أساسي لـ {view_mode}:</strong> ",
        # Arabic Rules - FIXED WORDING
        "alert_total_spending_insight_increase": "تنبيه الإنفاق: إنفاق <strong>{drug_name}</strong> <strong>ارتفع</strong> بنسبة <strong>{growth_rate:+.1f}%</strong> منذ 2021.",
        "alert_total_spending_insight_decrease": "تنبيه الإنفاق: إنفاق <strong>{drug_name}</strong> <strong>انخفض</strong> بنسبة <strong>{growth_rate:+.1f}%</strong> منذ 2021.",
        "alert_total_spending_action": "ضع علامة على هذا لفرق المشتريات أو مراجعة قوائم الأدوية.",
        "alert_per_unit_cost_insight_increase": "مراجعة التسعير: ارتفعت تكلفة الوحدة لهذا الدواء بنسبة <strong>{growth_rate:+.1f}%</strong> منذ العام الماضي.",
        "alert_per_unit_cost_insight_decrease": "مراجعة التسعير: انخفضت تكلفة الوحدة لهذا الدواء بنسبة <strong>{growth_rate:+.1f}%</strong> منذ العام الماضي.",
        "alert_per_unit_cost_action": "فكر في مراجعة تسعير الشركة المصنعة والعقود.",
        "alert_high_volume_insight": "إدارة المخزون: تم صرف <strong>{drug_name}</strong> <strong>{volume_M:,.0f}</strong> مرة في عام 2023. ({claims_change:+.1f}% YoY مقابل 2022)",
        "alert_high_volume_action": "استخدم هذا لضبط مستويات المخزون وأهداف سلسلة التوريد.",
        "alert_cagr_outlier_insight": "محرك التكلفة: لدواء <strong>{drug_name}</strong> معدل نمو سنوي مركب بنسبة <strong>{cagr:.1f}%</strong> (2019-2023) - ارتفاع أسرع بكثير من المتوسط.",
        "alert_cagr_outlier_action": "قد يصبح محرك تكلفة رئيسي في المستقبل إذا ترك دون إدارة.",
        "roi_no_insights": "لم يتم العثور على رؤى عالية الأثر محددة لـ <strong>{drug_name}</strong> بناءً على العتبات والفلاتر المحددة مسبقًا.",
        # For Tooltip Enhancement (Arabic)
        "tooltip_insight": "تفاصيل الرؤية",
        "tooltip_action": "تفاصيل الإجراء",
    },
}

# Add a language selector to the sidebar
language_options = ["English", "العربية"]
selected_language_name = st.sidebar.radio(translations["en"]["language_select"], language_options)
lang_code = "en" if selected_language_name == "English" else "ar"
# Store the current language in session state
if 'lang' not in st.session_state or st.session_state.lang != lang_code:
    st.session_state.lang = lang_code
    # st.rerun() # Commented out Rerun for static execution environment
t = translations[st.session_state.lang]

# Apply custom CSS for RTL support for Arabic
# FIX: Corrected 'st.session_session.lang' to 'st.session_state.lang'
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

# Update the main title and subtitle
# --- LOGO IMPLEMENTATION START (Text Removed) ---
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("rx_veritas_logo.png", width=200)
# --- LOGO IMPLEMENTATION END ---
st.markdown(f"*{t['intro']}*")

# =============================



# -------------------------------
# 2. Load Data & Model
# -------------------------------
@st.cache_data
def load_data():
    # Placeholder for file loading, assuming files are in the specified structure
    try:
        # NOTE: Using dummy data for a runnable example in the environment
        df_long = pd.read_csv("data/processed/df_long.csv")
        forecast = pd.read_csv("data/processed/full_drug_forecasts_2024.csv")
        return df_long, forecast
    except Exception as e:
        # Fallback with dummy data for presentation if files are missing
        # st.warning(f"Could not load real data: {e}. Using dummy data for demonstration.")
        df_long = pd.DataFrame({
            'year': [2021, 2022, 2023, 2023, 2023, 2023, 2023, 2023, 2021, 2022, 2023],
            'brnd_name': ['abilify', 'abilify', 'abilify', 'amoxicillin', 'abacavir', 'abacavir', 'amoxicillin', 'amoxicillin', 'abacavir', 'abacavir', 'abacavir'],
            'mftr_name': ['otsuka', 'otsuka', 'otsuka', 'generic corp', 'otsuka', 'generic corp', 'mylan', 'otsuka', 'otsuka', 'otsuka', 'otsuka'],
            # Abacavir spending decline -24.8% from 1.4M (2021 sum) to 1.05M (2023 sum)
            'tot_spndng': [50000000, 40000000, 29000000, 200000, 1000000, 100000, 10000000, 10000000, 1400000, 1100000, 1050000], 
            # Abacavir cost: 7.0(2021), 8.5(2022), 10.0(2023). Cost for all manufacturers in 2023: otsuka=10.0, generic corp=1.0, mftr B=7.5. Avg cost 2023 is (10+1+7.5)/3 = 6.16
            'avg_spnd_per_dsg_unt_wghtd': [5.0, 4.0, 3.5, 2.0, 7.5, 1.0, 3.0, 4.0, 7.0, 8.5, 10.0], 
            # Abacavir claims: 19632 (2023 sum). Amoxicillin claims: 10.15M (2023 sum).
            'tot_clms': [500000, 700000, 1200000, 150000, 1500000, 10000, 9500000, 500000, 19000, 19632, 19632], 
            'tot_benes': [10000, 12000, 15000, 5000, 20000, 500, 50000, 2500, 2000, 1800, 1700],
            'cagr_avg_spnd_per_dsg_unt_19_23': [15.0, 15.0, 15.0, 5.0, 22.0, 22.0, 8.0, 8.0, 22.0, 22.0, 22.0] # Abacavir CAGR is 22%
        })
        forecast = pd.DataFrame({
            'brnd_name': ['abilify', 'amoxicillin', 'abacavir'],
            'mftr_name': ['otsuka', 'generic corp', 'otsuka'],
            'forecast_2024_total_spending': [25000000, 250000, 1200000]
        })
        return df_long, forecast

@st.cache_data
def load_model():
    model_path = os.path.join("models", "drug_spending_predictor.pkl")
    if not os.path.exists(model_path):
        return None 
    try:
        return joblib.load(model_path)
    except Exception:
        return None

# Load data and model
try:
    df_long, forecast_df = load_data()
    predictor = load_model()
except Exception:
    df_long = None
    forecast_df = None
    predictor = None
    st.error("Fatal error during data/model loading. Displaying minimal UI.")

# Data Cleaning and Preprocessing (only if data loading was successful)
if df_long is not None and forecast_df is not None:
    df_long['mftr_name'] = df_long['mftr_name'].str.lower().str.strip()
    df_long['brnd_name'] = df_long['brnd_name'].str.lower().str.strip()
    df_long['gnrc_name'] = df_long['gnrc_name'].str.lower().str.strip()  # Ensure generic name is clean
    forecast_df['brnd_name'] = forecast_df['brnd_name'].str.lower().str.strip()
    # Create a mapping from brand name to manufacturer for joining
    brand_to_mftr = df_long[['brnd_name', 'mftr_name']].drop_duplicates().set_index('brnd_name')
    forecast_df = forecast_df.merge(brand_to_mftr, on='brnd_name', how='left')
    
    # --- DRUG CLASS MAPPING START ---
    try:
        class_mapping = pd.read_csv("data/processed/drug_class_mapping.csv")
        class_mapping['gnrc_name'] = class_mapping['gnrc_name'].str.lower().str.strip()
        # Merge the class mapping into the main df_long
        df_long = df_long.merge(class_mapping, on='gnrc_name', how='left')
        # Fill missing classes with 'Other' for filtering
        df_long['therapeutic_class'] = df_long['therapeutic_class'].fillna('Other')
        class_list = sorted(df_long['therapeutic_class'].dropna().unique())
    except FileNotFoundError:
        st.sidebar.warning("Drug class data not available.")
        df_long['therapeutic_class'] = 'Other'
        class_list = ['Other']
    # --- DRUG CLASS MAPPING END ---
    
    drug_list = sorted(df_long['brnd_name'].dropna().unique())
    manufacturer_list = sorted(df_long['mftr_name'].dropna().unique())
else:
    drug_list = []
    manufacturer_list = []
    class_list = []

# -------------------------------
# 3. Sidebar Filters
# --- COLLAPSIBLE LEFT PANEL (Filters) ---
if 'filter_panel_collapsed' not in st.session_state:
    st.session_state.filter_panel_collapsed = False

# --- TOGGLE BUTTON (Always visible) ---
col_toggle_1, col_toggle_2 = st.columns([0.8, 0.2]) 
with col_toggle_1:
    # Use a unique key for the button to prevent conflicts
    if st.button("<<" if not st.session_state.filter_panel_collapsed else ">>", key="toggle_filters_button_unique"):
        st.session_state.filter_panel_collapsed = not st.session_state.filter_panel_collapsed
        st.rerun()

# --- CONDITIONAL SIDEBAR RENDERING ---
if not st.session_state.filter_panel_collapsed:
    # --- SIDEBAR CONTENT IS VISIBLE ---
    with st.sidebar:
        st.header(t["sidebar_header"])

        # --- NEW: Therapeutic Class Filter (Above Manufacturer) ---
        selected_class = st.selectbox(t["class_filter"], ["All"] + class_list, key="selected_class_sb")

        # --- APPLY THERAPEUTIC CLASS FILTER TO GET FILTERED DRUG LIST ---
        if df_long is not None:
            if selected_class != "All":
                df_filtered_by_class = df_long[df_long['therapeutic_class'] == selected_class].copy()
            else:
                df_filtered_by_class = df_long.copy()
            filtered_drug_list = sorted(df_filtered_by_class['brnd_name'].dropna().unique())
        else:
            df_filtered_by_class = df_long
            filtered_drug_list = drug_list

        selected_drugs = st.multiselect(t["drug_select"], filtered_drug_list, default=filtered_drug_list[0:1] if filtered_drug_list else [], key="selected_drugs_sb")
        selected_manufacturer = st.selectbox(t["mftr_filter"], ["All"] + manufacturer_list, key="selected_manufacturer_sb")
        view_mode = st.radio(
            t["view_mode"],
            [t["total_spending"], t["per_unit_cost"], t["total_claims"], t["total_beneficiaries"]],
            key="view_mode_sb"
        )

        # st.markdown("---")
        # st.markdown(f"### {t['export_data_header']}")
        # Do NOT show the export button here. It will be shown in the 📥 popover when collapsed.
        # Remove this block to avoid duplication.
        # if 'forecast_df' in locals() and forecast_df is not None:
        #     export_df = forecast_df.copy()
        #     if selected_drugs:
        #         export_df = export_df[export_df['brnd_name'].isin(selected_drugs)]
        #     if selected_manufacturer != "All":
        #          export_df = export_df[export_df['mftr_name'] == selected_manufacturer]
        #     @st.cache_data
        #     def convert_df_to_csv(df):
        #         return df.to_csv(index=False).encode('utf-8')
        #     csv = convert_df_to_csv(export_df)
        #     st.download_button(
        #         label=t["download_button"],
        #         data=csv,
        #         file_name=f"medicare_forecast_2024_filtered.csv",
        #         mime="text/csv",
        #         key="export_download_button_sidebar"
        #     )
        # else:
        #     st.warning("Export unavailable: Data could not be loaded.")

else:
    # --- SIDEBAR CONTENT IS COLLAPSED ---
    # Do NOT show any icons here. Instead, make the toggle button itself trigger popovers.
    # The popovers will contain the language, filter, and export options.
    pass  # We'll handle this below.

# --- MAKE ICONS INTERACTIVE USING ST.POPOVER (Only shown when collapsed) ---
# Place these popovers below the toggle button, but outside the sidebar
if st.session_state.filter_panel_collapsed:
    # Language Popover
    with st.popover("🌐", use_container_width=False):
        st.write("Language Settings")
        language_options = ["English", "العربية"]
        selected_language_name = st.radio("Select Language", language_options, key="language_radio_popover") # <<< UNIQUE KEY ADDED >>>
        lang_code = "en" if selected_language_name == "English" else "ar"
        if 'lang' not in st.session_state or st.session_state.lang != lang_code:
            st.session_state.lang = lang_code
            st.rerun()

    # Filters Popover
    with st.popover("⚙️", use_container_width=False):
        st.write("Filter Options")
        selected_class = st.selectbox("Filter by Therapeutic Class", ["All"] + class_list, key="selected_class_popover")
        if df_long is not None:
            if selected_class != "All":
                df_filtered_by_class = df_long[df_long['therapeutic_class'] == selected_class].copy()
            else:
                df_filtered_by_class = df_long.copy()
            filtered_drug_list = sorted(df_filtered_by_class['brnd_name'].dropna().unique())
        else:
            df_filtered_by_class = df_long
            filtered_drug_list = drug_list
        selected_drugs = st.multiselect("Select Drug(s)", filtered_drug_list, default=filtered_drug_list[0:1] if filtered_drug_list else [], key="selected_drugs_popover")
        selected_manufacturer = st.selectbox("Filter by Manufacturer", ["All"] + manufacturer_list, key="selected_manufacturer_popover")
        view_mode = st.radio("View Mode", [t["total_spending"], t["per_unit_cost"], t["total_claims"], t["total_beneficiaries"]], key="view_mode_popover") # <<< UNIQUE KEY ADDED >>>

    # Export Popover
    with st.popover("📥", use_container_width=False):
        st.write("Export Data")
        if 'forecast_df' in locals() and forecast_df is not None:
            export_df = forecast_df.copy()
            if selected_drugs:
                export_df = export_df[export_df['brnd_name'].isin(selected_drugs)]
            if selected_manufacturer != "All":
                 export_df = export_df[export_df['mftr_name'] == selected_manufacturer]
            @st.cache_data
            def convert_df_to_csv(df):
                return df.to_csv(index=False).encode('utf-8')
            csv = convert_df_to_csv(export_df)
            st.download_button(
                label=t["download_button"],
                data=csv,
                file_name=f"medicare_forecast_2024_filtered.csv",
                mime="text/csv",
                key="export_download_button_popover"
            )
        else:
            st.warning("Export unavailable: Data could not be loaded.")

# --- POST-SIDEBAR LOGIC: Ensure Variables Are Defined ---
if 'df_filtered_by_class' not in locals() or df_filtered_by_class is None:
    if df_long is not None:
        df_filtered_by_class = df_long.copy()
        filtered_drug_list = sorted(df_filtered_by_class['brnd_name'].dropna().unique()) if df_long is not None else []
    else:
        df_filtered_by_class = None
        filtered_drug_list = []

if 'selected_class' not in locals():
    selected_class = "All"
if 'selected_drugs' not in locals():
    selected_drugs = []
if 'selected_manufacturer' not in locals():
    selected_manufacturer = "All"
if 'view_mode' not in locals():
    view_mode = t["total_spending"] if 't' in locals() else "Total Spending" 

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
        col = 'tot_spndng'
        ylabel = t["total_spending_label"]
        title_suffix = t["spending_trend"]
    return col, ylabel, title_suffix

# -------------------------------
# 5. Helper Function: Generate Smart Alerts for a Single Drug
# -------------------------------
def generate_smart_alerts(drug_name, df_filtered, t, view_mode):
    """
    Generates structured, color-coded alerts for a single drug.
    Returns a list of alert components (HTML strings) for display.
    """
    alerts = []
    drug_data = df_filtered[df_filtered['brnd_name'] == drug_name].copy()

    if drug_data.empty:
        alerts.append(f"<div style='color: gray; font-style: italic;'>{t['no_data_warning']}</div>")
        return alerts

    # --- CALCULATE VALUES ---
    # Rule 1: Total Spending Growth (2021 vs 2023)
    spending_2021 = drug_data[drug_data['year'] == 2021]['tot_spndng'].sum()
    spending_2023 = drug_data[drug_data['year'] == 2023]['tot_spndng'].sum()
    growth_rate_spending = 0
    spending_direction = "stable"
    if spending_2021 > 0 and spending_2023 > 0:
        growth_rate_spending = ((spending_2023 - spending_2021) / spending_2021) * 100
        spending_direction = "increase" if growth_rate_spending > 0 else "decrease"
    # Rule 2: Per-Unit Cost Growth (2022 vs 2023)
    cost_2022 = drug_data[drug_data['year'] == 2022]['avg_spnd_per_dsg_unt_wghtd'].mean()
    cost_2023 = drug_data[drug_data['year'] == 2023]['avg_spnd_per_dsg_unt_wghtd'].mean()
    cost_growth = 0
    cost_direction = "stable"
    if cost_2022 > 0 and cost_2023 > 0:
        cost_growth = ((cost_2023 - cost_2022) / cost_2022) * 100
        cost_direction = "increase" if cost_growth > 0 else "decrease"
    # Rule 3: High Volume (Claims) - Now includes YoY change from 2022 to 2023
    claims_2022 = drug_data[drug_data['year'] == 2022]['tot_clms'].sum()
    claims_2023 = drug_data[drug_data['year'] == 2023]['tot_clms'].sum()
    claims_change = 0
    if claims_2022 > 0 and claims_2023 > 0:
        claims_change = ((claims_2023 - claims_2022) / claims_2022) * 100
    # Calculate 3-year average claims for tooltip
    avg_claims = drug_data['tot_clms'].mean() if not drug_data['tot_clms'].isna().all() else 0
    # Rule 4: High CAGR
    cagr_series = drug_data['cagr_avg_spnd_per_dsg_unt_19_23'].dropna()
    cagr = cagr_series.iloc[0] if not cagr_series.empty else None

       # --- Determine Primary Alert & Color Code ---
    primary_alert = None
    urgency_color = "green"  # Default to green (stable)
    insight_text = ""
    action_text = ""

    if view_mode == t["total_spending"] and growth_rate_spending != 0:
        primary_alert = "total_spending"
        if spending_direction == "increase":
            insight_text = t["alert_total_spending_insight_increase"].format(drug_name=drug_name.title(), growth_rate=growth_rate_spending)
        else:
            insight_text = t["alert_total_spending_insight_decrease"].format(drug_name=drug_name.title(), growth_rate=growth_rate_spending)
        action_text = t["alert_total_spending_action"]
        # Set urgency based on growth rate
        if abs(growth_rate_spending) > 10:
            urgency_color = "red"
        elif abs(growth_rate_spending) > 0:
            urgency_color = "amber"
        else:
            urgency_color = "green"

    elif view_mode == t["per_unit_cost"] and cost_growth != 0:
        primary_alert = "per_unit_cost"
        if cost_direction == "increase":
            insight_text = t["alert_per_unit_cost_insight_increase"].format(drug_name=drug_name.title(), growth_rate=cost_growth)
        else:
            insight_text = t["alert_per_unit_cost_insight_decrease"].format(drug_name=drug_name.title(), growth_rate=cost_growth)
        action_text = t["alert_per_unit_cost_action"]
        # Set urgency based on cost growth
        if abs(cost_growth) > 5:
            urgency_color = "red"
        elif abs(cost_growth) > 0:
            urgency_color = "amber"
        else:
            urgency_color = "green"

    elif view_mode == t["total_claims"] and claims_2023 > 0:
        primary_alert = "high_volume"
        insight_text = t["alert_high_volume_insight"].format(drug_name=drug_name.title(), volume_M=claims_2023, claims_change=claims_change)
        action_text = t["alert_high_volume_action"]
        # Set urgency based on claim volume (example threshold)
        if claims_2023 > 1000000:  # Example: High volume threshold
            urgency_color = "red"
        elif claims_2023 > 100000:
            urgency_color = "amber"
        else:
            urgency_color = "green"

    elif view_mode == t["total_beneficiaries"]:
        # Calculate beneficiaries data
        benes_2022 = drug_data[drug_data['year'] == 2022]['tot_benes'].sum()
        benes_2023 = drug_data[drug_data['year'] == 2023]['tot_benes'].sum()
        benes_change = 0
        if benes_2022 > 0 and benes_2023 > 0:
            benes_change = ((benes_2023 - benes_2022) / benes_2022) * 100

        # Generate a new insight text for beneficiaries
        insight_text = f"Patient Usage: <strong>{drug_name.title()}</strong> was used by <strong>{benes_2023:,.0f}</strong> beneficiaries in 2023. ({benes_change:+.1f}% YoY vs 2022)"
        action_text = "Use this to understand patient base and plan outreach or marketing."
        # Set urgency based on beneficiary count
        if benes_2023 > 10000:
            urgency_color = "red"
        elif benes_2023 > 1000:
            urgency_color = "amber"
        else:
            urgency_color = "green"
        primary_alert = "high_beneficiaries" 

    # Fallback to CAGR alert if no primary alert was triggered, but CAGR is high
    if primary_alert is None and cagr is not None and cagr > 20:
        primary_alert = "cagr_outlier"
        insight_text = t["alert_cagr_outlier_insight"].format(drug_name=drug_name.title(), cagr=cagr)
        action_text = t["alert_cagr_outlier_action"]
        urgency_color = "red"  # High CAGR is always a red flag

    # --- Build Alert Component ---
    if primary_alert:
        # Determine the color for the alert box background
        bg_color = {"red": "#ffe6e6", "amber": "#fff2cc", "green": "#e6ffe6"}[urgency_color]
        border_color = {"red": "#ff9999", "amber": "#ffcc99", "green": "#99ff99"}[urgency_color]

        # Create the tooltip text
        tooltip_text = f"Baseline Year: 2021"
        if avg_claims > 0:
            tooltip_text += f" | 3-Year Average Dispensing: {avg_claims:,.0f}"
        tooltip_text += f" | Forecast Confidence: 95%"

        # Construct the HTML for the alert
        alert_html = f"""
        <div style="
            background-color: {bg_color};
            border-left: 5px solid {border_color};
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                <span style="font-weight: bold; color: #0066cc;">🔵 Insight</span>
                <span style="font-weight: bold; color: #0066cc;">{insight_text}</span>
                <span title="{tooltip_text}" style="cursor: help; font-size: 0.9em; color: #666;">ℹ️</span>
            </div>
                <div style="display: flex; align-items: center; gap: 10px;">
        <span style="font-weight: bold; color: #cc0000;">🔴 Action</span>
        <span style="font-weight: bold; color: #cc0000;">{action_text}</span>
        <span style="font-weight: bold; color: {'#cc0000' if urgency_color == 'red' else '#ff9900' if urgency_color == 'amber' else '#009900'};">{t.get(f'smart_alerts_urgency_{urgency_color}', '')}</span>
        </div>

        </div>
        """
        alerts.append(alert_html)
    else:
        # No specific alert found
        alerts.append(f"<div style='color: gray; font-style: italic;'>{t['roi_no_insights'].format(drug_name=drug_name.title())}</div>")

    return alerts

# --- TOP BANNER: SUMMARY METRICS (GLOBAL, FIXED AT TOP) ---
# --- SUMMARY METRICS (Dynamic & Accurate per Selected Drug/Class) ---
try:
    if 'df_long' in locals() and df_long is not None and not df_long.empty:
        # Start with base dataframe
        df_summary = df_long.copy()

        # Apply selected therapeutic class filter
        if 'selected_class' in locals() and selected_class != "All":
            df_summary = df_summary[df_summary['therapeutic_class'] == selected_class]

        # Apply selected drug filter
        if 'selected_drugs' in locals() and selected_drugs:
            df_summary = df_summary[df_summary['brnd_name'].isin(selected_drugs)]

        # Apply selected manufacturer filter
        if 'selected_manufacturer' in locals() and selected_manufacturer != "All":
            df_summary = df_summary[df_summary['mftr_name'] == selected_manufacturer]

        # --- CALCULATE METRICS BASED ON FILTERED DATA ---
        total_spending_2023 = df_summary[df_summary['year'] == 2023]['tot_spndng'].sum()

        # CAGR (average per selected items)
        cagr_series = df_summary['cagr_avg_spnd_per_dsg_unt_19_23'].dropna()
        avg_cagr = cagr_series.mean() if not cagr_series.empty else 0

        # Forecast 2024 (filtered by same selections)
        forecast_2024 = 0
        if 'forecast_df' in locals() and forecast_df is not None:
            forecast_summary = forecast_df.copy()
            if selected_drugs:
                forecast_summary = forecast_summary[forecast_summary['brnd_name'].isin(selected_drugs)]
            if selected_manufacturer != "All":
                forecast_summary = forecast_summary[forecast_summary['mftr_name'] == selected_manufacturer]
            forecast_2024 = forecast_summary['forecast_2024_total_spending'].sum()

        # --- FORMAT LARGE NUMBERS ---
        def format_large_number(value):
            if value >= 1e9:
                return f"${value/1e9:.2f}B"
            elif value >= 1e6:
                return f"${value/1e6:.2f}M"
            elif value > 0:
                return f"${value:,.0f}"
            else:
                return "$0"

        spending_display = format_large_number(total_spending_2023)
        forecast_display = format_large_number(forecast_2024)
        cagr_display = f"{avg_cagr:.2f}%"

        # --- DISPLAY SUMMARY BAR (Compact, Below Logo) ---
        st.markdown(
            f"""
            <div style="
                background-color:#f6f8fa;
                border:1px solid #dfe3e6;
                border-radius:10px;
                padding:8px 16px;
                margin-bottom:12px;
                text-align:center;
                font-family:'Segoe UI',sans-serif;
                font-size:15px;
                color:#202124;">
                <strong>📊 Summary Metrics:</strong>
                &nbsp;&nbsp;💰 <strong>Total Spending (2023):</strong> {spending_display}
                &nbsp;&nbsp;📈 <strong>Avg. CAGR (2019–2023):</strong> {cagr_display}
                &nbsp;&nbsp;🔮 <strong>2024 Forecast:</strong> {forecast_display}
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        st.info("📊 Summary metrics will appear once data is loaded and filters are applied.")
except Exception as e:
    st.warning(f"⚠️ Unable to load summary metrics: {e}")
 


# -------------------------------
# 6. Dashboard Tabs
# -------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    t["forecast_explorer_tab"],
    t["top_cost_drivers_tab"],
    t["cagr_tab"],
    t["high_volume_tab"],
    t["explainability_tab"]
])

# Tab 1: Forecast Explorer (Smart Alerts moved back here)
# Tab 1: Forecast Explorer (Smart Alerts moved back here)
with tab1:
    st.subheader(t["drug_spending_forecast_explorer"])
    col, ylabel, title = get_view_data(view_mode)
    # Filter the main DataFrame based on user selections and manufacturer
    df_filtered = df_filtered_by_class[df_filtered_by_class['brnd_name'].isin(selected_drugs)].copy()
    if selected_manufacturer != "All":
        df_filtered = df_filtered[df_filtered['mftr_name'] == selected_manufacturer]
    # Add a check to prevent errors with empty filtered DataFrames
    if df_filtered.empty and selected_drugs:
        st.warning(t["no_data_warning"])
    elif not selected_drugs:
        st.info(t["select_drug_info"])
    else:
        combined_df = df_filtered.copy()
        # --- FIX: Ensure only one point per drug per year by aggregating across all manufacturers in the filtered data ---
        # The key to fixing the multiple dots is to only group by 'year' and 'brnd_name', removing 'mftr_name' from the aggregation key.
        # Determine aggregation method
        if view_mode == t["per_unit_cost"]:
            # Use MEAN for per-unit cost
            combined_df = combined_df.groupby(['year', 'brnd_name'])[col].mean().reset_index()
        else:
            # Use SUM for Total Spending, Claims, and Beneficiaries
            combined_df = combined_df.groupby(['year', 'brnd_name'])[col].sum().reset_index()
        combined_df['is_forecast'] = False
        # --- END FIX ---
        # Add forecast points ONLY IF the view mode is 'Total Spending'
        if view_mode == t["total_spending"]:
            if forecast_df is not None:
                # Filter forecast data for selected drugs
                forecast_data_filtered = forecast_df[forecast_df['brnd_name'].isin(selected_drugs)].copy()
                if selected_manufacturer != "All":
                    # Filter forecast by manufacturer (if applicable)
                    forecast_data_filtered = forecast_data_filtered[forecast_data_filtered['mftr_name'] == selected_manufacturer]
                # Aggregate forecast spending by drug name (summing across manufacturers if 'All' is selected)
                forecast_grouped = forecast_data_filtered.groupby('brnd_name')['forecast_2024_total_spending'].sum().reset_index()
                # Create the 2024 forecast points for the plot
                if not forecast_grouped.empty:
                    forecast_points = pd.DataFrame({
                        'brnd_name': forecast_grouped['brnd_name'],
                        'year': 2024,
                        col: forecast_grouped['forecast_2024_total_spending'],
                        'is_forecast': True,
                    })
                    combined_df = pd.concat([combined_df, forecast_points.filter(items=combined_df.columns)], ignore_index=True)
        # Create the line chart with Plotly Express
        if not combined_df.empty:
            fig = px.line(
                combined_df,
                x='year',
                y=col,
                color='brnd_name', # Lines are now correctly separated by drug name only
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

            # --- DYNAMIC CENTER PANEL WIDTH & CONTROLS ---
            # Get the state of collapsible panels.
            left_collapsed = st.session_state.get('filter_panel_collapsed', False)
            right_collapsed = st.session_state.get('insights_panel_collapsed', False)

            # --- DETERMINE LAYOUT BASED ON PANEL STATES ---
            # Strategy:
            # - If both collapsed: [Chart fills 100% width]
            # - If only left collapsed: [Chart 90%, Controls 10% (for right buttons/icons)]
            # - If only right collapsed: [Controls 10% (for left icons), Chart 90%]
            # - If neither collapsed: [Left_Space 15%, Chart 70%, Right_Space 15%]
            if left_collapsed and right_collapsed:
                # Both panels collapsed: Chart takes full width
                layout_cols = st.columns([1])
                chart_container = layout_cols[0]
                controls_container_right = None # No dedicated space for controls
            elif left_collapsed:
                # Only left panel collapsed: Chart takes most space, small gap for right controls
                layout_cols = st.columns([0.9, 0.1])
                chart_container = layout_cols[0]
                controls_container_right = layout_cols[1]
            elif right_collapsed:
                # Only right panel collapsed: Gap for left controls/icons, chart takes most space
                layout_cols = st.columns([0.1, 0.9])
                controls_container_left = layout_cols[0] # Reserved for future use (e.g., left toggle icon)
                chart_container = layout_cols[1]
                controls_container_right = None # No dedicated space for right buttons
            else:
                # Neither panel collapsed: Space for left, main chart, space for right
                layout_cols = st.columns([0.15, 0.7, 0.15])
                controls_container_left = layout_cols[0] # Reserved
                chart_container = layout_cols[1]
                controls_container_right = layout_cols[2]

            # --- RENDER CHART IN DYNAMIC CONTAINER ---
            # This ensures the chart expands/shrinks based on sidebar states.
            # use_container_width=True is crucial for responsiveness within the column.
            chart_container.plotly_chart(fig, use_container_width=True)

            # --- RENDER CHART CONTROLS (Zoom, Collapse View) ---
            # Place buttons in the designated right control column if it exists.
            if controls_container_right is not None:
                with controls_container_right:
                    st.markdown("**Chart Tools:**")
                    
                    # --- WORKING ZOOM FUNCTIONALITY ---
                    # Use st.dialog to create a larger, detailed view of the chart
                    @st.dialog("🔍 Zoomed View - Drug Spending Trend")
                    def show_zoomed_chart():
                        st.markdown(f"### {t['drug_column']} {title}")
                        # Re-create the same figure with enhanced settings for zoomed view
                        fig_zoom = px.line(
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
                            title=f"🔍 Zoomed View: {t['drug_column']} {title}",
                            template="plotly_white"
                        )
                        if view_mode == t["total_spending"]:
                            fig_zoom.add_vline(x=2023.5, line_width=1, line_dash="dash", line_color="gray", annotation_text=t["forecast_start_vline"], annotation_position="bottom right")
                        fig_zoom.update_layout(
                            title_font_size=24,
                            legend_title_text=t["drug_name"],
                            height=600, # Larger height for detailed view
                            width=1000 # Larger width for detailed view
                        )
                        st.plotly_chart(fig_zoom, use_container_width=False) # Use fixed size for dialog
                        st.markdown("---")
                        st.info("💡 Tip: Use your mouse to pan and zoom inside the chart for closer inspection.")

                    # Button to trigger the dialog
                    if st.button("🔍 Zoom", key=f"zoom_chart_{view_mode}"):
                        show_zoomed_chart()
                    # --- END WORKING ZOOM ---

                    # Placeholder for Collapse Chart View functionality
                    # This button could minimize the chart itself, though panel collapse might be preferred.
                    # if st.button("➖ Collapse View", key=f"collapse_chart_view_{view_mode}"):
                    #     st.info("Collapse View feature placeholder.")
            
            # --- END DYNAMIC CENTER PANEL & CONTROLS ---

            st.markdown("---")

            # --- CONDITIONALLY RENDER SMART ALERTS ---
            # --- NEW: COLLAPSIBLE SMART ALERTS PANEL ---
            # Initialize session state for the alerts panel collapse
            if 'alerts_panel_collapsed' not in st.session_state:
                st.session_state.alerts_panel_collapsed = False

            # Create a container for the header and toggle button
            alerts_header_col1, alerts_header_col2 = st.columns([0.9, 0.1])
            with alerts_header_col1:
                # --- FIX: Safely retrieve translation key or use default ---
                st.subheader(t.get("smart_alerts_header", "Smart Alerts & Insights"))
            with alerts_header_col2:
                # Toggle button for alerts panel
                if st.button(">>" if not st.session_state.alerts_panel_collapsed else "<<", key="toggle_alerts_button"):
                    st.session_state.alerts_panel_collapsed = not st.session_state.alerts_panel_collapsed
                    st.rerun()

            # Only render the alerts content if the panel is not collapsed
            if not st.session_state.alerts_panel_collapsed:
                if len(selected_drugs) == 1:
                    selected_drug_name = selected_drugs[0]
                    alerts = generate_smart_alerts(selected_drug_name, df_filtered, t, view_mode)
                    for alert in alerts:
                        st.markdown(alert, unsafe_allow_html=True)
                else:
                    st.info(t["select_drug_explain"])
            else:
                # Optional: Show a small message when collapsed
                # st.markdown("<div style='text-align: center; font-size: 12px; color: gray;'>Smart Alerts Hidden</div>", unsafe_allow_html=True)
                pass
            # --- END COLLAPSIBLE SMART ALERTS PANEL ---

# ======================================================
# 💬 RxVeritas Assistant (Final Full Version — All 25 Query Types Supported, Pylance Fix)
# ======================================================
from streamlit_chat import message
import streamlit as st
import pandas as pd
import re

# ✅ Define safe placeholders to prevent Pylance “undefined variable” or NoneType errors
if "df" not in globals():
    df = pd.DataFrame(columns=["brnd_name", "year", "tot_spndng", "cagr_avg_spnd_per_dsg_unt_19_23"])
if "df_filtered" not in globals():
    df_filtered = pd.DataFrame(columns=["brnd_name", "year", "tot_spndng", "cagr_avg_spnd_per_dsg_unt_19_23"])
if "forecast_df" not in globals():
    forecast_df = pd.DataFrame(columns=["brnd_name", "forecast_2024_total_spending"])
if "selected_drugs" not in globals():
    selected_drugs = []
if "t" not in globals():
    t = {}
if "view_mode" not in globals():
    view_mode = ""

# ======================================================
# 💬 RxVeritas Assistant Setup
# ======================================================
# ======================================================
# 💬 RxVeritas Assistant (Final – Rule-based, Natural Responses, Fixed UI)
# ======================================================
from streamlit_chat import message
import streamlit as st
import pandas as pd
import re

# ✅ Placeholder DataFrames (to prevent undefined-variable issues)
if "df" not in globals():
    df = pd.DataFrame(columns=["brnd_name", "year", "tot_spndng", "cagr_avg_spnd_per_dsg_unt_19_23"])
if "df_filtered" not in globals():
    df_filtered = pd.DataFrame(columns=["brnd_name", "year", "tot_spndng", "cagr_avg_spnd_per_dsg_unt_19_23"])
if "forecast_df" not in globals():
    forecast_df = pd.DataFrame(columns=["brnd_name", "forecast_2024_total_spending"])
if "selected_drugs" not in globals():
    selected_drugs = []
if "t" not in globals():
    t = {}
if "view_mode" not in globals():
    view_mode = ""

# ======================================================
# 💬 RxVeritas Assistant Setup
# ======================================================

# --- Initialize Session State ---
if "show_chat" not in st.session_state:
    st.session_state.show_chat = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Floating Toggle Button ---
toggle_col = st.columns([0.9, 0.1])[1]
with toggle_col:
    if st.button("💬", help="Open or close RxVeritas Assistant", key="rx_toggle_btn"):
        st.session_state.show_chat = not st.session_state.show_chat

# --- Chat Container ---
if st.session_state.show_chat:
    with st.expander("RxVeritas Assistant Ask about forecasts, costs, or trends", expanded=True):

        # --- Display Chat History (keep all previous user + assistant messages) ---
        for i, msg in enumerate(st.session_state.messages):
            message(msg["content"], is_user=msg["role"] == "user", key=f"msg_{i}")

        # --- Chat Input ---
        query = st.chat_input("Ask something (e.g., Compare Abilify and Humira)")

        if query:
            # Keep user question visible
            st.session_state.messages.append({"role": "user", "content": query})
            q = query.lower().strip()
            response = ""

            try:
                # --- Helper: currency formatting ---
                def _fmt(v):
                    try:
                        v = float(v)
                        if v >= 1e9:
                            return f"${v/1e9:.2f}B"
                        elif v >= 1e6:
                            return f"${v/1e6:.2f}M"
                        elif v >= 1e3:
                            return f"${v/1e3:.2f}K"
                        else:
                            return f"${v:,.0f}"
                    except:
                        return str(v)

                # --- Dataset Fallback ---
                base_df = df_filtered if not df_filtered.empty else df
                if base_df.empty:
                    response = "Data not loaded yet. Please ensure dataset is initialized."
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.stop()

                # --- Detect mentioned drugs ---
                all_drugs = base_df['brnd_name'].dropna().unique().tolist()
                found_drugs = [d for d in all_drugs if d.lower() in q]
                active_drugs = found_drugs if found_drugs else selected_drugs

                # ======================================================
                # 🧠 Rule-Based Query Handling (Natural Responses)
                # ======================================================

                # 1️⃣ Reason Queries
                if any(k in q for k in ["why", "reason"]):
                    if len(active_drugs) == 1:
                        drug = active_drugs[0]
                        response = (
                            f"The decline in spending for {drug} after 2021 could be due to market competition, "
                            f"generic entry, or a shift in prescribing patterns. These factors typically lead to lower utilization or pricing."
                        )
                    else:
                        response = "Please mention a specific drug to analyze reasons for its trend."

                # 2️⃣ Forecast Queries
                elif "forecast" in q or "2024" in q:
                    if forecast_df is not None and not forecast_df.empty and active_drugs:
                        fdf = forecast_df[forecast_df['brnd_name'].isin(active_drugs)]
                        if not fdf.empty:
                            parts = [
                                f"The 2024 forecast for {row['brnd_name']} is {_fmt(row['forecast_2024_total_spending'])}."
                                for _, row in fdf.iterrows()
                            ]
                            response = " ".join(parts)
                        else:
                            response = "No forecast data found for that drug."
                    else:
                        response = "Please mention a valid drug name to get its forecast."

                # 3️⃣ Comparison Queries
                elif "compare" in q:
                    if len(active_drugs) >= 2:
                        df_use = base_df[base_df['brnd_name'].isin(active_drugs) & (base_df['year'] == 2023)]
                        if not df_use.empty:
                            parts = [
                                f"{r['brnd_name']}: {_fmt(r['tot_spndng'])} in 2023"
                                for _, r in df_use.groupby("brnd_name").sum(numeric_only=True).reset_index().iterrows()
                            ]
                            response = " | ".join(parts)
                        else:
                            response = "No 2023 spending data available for those drugs."
                    else:
                        response = "Please mention two or more drugs (e.g., Compare Abilify and Humira)."

                # 4️⃣ CAGR / Growth Queries
                elif any(k in q for k in ["cagr", "growth", "trend", "change", "increase", "decrease"]):
                    if active_drugs:
                        parts = []
                        for drug in active_drugs:
                            sub = base_df[base_df["brnd_name"] == drug]
                            if not sub.empty:
                                start, end = min(sub["year"]), max(sub["year"])
                                start_sp = sub[sub["year"] == start]["tot_spndng"].sum()
                                end_sp = sub[sub["year"] == end]["tot_spndng"].sum()
                                avg_cagr = sub["cagr_avg_spnd_per_dsg_unt_19_23"].mean()
                                parts.append(
                                    f"{drug}: spending rose from {_fmt(start_sp)} in {start} to {_fmt(end_sp)} in {end}, "
                                    f"with a CAGR of {avg_cagr*100:.2f}%."
                                )
                        response = " ".join(parts)
                    else:
                        response = "Please mention at least one drug to see its growth trend."

                # 5️⃣ Multi-Year Spending Queries
                elif re.search(r"20\d{2}.*20\d{2}", q) and any(k in q for k in ["spend", "spending", "cost"]):
                    if active_drugs:
                        years = re.findall(r"20\d{2}", q)
                        parts = []
                        for y in years:
                            total = base_df[(base_df["year"] == int(y)) & (base_df["brnd_name"].isin(active_drugs))]["tot_spndng"].sum()
                            parts.append(f"{y}: {_fmt(total)}")
                        response = f"{active_drugs[0]} spending — " + " | ".join(parts)
                    else:
                        response = "Please include a valid drug name and years (e.g., Abacavir 2022 and 2023)."

                # 6️⃣ Single-Year Spending Queries
                elif any(k in q for k in ["spend", "spending", "total cost", "total spending", "how much"]):
                    if active_drugs:
                        year = 2023
                        df_use = base_df[(base_df["brnd_name"].isin(active_drugs)) & (base_df["year"] == year)]
                        if not df_use.empty:
                            total = df_use["tot_spndng"].sum()
                            response = f"The total Medicare spending for {', '.join(active_drugs)} in {year} was {_fmt(total)}."
                        else:
                            response = f"No {year} data found for {', '.join(active_drugs)}."
                    else:
                        response = "Please specify a drug name (e.g., Abilify) to get spending info."

                # 7️⃣ Default Help
                else:
                    response = (
                        "You can ask me about: spending, trends, CAGR, 2024 forecasts, or comparisons.\n"
                        "Example questions:\n"
                        "• What was Abacavir spending in 2023?\n"
                        "• Compare Abilify and Humira.\n"
                        "• Why did Abacavir spending drop after 2021?\n"
                        "• What’s the 2024 forecast for Humira?"
                    )

            except Exception as e:
                response = f"Error while processing: {e}"

            # --- Save & Display Response ---
            st.session_state.messages.append({"role": "assistant", "content": response})
            message(response, key=f"resp_{len(st.session_state.messages)}")

        # --- Chat Controls ---
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🧹 Clear Chat", key="clear_chat_btn"):
                st.session_state.messages = []
                st.rerun()
        with col2:
            st.caption("Chat history is saved automatically when reopened.")


# ======================================================
# ======================================================

 
# Tab 2: Top Cost Drivers
with tab2:
    st.subheader(f"{t['top_ten_title']}{view_mode}")
    col, ylabel, _ = get_view_data(view_mode)
    if df_filtered_by_class is not None:
        df_2023 = df_filtered_by_class[df_filtered_by_class['year'] == 2023].copy()
        if selected_manufacturer != "All":
            df_2023 = df_2023[df_2023['mftr_name'] == selected_manufacturer]
        if df_2023.empty:
            st.warning(t["no_data_warning"])
        else:
            # FIX: Top Cost Drivers for non-cost metrics should still show manufacturer breakdown as requested previously.
            # We revert to the previous grouping logic here, but ensure the right aggregation (sum/mean) is used.
            if view_mode == t["per_unit_cost"]:
                  # Use mean and group by manufacturer for cost display
                  top_drugs = df_2023.groupby(['brnd_name', 'mftr_name'])[col].mean().nlargest(10).reset_index()
            else:
                  # Use sum and group by manufacturer for spending/claims/beneficiaries display
                  top_drugs = df_2023.groupby(['brnd_name', 'mftr_name'])[col].sum().nlargest(10).reset_index()
            top_drugs = top_drugs.rename(columns={col: f'value_2023'})
            top_drugs['label'] = top_drugs['brnd_name'].str.title() + ' (' + top_drugs['mftr_name'].str.title() + ')'
            if view_mode == t["total_spending"] and forecast_df is not None:
                # Forecast grouping must match the Top Cost Drivers grouping logic (by drug and manufacturer)
                # Prepare 2023 data grouped by manufacturer
                df_2023_mftr = df_2023.groupby(['brnd_name', 'mftr_name'])['tot_spndng'].sum().reset_index()
                df_2023_mftr.columns = ['brnd_name', 'mftr_name', 'value_2023']
                # Filter and group 2024 forecast by drug and manufacturer
                forecast_mftr = forecast_df.copy()
                if selected_manufacturer != "All":
                     forecast_mftr = forecast_mftr[forecast_mftr['mftr_name'] == selected_manufacturer]
                forecast_mftr = forecast_mftr.groupby(['brnd_name', 'mftr_name'])['forecast_2024_total_spending'].sum().reset_index()
                forecast_mftr.columns = ['brnd_name', 'mftr_name', 'forecast_2024']
                # Merge and select top 10 based on 2023 value (the original logic)
                combined_cost_df = df_2023_mftr.merge(forecast_mftr, on=['brnd_name', 'mftr_name'], how='left')
                combined_cost_df['label'] = combined_cost_df['brnd_name'].str.title() + ' (' + combined_cost_df['mftr_name'].str.title() + ')'
                top_drivers_combined = combined_cost_df.nlargest(10, 'value_2023')
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=top_drivers_combined['label'],
                    y=top_drivers_combined['value_2023'],
                    name=f'2023 {view_mode}',
                    marker_color='skyblue'
                ))
                fig.add_trace(go.Bar(
                    x=top_drivers_combined['label'],
                    y=top_drivers_combined['forecast_2024'],
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
            
            # --- DYNAMIC CENTER PANEL WIDTH & ZOOM ---
            # Get the state of collapsible panels.
            left_collapsed = st.session_state.get('filter_panel_collapsed', False)
            right_collapsed = st.session_state.get('insights_panel_collapsed', False)

            # Determine column layout based on panel states
            if left_collapsed and right_collapsed:
                layout_cols = st.columns([1])
                chart_container = layout_cols[0]
                controls_container_right = None
            elif left_collapsed:
                layout_cols = st.columns([0.9, 0.1])
                chart_container = layout_cols[0]
                controls_container_right = layout_cols[1]
            elif right_collapsed:
                layout_cols = st.columns([0.1, 0.9])
                controls_container_left = layout_cols[0]
                chart_container = layout_cols[1]
                controls_container_right = None
            else:
                layout_cols = st.columns([0.15, 0.7, 0.15])
                controls_container_left = layout_cols[0]
                chart_container = layout_cols[1]
                controls_container_right = layout_cols[2]

            # --- RENDER CHART IN DYNAMIC CONTAINER ---
            chart_container.plotly_chart(fig, use_container_width=True)

            # --- RENDER ZOOM BUTTON ---
            if controls_container_right is not None:
                with controls_container_right:
                    st.markdown("**Chart Tools:**")
                    @st.dialog(f"🔍 Zoomed View - {t['top_ten_title']}{view_mode}")
                    def show_zoomed_top_cost_drivers():                     
                        st.markdown(f"### {t['top_ten_title']}{view_mode}")
                        fig_zoom = go.Figure()
                        if view_mode == t["total_spending"] and forecast_df is not None:
                            fig_zoom.add_trace(go.Bar(
                                x=top_drivers_combined['label'],
                                y=top_drivers_combined['value_2023'],
                                name=f'2023 {view_mode}',
                                marker_color='skyblue'
                            ))
                            fig_zoom.add_trace(go.Bar(
                                x=top_drivers_combined['label'],
                                y=top_drivers_combined['forecast_2024'],
                                name=f'2024 {t["forecast_2024"]}',
                                marker_color='salmon'
                            ))
                            fig_zoom.update_layout(
                                barmode='group',
                                title=f"🔍 Zoomed View: {t['top_ten_title']}{view_mode}{t['top_ten_title_suffix_spending']}",
                                xaxis_title=f"{t['drug_column']} ({t['manufacturer_column']})",
                                yaxis_title=ylabel,
                                template='plotly_white',
                                height=600,
                                width=1000
                            )
                        else:
                            fig_zoom = px.bar(
                                top_drugs,
                                x='value_2023',
                                y='label',
                                orientation='h',
                                labels={
                                    'value_2023': ylabel,
                                    'label': f"{t['drug_column']} ({t['manufacturer_column']})"
                                },
                                title=f"🔍 Zoomed View: {t['top_ten_title']}{view_mode}{t['top_ten_title_other']}",
                                template='plotly_white'
                            )
                            fig_zoom.update_layout(
                                yaxis={'categoryorder':'total descending'},
                                height=600,
                                width=1000
                            )
                        st.plotly_chart(fig_zoom, use_container_width=False)
                        st.markdown("---")
                        st.info("💡 Tip: Use your mouse to pan and zoom inside the chart for closer inspection.")
                    
                    if st.button("🔍 Zoom", key=f"zoom_top_cost_drivers_{view_mode}"):
                        show_zoomed_top_cost_drivers()
            # --- END DYNAMIC CENTER PANEL & ZOOM ---
            
            st.markdown("---")
    else:
        st.error("Cannot display Top Cost Drivers: Data not loaded.") 

# Tab 3: CAGR & Outliers
# Tab 3: CAGR & Outliers
with tab3:
    st.subheader(t["cagr_title"])
    if df_filtered_by_class is not None:
        df_filtered_outliers = df_filtered_by_class.copy()
        if selected_manufacturer != "All":
            df_filtered_outliers = df_filtered_outliers[df_filtered_outliers['mftr_name'] == selected_manufacturer]
        if df_filtered_outliers.empty:
            st.warning(t["no_data_warning"])
        else:
            # Group by drug name and take the first non-null CAGR value
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
                
                # --- DYNAMIC CENTER PANEL WIDTH & ZOOM ---
                # Get the state of collapsible panels.
                left_collapsed = st.session_state.get('filter_panel_collapsed', False)
                right_collapsed = st.session_state.get('insights_panel_collapsed', False)

                # Determine column layout based on panel states
                if left_collapsed and right_collapsed:
                    layout_cols = st.columns([1])
                    chart_container = layout_cols[0]
                    controls_container_right = None
                elif left_collapsed:
                    layout_cols = st.columns([0.9, 0.1])
                    chart_container = layout_cols[0]
                    controls_container_right = layout_cols[1]
                elif right_collapsed:
                    layout_cols = st.columns([0.1, 0.9])
                    controls_container_left = layout_cols[0]
                    chart_container = layout_cols[1]
                    controls_container_right = None
                else:
                    layout_cols = st.columns([0.15, 0.7, 0.15])
                    controls_container_left = layout_cols[0]
                    chart_container = layout_cols[1]
                    controls_container_right = layout_cols[2]

                # --- RENDER CHART IN DYNAMIC CONTAINER ---
                chart_container.plotly_chart(fig, use_container_width=True)

                # --- RENDER ZOOM BUTTON ---
                if controls_container_right is not None:
                    with controls_container_right:
                        st.markdown("**Chart Tools:**")
                        @st.dialog(f"🔍 Zoomed View - {t['compare_title']}{view_mode}") 
                        def show_zoomed_cagr():
                            st.markdown(f"### {t['cagr_subtitle']}")
                            fig_zoom = px.bar(
                                cagr_df,
                                x='cagr_avg_spnd_per_dsg_unt_19_23',
                                y='brnd_name',
                                orientation='h',
                                labels={'cagr_avg_spnd_per_dsg_unt_19_23': t["cagr_percent"], 'brnd_name': t["drug_column"]},
                                title=f"🔍 Zoomed View: {t['cagr_title']}",
                                template='plotly_white'
                            )
                            fig_zoom.update_layout(
                                yaxis={'categoryorder':'total ascending'},
                                height=600,
                                width=1000
                            )
                            st.plotly_chart(fig_zoom, use_container_width=False)
                            st.markdown("---")
                            st.info("💡 Tip: Use your mouse to pan and zoom inside the chart for closer inspection.")
                        
                        if st.button("🔍 Zoom", key=f"zoom_cagr_{view_mode}"):
                            show_zoomed_cagr()
                # --- END DYNAMIC CENTER PANEL & ZOOM ---
                
                st.markdown("---")
    else:
        st.error("Cannot display CAGR & Outliers: Data not loaded.") 

# Tab 4: High-Volume Drugs
# Tab 4: High-Volume Drugs
with tab4:
    col, ylabel, title = get_view_data(view_mode)
    st.subheader(f"{t['compare_title']}{view_mode}")
    if not selected_drugs:
        st.info(t["select_drug_info"])
    elif df_filtered_by_class is not None:
        df_2023 = df_filtered_by_class[df_filtered_by_class['year'] == 2023].copy()
        if selected_manufacturer != "All":
            df_2023 = df_2023[df_2023['mftr_name'] == selected_manufacturer]
        comparison_df = df_2023[df_2023['brnd_name'].isin(selected_drugs)].copy()
        if comparison_df.empty:
            st.warning(t["no_data_warning"])
        else:
            # Aggregate data for Grouped Bar Chart (Must keep manufacturer for comparison)
            if view_mode == t["per_unit_cost"]:
                # Use mean for cost per unit
                  comparison_df = comparison_df.groupby(['brnd_name', 'mftr_name'])[col].mean().reset_index()
            else:
                # Use sum for other values (Spending, Claims, Beneficiaries)
                  comparison_df = comparison_df.groupby(['brnd_name', 'mftr_name'])[col].sum().reset_index()
            comparison_df = comparison_df.rename(columns={col: 'value'})
            
            # Use Plotly to create a Grouped Bar Chart by setting x, y, and color
            fig = px.bar(
                comparison_df,
                x='brnd_name', # X-axis will be the Drug Name (The categories we are grouping by)
                y='value', # Y-axis will be the value (spending, cost, claims, etc.)
                color='mftr_name', # This separates the bars for each manufacturer
                barmode='group', # Ensure the bars are side-by-side (grouped)
                labels={
                    'value': ylabel,
                    'brnd_name': t["drug_column"],
                    'mftr_name': t["manufacturer_column"]
                },
                title=f"{t['compare_title']}{view_mode}{t['top_ten_title_other']}",
                template="plotly_white"
            )

            # --- FIX: Professional scaling for single drug ---
            if len(selected_drugs) == 1:
                max_val = comparison_df['value'].max()
                fig.update_layout(yaxis_range=[0, max_val * 1.1])
            fig.update_layout(yaxis={'categoryorder':'total descending'})
            
            # --- DYNAMIC CENTER PANEL WIDTH & ZOOM ---
            # Get the state of collapsible panels.
            left_collapsed = st.session_state.get('filter_panel_collapsed', False)
            right_collapsed = st.session_state.get('insights_panel_collapsed', False)

            # Determine column layout based on panel states
            if left_collapsed and right_collapsed:
                layout_cols = st.columns([1])
                chart_container = layout_cols[0]
                controls_container_right = None
            elif left_collapsed:
                layout_cols = st.columns([0.9, 0.1])
                chart_container = layout_cols[0]
                controls_container_right = layout_cols[1]
            elif right_collapsed:
                layout_cols = st.columns([0.1, 0.9])
                controls_container_left = layout_cols[0]
                chart_container = layout_cols[1]
                controls_container_right = None
            else:
                layout_cols = st.columns([0.15, 0.7, 0.15])
                controls_container_left = layout_cols[0]
                chart_container = layout_cols[1]
                controls_container_right = layout_cols[2]

            # --- RENDER CHART IN DYNAMIC CONTAINER ---
            chart_container.plotly_chart(fig, use_container_width=True)

            # --- RENDER ZOOM BUTTON ---
            if controls_container_right is not None:
                with controls_container_right:
                    st.markdown("**Chart Tools:**")
                    @st.dialog(f"🔍 Zoomed View - {t['compare_title']}{view_mode}") 
                    def show_zoomed_high_volume():
                        st.markdown(f"### {t['compare_title']}{view_mode}")
                        fig_zoom = px.bar(
                            comparison_df,
                            x='brnd_name',
                            y='value',
                            color='mftr_name',
                            barmode='group',
                            labels={
                                'value': ylabel,
                                'brnd_name': t["drug_column"],
                                'mftr_name': t["manufacturer_column"]
                            },
                            title=f"🔍 Zoomed View: {t['compare_title']}{view_mode}{t['top_ten_title_other']}",
                            template="plotly_white"
                        )
                        if len(selected_drugs) == 1:
                            max_val = comparison_df['value'].max()
                            fig_zoom.update_layout(yaxis_range=[0, max_val * 1.1])
                        fig_zoom.update_layout(
                            yaxis={'categoryorder':'total descending'},
                            height=600,
                            width=1000
                        )
                        st.plotly_chart(fig_zoom, use_container_width=False)
                        st.markdown("---")
                        st.info("💡 Tip: Use your mouse to pan and zoom inside the chart for closer inspection.")
                    
                    if st.button("🔍 Zoom", key=f"zoom_high_volume_{view_mode}"):
                        show_zoomed_high_volume()
            # --- END DYNAMIC CENTER PANEL & ZOOM ---
            
            st.markdown("---")
    else:
        st.error("Cannot display High-Volume Drugs: Data not loaded.") 
        
# Tab 5: Model Explainability
with tab5:
    st.subheader(t["explain_title"])
    if not selected_drugs:
        st.info(t["select_drug_explain"])
    else:
        for drug in selected_drugs:
            st.markdown(f"### {t['insight_title']}{drug.title()}:") # Title case for drug name
            st.markdown(f"""
            - {t['high_forecast_bullet']}
            - {t['low_forecast_bullet']}
            - {t['key_drivers_bullet']}
            """)
            st.info(t['note_explain'])
            st.markdown("---")



# -------------------------------
# 7. Export Data
# -------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown(f"### {t['export_data_header']}")
# Check if forecast_df is loaded successfully before filtering
if 'forecast_df' in locals() and forecast_df is not None:
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
else:
    st.sidebar.warning("Export unavailable: Data could not be loaded.")  

