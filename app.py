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
        "procurement_tab": "Procurement Intelligence",
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
        "procurement_tab": "ذكاء المشتريات", 
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
        # --- NEW: Data Type Filter (Moved Above View Mode) ---
        data_type = st.sidebar.selectbox("Filter by Data Type", ["Forecast Data", "Procurement Data"], key="data_type_sb")
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
# -------------------------------
# 6. Dashboard Tabs
# -------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    t["forecast_explorer_tab"],
    t["top_cost_drivers_tab"],
    t["cagr_tab"],
    t["high_volume_tab"],
    t["procurement_tab"],
    t["explainability_tab"]
      # <-- ADD THIS NEW TAB
]) 

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
        # --- NEW: Data Type Check ---
        if data_type == "Forecast Data":
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
                            st.plotly_chart(fig_zoom, use_container_width=False) # Use fixed size for modal
                            st.markdown("---")
                            st.info("💡 Tip: Use your mouse to pan and zoom inside the chart for closer inspection.")

                        # Button to trigger the modal
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
        else:
            st.info("Switch to 'Forecast Data' mode to view this chart.")  

# ======================================================
# 💬 RxVeritas Assistant - Enhanced to Cover 55+ Questions
# ======================================================
from streamlit_chat import message
import streamlit as st
import pandas as pd
import re
import numpy as np

# ------------------------------
# Emoji removal regex
# ------------------------------
EMOJI_RE = re.compile(
    "["                   
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\u2600-\u26FF"
    "\u2700-\u27BF"
    "]+",
    flags=re.UNICODE,
)

def strip_emojis(text: str) -> str:
    if not isinstance(text, str):
        return text
    out = EMOJI_RE.sub("", text)
    out = re.sub(r"\s{2,}", " ", out).strip()
    return out

# ------------------------------
# CSS for Toggle Button
# ------------------------------
st.markdown(
    """
    <style>
    button[title="Open or close RxVeritas Assistant"] {
        background-color: #0b5ed7 !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 6px 10px !important;
        font-weight: 600 !important;
    }
    .streamlit-expanderHeader {
        font-weight:600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------
# Init session state with unique keys
# ------------------------------
if "show_chat_v2" not in st.session_state:
    st.session_state.show_chat_v2 = False
if "chat_messages_v2" not in st.session_state:
    st.session_state.chat_messages_v2 = []

# ------------------------------
# FLOATING TOGGLE BUTTON (Single instance)
# ------------------------------
toggle_col = st.columns([0.9, 0.1])[1]
with toggle_col:
    if st.button("💬", help="Open or close RxVeritas Assistant", key="rx_toggle_btn_final"):
        st.session_state.show_chat_v2 = not st.session_state.show_chat_v2

# ------------------------------
# CHAT EXPANDER (Single instance)
# ------------------------------
if st.session_state.show_chat_v2:
    with st.expander("RxVeritas Assistant Ask about forecasts, costs, or trends", expanded=True):

        # Display chat history with unique keys
        for i, msg in enumerate(st.session_state.chat_messages_v2):
            # Use a combination of index and content hash for truly unique keys
            content_hash = hash(msg["content"]) % 10000
            message_key = f"chat_msg_{i}_{content_hash}"
            message(msg["content"], is_user=(msg["role"] == "user"), key=message_key)

        # Chat input with unique key
        query = st.chat_input("Ask something (e.g., What was Abacavir spending in 2023?)", key="chat_input_final")

        if query:
            st.session_state.chat_messages_v2.append({"role": "user", "content": query})

            q = query.lower().strip()
            response = ""

            # Use the actual data from your main app
            base_df = df_filtered if (df_filtered is not None and not df_filtered.empty) else df_long
            forecast_data = forecast_df  # Use the main forecast_df from your app

            def _fmt(v):
                try:
                    v = float(v)
                    if v >= 1e9: return f"${v/1e9:.2f}B"
                    if v >= 1e6: return f"${v/1e6:.2f}M"
                    if v >= 1e3: return f"${v/1e3:.2f}K"
                    return f"${v:,.0f}"
                except:
                    return str(v)
            
            def get_drug_data(drug_name, df):
                """Get filtered drug data with manufacturer filtering"""
                drug_data = df[df['brnd_name'].str.lower() == drug_name.lower()]
                if selected_manufacturer != "All":
                    drug_data = drug_data[drug_data['mftr_name'] == selected_manufacturer]
                return drug_data

            def get_drug_trend_analysis(drug_data, metric_col, metric_name):
                """Analyze trend for a given metric"""
                if metric_col in ['avg_spnd_per_dsg_unt_wghtd']:
                    yearly_data = drug_data.groupby('year')[metric_col].mean()
                else:
                    yearly_data = drug_data.groupby('year')[metric_col].sum()
                    
                if len(yearly_data) > 1:
                    first_val = yearly_data.iloc[0]
                    last_val = yearly_data.iloc[-1]
                    change_pct = ((last_val - first_val) / first_val * 100) if first_val > 0 else 0
                    
                    if change_pct > 10:
                        trend = "increasing"
                    elif change_pct < -10:
                        trend = "decreasing"
                    else:
                        trend = "stable"
                    
                    return trend, change_pct, yearly_data
                return "insufficient data", 0, yearly_data

            def calculate_cagr(start_val, end_val, years):
                """Calculate CAGR given start, end values and number of years"""
                if start_val > 0 and years > 0:
                    return ((end_val / start_val) ** (1/years) - 1) * 100
                return 0

            def get_stability_analysis(values):
                """Analyze stability/volatility of a metric"""
                if len(values) < 2:
                    return "insufficient data", 0
                
                cv = (np.std(values) / np.mean(values)) * 100
                if cv < 15:
                    return "stable", cv
                elif cv < 30:
                    return "moderately volatile", cv
                else:
                    return "highly volatile", cv

            def get_therapeutic_class_insights():
                """Get therapeutic class analysis if available"""
                try:
                    if 'therapeutic_class' in base_df.columns:
                        class_trends = base_df.groupby('therapeutic_class').agg({
                            'tot_spndng': 'sum',
                            'cagr_avg_spnd_per_dsg_unt_19_23': 'mean'
                        }).round(2)
                        return class_trends
                    return None
                except:
                    return None

            try:
                if base_df is None or base_df.empty:
                    response = "Data not loaded yet. Please ensure the dataset is loaded first."
                else:
                    all_drugs = base_df["brnd_name"].dropna().astype(str).str.lower().unique().tolist()
                    found_drugs = [d for d in all_drugs if d.lower() in q]
                    active_drugs = found_drugs if found_drugs else [d.lower() for d in selected_drugs]

                    # ======================================================
                    # A. TOTAL SPENDING QUESTIONS (1-10) - 6/10 COVERED
                    # ======================================================
                    
                    # 1. What is the total Medicare spending forecast for [Drug Name] in 2024?
                    if re.search(r'forecast.*2024|2024.*forecast', q) and any(drug in q for drug in active_drugs):
                        if not active_drugs:
                            response = "Please mention a drug for forecast (e.g., 'What is the 2024 forecast for Abilify?')."
                        else:
                            parts = []
                            for drug in active_drugs:
                                if forecast_data is not None and not forecast_data.empty:
                                    forecast_filtered = forecast_data.copy()
                                    if selected_manufacturer != "All":
                                        forecast_filtered = forecast_filtered[forecast_filtered['mftr_name'] == selected_manufacturer]
                                    
                                    forecast_grouped = forecast_filtered.groupby('brnd_name')['forecast_2024_total_spending'].sum().reset_index()
                                    
                                    match = forecast_grouped[forecast_grouped['brnd_name'].str.lower() == drug.lower()]
                                    if not match.empty:
                                        val = match.iloc[0]["forecast_2024_total_spending"]
                                        parts.append(f"{drug.title()}: {_fmt(val)} (2024 forecast)")
                                    else:
                                        parts.append(f"{drug.title()}: no forecast available")
                                else:
                                    parts.append(f"{drug.title()}: forecast data not loaded")
                            response = " | ".join(parts)

                    # 2. How has total spending on [Drug Name] changed over the past five years?
                    elif re.search(r'changed over.*five years|past five years.*spending|last five years.*spending', q):
                        if not active_drugs:
                            response = "Please mention a drug (e.g., 'How has spending on Abilify changed over the past five years?')."
                        else:
                            drug = active_drugs[0]
                            drug_data = get_drug_data(drug, base_df)
                            
                            if drug_data.empty:
                                response = f"No data available for {drug.title()}."
                            else:
                                # Get spending for each year from 2019-2023
                                years_data = []
                                for year in [2019, 2020, 2021, 2022, 2023]:
                                    year_data = drug_data[drug_data['year'] == year]
                                    if not year_data.empty:
                                        spending = year_data['tot_spndng'].sum()
                                        years_data.append((year, spending))
                                
                                if len(years_data) >= 2:
                                    first_year, first_spending = years_data[0]
                                    last_year, last_spending = years_data[-1]
                                    change_pct = ((last_spending - first_spending) / first_spending * 100) if first_spending > 0 else 0
                                    
                                    year_details = ", ".join([f"{year}: {_fmt(spend)}" for year, spend in years_data])
                                    response = f"{drug.title()} spending trend ({first_year}-{last_year}): {year_details}. Overall change: {change_pct:+.1f}%"
                                else:
                                    response = f"Insufficient historical data for {drug.title()}."

                    # 3. Which year had the highest total spending for [Drug Name]?
                    elif re.search(r'highest.*spending.*year|year.*highest.*spending', q):
                        if not active_drugs:
                            response = "Please mention a drug (e.g., 'Which year had highest spending for Abilify?')."
                        else:
                            drug = active_drugs[0]
                            drug_data = get_drug_data(drug, base_df)
                            
                            if drug_data.empty:
                                response = f"No data available for {drug.title()}."
                            else:
                                yearly_spending = drug_data.groupby('year')['tot_spndng'].sum()
                                if not yearly_spending.empty:
                                    max_year = yearly_spending.idxmax()
                                    max_spending = yearly_spending.max()
                                    response = f"{drug.title()} had highest spending in {max_year}: {_fmt(max_spending)}"
                                else:
                                    response = f"No spending data found for {drug.title()}."

                    # 4. Does total spending for [Drug Name] show a steady trend or volatility?
                    elif re.search(r'steady.*trend|volatility.*spending|stable.*spending', q):
                        if not active_drugs:
                            response = "Please mention a drug to analyze trend stability."
                        else:
                            drug = active_drugs[0]
                            drug_data = get_drug_data(drug, base_df)
                            
                            if drug_data.empty:
                                response = f"No data available for {drug.title()}."
                            else:
                                yearly_spending = drug_data.groupby('year')['tot_spndng'].sum()
                                if len(yearly_spending) > 1:
                                    stability, cv = get_stability_analysis(yearly_spending.values)
                                    response = f"{drug.title()} spending shows {stability} (volatility index: {cv:.1f}%)."
                                else:
                                    response = f"Insufficient data to determine trend stability for {drug.title()}."

                    # 5. Which drugs contribute most to overall total spending this year?
                    elif re.search(r'drugs contribute.*spending|most.*spending.*drugs', q):
                        df_2023 = base_df[base_df['year'] == 2023]
                        if not df_2023.empty:
                            top_spenders = df_2023.groupby('brnd_name')['tot_spndng'].sum().nlargest(5)
                            total_2023 = df_2023['tot_spndng'].sum()
                            
                            spend_list = []
                            for drug, spending in top_spenders.items():
                                share = (spending / total_2023 * 100) if total_2023 > 0 else 0
                                spend_list.append(f"• {drug.title()}: {_fmt(spending)} ({share:.1f}% of total)")
                            
                            response = f"**Top 5 Drugs by 2023 Spending:**\n\n" + "\n".join(spend_list)
                        else:
                            response = "No 2023 spending data available."

                    # 6. Has total spending increased or decreased for this therapeutic class?
                    elif re.search(r'therapeutic class.*spending|class.*spending.*trend', q):
                        class_insights = get_therapeutic_class_insights()
                        if class_insights is not None and not class_insights.empty:
                            # Get top class by spending
                            top_class = class_insights.nlargest(1, 'tot_spndng')
                            class_name = top_class.index[0]
                            class_spending = top_class.iloc[0]['tot_spndng']
                            class_cagr = top_class.iloc[0]['cagr_avg_spnd_per_dsg_unt_19_23'] * 100
                            
                            trend = "increasing" if class_cagr > 0 else "decreasing"
                            response = f"The {class_name} therapeutic class shows {trend} spending trend with {_fmt(class_spending)} total and {class_cagr:.1f}% CAGR."
                        else:
                            response = "Therapeutic class analysis not available in current data."

                    # ======================================================
                    # B. PER-UNIT COST QUESTIONS (11-20) - 7/10 COVERED
                    # ======================================================
                    
                    # 11. What is the average per-unit cost of [Drug Name] for 2023?
                    elif re.search(r'per-unit cost|unit cost.*2023|average cost.*2023', q):
                        if not active_drugs:
                            response = "Please mention a drug for cost analysis (e.g., 'What is the per-unit cost of Abilify?')."
                        else:
                            drug = active_drugs[0]
                            drug_data = get_drug_data(drug, base_df)
                            drug_2023 = drug_data[drug_data['year'] == 2023]
                            
                            if not drug_2023.empty:
                                avg_cost = drug_2023['avg_spnd_per_dsg_unt_wghtd'].mean()
                                response = f"{drug.title()} average per-unit cost in 2023: {_fmt(avg_cost)}"
                            else:
                                response = f"No 2023 cost data available for {drug.title()}."

                    # 12. How has the per-unit cost of [Drug Name] evolved since 2019?
                    elif re.search(r'per-unit cost.*evolved|unit cost.*since 2019|cost.*evolved.*2019', q):
                        if not active_drugs:
                            response = "Please mention a drug for cost evolution analysis."
                        else:
                            drug = active_drugs[0]
                            drug_data = get_drug_data(drug, base_df)
                            
                            if drug_data.empty:
                                response = f"No data available for {drug.title()}."
                            else:
                                # Get cost data for 2019-2023
                                cost_data = []
                                for year in [2019, 2020, 2021, 2022, 2023]:
                                    year_data = drug_data[drug_data['year'] == year]
                                    if not year_data.empty:
                                        avg_cost = year_data['avg_spnd_per_dsg_unt_wghtd'].mean()
                                        cost_data.append((year, avg_cost))
                                
                                if len(cost_data) >= 2:
                                    cost_details = ", ".join([f"{year}: {_fmt(cost)}" for year, cost in cost_data])
                                    first_year, first_cost = cost_data[0]
                                    last_year, last_cost = cost_data[-1]
                                    change_pct = ((last_cost - first_cost) / first_cost * 100) if first_cost > 0 else 0
                                    
                                    response = f"{drug.title()} unit cost evolution ({first_year}-{last_year}): {cost_details}. Overall change: {change_pct:+.1f}%"
                                else:
                                    response = f"Insufficient cost history for {drug.title()}."

                    # 13. Does the per-unit cost of [Drug Name] show stability or fluctuation?
                    elif re.search(r'stability.*cost|fluctuation.*cost|stable.*unit cost', q):
                        if not active_drugs:
                            response = "Please mention a drug for cost stability analysis."
                        else:
                            drug = active_drugs[0]
                            drug_data = get_drug_data(drug, base_df)
                            
                            if drug_data.empty:
                                response = f"No data available for {drug.title()}."
                            else:
                                yearly_costs = drug_data.groupby('year')['avg_spnd_per_dsg_unt_wghtd'].mean()
                                if len(yearly_costs) > 1:
                                    stability, cv = get_stability_analysis(yearly_costs.values)
                                    response = f"{drug.title()} per-unit cost shows {stability} (volatility index: {cv:.1f}%)."
                                else:
                                    response = f"Insufficient data to determine cost stability for {drug.title()}."

                    # 14. Which drugs currently have the highest per-unit costs?
                    elif re.search(r'highest.*per-unit costs|drugs.*highest.*cost', q):
                        df_2023 = base_df[base_df['year'] == 2023]
                        if not df_2023.empty:
                            # Get top 5 drugs by unit cost
                            top_costs = df_2023.groupby('brnd_name')['avg_spnd_per_dsg_unt_wghtd'].mean().nlargest(5)
                            cost_list = [f"{drug.title()}: {_fmt(cost)}" for drug, cost in top_costs.items()]
                            response = f"Top 5 drugs by per-unit cost in 2023:\n" + "\n".join([f"• {item}" for item in cost_list])
                        else:
                            response = "No 2023 cost data available."

                    # 15. Which year recorded the lowest per-unit cost for [Drug Name]?
                    elif re.search(r'lowest.*per-unit cost|year.*lowest.*cost', q):
                        if not active_drugs:
                            response = "Please mention a drug to find lowest cost year."
                        else:
                            drug = active_drugs[0]
                            drug_data = get_drug_data(drug, base_df)
                            
                            if drug_data.empty:
                                response = f"No data available for {drug.title()}."
                            else:
                                yearly_costs = drug_data.groupby('year')['avg_spnd_per_dsg_unt_wghtd'].mean()
                                if not yearly_costs.empty:
                                    min_year = yearly_costs.idxmin()
                                    min_cost = yearly_costs.min()
                                    response = f"{drug.title()} had lowest per-unit cost in {min_year}: {_fmt(min_cost)}"
                                else:
                                    response = f"No cost data found for {drug.title()}."

                    # 16. How does [Drug Name] compare in unit cost to other drugs in its category?
                    elif re.search(r'compare.*unit cost|unit cost.*compare', q):
                        if not active_drugs:
                            response = "Please mention a drug for cost comparison analysis."
                        else:
                            drug = active_drugs[0]
                            drug_data = get_drug_data(drug, base_df)
                            drug_2023 = drug_data[drug_data['year'] == 2023]
                            
                            if not drug_2023.empty:
                                drug_cost = drug_2023['avg_spnd_per_dsg_unt_wghtd'].mean()
                                
                                # Compare with all drugs
                                all_drugs_2023 = base_df[base_df['year'] == 2023]
                                avg_all_cost = all_drugs_2023['avg_spnd_per_dsg_unt_wghtd'].mean()
                                
                                if drug_cost > avg_all_cost:
                                    comparison = f"higher than average ({_fmt(avg_all_cost)})"
                                else:
                                    comparison = f"lower than average ({_fmt(avg_all_cost)})"
                                
                                response = f"{drug.title()} unit cost {_fmt(drug_cost)} is {comparison} across all drugs."
                            else:
                                response = f"No 2023 cost data available for {drug.title()}."

                    # 19. Which manufacturer offers the lowest per-unit cost for [Drug Name]?
                    elif re.search(r'lowest.*per-unit cost|lowest.*unit cost|manufacturer.*lowest cost', q):
                        if not active_drugs:
                            response = "Please mention a drug to find lowest cost manufacturer."
                        else:
                            drug = active_drugs[0]
                            drug_data = base_df[base_df['brnd_name'].str.lower() == drug.lower()]
                            
                            if not drug_data.empty:
                                # Get average cost per manufacturer
                                mfr_costs = drug_data.groupby('mftr_name')['avg_spnd_per_dsg_unt_wghtd'].mean()
                                if not mfr_costs.empty:
                                    min_mfr = mfr_costs.idxmin()
                                    min_cost = mfr_costs.min()
                                    max_mfr = mfr_costs.idxmax()
                                    max_cost = mfr_costs.max()
                                    cost_diff = max_cost - min_cost
                                    response = f"{min_mfr.title()} offers the lowest per-unit cost for {drug.title()}: {_fmt(min_cost)} (vs {max_mfr.title()}: {_fmt(max_cost)}, difference: {_fmt(cost_diff)})"
                                else:
                                    response = f"No manufacturer cost data found for {drug.title()}."
                            else:
                                response = f"No data available for {drug.title()}."

                    # ======================================================
                    # C. CLAIM VOLUME & BENEFICIARIES (21-30) - 8/10 COVERED
                    # ======================================================
                    
                    # 21. What is the total claim volume for [Drug Name] in 2023?
                    elif re.search(r'claim volume.*2023|total claims.*2023|number of claims.*2023', q):
                        if not active_drugs:
                            response = "Please mention a drug for claim volume analysis."
                        else:
                            drug = active_drugs[0]
                            drug_data = get_drug_data(drug, base_df)
                            drug_2023 = drug_data[drug_data['year'] == 2023]
                            
                            if not drug_2023.empty:
                                total_claims = drug_2023['tot_clms'].sum()
                                response = f"{drug.title()} total claims in 2023: {total_claims:,}"
                            else:
                                response = f"No 2023 claim data available for {drug.title()}."

                    # 22. How many beneficiaries were prescribed [Drug Name] last year?
                    elif re.search(r'beneficiaries.*prescribed|patients.*prescribed|how many.*beneficiaries', q):
                        if not active_drugs:
                            response = "Please mention a drug for beneficiary analysis."
                        else:
                            drug = active_drugs[0]
                            drug_data = get_drug_data(drug, base_df)
                            drug_2023 = drug_data[drug_data['year'] == 2023]
                            
                            if not drug_2023.empty:
                                total_benes = drug_2023['tot_benes'].sum()
                                response = f"{drug.title()} beneficiaries in 2023: {total_benes:,}"
                            else:
                                response = f"No 2023 beneficiary data available for {drug.title()}."

                    # 23. Has the number of claims for [Drug Name] been increasing or decreasing since 2019?
                    elif re.search(r'claims.*increasing.*decreasing|number of claims.*trend|claims.*since 2019', q):
                        if not active_drugs:
                            response = "Please mention a drug for claims trend analysis."
                        else:
                            drug = active_drugs[0]
                            drug_data = get_drug_data(drug, base_df)
                            
                            if drug_data.empty:
                                response = f"No data available for {drug.title()}."
                            else:
                                trend, change_pct, yearly_data = get_drug_trend_analysis(drug_data, 'tot_clms', 'claims')
                                
                                if trend != "insufficient data":
                                    years_list = ", ".join([f"{year}: {claims:,}" for year, claims in yearly_data.items()])
                                    response = f"{drug.title()} claims trend: {trend} ({change_pct:+.1f}% change since {yearly_data.index[0]}). Yearly data: {years_list}"
                                else:
                                    response = f"Insufficient claims history for {drug.title()}."

                    # 24. What is the CAGR (compound annual growth rate) of claims for [Drug Name]?
                    elif re.search(r'cagr.*claims|claims.*cagr', q):
                        if not active_drugs:
                            response = "Please mention a drug for claims CAGR analysis."
                        else:
                            drug = active_drugs[0]
                            drug_data = get_drug_data(drug, base_df)
                            
                            if drug_data.empty:
                                response = f"No data available for {drug.title()}."
                            else:
                                yearly_claims = drug_data.groupby('year')['tot_clms'].sum()
                                if len(yearly_claims) >= 2:
                                    start_year = yearly_claims.index.min()
                                    end_year = yearly_claims.index.max()
                                    start_val = yearly_claims.loc[start_year]
                                    end_val = yearly_claims.loc[end_year]
                                    years = end_year - start_year
                                    cagr_claims = calculate_cagr(start_val, end_val, years)
                                    response = f"{drug.title()} claims CAGR ({start_year}-{end_year}): {cagr_claims:.2f}%"
                                else:
                                    response = f"Insufficient data to calculate claims CAGR for {drug.title()}."

                    # 25. Which drugs had the highest claim volume in 2023?
                    elif re.search(r'highest.*claim volume|drugs.*most claims', q):
                        df_2023 = base_df[base_df['year'] == 2023]
                        if not df_2023.empty:
                            # Get top 5 drugs by claim volume
                            top_claims = df_2023.groupby('brnd_name')['tot_clms'].sum().nlargest(5)
                            claim_list = [f"{drug.title()}: {claims:,} claims" for drug, claims in top_claims.items()]
                            response = f"Top 5 drugs by claim volume in 2023:\n" + "\n".join([f"• {item}" for item in claim_list])
                        else:
                            response = "No 2023 claim data available."

                    # 26. Does claim volume for [Drug Name] correspond with its total spending trend?
                    elif re.search(r'claim volume.*correspond.*spending|claims.*spending.*trend', q):
                        if not active_drugs:
                            response = "Please mention a drug for claim-spending correlation analysis."
                        else:
                            drug = active_drugs[0]
                            drug_data = get_drug_data(drug, base_df)
                            
                            if drug_data.empty:
                                response = f"No data available for {drug.title()}."
                            else:
                                spending_trend, spending_change, spending_data = get_drug_trend_analysis(drug_data, 'tot_spndng', 'spending')
                                claims_trend, claims_change, claims_data = get_drug_trend_analysis(drug_data, 'tot_clms', 'claims')
                                
                                if spending_trend == claims_trend:
                                    correlation = "positive correlation"
                                elif spending_trend == "increasing" and claims_trend == "decreasing":
                                    correlation = "inverse relationship (spending up, claims down)"
                                elif spending_trend == "decreasing" and claims_trend == "increasing":
                                    correlation = "inverse relationship (spending down, claims up)"
                                else:
                                    correlation = "mixed relationship"
                                
                                response = f"{drug.title()} shows {correlation}. Spending: {spending_trend} ({spending_change:+.1f}%), Claims: {claims_trend} ({claims_change:+.1f}%)"

                    # 27. Why might claim volume for [Drug Name] be high while spending per unit remains low?
                    elif re.search(r'claim volume.*high.*spending.*low|high claims.*low spending', q):
                        if not active_drugs:
                            response = "Please mention a specific drug for this analysis."
                        else:
                            drug = active_drugs[0]
                            response = f"High claim volume with low per-unit spending for {drug.title()} may indicate:\n\n• **Generic availability**: Lower-cost generic versions dominating the market\n• **High-volume, low-margin product**: Widespread use with competitive pricing\n• **Preventive care focus**: Drugs used for prevention rather than treatment\n• **Formulary positioning**: Preferred status driving volume despite lower costs\n• **Manufacturer strategy**: Volume-based pricing to maintain market share"

                    # 29. What's the average number of claims per beneficiary for [Drug Name]?
                    elif re.search(r'claims per beneficiary|average claims per', q):
                        if not active_drugs:
                            response = "Please mention a drug for claims per beneficiary analysis."
                        else:
                            drug = active_drugs[0]
                            drug_data = get_drug_data(drug, base_df)
                            drug_2023 = drug_data[drug_data['year'] == 2023]
                            
                            if not drug_2023.empty:
                                total_claims = drug_2023['tot_clms'].sum()
                                total_benes = drug_2023['tot_benes'].sum()
                                if total_benes > 0:
                                    claims_per_bene = total_claims / total_benes
                                    response = f"{drug.title()} average claims per beneficiary in 2023: {claims_per_bene:.2f}"
                                else:
                                    response = f"No beneficiary data available for {drug.title()}."
                            else:
                                response = f"No 2023 data available for {drug.title()}."

                    # ======================================================
                    # D. CAGR & OUTLIERS (31-40) - 6/10 COVERED
                    # ======================================================
                    
                    # 31. What is the CAGR for total spending on [Drug Name] from 2019–2023?
                    elif re.search(r'cagr.*spending|spending.*cagr', q):
                        if not active_drugs:
                            response = "Please mention a drug for CAGR analysis."
                        else:
                            drug = active_drugs[0]
                            drug_data = get_drug_data(drug, base_df)
                            
                            if drug_data.empty:
                                response = f"No data available for {drug.title()}."
                            else:
                                cagr_vals = drug_data['cagr_avg_spnd_per_dsg_unt_19_23'].dropna()
                                if not cagr_vals.empty:
                                    avg_cagr = cagr_vals.mean()
                                    response = f"{drug.title()} CAGR (2019-2023): {avg_cagr*100:.2f}%"
                                else:
                                    # Calculate CAGR manually if not available
                                    yearly_spending = drug_data.groupby('year')['tot_spndng'].sum()
                                    if len(yearly_spending) >= 2:
                                        start_year = yearly_spending.index.min()
                                        end_year = yearly_spending.index.max()
                                        start_val = yearly_spending.loc[start_year]
                                        end_val = yearly_spending.loc[end_year]
                                        years = end_year - start_year
                                        cagr_calc = calculate_cagr(start_val, end_val, years)
                                        response = f"{drug.title()} CAGR ({start_year}-{end_year}): {cagr_calc:.2f}%"
                                    else:
                                        response = f"Insufficient data to calculate CAGR for {drug.title()}."

                    # 32. Does the CAGR for [Drug Name] suggest rapid growth or market maturity?
                    elif re.search(r'cagr.*suggest|rapid growth.*cagr|market maturity.*cagr', q):
                        if not active_drugs:
                            response = "Please mention a drug for CAGR interpretation."
                        else:
                            drug = active_drugs[0]
                            drug_data = get_drug_data(drug, base_df)
                            
                            if drug_data.empty:
                                response = f"No data available for {drug.title()}."
                            else:
                                cagr_vals = drug_data['cagr_avg_spnd_per_dsg_unt_19_23'].dropna()
                                if not cagr_vals.empty:
                                    avg_cagr = cagr_vals.mean() * 100
                                    if avg_cagr > 20:
                                        interpretation = "rapid growth"
                                    elif avg_cagr > 10:
                                        interpretation = "moderate growth"
                                    elif avg_cagr > 0:
                                        interpretation = "slow growth"
                                    elif avg_cagr > -10:
                                        interpretation = "market maturity/decline"
                                    else:
                                        interpretation = "significant decline"
                                    
                                    response = f"{drug.title()} CAGR of {avg_cagr:.1f}% suggests {interpretation}."
                                else:
                                    response = f"No CAGR data available for {drug.title()}."

                    # 33. Which drugs have the highest CAGR in total spending?
                    elif re.search(r'highest.*cagr.*drugs|drugs.*highest.*cagr', q):
                        if base_df is not None and not base_df.empty:
                            # Get drugs with CAGR data
                            cagr_data = base_df[['brnd_name', 'cagr_avg_spnd_per_dsg_unt_19_23']].dropna()
                            if not cagr_data.empty:
                                top_cagr = cagr_data.groupby('brnd_name')['cagr_avg_spnd_per_dsg_unt_19_23'].mean().nlargest(5)
                                cagr_list = [f"{drug.title()}: {cagr*100:.1f}%" for drug, cagr in top_cagr.items()]
                                response = f"Top 5 drugs by CAGR (2019-2023):\n" + "\n".join([f"• {item}" for item in cagr_list])
                            else:
                                response = "No CAGR data available."
                        else:
                            response = "Data not loaded."

                    # 34. Which drugs show negative CAGR — indicating declining spending trends?
                    elif re.search(r'negative.*cagr|declining.*cagr', q):
                        if base_df is not None and not base_df.empty:
                            # Get drugs with negative CAGR
                            cagr_data = base_df[['brnd_name', 'cagr_avg_spnd_per_dsg_unt_19_23']].dropna()
                            if not cagr_data.empty:
                                negative_cagr = cagr_data.groupby('brnd_name')['cagr_avg_spnd_per_dsg_unt_19_23'].mean()
                                negative_cagr = negative_cagr[negative_cagr < 0].nsmallest(5)
                                if not negative_cagr.empty:
                                    neg_list = [f"{drug.title()}: {cagr*100:.1f}%" for drug, cagr in negative_cagr.items()]
                                    response = f"Drugs with negative CAGR (declining spending):\n" + "\n".join([f"• {item}" for item in neg_list])
                                else:
                                    response = "No drugs with negative CAGR found."
                            else:
                                response = "No CAGR data available."
                        else:
                            response = "Data not loaded."

                    # 35. How does CAGR differ between [Drug Name] and other drugs in the same therapeutic class?
                    elif re.search(r'cagr.*differ.*class|cagr.*therapeutic class', q):
                        if not active_drugs:
                            response = "Please mention a drug for class CAGR comparison."
                        else:
                            drug = active_drugs[0]
                            class_insights = get_therapeutic_class_insights()
                            
                            if class_insights is not None and not class_insights.empty:
                                # Find drug's therapeutic class
                                drug_class = base_df[base_df['brnd_name'].str.lower() == drug.lower()]['therapeutic_class'].iloc[0] if 'therapeutic_class' in base_df.columns else "Unknown"
                                
                                if drug_class in class_insights.index:
                                    class_cagr = class_insights.loc[drug_class, 'cagr_avg_spnd_per_dsg_unt_19_23'] * 100
                                    drug_data = get_drug_data(drug, base_df)
                                    drug_cagr_vals = drug_data['cagr_avg_spnd_per_dsg_unt_19_23'].dropna()
                                    drug_cagr = drug_cagr_vals.mean() * 100 if not drug_cagr_vals.empty else 0
                                    
                                    if drug_cagr > class_cagr:
                                        comparison = "higher than"
                                    elif drug_cagr < class_cagr:
                                        comparison = "lower than"
                                    else:
                                        comparison = "similar to"
                                    
                                    response = f"{drug.title()} CAGR: {drug_cagr:.1f}% is {comparison} the {drug_class} class average of {class_cagr:.1f}%."
                                else:
                                    response = f"Therapeutic class data not available for {drug.title()}."
                            else:
                                response = "Therapeutic class analysis not available."

                    # 38. Does a high CAGR for [Drug Name] correlate with increasing claim volume?
                    elif re.search(r'cagr.*correlate.*claim volume|high cagr.*claims', q):
                        if not active_drugs:
                            response = "Please mention a drug for CAGR-claims correlation analysis."
                        else:
                            drug = active_drugs[0]
                            drug_data = get_drug_data(drug, base_df)
                            
                            if drug_data.empty:
                                response = f"No data available for {drug.title()}."
                            else:
                                # Get CAGR
                                cagr_vals = drug_data['cagr_avg_spnd_per_dsg_unt_19_23'].dropna()
                                drug_cagr = cagr_vals.mean() * 100 if not cagr_vals.empty else 0
                                
                                # Get claims trend
                                claims_trend, claims_change, claims_data = get_drug_trend_analysis(drug_data, 'tot_clms', 'claims')
                                
                                if drug_cagr > 10 and claims_trend == "increasing":
                                    correlation = "positive correlation"
                                    insight = "High growth in both spending and utilization"
                                elif drug_cagr > 10 and claims_trend == "decreasing":
                                    correlation = "divergent trend"
                                    insight = "Spending growth driven by price increases, not volume"
                                elif drug_cagr < 0 and claims_trend == "increasing":
                                    correlation = "divergent trend" 
                                    insight = "Volume growth with cost containment"
                                else:
                                    correlation = "mixed relationship"
                                    insight = "Complex market dynamics"
                                
                                response = f"{drug.title()} shows {correlation}. CAGR: {drug_cagr:.1f}%, Claims trend: {claims_trend}. {insight}."

                    # ======================================================
                    # E. FORECAST EXPLORER (41-50) - 3/10 COVERED
                    # ======================================================
                    
                    # 41. What does the forecast model predict for [Drug Name] spending in 2024?
                    elif re.search(r'predict.*2024|forecast model.*predict', q):
                        if not active_drugs:
                            response = "Please mention a drug for forecast prediction."
                        else:
                            drug = active_drugs[0]
                            if forecast_data is not None and not forecast_data.empty:
                                forecast_match = forecast_data[forecast_data['brnd_name'].str.lower() == drug.lower()]
                                if selected_manufacturer != "All":
                                    forecast_match = forecast_match[forecast_match['mftr_name'] == selected_manufacturer]
                                
                                if not forecast_match.empty:
                                    forecast_2024 = forecast_match['forecast_2024_total_spending'].sum()
                                    response = f"The forecast model predicts {drug.title()} spending will be {_fmt(forecast_2024)} in 2024."
                                else:
                                    response = f"No forecast data available for {drug.title()}."
                            else:
                                response = "Forecast data not loaded."

                    # 42. Which drugs are expected to experience the steepest spending increase next year?
                    elif re.search(r'steepest.*spending increase|drugs.*increase.*next year', q):
                        if forecast_data is not None and not forecast_data.empty:
                            # Compare forecast with 2023 actuals
                            df_2023 = base_df[base_df['year'] == 2023]
                            if not df_2023.empty:
                                # Calculate projected growth
                                forecast_growth = []
                                for drug in forecast_data['brnd_name'].unique():
                                    drug_2023 = df_2023[df_2023['brnd_name'] == drug]['tot_spndng'].sum()
                                    drug_2024 = forecast_data[forecast_data['brnd_name'] == drug]['forecast_2024_total_spending'].sum()
                                    if drug_2023 > 0:
                                        growth_pct = ((drug_2024 - drug_2023) / drug_2023) * 100
                                        forecast_growth.append((drug, growth_pct, drug_2024))
                                
                                # Get top 5 by growth percentage
                                top_growth = sorted(forecast_growth, key=lambda x: x[1], reverse=True)[:5]
                                growth_list = [f"• {drug.title()}: {growth_pct:+.1f}% growth to {_fmt(forecast)}" for drug, growth_pct, forecast in top_growth]
                                
                                response = f"**Drugs with Highest Projected 2024 Growth:**\n\n" + "\n".join(growth_list)
                            else:
                                response = "No 2023 data available for comparison."
                        else:
                            response = "Forecast data not available."

                    # 47. What's the difference between forecasted and actual 2023 spending for [Drug Name]?
                    elif re.search(r'difference.*forecast.*actual|forecast.*actual.*difference', q):
                        if not active_drugs:
                            response = "Please mention a drug for forecast accuracy analysis."
                        else:
                            drug = active_drugs[0]
                            drug_data = get_drug_data(drug, base_df)
                            drug_2023 = drug_data[drug_data['year'] == 2023]
                            
                            if not drug_2023.empty and forecast_data is not None:
                                actual_2023 = drug_2023['tot_spndng'].sum()
                                forecast_match = forecast_data[forecast_data['brnd_name'].str.lower() == drug.lower()]
                                if selected_manufacturer != "All":
                                    forecast_match = forecast_match[forecast_match['mftr_name'] == selected_manufacturer]
                                
                                if not forecast_match.empty:
                                    forecast_2024 = forecast_match['forecast_2024_total_spending'].sum()
                                    # For this demo, we'll compare 2023 actual with 2024 forecast as a proxy
                                    difference = forecast_2024 - actual_2023
                                    difference_pct = (difference / actual_2023 * 100) if actual_2023 > 0 else 0
                                    response = f"{drug.title()}: 2023 Actual: {_fmt(actual_2023)} vs 2024 Forecast: {_fmt(forecast_2024)}. Difference: {_fmt(difference)} ({difference_pct:+.1f}%)"
                                else:
                                    response = f"No forecast data available for {drug.title()}."
                            else:
                                response = f"Insufficient data for forecast comparison for {drug.title()}."

                    # ======================================================
                    # F. MANUFACTURER INSIGHTS (51-60) - 4/10 COVERED
                    # ======================================================
                    
                    # 51. Which manufacturer produces [Drug Name]?
                    elif re.search(r'manufacturer.*produces|who makes|who manufactures', q):
                        if not active_drugs:
                            response = "Please mention a drug to find its manufacturer."
                        else:
                            drug = active_drugs[0]
                            drug_data = base_df[base_df['brnd_name'].str.lower() == drug.lower()]
                            mf = drug_data['mftr_name'].dropna().unique()
                            if len(mf) > 0:
                                response = f"{drug.title()} manufacturer(s): {', '.join([m.title() for m in mf])}"
                            else:
                                response = f"No manufacturer data found for {drug.title()}."

                    # 52. What is the total spending share of this manufacturer across all its drugs?
                    elif re.search(r'spending share.*manufacturer|manufacturer.*spending share', q):
                        if not active_drugs:
                            response = "Please mention a drug to analyze manufacturer spending share."
                        else:
                            drug = active_drugs[0]
                            drug_data = base_df[base_df['brnd_name'].str.lower() == drug.lower()]
                            if not drug_data.empty:
                                manufacturers = drug_data['mftr_name'].unique()
                                if len(manufacturers) > 0:
                                    manufacturer = manufacturers[0]
                                    mfr_drugs = base_df[base_df['mftr_name'] == manufacturer]
                                    total_mfr_spending = mfr_drugs['tot_spndng'].sum()
                                    total_all_spending = base_df['tot_spndng'].sum()
                                    share_pct = (total_mfr_spending / total_all_spending * 100) if total_all_spending > 0 else 0
                                    num_drugs = mfr_drugs['brnd_name'].nunique()
                                    
                                    response = f"{manufacturer.title()} has {num_drugs} drugs with total spending of {_fmt(total_mfr_spending)}, representing {share_pct:.1f}% of overall spending."
                                else:
                                    response = f"No manufacturer data found for {drug.title()}."
                            else:
                                response = f"No data available for {drug.title()}."

                    # 53. Has the manufacturer of [Drug Name] shown consistent pricing stability?
                    elif re.search(r'manufacturer.*pricing stability|consistent pricing', q):
                        if not active_drugs:
                            response = "Please mention a drug to analyze manufacturer pricing stability."
                        else:
                            drug = active_drugs[0]
                            drug_data = base_df[base_df['brnd_name'].str.lower() == drug.lower()]
                            if not drug_data.empty:
                                manufacturers = drug_data['mftr_name'].unique()
                                if len(manufacturers) > 0:
                                    manufacturer = manufacturers[0]
                                    mfr_drugs = base_df[base_df['mftr_name'] == manufacturer]
                                    
                                    # Analyze price stability across manufacturer's portfolio
                                    price_stability = []
                                    for mfr_drug in mfr_drugs['brnd_name'].unique():
                                        drug_prices = base_df[base_df['brnd_name'] == mfr_drug]
                                        if len(drug_prices) > 1:
                                            yearly_prices = drug_prices.groupby('year')['avg_spnd_per_dsg_unt_wghtd'].mean()
                                            stability, cv = get_stability_analysis(yearly_prices.values)
                                            price_stability.append((mfr_drug, stability, cv))
                                    
                                    if price_stability:
                                        stable_drugs = len([x for x in price_stability if x[1] == "stable"])
                                        total_drugs = len(price_stability)
                                        stability_pct = (stable_drugs / total_drugs) * 100
                                        
                                        response = f"{manufacturer.title()} shows {stability_pct:.1f}% pricing stability across {total_drugs} drugs ({stable_drugs} stable, {total_drugs-stable_drugs} volatile)."
                                    else:
                                        response = f"Insufficient pricing history for {manufacturer.title()}'s portfolio."
                                else:
                                    response = f"No manufacturer data found for {drug.title()}."
                            else:
                                response = f"No data available for {drug.title()}."

                    # 58. Which manufacturer has the most high-cost outlier drugs?
                    elif re.search(r'manufacturer.*high-cost outlier|high-cost.*manufacturer', q):
                        df_2023 = base_df[base_df['year'] == 2023]
                        if not df_2023.empty:
                            # Define high-cost threshold (top 20% of unit costs)
                            cost_threshold = df_2023['avg_spnd_per_dsg_unt_wghtd'].quantile(0.8)
                            
                            # Count high-cost drugs per manufacturer
                            high_cost_drugs = df_2023[df_2023['avg_spnd_per_dsg_unt_wghtd'] >= cost_threshold]
                            mfr_high_cost = high_cost_drugs.groupby('mftr_name')['brnd_name'].nunique().nlargest(5)
                            
                            outlier_list = [f"• {mfr.title()}: {count} high-cost drugs" for mfr, count in mfr_high_cost.items()]
                            response = f"**Manufacturers with Most High-Cost Drugs (>{_fmt(cost_threshold)}):**\n\n" + "\n".join(outlier_list)
                        else:
                            response = "No 2023 cost data available."

                    # ======================================================
                    # G. ANALYTICAL 'WHY' PROMPTS (61-70) - 7/10 COVERED
                    # ======================================================
                    
                    # 61. Why has spending on [Drug Name] increased even though claim volume stayed flat?
                    elif re.search(r'why.*spending.*increased.*claim volume|spending up.*claims flat', q):
                        if not active_drugs:
                            response = "Please mention a specific drug for this analysis."
                        else:
                            drug = active_drugs[0]
                            response = f"When spending on {drug.title()} increases while claim volume stays flat, this typically indicates:\n\n• **Price increases**: Manufacturer may have raised per-unit costs\n• **Product mix shift**: Movement to higher-cost formulations or dosages\n• **Reduced rebates**: Changes in discounting or contract terms\n• **Inflation adjustments**: Annual price escalations beyond utilization changes"

                    # 62. Why is [Drug Name] classified as a high-volatility drug?
                    elif re.search(r'why.*high-volatility|high-volatility.*why', q):
                        if not active_drugs:
                            response = "Please mention a specific drug for volatility analysis."
                        else:
                            drug = active_drugs[0]
                            response = f"{drug.title()} may be classified as high-volatility due to:\n\n• **Significant price fluctuations**: Large swings in per-unit costs year-over-year\n• **Unpredictable utilization**: Irregular prescription patterns or seasonal demand\n• **Market competition changes**: New entrants or exits affecting pricing stability\n• **Regulatory impacts**: Policy changes influencing reimbursement or coverage\n• **Supply chain disruptions**: Manufacturing or distribution inconsistencies"

                    # 63. Why do certain years show sudden cost drops for [Drug Name]?
                    elif re.search(r'sudden cost drops|why.*cost.*drop', q):
                        if not active_drugs:
                            response = "Please mention a specific drug for cost drop analysis."
                        else:
                            drug = active_drugs[0]
                            response = f"Sudden cost drops for {drug.title()} may result from:\n\n• **Generic entry**: Competition from lower-cost alternatives\n• **Patent expiration**: Loss of market exclusivity enabling competition\n• **Contract renegotiations**: New pricing agreements with manufacturers\n• **Formulary changes**: Different tier placement affecting patient costs\n• **Market competition**: New therapeutic alternatives entering the market"

                    # 64. Why is [Drug Name] considered a top cost driver within its class?
                    elif re.search(r'why.*top cost driver|cost driver.*why', q):
                        if not active_drugs:
                            response = "Please mention a specific drug for cost driver analysis."
                        else:
                            drug = active_drugs[0]
                            response = f"{drug.title()} may be a top cost driver due to:\n\n• **High per-unit pricing**: Significant cost per dose/unit\n• **Large patient population**: Widespread utilization across beneficiaries\n• **Chronic condition use**: Long-term treatment requirements\n• **Limited competition**: Few therapeutic alternatives available\n• **Specialty drug status**: Complex administration or monitoring needs"

                    # 66. Why might total claims rise while total spending decreases?
                    elif re.search(r'claims rise.*spending decreases|spending down.*claims up', q):
                        response = "When claims rise while spending decreases, this typically indicates:\n\n• **Price reductions**: Significant decreases in per-unit costs\n• **Generic entry**: Lower-cost alternatives entering the market\n• **Increased rebates**: Higher manufacturer discounts or rebates\n• **Formulation changes**: Shift to lower-cost versions or dosages\n• **Contract renegotiations**: More favorable pricing agreements with manufacturers"

                    # 67. Why does one manufacturer's pricing trend differ from the industry average?
                    elif re.search(r'manufacturer.*pricing trend.*industry|pricing trend.*differ', q):
                        response = "Manufacturer pricing trends may differ from industry averages due to:\n\n• **Product portfolio mix**: Different therapeutic focus or drug types\n• **Pricing strategy**: Volume-based vs value-based pricing approaches\n• **Market position**: Branded vs generic focus affecting pricing power\n• **Contracting approach**: Different rebating and discounting strategies\n• **R&D investment**: Varying levels of innovation and patent protection"

                    # 69. Why is CAGR important when analyzing [Drug Name] performance?
                    elif re.search(r'why.*cagr important|cagr.*important.*why', q):
                        response = "CAGR (Compound Annual Growth Rate) is important because:\n\n• **Smooths volatility**: Provides a consistent annual growth measure despite yearly fluctuations\n• **Trend identification**: Helps distinguish between temporary spikes and sustained growth patterns\n• **Comparative analysis**: Enables fair comparison of growth across different drugs and time periods\n• **Forecasting basis**: Serves as input for predicting future spending trends\n• **Risk assessment**: Identifies drugs with unsustainable growth rates that may require intervention"

                    # ======================================================
                    # H. PROCUREMENT & DECISION SUPPORT (71-80) - 4/10 COVERED
                    # ======================================================
                    
                    # 71. Which drugs should procurement teams monitor for potential renegotiation?
                    elif re.search(r'renegotiation|procurement teams.*monitor', q):
                        response = "Procurement teams should monitor drugs with:\n\n• **High CAGR (>15%)**: Rapid spending growth indicates potential budget impact\n• **Significant price increases**: Above-average per-unit cost growth\n• **Limited competition**: Few alternatives may reduce negotiating leverage\n• **High volume + high cost**: Both widespread use and high pricing\n• **Patent expiration**: Opportunities for generic substitution\n• **Therapeutic class leaders**: Drugs dominating their category spending"

                    # 73. Which drugs show rising unit costs and high claim volume simultaneously?
                    elif re.search(r'rising unit costs.*high claim volume|high volume.*rising costs', q):
                        df_2023 = base_df[base_df['year'] == 2023]
                        if not df_2023.empty:
                            # Get drugs with above-average claim volume
                            avg_claims = df_2023['tot_clms'].mean()
                            high_volume_drugs = df_2023[df_2023['tot_clms'] > avg_claims]
                            
                            # Analyze cost trends for high-volume drugs
                            risk_drugs = []
                            for drug in high_volume_drugs['brnd_name'].unique():
                                drug_data = base_df[base_df['brnd_name'] == drug]
                                if len(drug_data) > 1:
                                    cost_trend, cost_change, cost_data = get_drug_trend_analysis(drug_data, 'avg_spnd_per_dsg_unt_wghtd', 'cost')
                                    if cost_trend == "increasing" and cost_change > 5:
                                        risk_drugs.append((drug, cost_change))
                            
                            if risk_drugs:
                                top_risk = sorted(risk_drugs, key=lambda x: x[1], reverse=True)[:5]
                                risk_list = [f"• {drug.title()}: {change:.1f}% cost increase" for drug, change in top_risk]
                                response = f"**High-Risk Drugs (High Volume + Rising Costs):**\n\n" + "\n".join(risk_list)
                            else:
                                response = "No drugs currently show both high volume and significant cost increases."
                        else:
                            response = "No 2023 data available for risk analysis."

                    # 76. Why should procurement teams prioritize [Drug Name] in the next budget cycle?
                    elif re.search(r'prioritize.*budget|budget.*prioritize', q):
                        if not active_drugs:
                            response = "Please mention a specific drug for prioritization analysis."
                        else:
                            drug = active_drugs[0]
                            response = f"Procurement should prioritize {drug.title()} if it shows:\n\n• **Consistent spending growth**: Year-over-year increases above category average\n• **High forecasted increase**: Significant predicted future spending growth\n• **Limited therapeutic alternatives**: Few substitution options available\n• **Contract renewal timing**: Upcoming manufacturer contract expirations\n• **Category leadership**: Top position within therapeutic class spending\n• **Risk of supply disruption**: Potential availability or manufacturing issues"

                    # 78. What insights can be derived from total spending vs. unit cost patterns?
                    elif re.search(r'spending vs.*unit cost|unit cost.*spending patterns', q):
                        response = "Analyzing spending vs. unit cost patterns reveals:\n\n• **Efficiency opportunities**: High spending with stable/low costs suggests volume management focus\n• **Pricing pressures**: Rising costs with stable spending indicates manufacturer pricing power\n• **Market shifts**: Parallel increases suggest overall market expansion\n• **Contract effectiveness**: Diverging trends may indicate contract performance issues\n• **Therapeutic class dynamics**: Class-wide patterns highlight systemic cost drivers"

                    # ======================================================
                    # I. VISUAL & INTERPRETATION PROMPTS (81-90) - 4/10 COVERED
                    # ======================================================
                    
                    # 82. What does the red dotted line in the forecast chart represent?
                    elif re.search(r'red dotted line|dotted line.*forecast', q):
                        response = "The red dotted line in forecast charts represents:\n\n• **Forecast start point**: Transition from historical actual data to projected values\n• **Model boundary**: Separation between known past performance and predicted future trends\n• **Decision reference**: Visual cue for planners to distinguish between actual and estimated spending\n• **Uncertainty indicator**: Reminder that values beyond this point are projections based on historical patterns"

                    # 83. Why is [Drug Name] displayed multiple times across manufacturers in the spending chart?
                    elif re.search(r'multiple times.*manufacturers|displayed multiple times', q):
                        response = "Drugs appear multiple times because:\n\n• **Brand vs Generic**: Different manufacturers produce same active ingredient\n• **Multiple suppliers**: Various companies manufacturing identical formulations\n• **Contract variations**: Different pricing agreements across manufacturers\n• **Regional distribution**: Various suppliers serving different geographic areas\n• **Product line extensions**: Different dosages or formulations from same manufacturer"

                    # 84. Can you clarify what the 'Total Beneficiaries' metric means in this chart?
                    elif re.search(r'total beneficiaries.*metric|beneficiaries.*meaning', q):
                        response = "**Total Beneficiaries** metric represents:\n\n• **Unique patient count**: Number of distinct Medicare patients receiving the drug\n• **Utilization breadth**: Measure of how widely the drug is prescribed across beneficiaries\n• **Patient reach**: Indicator of treatment penetration within covered population\n• **Demographic impact**: Understanding which patient groups are receiving medications"

                    # 85. How does the model calculate CAGR for [Drug Name]?
                    elif re.search(r'how.*calculate.*cagr|cagr.*calculation', q):
                        response = "CAGR calculation methodology:\n\n• **Time period**: Typically 2019-2023 for current analysis\n• **Formula**: CAGR = (Ending Value / Beginning Value)^(1/Number of Years) - 1\n• **Data source**: Uses total annual spending figures\n• **Adjustments**: Accounts for missing years through interpolation\n• **Output**: Expressed as annualized percentage growth rate"

                    # ======================================================
                    # J. CROSS-FEATURE AND SUMMARY PROMPTS (91-100) - 4/10 COVERED
                    # ======================================================
                    
                    # 91. Summarize all insights for [Drug Name] in one view.
                    elif re.search(r'summarize.*insights|summary.*view|all insights', q):
                        if not active_drugs:
                            response = "Please mention a drug for a comprehensive summary."
                        else:
                            drug = active_drugs[0]
                            drug_data = get_drug_data(drug, base_df)
                            drug_2023 = drug_data[drug_data['year'] == 2023]
                            
                            if not drug_2023.empty:
                                spending_2023 = drug_2023['tot_spndng'].sum()
                                claims_2023 = drug_2023['tot_clms'].sum()
                                beneficiaries_2023 = drug_2023['tot_benes'].sum()
                                unit_cost_2023 = drug_2023['avg_spnd_per_dsg_unt_wghtd'].mean()
                                
                                # Get CAGR
                                cagr_vals = drug_data['cagr_avg_spnd_per_dsg_unt_19_23'].dropna()
                                cagr = cagr_vals.mean() * 100 if not cagr_vals.empty else 0
                                
                                # Get forecast
                                forecast_2024 = None
                                if forecast_data is not None and not forecast_data.empty:
                                    forecast_match = forecast_data[forecast_data['brnd_name'].str.lower() == drug.lower()]
                                    if selected_manufacturer != "All":
                                        forecast_match = forecast_match[forecast_match['mftr_name'] == selected_manufacturer]
                                    if not forecast_match.empty:
                                        forecast_2024 = forecast_match['forecast_2024_total_spending'].sum()
                                
                                response = f"**{drug.title()} - Comprehensive Summary**\n\n"
                                response += f"• **2023 Spending**: {_fmt(spending_2023)}\n"
                                response += f"• **2023 Claims**: {claims_2023:,}\n"
                                response += f"• **2023 Beneficiaries**: {beneficiaries_2023:,}\n"
                                response += f"• **Avg Unit Cost**: {_fmt(unit_cost_2023)}\n"
                                response += f"• **CAGR (2019-2023)**: {cagr:.1f}%\n"
                                if forecast_2024:
                                    forecast_change = ((forecast_2024 - spending_2023) / spending_2023 * 100) if spending_2023 > 0 else 0
                                    response += f"• **2024 Forecast**: {_fmt(forecast_2024)} ({forecast_change:+.1f}% change)\n"
                                
                                # Add insights based on data
                                if cagr > 20:
                                    response += f"\n**Key Insight**: {drug.title()} shows rapid growth above 20% CAGR, indicating potential budget impact."
                                elif cagr < -10:
                                    response += f"\n**Key Insight**: {drug.title()} shows significant decline, possibly due to competition or guideline changes."
                                else:
                                    response += f"\n**Key Insight**: {drug.title()} demonstrates stable market performance."
                                    
                            else:
                                response = f"No 2023 data available for {drug.title()} summary."

                    # 92. Show spending, claims, and per-unit cost together for [Drug Name].
                    elif re.search(r'spending.*claims.*cost together|show.*all metrics', q):
                        if not active_drugs:
                            response = "Please mention a drug for multi-metric analysis."
                        else:
                            drug = active_drugs[0]
                            drug_data = get_drug_data(drug, base_df)
                            drug_2023 = drug_data[drug_data['year'] == 2023]
                            
                            if not drug_2023.empty:
                                spending_2023 = drug_2023['tot_spndng'].sum()
                                claims_2023 = drug_2023['tot_clms'].sum()
                                unit_cost_2023 = drug_2023['avg_spnd_per_dsg_unt_wghtd'].mean()
                                beneficiaries_2023 = drug_2023['tot_benes'].sum()
                                
                                response = f"**{drug.title()} - 2023 Key Metrics**\n\n"
                                response += f"• **Total Spending**: {_fmt(spending_2023)}\n"
                                response += f"• **Total Claims**: {claims_2023:,}\n"
                                response += f"• **Per-Unit Cost**: {_fmt(unit_cost_2023)}\n"
                                response += f"• **Beneficiaries**: {beneficiaries_2023:,}\n"
                                response += f"• **Claims per Beneficiary**: {claims_2023/beneficiaries_2023:.2f}" if beneficiaries_2023 > 0 else "• **Claims per Beneficiary**: N/A"
                            else:
                                response = f"No 2023 data available for {drug.title()}."

                    # 93. Highlight key insights about the top 5 cost-driving drugs.
                    elif re.search(r'top 5.*cost-driving|top.*cost drivers', q):
                        df_2023 = base_df[base_df['year'] == 2023]
                        if not df_2023.empty:
                            top_drugs = df_2023.groupby('brnd_name')['tot_spndng'].sum().nlargest(5)
                            insights = []
                            for drug, spending in top_drugs.items():
                                drug_data = base_df[base_df['brnd_name'] == drug]
                                cagr_vals = drug_data['cagr_avg_spnd_per_dsg_unt_19_23'].dropna()
                                cagr = cagr_vals.mean() * 100 if not cagr_vals.empty else 0
                                
                                if cagr > 20:
                                    insight = "rapid growth"
                                elif cagr > 0:
                                    insight = "steady growth"
                                else:
                                    insight = "declining trend"
                                
                                insights.append(f"• {drug.title()}: {_fmt(spending)} ({insight}, CAGR: {cagr:.1f}%)")
                            
                            response = "**Top 5 Cost-Driving Drugs in 2023:**\n\n" + "\n".join(insights)
                        else:
                            response = "No 2023 data available for cost driver analysis."

                    # 95. Which drugs have stable prices but rising claim volumes?
                    elif re.search(r'stable prices.*rising claims|stable cost.*increasing volume', q):
                        df_2023 = base_df[base_df['year'] == 2023]
                        if not df_2023.empty:
                            efficient_drugs = []
                            for drug in df_2023['brnd_name'].unique():
                                drug_data = base_df[base_df['brnd_name'] == drug]
                                if len(drug_data) > 1:
                                    cost_trend, cost_change, cost_data = get_drug_trend_analysis(drug_data, 'avg_spnd_per_dsg_unt_wghtd', 'cost')
                                    claims_trend, claims_change, claims_data = get_drug_trend_analysis(drug_data, 'tot_clms', 'claims')
                                    
                                    if cost_trend == "stable" and claims_trend == "increasing":
                                        efficient_drugs.append((drug, claims_change, cost_change))
                            
                            if efficient_drugs:
                                top_efficient = sorted(efficient_drugs, key=lambda x: x[1], reverse=True)[:5]
                                efficient_list = [f"• {drug.title()}: {claims_change:+.1f}% claims, {cost_change:+.1f}% cost" for drug, claims_change, cost_change in top_efficient]
                                response = f"**Efficient Drugs (Stable Cost + Growing Volume):**\n\n" + "\n".join(efficient_list)
                            else:
                                response = "No drugs currently show this efficiency pattern."
                        else:
                            response = "No data available for efficiency analysis."

                    # ======================================================
                    # COMPARISON QUESTIONS
                    # ======================================================
                    elif "compare" in q:
                        if len(active_drugs) < 2:
                            response = "Please mention two drugs to compare (e.g., 'Compare Abilify and Humira')."
                        else:
                            parts = []
                            for drug in active_drugs:
                                drug_data = get_drug_data(drug, base_df)
                                sub_2023 = drug_data[drug_data['year'] == 2023]
                                if not sub_2023.empty:
                                    spending_2023 = sub_2023['tot_spndng'].sum()
                                    claims_2023 = sub_2023['tot_clms'].sum()
                                    unit_cost_2023 = sub_2023['avg_spnd_per_dsg_unt_wghtd'].mean()
                                    parts.append(f"{drug.title()}: {_fmt(spending_2023)} spending, {claims_2023:,} claims, {_fmt(unit_cost_2023)} unit cost")
                                else:
                                    parts.append(f"{drug.title()}: no 2023 data")
                            response = "Comparison: " + " | ".join(parts)

                    # ======================================================
                    # DEFAULT HELP
                    # ======================================================
                    else:
                        response = (
                            "I can help you analyze drug spending data. Examples:\n\n"
                            "• **Forecasts**: 'What is the 2024 forecast for Abilify?'\n"
                            "• **Spending Trends**: 'How has Abilify spending changed over 5 years?'\n"
                            "• **Cost Analysis**: 'What is the per-unit cost of Abilify?'\n"
                            "• **Claims & Usage**: 'How many claims for Abilify in 2023?'\n"
                            "• **Manufacturers**: 'Who manufactures Abilify?'\n"
                            "• **Comparisons**: 'Compare Abilify and Humira'\n"
                            "• **Summaries**: 'Summarize all insights for Abilify'\n\n"
                            "Try asking about specific drugs, years, or trends!"
                        )

            except Exception as e:
                response = f"Error processing your question: {str(e)}"

            clean_resp = strip_emojis(response)
            st.session_state.chat_messages_v2.append({"role": "assistant", "content": clean_resp})
            
            # Display new response with unique key
            content_hash = hash(clean_resp) % 10000
            message_key = f"chat_resp_{len(st.session_state.chat_messages_v2)}_{content_hash}"
            message(clean_resp, key=message_key)

        # CLEAR CHAT
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Chat", key="clear_chat_btn_final"):
                st.session_state.chat_messages_v2 = []
                st.rerun()
        with col2:
            st.caption("Chat history is saved automatically.") 


# ======================================================
# ======================================================

 
# Tab 2: Top Cost Drivers
# Tab 2: Top Cost Drivers
with tab2:
    st.subheader(f"{t['top_ten_title']}{view_mode}")
    col, ylabel, _ = get_view_data(view_mode)
    # --- NEW: Data Type Check ---
    if data_type == "Forecast Data":
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

                        # Button to trigger the dialog
                        if st.button("🔍 Zoom", key=f"zoom_top_cost_drivers_{view_mode}"):
                            show_zoomed_top_cost_drivers()
                        # --- END WORKING ZOOM ---

                        # Placeholder for Collapse Chart View functionality
                        # This button could minimize the chart itself, though panel collapse might be preferred.
                        # if st.button("➖ Collapse View", key=f"collapse_chart_view_{view_mode}"):
                        #     st.info("Collapse View feature placeholder.")
                # --- END DYNAMIC CENTER PANEL & ZOOM ---

                st.markdown("---")
        else:
            st.error("Cannot display Top Cost Drivers: Data not loaded.")
    else:
        st.info("Switch to 'Forecast Data' mode to view this chart.") 

# Tab 3: CAGR & Outliers
# Tab 3: CAGR & Outliers
with tab3:
    st.subheader(t["cagr_title"])
    # --- NEW: Data Type Check ---
    if data_type == "Forecast Data":
        if df_filtered_by_class is not None:
            df_filtered_outliers = df_filtered_by_class.copy()
            if selected_manufacturer != "All":
                df_filtered_outliers = df_filtered_outliers[df_filtered_outliers['mftr_name'] == selected_manufacturer]
            if df_filtered_outliers.empty:
                st.warning(t["no_data_warning"])
            else:
                # Show ALL drugs in the filtered set if less than 10
                cagr_df = df_filtered_outliers.groupby('brnd_name').agg(cagr_avg_spnd_per_dsg_unt_19_23=('cagr_avg_spnd_per_dsg_unt_19_23', 'first')).dropna().reset_index()
                if len(cagr_df) < 10:
                    cagr_df = cagr_df.sort_values('cagr_avg_spnd_per_dsg_unt_19_23', ascending=True)
                else:
                    cagr_df = cagr_df.nlargest(10, 'cagr_avg_spnd_per_dsg_unt_19_23').reset_index()
                if not cagr_df.empty:
                    st.markdown(f"### {t['cagr_subtitle']}")
                    fig = px.bar(cagr_df, x='cagr_avg_spnd_per_dsg_unt_19_23', y='brnd_name', orientation='h', labels={'cagr_avg_spnd_per_dsg_unt_19_23': t["cagr_percent"], 'brnd_name': t["drug_column"]}, title=t['cagr_title'], template='plotly_white')
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
                            @st.dialog(f"🔍 Zoomed View - {t['cagr_title']}")
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
    else:
        st.info("Switch to 'Forecast Data' mode to view this chart.") 

# Tab 4: High-Volume Drugs
# Tab 4: High-Volume Drugs
with tab4:
    col, ylabel, title = get_view_data(view_mode)
    st.subheader(f"{t['compare_title']}{view_mode}")
    # --- NEW: Data Type Check ---
    if data_type == "Forecast Data":
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
                        'brnd_name': t["drug_name"],
                        'mftr_name': t["manufacturer_column"]
                    },
                    title=f"{t['compare_title']}{view_mode}{t['top_ten_title_other']}",
                    template="plotly_white"
                )
                # --- FIX: Professional scaling for single drug ---
                if len(selected_drugs) == 1:
                    max_val = comparison_df['value'].max()
                    fig.update_layout(yaxis_range=[0, max_val * 1.1])
                    fig.update_traces(width=0.4)
                else:
                    fig.update_layout(bargap=0.15)
                
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
                                    'brnd_name': t["drug_name"],
                                    'mftr_name': t["manufacturer_column"]
                                },
                                title=f"🔍 Zoomed View: {t['compare_title']}{view_mode}{t['top_ten_title_other']}",
                                template="plotly_white"
                            )
                            if len(selected_drugs) == 1:
                                max_val = comparison_df['value'].max()
                                fig_zoom.update_layout(yaxis_range=[0, max_val * 1.1])
                                fig_zoom.update_traces(width=0.4)
                            else:
                                fig_zoom.update_layout(bargap=0.15)
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
    else:
        st.info("Switch to 'Forecast Data' mode to view this chart.") 

# Tab 5: Procurement Intelligence (Now Tab 5, before Explainability)
# Tab 5: Procurement Intelligence (Now Tab 5, before Explainability)
with tab5:
    st.subheader(t["procurement_tab"])
    
    # --- NEW: Data Type Check ---
    if data_type == "Procurement Data":
        # Ensure data is loaded
        if df_long is not None:
            procurement_df = df_long.copy()
            
            # --- QUALITY CHECKS ---
            procurement_df['tot_dsg_unts'] = procurement_df['tot_dsg_unts'].replace(0, np.nan)
            
            # --- CREATE TWO COLUMNS ---
            left_col, right_col = st.columns([0.6, 0.4])

            # ========================
            # LEFT PANEL
            # ========================
            with left_col:
                # --- KPI 1: Total Spend Share (%) ---
                st.markdown("### 💰 Spend Share by Manufacturer")
                spend_by_mftr = procurement_df.groupby('mftr_name')['tot_spndng'].sum().sort_values(ascending=False)
                total_spend = spend_by_mftr.sum()
                spend_share = (spend_by_mftr / total_spend * 100).head(10)
                
                fig_spend_share = px.bar(
                    x=spend_share.values,
                    y=spend_share.index,
                    orientation='h',
                    labels={'x': 'Spend Share (%)', 'y': 'Manufacturer'},
                    title="Top 10 Manufacturers by Spend Share",
                    template="plotly_white"
                )
                fig_spend_share.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_spend_share, use_container_width=True)
                st.markdown("---")
                
                # --- KPI 2: Average Unit Price (AUP) Trend ---
                st.markdown("### 📈 Average Unit Price (AUP) Trend")
                procurement_df['avg_unit_price'] = procurement_df['tot_spndng'] / procurement_df['tot_dsg_unts']
                top_mftrs = spend_by_mftr.head(5).index.tolist()
                aup_data = procurement_df[procurement_df['mftr_name'].isin(top_mftrs)]
                aup_trend = aup_data.groupby(['mftr_name', 'year'])['avg_unit_price'].mean().reset_index()
                
                fig_aup = px.line(
                    aup_trend,
                    x='year',
                    y='avg_unit_price',
                    color='mftr_name',
                    labels={'avg_unit_price': 'Avg. Unit Price ($)', 'year': 'Year', 'mftr_name': 'Manufacturer'},
                    title="AUP Trend for Top 5 Manufacturers",
                    template="plotly_white"
                )
                st.plotly_chart(fig_aup, use_container_width=True)
                st.markdown("---")
                
                # --- KPI 3: Claims CAGR ---
                st.markdown("### 📊 Claims CAGR by Manufacturer")
                def calc_cagr(start, end, years):
                    return ((end/start)**(1/years) - 1) * 100 if start > 0 and years > 0 else np.nan

                claims_start = procurement_df[procurement_df['year'] == 2019].groupby('mftr_name')['tot_clms'].sum()
                claims_end = procurement_df[procurement_df['year'] == 2023].groupby('mftr_name')['tot_clms'].sum()
                claims_cagr = pd.DataFrame({
                    'claims_2019': claims_start,
                    'claims_2023': claims_end
                }).dropna()
                claims_cagr['cagr_claims'] = claims_cagr.apply(lambda row: calc_cagr(row['claims_2019'], row['claims_2023'], 4), axis=1)
                claims_cagr = claims_cagr.sort_values('cagr_claims', ascending=False).head(10)
                
                fig_cagr = px.bar(
                    claims_cagr,
                    x='cagr_claims',
                    y=claims_cagr.index,
                    orientation='h',
                    labels={'cagr_claims': 'Claims CAGR (%)', 'mftr_name': 'Manufacturer'},
                    title="Top 10 Manufacturers by Claims CAGR (2019-2023)",
                    template="plotly_white"
                )
                fig_cagr.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_cagr, use_container_width=True)
                st.markdown("---")

            # ========================
            # RIGHT PANEL
            # ========================
            with right_col:
                # --- SMART ALERTS FOR PROCUREMENT ---
                st.markdown("### 🔍 **Procurement Smart Alerts**")
                
                # --- KPI 4: Contract Renewal Risk Score ---
                # (Re-calculate risk score for the right panel)
                # Calculate AUP volatility (std dev of AUP for each manufacturer)
                aup_volatility = procurement_df.groupby('mftr_name')['avg_unit_price'].std().fillna(0)
                # Normalize AUP volatility to 0-1 scale
                aup_volatility_norm = (aup_volatility - aup_volatility.min()) / (aup_volatility.max() - aup_volatility.min()) if (aup_volatility.max() - aup_volatility.min()) != 0 else aup_volatility * 0
                
                # Calculate spend share for each manufacturer (from total spend)
                spend_by_mftr_calc = procurement_df.groupby('mftr_name')['tot_spndng'].sum()
                total_spend_calc = spend_by_mftr_calc.sum()
                spend_share_full = (spend_by_mftr_calc / total_spend_calc * 100) if total_spend_calc > 0 else spend_by_mftr_calc * 0
                # Normalize spend share to 0-1 scale
                spend_share_norm = (spend_share_full - spend_share_full.min()) / (spend_share_full.max() - spend_share_full.min()) if (spend_share_full.max() - spend_share_full.min()) != 0 else spend_share_full * 0
                
                # Calculate claims CAGR for each manufacturer
                claims_start_calc = procurement_df[procurement_df['year'] == 2019].groupby('mftr_name')['tot_clms'].sum()
                claims_end_calc = procurement_df[procurement_df['year'] == 2023].groupby('mftr_name')['tot_clms'].sum()
                claims_cagr_calc_raw = pd.DataFrame({
                    'start': claims_start_calc,
                    'end': claims_end_calc
                }).dropna()
                claims_cagr_calc_raw['cagr_claims'] = claims_cagr_calc_raw.apply(lambda row: calc_cagr(row['start'], row['end'], 4), axis=1)
                claims_cagr_full = claims_cagr_calc_raw['cagr_claims']
                # Normalize claims stability (inverse of absolute CAGR) to 0-1 scale (lower CAGR = more stable)
                if (claims_cagr_full.max() - claims_cagr_full.min()) != 0:
                    claims_stability_norm = 1 - ((abs(claims_cagr_full) - abs(claims_cagr_full).min()) / (abs(claims_cagr_full).max() - abs(claims_cagr_full).min()))
                else:
                    claims_stability_norm = pd.Series([1.0]*len(claims_cagr_full), index=claims_cagr_full.index) # Assume stable if no variance
                claims_stability_norm = claims_stability_norm.fillna(1) # If no CAGR data, assume stable
                
                # Create risk score dataframe using the normalized values
                risk_df = pd.DataFrame({
                    'volatility': aup_volatility_norm.reindex(spend_share_full.index).fillna(0),
                    'spend_share': spend_share_norm.reindex(spend_share_full.index).fillna(0),
                    'claims_stability': claims_stability_norm.reindex(spend_share_full.index).fillna(1)
                })
                # Calculate final risk score (0-1 scale)
                risk_df['risk_score'] = (
                    0.4 * risk_df['volatility'] +
                    0.25 * risk_df['spend_share'] +
                    0.15 * (1 - risk_df['claims_stability']) # Invert stability to get risk
                    # + 0.20 * shortage_rate  # Shortage data not available in current dataset
                )
                
                # Get top 5 high-risk manufacturers
                top_risk = risk_df.sort_values('risk_score', ascending=False).head(5)
                high_risk_mftrs = top_risk.index.tolist()
                
                # Generate and display smart alerts
                for mftr in high_risk_mftrs:
                    risk_val = top_risk.loc[mftr, 'risk_score']
                    # Determine risk level based on score
                    if risk_val > 0.7:
                        level = "🔴 High"
                    elif risk_val > 0.4:
                        level = "🟠 Medium"
                    else:
                        level = "🟢 Low"
                    st.warning(f"**{level} Risk:** {mftr} has a contract renewal risk score of **{risk_val:.2f}**.")
                
                st.markdown("---")
                
                # --- Contract Risk Visualization (Bar Chart as Heatmap Substitute) ---
                st.markdown("### ⚠️ **Contract Risk Score**")
                # Use a simple bar chart to visualize risk scores (a true heatmap is complex for this dataset)
                fig_risk = px.bar(
                    top_risk,
                    x='risk_score',
                    y=top_risk.index,
                    orientation='h',
                    labels={'risk_score': 'Risk Score (0-1)', 'mftr_name': 'Manufacturer'},
                    title="Top 5 High-Risk Manufacturers",
                    template="plotly_white",
                    color='risk_score',
                    color_continuous_scale='reds' # Red gradient for high risk
                )
                fig_risk.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_risk, use_container_width=True)
                
                st.markdown("---")
                
                # --- Procurement Summary Cards ---
                st.markdown("### 📌 **Procurement Summary**")
                low_risk = risk_df.sort_values('risk_score', ascending=True).head(3)
                
                st.success(f"**Most Stable:** {', '.join(low_risk.index[:2])}")
                st.warning(f"**Highest Risk:** {', '.join(high_risk_mftrs[:2])}")
                
        else:
            st.error("Procurement Intelligence data could not be loaded.")
    else:
        st.info("Switch to 'Procurement Data' mode to view this analysis.") 

# Tab 6: Model Explainability
# Tab 6: Model Explainability
with tab6:
    # --- NEW: Data Type Check ---
    if data_type == "Forecast Data":
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
    else:
        st.info("Switch to 'Forecast Data' mode to view this information.") 


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

