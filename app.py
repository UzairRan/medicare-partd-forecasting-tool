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
        "smart_alerts_tab": "Smart Alerts & Insights", # New Tab Title
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
        # Rule 1: Total Spending Alert (Uses 2021 vs 2023)
        "alert_total_spending_insight": "Spending Alert: <strong>{drug_name}</strong> spending has grown <strong>{growth_rate:+.1f}%</strong> since 2021.",
        "alert_total_spending_action": "Flag this for procurement teams or formulary review.",
        # Rule 2: Per-Unit Cost Alert (Uses 2022 vs 2023)
        "alert_per_unit_cost_insight": "Pricing Review: This drug’s per-unit cost rose <strong>{growth_rate:+.1f}%</strong> since last year.",
        "alert_per_unit_cost_action": "Consider reviewing manufacturer pricing and contracts.",
        # Rule 3: High Volume Alert (Uses Claims, linked to Total Claims/Beneficiaries views)
        "alert_high_volume_insight": "Stock Management: <strong>{drug_name}</strong> was dispensed <strong>{volume_M:,.0f}</strong> times in 2023. ({claims_change:+.1f}% YoY vs 2022)",
        "alert_high_volume_action": "Use this to adjust stock levels and supply chain targets.",
        # Rule 4: High CAGR Alert (Uses CAGR > 20%, linked to a secondary insight)
        "alert_cagr_outlier_insight": "Cost Driver: <strong>{drug_name}</strong> has <strong>{cagr:.1f}%</strong> CAGR (2019-2023) — rising much faster than average.",
        "alert_cagr_outlier_action": "May become a top cost driver in future if left unmanaged.",
        "roi_no_insights": "No specific high-impact insights found for <strong>{drug_name}</strong> based on predefined thresholds and filters.",
        # New Smart Alerts Tab Text
        "smart_alerts_header": "Smart Alerts & Insights",
        "smart_alerts_intro": "",  # Removed as requested
        "smart_alerts_value_prop_header": "Value Proposition (Who Benefits & How):",
        "smart_alerts_insight_header": "Context-Aware Insights for <strong>{drug_name}</strong>:",
        "smart_alerts_insight_placeholder": "Select a single drug to see its insights.",
        "smart_alerts_yoy": "YoY Change",
        "smart_alerts_direction_up": "↑",
        "smart_alerts_direction_down": "↓",
        "smart_alerts_urgency_red": "🔴 Immediate Action Required",
        "smart_alerts_urgency_amber": "🟠 Monitor / Review",
        "smart_alerts_urgency_green": "🟢 Stable - No Action Needed",
        "smart_alerts_tooltip_base": "Baseline Year: 2021",
        "smart_alerts_tooltip_avg": "3-Year Average Dispensing: {avg_claims:,.0f}",
        "smart_alerts_tooltip_confidence": "Forecast Confidence: 95%",
        "smart_alerts_insight_color": "🔵 Insight",
        "smart_alerts_action_color": "🔴 Action",
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
        "smart_alerts_tab": "تنبيهات ورؤى ذكية", # New Tab Title
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
        # Arabic Rules
        "alert_total_spending_insight": "تنبيه الإنفاق: إنفاق <strong>{drug_name}</strong> ارتفع بنسبة <strong>{growth_rate:+.1f}%</strong> منذ 2021.",
        "alert_total_spending_action": "ضع علامة على هذا لفرق المشتريات أو مراجعة قوائم الأدوية.",
        "alert_per_unit_cost_insight": "مراجعة التسعير: ارتفعت تكلفة الوحدة لهذا الدواء بنسبة <strong>{growth_rate:+.1f}%</strong> منذ العام الماضي.",
        "alert_per_unit_cost_action": "فكر في مراجعة تسعير الشركة المصنعة والعقود.",
        "alert_high_volume_insight": "إدارة المخزون: تم صرف <strong>{drug_name}</strong> <strong>{volume_M:,.0f}</strong> مرة في عام 2023. ({claims_change:+.1f}% YoY مقابل 2022)",
        "alert_high_volume_action": "استخدم هذا لضبط مستويات المخزون وأهداف سلسلة التوريد.",
        "alert_cagr_outlier_insight": "محرك التكلفة: لدواء <strong>{drug_name}</strong> معدل نمو سنوي مركب بنسبة <strong>{cagr:.1f}%</strong> (2019-2023) - ارتفاع أسرع بكثير من المتوسط.",
        "alert_cagr_outlier_action": "قد يصبح محرك تكلفة رئيسي في المستقبل إذا ترك دون إدارة.",
        "roi_no_insights": "لم يتم العثور على رؤى عالية الأثر محددة لـ <strong>{drug_name}</strong> بناءً على العتبات والفلاتر المحددة مسبقًا.",
        # New Smart Alerts Tab Text (Arabic)
        "smart_alerts_header": "تنبيهات ورؤى ذكية",
        "smart_alerts_intro": "", # Removed as requested
        "smart_alerts_value_prop_header": "القيمة المقترحة (من يستفيد وكيف؟):",
        "smart_alerts_insight_header": "رؤى سياقية لـ <strong>{drug_name}</strong>:",
        "smart_alerts_insight_placeholder": "اختر دواءً واحدًا لرؤية رؤاه.",
        "smart_alerts_yoy": "التغيير السنوي",
        "smart_alerts_direction_up": "↑",
        "smart_alerts_direction_down": "↓",
        "smart_alerts_urgency_red": "🔴 إجراء فوري مطلوب",
        "smart_alerts_urgency_amber": "🟠 راقب / راجع",
        "smart_alerts_urgency_green": "🟢 مستقر - لا حاجة لإجراء",
        "smart_alerts_tooltip_base": "السنة الأساسية: 2021",
        "smart_alerts_tooltip_avg": "متوسط التوزيع لمدة 3 سنوات: {avg_claims:,.0f}",
        "smart_alerts_tooltip_confidence": "ثقة التنبؤ: 95%",
        "smart_alerts_insight_color": "🔵 رؤية",
        "smart_alerts_action_color": "🔴 إجراء",
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
# --- LOGO IMPLEMENTATION START ---
col1, col2 = st.columns([1, 4])
with col1:
    # Display the logo with a larger width for better visibility
    st.image("rx_veritas_logo.png", width=200)
with col2:
    st.markdown(f"<h1 style='font-size: 36px; margin-top: 10px;'>{t['title']}</h1>", unsafe_allow_html=True)
# --- LOGO IMPLEMENTATION END ---
st.markdown(f"*{t['intro']}*")

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
# -------------------------------
st.sidebar.header(t["sidebar_header"])
# Drug selection (changed to a multi-select for side-by-side comparison)
selected_drugs = st.sidebar.multiselect(t["drug_select"], drug_list, default=drug_list[4:5]) # Default to 'abacavir' for demo

# --- NEW: Therapeutic Class Filter (Above Manufacturer) ---
# Use a selectbox instead of multiselect to prevent auto-selection
selected_class = st.sidebar.selectbox(t["class_filter"], ["All"] + class_list)

# Manufacturer filter
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
    if spending_2021 > 0 and spending_2023 > 0:
        growth_rate_spending = ((spending_2023 - spending_2021) / spending_2021) * 100
    # Rule 2: Per-Unit Cost Growth (2022 vs 2023)
    cost_2022 = drug_data[drug_data['year'] == 2022]['avg_spnd_per_dsg_unt_wghtd'].mean()
    cost_2023 = drug_data[drug_data['year'] == 2023]['avg_spnd_per_dsg_unt_wghtd'].mean()
    cost_growth = 0
    if cost_2022 > 0 and cost_2023 > 0:
        cost_growth = ((cost_2023 - cost_2022) / cost_2022) * 100
    # Rule 3: High Volume (Claims) - Now includes YoY change from 2022 to 2023
    claims_2022 = drug_data[drug_data['year'] == 2022]['tot_clms'].sum()
    claims_2023 = drug_data[drug_data['year'] == 2023]['tot_clms'].sum()
    claims_change = 0
    if claims_2022 > 0 and claims_2023 > 0:
        claims_change = ((claims_2023 - claims_2022) / claims_2022) * 100
    # Rule 3b: High Beneficiaries (Beneficiaries) - Now includes YoY change from 2022 to 2023
    benes_2022 = drug_data[drug_data['year'] == 2022]['tot_benes'].sum()
    benes_2023 = drug_data[drug_data['year'] == 2023]['tot_benes'].sum()
    benes_change = 0
    if benes_2022 > 0 and benes_2023 > 0:
        benes_change = ((benes_2023 - benes_2022) / benes_2022) * 100
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
        insight_text = t["alert_total_spending_insight"].format(drug_name=drug_name.title(), growth_rate=growth_rate_spending)
        action_text = t["alert_total_spending_action"]
        # Set urgency based on growth rate
        if growth_rate_spending > 10:
            urgency_color = "red"
        elif growth_rate_spending > 0:
            urgency_color = "amber"
        else:
            urgency_color = "green"

    elif view_mode == t["per_unit_cost"] and cost_growth != 0:
        primary_alert = "per_unit_cost"
        insight_text = t["alert_per_unit_cost_insight"].format(drug_name=drug_name.title(), growth_rate=cost_growth)
        action_text = t["alert_per_unit_cost_action"]
        # Set urgency based on cost growth
        if cost_growth > 5:
            urgency_color = "red"
        elif cost_growth > 0:
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

    elif view_mode == t["total_beneficiaries"] and benes_2023 > 0:
        primary_alert = "high_beneficiaries"
        # Create a new insight text for beneficiaries
        insight_text = f"Patient Usage: <strong>{drug_name.title()}</strong> was used by <strong>{benes_2023:,.0f}</strong> beneficiaries in 2023. ({benes_change:+.1f}% YoY vs 2022)"
        action_text = "Use this to understand patient base and plan outreach or marketing."
        # Set urgency based on beneficiary count (example threshold)
        if benes_2023 > 10000:  # Example: Large patient base
            urgency_color = "red"
        elif benes_2023 > 1000:
            urgency_color = "amber"
        else:
            urgency_color = "green"

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
        tooltip_text = f"{t['smart_alerts_tooltip_base']}"
        if avg_claims > 0:
            tooltip_text += f" | {t['smart_alerts_tooltip_avg'].format(avg_claims=avg_claims)}"
        tooltip_text += f" | {t['smart_alerts_tooltip_confidence']}"

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
                <span style="font-weight: bold; color: #0066cc;">{t['smart_alerts_insight_color']}</span>
                <span style="font-weight: bold; color: #0066cc;">{insight_text}</span>
                <span title="{tooltip_text}" style="cursor: help; font-size: 0.9em; color: #666;">ℹ️</span>
            </div>
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-weight: bold; color: #cc0000;">{t['smart_alerts_action_color']}</span>
                <span style="font-weight: bold; color: #cc0000;">{action_text}</span>
                <span style="font-weight: bold; color: {'#cc0000' if urgency_color == 'red' else '#ff9900' if urgency_color == 'amber' else '#009900'};">
                    {t[f'smart_alerts_urgency_{urgency_color}']}
                </span>
            </div>
        </div>
        """
        alerts.append(alert_html)
    else:
        # No specific alert found
        alerts.append(f"<div style='color: gray; font-style: italic;'>{t['roi_no_insights'].format(drug_name=drug_name.title())}</div>")

    return alerts

# -------------------------------
# 6. Dashboard Tabs
# -------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    t["forecast_explorer_tab"],
    t["top_cost_drivers_tab"],
    t["cagr_tab"],
    t["high_volume_tab"],
    t["explainability_tab"],
    t["smart_alerts_tab"]  # New Tab
])

# --- APPLY THERAPEUTIC CLASS FILTER TO ALL TABS ---
# Filter the main DataFrame based on user selections
if df_long is not None:
    if selected_class != "All":
        df_filtered_by_class = df_long[df_long['therapeutic_class'] == selected_class].copy()
    else:
        df_filtered_by_class = df_long.copy()
else:
    df_filtered_by_class = df_long

# Tab 1: Forecast Explorer
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
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("---")

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
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("---")
    else:
        st.error("Cannot display Top Cost Drivers: Data not loaded.")

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
                st.plotly_chart(fig, use_container_width=True)
                st.markdown("---")
    else:
        st.error("Cannot display CAGR & Outliers: Data not loaded.")

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
                    'brnd_name': t["drug_name"],
                    'mftr_name': t["manufacturer_column"]
                },
                title=f"{t['compare_title']}{view_mode}{t['top_ten_title_other']}",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
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

# Tab 6: Smart Alerts & Insights (New Tab)
with tab6:
    st.subheader(t["smart_alerts_header"])
    # Check if exactly one drug is selected
    if len(selected_drugs) != 1:
        st.info(t["smart_alerts_insight_placeholder"])
        # Show general value prop even if no drug is selected
        st.markdown(f"### {t['smart_alerts_value_prop_header']}")
        st.markdown(f"""
        - {t['benefit_hospital']}
        - {t['benefit_payer']}
        - {t['benefit_pharmacy']}
        - {t['benefit_government']}
        """)
    else:
        selected_drug_name = selected_drugs[0]
        # Do NOT show the redundant heading "Context-Aware Insights for <strong>Abacavir</strong>:"
        # Instead, directly show the alert
        # Filter data for the selected drug
        df_filtered = df_filtered_by_class[df_filtered_by_class['brnd_name'].isin(selected_drugs)].copy()
        if selected_manufacturer != "All":
            df_filtered = df_filtered[df_filtered['mftr_name'] == selected_manufacturer]
        # Generate and display alerts
        alerts = generate_smart_alerts(selected_drug_name, df_filtered, t, view_mode)
        for alert in alerts:
            st.markdown(alert, unsafe_allow_html=True)

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