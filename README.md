# 💊 Medicare Part D Drug Spending Forecasting Tool

# AI-Powered Forecasting for Future Healthcare Trends


## The Problem

Current CMS Medicare tools are retrospective they show historical data but lack predictive capabilities.

This leaves stakeholders reacting to cost spikes and utilization shifts without foresight. This idea solution transforms static reports into forward-looking intelligence



## Data Source
All data is sourced from the Centers for Medicare & Medicaid Services (CMS),  the official U.S. government repository for healthcare spending, utilization, and prescription drug coverage under Medicare Part D.

CMS: https://www.cms.gov/


## Methodology: Data Science Lifecycle

This project follows the full data science lifecycle to build a scalable, production-ready forecasting system:


**1- Problem Definition:**  Identify gaps in Medicare forecasting tools


**2- Data Acquisition:**  Ingest raw CMS datasets


**3- Data Preparation:**  Clean, reshape, and standardize for time-series modeling


**4- Exploratory Analysis:**  Analyze top spenders, growth trends, and anomalies


**5- Feature Engineering:**  Create lag variables, rolling averages, and log transforms


**6- Modeling:**  Train LightGBM with Optuna tuning for high accuracy


**7- Evaluation:**  Validate with MAE, RMSE, and MAPE metrics


**8- Deployment Preparation:** Modularize code, add validation, and monitoring


**9- Dashboard Development:**  Build an interactive Streamlit app for stakeholders


**10- Documentation & Versioning:**  Maintain a clean GitHub repo with a  reproducible structure



# MVP: Streamlit Dashboard Features

Interactive dashboard turns raw data into actionable insights:


**Drug-Level Forecast Explorer:**  View historical trends + 2024 predictions


**Top Cost Drivers:**  Compare 2023 vs 2024 spending


**CAGR & Outliers:**  Spot fast-growing drugs and price spikes


**High-Volume Drugs:** Track most prescribed medications


 **Model Explainability:** SHAP-ready insights into forecast drivers
 

 **Dynamic Filters:**  Slice by drug, manufacturer, and year


**💾Export Data:**  Download forecasts as CSV


# ✅ Pre-Deployment Best Practices

Ensured code quality and reliability with:


**📁Modular Codebase** (src/ingest.py, preprocess.py, features.py, model.py)


 **Data Validation**  Catch missing values and schema issues
 

 **Unit Tests**  Verify feature logic and transformations
 

 **Prediction Monitoring**  Log outputs for drift detection
 

 **Reusable Pipeline**  Encapsulated in DrugSpendingPredictor class
 

 **Production-Ready Structure**  Clean, documented, and scalable


# Future Enhancements

We’re building toward a next gen healthcare forecasting platform:


**SHAP Explainability:**  Visualize feature impact on predictions


 **RxNorm Integration:**  Map drugs to therapeutic classes


 **CMS API Automation:** Auto-refresh data quarterly


 **Regional Filtering:** Break down by state or provider region


 **PDF Reports:**  Auto-generate summaries for stakeholders


 **Dockerization:**  Containerize pipeline for scalable deployment



# 🛠️ Tech Stack

**Python** | Colab Notebook | Pandas, NumPy, LightGBM, Optuna 


**Streamlit** | Interactive Dashboard


**GitHub** | Version Control & Collaboration


**Streamlit Community Cloud** | Deployment 


# 🎯 Why This Matters

This tool shifts Medicare analytics from "What happened?" to "What’s next?"  empowering policymakers, insurers, and providers to anticipate costs, spot outliers, and act before spending spirals.


Built with care, validated rigorously, and designed for impact.





