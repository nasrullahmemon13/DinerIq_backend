"""
DineIQ Analytics - Master Python Pipeline Runner (SRS Step 13)
Coordinates all independent data science components:
 1. Econometric Price Elasticity of Demand (Statsmodels OLS)
 2. Inferential Hypothesis Testing (SciPy t-test, ANOVA, Chi-Square)
 3. Time-Series Sales Forecasting & Seasonal Decomposition (Statsmodels ARIMA)
 4. Customer Churn Prediction with Gradient Boosting (XGBoost & Scikit-learn)

Enforces SRS Explicit Rule:
"Spark-generated predictions must not simply be exported and reused as Python results"
Reads strictly from operational data stores and builds independent models.
"""
import os
import sys
import time
import json
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "python_pipeline")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "python")

if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from data_loader import load_operational_data
from statistical_analysis import analyze_price_elasticity, perform_hypothesis_tests
from sales_forecaster import build_sales_forecast_model
from churn_xgboost import train_xgboost_churn_model

def execute_python_pipeline():
    start_time = time.time()
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("=" * 80)
    print("DineIQ Analytics - SRS Step 13: Independent Python Data Science Pipeline")
    print("Libraries: Pandas, NumPy, Scikit-learn, XGBoost, SciPy, Statsmodels")
    print("=" * 80)

    print("\n>>> Phase 1: Ingesting Raw/Cleaned Parquet Operational Datasets...")
    t0 = time.time()
    data = load_operational_data()
    orders = data["orders"]
    order_items = data["order_items"]
    customers = data["customers"]
    restaurants = data["restaurants"]
    print(f"[Ingestion OK] Ingested {len(orders):,} orders, {len(order_items):,} items, {len(customers):,} customers in {time.time() - t0:.2f}s")

    print("\n>>> Phase 2: Econometric Modeling — Price Elasticity of Demand (Statsmodels)...")
    elasticity_results = analyze_price_elasticity(orders, order_items)
    print(f"  Estimated Elasticity (Ed): {elasticity_results['price_elasticity_coefficient']} ({elasticity_results['elasticity_category']})")
    print(f"  Statistical Significance: p-value = {elasticity_results['p_value']} (Significant: {elasticity_results['is_statistically_significant']})")

    print("\n>>> Phase 3: Inferential Hypothesis Testing (SciPy)...")
    hypo_results = perform_hypothesis_tests(orders, restaurants)
    ttest = hypo_results["two_sample_ttest_weekend_vs_weekday"]
    anova = hypo_results["anova_location_tiers"]
    chi2 = hypo_results["chi_square_channel_payment"]
    print(f"  Weekend vs Weekday t-test: t={ttest['t_statistic']}, p={ttest['p_value']:.4e} (Reject H0: {ttest['reject_null_at_5pct']})")
    print(f"  Location Tier ANOVA: F={anova['f_statistic']}, p={anova['p_value']:.4e} (Reject H0: {anova['reject_null_at_5pct']})")
    print(f"  Channel vs Payment Chi2: chi2={chi2['chi2_statistic']}, p={chi2['p_value']:.4e} (Reject H0: {chi2['reject_null_at_5pct']})")

    print("\n>>> Phase 4: Time-Series Sales Forecasting (Statsmodels ARIMA)...")
    ts_results = build_sales_forecast_model(orders)
    forecast_df = ts_results.pop("forecast_df")
    forecast_csv = os.path.join(REPORTS_DIR, "sales_forecast_30d.csv")
    forecast_df.to_csv(forecast_csv, index=False)
    print(f"  Holdout MAPE: {ts_results['holdout_evaluation_30d']['mape_pct']}% | RMSE: ${ts_results['holdout_evaluation_30d']['rmse']}")
    print(f"  30-Day Total Projected Revenue: ${ts_results['forecast_30d_summary']['total_projected_30d_revenue']:,.2f}")
    print(f"  [SAVED] 30-Day Projections: {forecast_csv}")

    print("\n>>> Phase 5: Customer Retention Modeling (XGBoost & Scikit-learn)...")
    churn_results = train_xgboost_churn_model(orders, customers)
    xgb_m = churn_results["xgb_metrics"]
    rf_m = churn_results["rf_baseline_metrics"]
    print(f"  XGBoost Accuracy: {xgb_m['accuracy']:.4f} | F1-Score: {xgb_m['f1_score']:.4f} | ROC-AUC: {xgb_m['roc_auc']:.4f}")
    print(f"  RandomForest Baseline F1: {rf_m['f1_score']:.4f} | ROC-AUC: {rf_m['roc_auc']:.4f}")

    print("\n>>> Phase 6: Compiling Independent Python Pipeline Reports...")
    pipeline_report_data = {
        "execution_timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "libraries_utilized": ["Pandas", "NumPy", "Scikit-learn", "XGBoost", "SciPy", "Statsmodels"],
        "price_elasticity": {k: v for k, v in elasticity_results.items() if k != "ols_summary_str"},
        "hypothesis_tests": hypo_results,
        "time_series_forecasting": ts_results,
        "churn_xgboost": {
            "xgb_metrics": xgb_m,
            "rf_baseline_metrics": rf_m,
            "top_features": churn_results["feature_importances"][:5]
        },
        "artifacts_persisted": [
            churn_results["model_path"],
            forecast_csv
        ]
    }

    json_report_path = os.path.join(REPORTS_DIR, "python_pipeline_report.json")
    with open(json_report_path, "w", encoding="utf-8") as f:
        json.dump(pipeline_report_data, f, indent=2)

    md_report_path = os.path.join(REPORTS_DIR, "python_pipeline_report.md")
    with open(md_report_path, "w", encoding="utf-8") as f:
        f.write("# DineIQ Analytics - Step 13: Independent Python Data Science Pipeline\n\n")
        f.write(f"**Execution Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Compliance Rule:** Independent execution using Pandas, NumPy, Scikit-learn, XGBoost, SciPy, Statsmodels. No Spark predictions reused.\n\n")

        f.write("## 1. Econometric Price Elasticity of Demand (Statsmodels OLS)\n\n")
        f.write(f"- **Demand Model Specification:** `ln(Q) = beta_0 + beta_1 * ln(P) + beta_2 * Promo + epsilon`\n")
        f.write(f"- **Estimated Elasticity ($E_d$):** `{elasticity_results['price_elasticity_coefficient']}`\n")
        f.write(f"- **Elasticity Regime:** {elasticity_results['elasticity_category']}\n")
        f.write(f"- **p-value:** `{elasticity_results['p_value']}` (Statistically Significant: `{elasticity_results['is_statistically_significant']}`)\n")
        f.write(f"- **Promotional Lift Coefficient:** `{elasticity_results['promo_lift_coefficient']}`\n")
        f.write(f"- **Model $R^2$:** `{elasticity_results['r_squared']}` across {elasticity_results['total_observations']:,} weekly item observations\n\n")

        f.write("## 2. Inferential Hypothesis Testing (SciPy)\n\n")
        f.write("| Hypothesis Test | Test Method | Test Statistic | p-value | Decision ($\alpha=0.05$) |\n")
        f.write("|---|---|---:|---:|---|\n")
        f.write(f"| Weekend vs Weekday Spend | Welch's t-test | t = {ttest['t_statistic']} | {ttest['p_value']:.4e} | {'Reject H0 (Statistically Significant)' if ttest['reject_null_at_5pct'] else 'Fail to Reject H0'} |\n")
        f.write(f"| Location Tier Spend Variance | One-Way ANOVA | F = {anova['f_statistic']} | {anova['p_value']:.4e} | {'Reject H0 (Statistically Significant)' if anova['reject_null_at_5pct'] else 'Fail to Reject H0'} |\n")
        f.write(f"| Channel vs Payment Independence | Pearson's Chi-Square | $\\\\chi^2$ = {chi2['chi2_statistic']} | {chi2['p_value']:.4e} | {'Reject H0 (Statistically Significant)' if chi2['reject_null_at_5pct'] else 'Fail to Reject H0'} |\n\n")

        f.write("## 3. Time-Series Sales Forecasting (Statsmodels ARIMA)\n\n")
        f.write(f"- **Model Architecture:** ARIMA(1, 1, 1) with 7-Day Day-of-Week Seasonality\n")
        f.write(f"- **Historical Horizon:** {ts_results['total_historical_days']} daily points\n")
        f.write(f"- **Holdout Forecast Accuracy (30-day):** MAPE = `{ts_results['holdout_evaluation_30d']['mape_pct']}%`, RMSE = `${ts_results['holdout_evaluation_30d']['rmse']}`\n")
        f.write(f"- **Projected 30-Day Revenue:** `${ts_results['forecast_30d_summary']['total_projected_30d_revenue']:,.2f}` (Mean daily: `${ts_results['forecast_30d_summary']['mean_projected_daily_revenue']:,.2f}`)\n")
        f.write(f"- **Forecast File:** [`reports/python_pipeline/sales_forecast_30d.csv`](sales_forecast_30d.csv)\n\n")

        f.write("## 4. Machine Learning Customer Churn Pipeline (XGBoost & Scikit-learn)\n\n")
        f.write("| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        f.write(f"| **XGBoost Classifier** | **{xgb_m['accuracy']:.4f}** | **{xgb_m['precision']:.4f}** | **{xgb_m['recall']:.4f}** | **{xgb_m['f1_score']:.4f}** | **{xgb_m['roc_auc']:.4f}** |\n")
        f.write(f"| Scikit-Learn Random Forest | {rf_m['accuracy']:.4f} | {rf_m['precision']:.4f} | {rf_m['recall']:.4f} | {rf_m['f1_score']:.4f} | {rf_m['roc_auc']:.4f} |\n\n")
        f.write(f"- **Top Predictive Features:**\n")
        for i, row in enumerate(churn_results["feature_importances"][:5], 1):
            f.write(f"  {i}. `{row['feature_name']}` (Gain: {row['importance_gain']:.4f})\n")

    print(f"\n[OK] Reports compiled to {md_report_path} and {json_report_path}")
    print(f"Independent Python Pipeline Completed in {time.time() - start_time:.2f} seconds!")
    print("=" * 80)
    return pipeline_report_data

if __name__ == "__main__":
    execute_python_pipeline()
