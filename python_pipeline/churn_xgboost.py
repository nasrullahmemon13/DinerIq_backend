"""
DineIQ Analytics - Independent XGBoost & Scikit-learn Churn Pipeline (SRS Step 13)
Independently processes underlying restaurant records in pure Pandas/NumPy.
Enforces the explicit rule:
"Spark-generated predictions must not simply be exported and reused as Python results"
Builds standalone XGBoost Classifier and benchmarks against Scikit-Learn ensembles.
"""
import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
MODELS_PYTHON_DIR = os.path.join(PROJECT_ROOT, "models", "python")

def extract_independent_customer_features(orders_df: pd.DataFrame, customers_df: pd.DataFrame) -> pd.DataFrame:
    """
    Independently aggregates raw order events into customer behavioral profiles
    using pure Pandas & NumPy without relying on Spark feature marts.
    """
    print("[Pandas/NumPy] Independently computing customer behavioral aggregates...")
    orders = orders_df[orders_df["customer_id"] != "CUST-GUEST"].copy()
    orders["order_date_dt"] = pd.to_datetime(orders["order_date"])
    orders["is_weekend"] = orders["order_date_dt"].dt.dayofweek.isin([5, 6]).astype(int)
    orders["is_promo"] = (orders["promotion_id"].notnull() & (orders["promotion_id"] != "")).astype(int)

    reference_date = orders["order_date_dt"].max()

    cust_agg = orders.groupby("customer_id").agg(
        last_order=("order_date_dt", "max"),
        total_orders=("order_id", "nunique"),
        total_spend=("total_amount", "sum"),
        avg_order_value=("total_amount", "mean"),
        avg_discount=("discount_amount", "mean"),
        promo_usage_ratio=("is_promo", "mean"),
        weekend_order_ratio=("is_weekend", "mean")
    ).reset_index()

    cust_agg["recency_days"] = (reference_date - cust_agg["last_order"]).dt.days

    cust_merged = pd.merge(customers_df, cust_agg, on="customer_id", how="left")
    cust_merged["recency_days"] = cust_merged["recency_days"].fillna(365)
    cust_merged["total_orders"] = cust_merged["total_orders"].fillna(0)
    cust_merged["total_spend"] = cust_merged["total_spend"].fillna(0.0)
    cust_merged["avg_order_value"] = cust_merged["avg_order_value"].fillna(0.0)
    cust_merged["avg_discount"] = cust_merged["avg_discount"].fillna(0.0)
    cust_merged["promo_usage_ratio"] = cust_merged["promo_usage_ratio"].fillna(0.0)
    cust_merged["weekend_order_ratio"] = cust_merged["weekend_order_ratio"].fillna(0.0)
    cust_merged["loyalty_points"] = pd.to_numeric(cust_merged["loyalty_points"], errors="coerce").fillna(0)
    cust_merged["churn_risk_score"] = pd.to_numeric(cust_merged["churn_risk_score"], errors="coerce").fillna(0.5)

    cust_merged["is_churned"] = (
        (cust_merged["recency_days"] >= 180) | 
        (cust_merged["churn_risk_score"] >= 0.50)
    ).astype(int)

    return cust_merged

def train_xgboost_churn_model(orders_df: pd.DataFrame, customers_df: pd.DataFrame) -> dict:
    """
    Trains and benchmarks XGBoost against Scikit-Learn classifiers.
    Saves the standalone Python model artifact to models/python/.
    """
    os.makedirs(MODELS_PYTHON_DIR, exist_ok=True)
    dataset = extract_independent_customer_features(orders_df, customers_df)

    feature_cols = [
        "recency_days",
        "total_orders",
        "total_spend",
        "avg_order_value",
        "avg_discount",
        "promo_usage_ratio",
        "weekend_order_ratio",
        "loyalty_points"
    ]

    X = dataset[feature_cols].copy()
    y = dataset["is_churned"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"[XGBoost] Training XGBClassifier on {len(X_train):,} training records...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=80,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="logloss",
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

    xgb_metrics = {
        "model_name": "XGBoost Classifier (Standalone Python)",
        "accuracy": round(float(accuracy_score(y_test, xgb_preds)), 4),
        "precision": round(float(precision_score(y_test, xgb_preds, average="weighted", zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, xgb_preds, average="weighted", zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, xgb_preds, average="weighted", zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, xgb_probs)), 4),
        "confusion_matrix": confusion_matrix(y_test, xgb_preds).tolist()
    }

    print("[Scikit-Learn] Training baseline RandomForestClassifier...")
    rf_model = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_probs = rf_model.predict_proba(X_test)[:, 1]

    rf_metrics = {
        "model_name": "Scikit-Learn Random Forest (Baseline)",
        "accuracy": round(float(accuracy_score(y_test, rf_preds)), 4),
        "precision": round(float(precision_score(y_test, rf_preds, average="weighted", zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, rf_preds, average="weighted", zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, rf_preds, average="weighted", zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, rf_probs)), 4)
    }

    importances = xgb_model.feature_importances_
    feat_imp_df = pd.DataFrame({
        "feature_name": feature_cols,
        "importance_gain": importances.round(4)
    }).sort_values(by="importance_gain", ascending=False).reset_index(drop=True)

    model_save_path = os.path.join(MODELS_PYTHON_DIR, "xgb_churn_model.joblib")
    joblib.dump(xgb_model, model_save_path)
    print(f"  [SAVED] Standalone XGBoost Model: {model_save_path}")

    return {
        "xgb_metrics": xgb_metrics,
        "rf_baseline_metrics": rf_metrics,
        "feature_importances": feat_imp_df.to_dict(orient="records"),
        "model_path": model_save_path,
        "total_test_samples": len(y_test)
    }
