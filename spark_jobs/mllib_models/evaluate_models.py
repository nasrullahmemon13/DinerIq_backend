"""
DineIQ Analytics - ML Evaluation Metrics Suite (SRS Step 12)
Provides comprehensive evaluation for binary classification and clustering:
- Accuracy, Precision, Recall, F1-Score (weighted & macro)
- ROC-AUC Score and Log Loss
- Silhouette score for unsupervised clustering
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix
)
from sklearn.metrics import silhouette_score

def evaluate_classifier(model, X_test, y_test, model_name: str) -> dict:
    """Calculates comprehensive classification metrics for a trained model."""
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test)
        y_prob = 1 / (1 + np.exp(-y_prob))
    else:
        y_prob = y_pred.astype(float)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
    rec = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    try:
        auc = float(roc_auc_score(y_test, y_prob))
    except Exception:
        auc = 0.50

    try:
        loss = float(log_loss(y_test, y_prob))
    except Exception:
        loss = 0.0

    cm = confusion_matrix(y_test, y_pred).tolist()

    return {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "f1_macro": round(f1_macro, 4),
        "roc_auc": round(auc, 4),
        "log_loss": round(loss, 4),
        "confusion_matrix": cm
    }

def evaluate_clustering(model, X_sample, k: int) -> dict:
    """Evaluates K-Means clustering model using Silhouette Score and inertia."""
    inertia = float(model.inertia_) if hasattr(model, "inertia_") else 0.0
    labels = model.labels_ if hasattr(model, "labels_") else model.predict(X_sample)

    sub_sample_size = min(3000, len(X_sample))
    idx = np.random.choice(len(X_sample), sub_sample_size, replace=False)
    sil_score = float(silhouette_score(X_sample[idx], labels[idx]))

    return {
        "model_name": f"K-Means Clustering (k={k})",
        "k_clusters": k,
        "inertia": round(inertia, 2),
        "silhouette_score": round(sil_score, 4)
    }
