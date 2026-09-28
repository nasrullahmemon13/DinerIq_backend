"""
DineIQ Analytics - Data Export Engine with Permissions Control
Implements SRS Step 50:
"Users with suitable permissions should be able to export selected analytical
results in CSV or Excel-compatible format."
"""

import io
import os
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import openpyxl

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

PERMITTED_ROLES = {"admin", "analyst", "executive", "manager", "regional_manager"}
RESTRICTED_ROLES = {"viewer", "guest", "anonymous"}

EXPORTABLE_DATASETS = {
    "menu_performance": os.path.join(PROJECT_ROOT, "processed_data", "menu_classification", "menu_classification.parquet"),
    "menu_classification": os.path.join(PROJECT_ROOT, "processed_data", "menu_classification", "menu_classification.parquet"),
    "location_performance": os.path.join(PROJECT_ROOT, "processed_data", "locations", "location_comparison_matrix.parquet"),
    "customer_segmentation": os.path.join(PROJECT_ROOT, "processed_data", "customer_segmentation", "customer_segments.parquet"),
    "wastage_loss": os.path.join(PROJECT_ROOT, "processed_data", "wastage", "wastage_by_item.parquet"),
    "recommendations": os.path.join(PROJECT_ROOT, "processed_data", "recommendations", "recommendations.parquet"),
    "price_sensitivity": os.path.join(PROJECT_ROOT, "processed_data", "pricing", "price_sensitivity_analysis.parquet"),
    "promotion_evaluations": os.path.join(PROJECT_ROOT, "processed_data", "promotion", "promotion_evaluations.parquet"),
    "anomalies": os.path.join(PROJECT_ROOT, "processed_data", "anomaly", "sales_anomalies.parquet"),
    "what_if_benchmark": os.path.join(PROJECT_ROOT, "processed_data", "what_if", "what_if_scenario_benchmark.parquet")
}


class PermissionDeniedError(Exception):
    """Raised when a user lacks export authorization."""
    pass


class DataExportEngine:
    """
    Enforces role-based export authorization and formats data into CSV or Excel workbooks.
    """

    @staticmethod
    def verify_permission(user_role: Optional[str] = "analyst") -> bool:
        """
        Validates if user role has export authorization.
        Permitted roles: admin, analyst, executive, manager.
        """
        if not user_role:
            return False
        return user_role.lower() in PERMITTED_ROLES

    @classmethod
    def export_dataset(
        cls,
        dataset_key: str,
        export_format: str = "csv",
        user_role: str = "analyst",
        filter_df: Optional[pd.DataFrame] = None,
        filters: Optional[Dict[str, str]] = None
    ) -> Tuple[bytes, str, str]:
        """
        Exports dataset in CSV or Excel format.
        Returns (content_bytes, media_type, filename).
        Raises PermissionDeniedError if user role lacks permission.
        """
        if not cls.verify_permission(user_role):
            raise PermissionDeniedError(
                f"Role '{user_role}' lacks permission to export analytical data. "
                f"Required permission: can_export_data (available to {', '.join(sorted(PERMITTED_ROLES))})."
            )

        if filter_df is not None:
            df = filter_df
            base_filename = f"dineiq_filtered_{dataset_key}"
        elif dataset_key in EXPORTABLE_DATASETS:
            path = EXPORTABLE_DATASETS[dataset_key]
            if os.path.exists(path):
                df = pd.read_parquet(path)
            else:
                raise FileNotFoundError(f"Dataset {dataset_key} has not been generated. Run its analytical pipeline first.")
            base_filename = f"dineiq_{dataset_key}"
        else:
            raise ValueError(f"Unknown dataset key: {dataset_key}. Available: {list(EXPORTABLE_DATASETS.keys())}")

        for column, value in (filters or {}).items():
            if value is None or value == "" or value.upper() == "ALL":
                continue
            if column not in df.columns:
                raise ValueError(f"Dataset {dataset_key} does not support filter {column}.")
            df = df[df[column].astype(str) == value]

        df = df.copy()
        def safe_cell(value):
            if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
                return "'" + value
            return value
        df = df.map(safe_cell)
        export_format = export_format.lower()

        if export_format == "csv":
            csv_str = df.to_csv(index=False)
            content_bytes = csv_str.encode("utf-8-sig")                                          
            media_type = "text/csv; charset=utf-8"
            filename = f"{base_filename}.csv"
            return content_bytes, media_type, filename

        elif export_format in ["excel", "xlsx"]:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                sheet = dataset_key[:30]
                df.to_excel(writer, index=False, sheet_name=sheet)
            content_bytes = buffer.getvalue()
            media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            filename = f"{base_filename}.xlsx"
            return content_bytes, media_type, filename

        else:
            raise ValueError(f"Unsupported format '{export_format}'. Supported formats: 'csv', 'excel', 'xlsx'.")

    @classmethod
    def list_exportable_datasets(cls) -> List[Dict[str, Any]]:
        """Returns catalog of exportable analytical datasets."""
        datasets = []
        for key, path in EXPORTABLE_DATASETS.items():
            exists = os.path.exists(path)
            record_count = 0
            if exists:
                try:
                    df = pd.read_parquet(path)
                    record_count = len(df)
                except Exception:
                    pass

            datasets.append({
                "dataset_key": key,
                "name": key.replace("_", " ").title(),
                "record_count": record_count,
                "supported_formats": ["csv", "excel"],
                "required_permission": "can_export_data",
                "is_available": exists
            })
        return datasets
