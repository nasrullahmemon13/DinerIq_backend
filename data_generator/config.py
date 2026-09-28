"""
DineIQ Analytics - Data Generator Configuration & Shared Constants
Per SRS v1.0 specifications and hint requirements.
"""
import os
from datetime import datetime, date

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "raw_data")

VOLUME_TARGETS = {
    "restaurants": 20,
    "menu_categories": 10,
    "menu_items": 150,
    "customers": 50000,
    "orders": 100000,
    "order_items": 1000000,
    "ratings": 100000,
    "wastage": 50000,
}

START_DATE = date(2025, 1, 1)
END_DATE = date(2025, 12, 31)

RANDOM_SEED = 42

def ensure_dir(path: str):
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)
