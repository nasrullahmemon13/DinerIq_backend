"""
DineIQ Analytics - Ratings & Customer Satisfaction Intelligence Package
Covers SRS Step 29 (Rating and Satisfaction Analysis) and Step 30 (Rating Anomaly Detection).
"""
from python_pipeline.ratings.rating_analysis import (
    RatingSatisfactionAnalyzer,
    RatingAnomalyDetector,
    run_rating_pipeline
)

__all__ = [
    "RatingSatisfactionAnalyzer",
    "RatingAnomalyDetector",
    "run_rating_pipeline"
]
