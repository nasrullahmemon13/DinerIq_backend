"""
DineIQ Analytics - Menu Intelligence Package
Covers SRS Step 32 (Slow-Moving Dish Detection).
"""
from python_pipeline.menu.slow_moving_dishes import (
    SlowMovingDishDetector,
    run_slow_moving_pipeline
)

__all__ = [
    "SlowMovingDishDetector",
    "run_slow_moving_pipeline"
]
