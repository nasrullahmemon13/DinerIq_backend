"""
DineIQ Analytics - Slow-Moving Dish Detection Package (SRS Step 32)
"""
from python_pipeline.menu.slow_moving_dishes import (
    SlowMovingDishDetector,
    run_slow_moving_pipeline
)

__all__ = [
    "SlowMovingDishDetector",
    "run_slow_moving_pipeline"
]
