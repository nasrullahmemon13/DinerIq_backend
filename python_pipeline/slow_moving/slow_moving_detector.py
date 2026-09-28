"""
DineIQ Analytics - Standalone Slow-Moving Dish Detection Module
Implements SRS Step 32:
Identifies slow-moving dishes using the EXACT combination SRS lists:
1. Low sales volume
2. Low purchase frequency
3. Long gaps between purchases
4. Low repeat purchase
5. High wastage
6. Weak profitability
7. Poor trend
"""

from python_pipeline.menu.slow_moving_dishes import (
    SlowMovingDishDetector,
    run_slow_moving_pipeline
)

if __name__ == "__main__":
    run_slow_moving_pipeline()
