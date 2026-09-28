"""
DineIQ Analytics - 22 Core Feature Formulas Test Suite (SRS Step 7)
Verifies:
- Mathematical accuracy and consistency of all 22 required SRS features
- Zero impossible values (negative margins where revenue>cost, invalid ratios, out-of-bound percentages)
"""
import os
import pandas as pd
import numpy as np
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PARQUET_FEAT_DIR = os.path.join(PROJECT_ROOT, "parquet_data", "features")


def test_menu_features_formulas():
    """Verify all 11 menu-level feature formulas."""
    df = pd.read_parquet(os.path.join(PARQUET_FEAT_DIR, "menu_features.parquet"))
    assert len(df) == 150

                                                      
                                                        
    diff_margin = np.abs(df["contribution_margin"] - (df["item_revenue"] - df["cost"]))
    assert (diff_margin > 0.05).sum() == 0, "Contribution margin formula mismatch"

                                                                      
    calc_profit_pct = (df["contribution_margin"] / df["item_revenue"]) * 100
    diff_profit = np.abs(df["profit_percentage"] - calc_profit_pct)
    assert (diff_profit > 0.1).sum() == 0, "Profit percentage formula mismatch"

                                             
    assert (df["order_frequency"] <= 0).sum() == 0, "Non-positive order frequency found"
    assert (df["item_popularity"] <= 0).sum() == 0, "Non-positive item popularity found"
    assert (df["item_popularity"] < df["order_frequency"]).sum() == 0, "Popularity cannot be less than order frequency"

                                                  
    assert df["repeat_purchase_rate"].between(0.0, 1.0).all(), "Repeat purchase rate out of [0, 1]"

                                            
    assert df["average_rating"].between(1.0, 5.0).all(), "Average rating out of [1.0, 5.0]"

                                         
    assert df["rating_trend"].notna().all(), "NaNs found in rating trend"

                                                    
    assert df["wastage_percentage"].between(0.0, 100.0).all(), "Wastage percentage out of [0, 100]"

                                                        
    assert df["price_change_percentage"].notna().all(), "NaNs found in price change percentage"


def test_customer_features_formulas():
    """Verify all 11 customer-level feature formulas."""
    df = pd.read_parquet(os.path.join(PARQUET_FEAT_DIR, "customer_master_features.parquet"))
    assert len(df) == 50000

                                                         
    assert df["promotion_dependency"].between(0.0, 1.0).all(), "Promotion dependency out of [0, 1]"

                                                     
    assert df["discount_percentage"].between(0.0, 100.0).all(), "Discount percentage out of [0, 100]"

                                             
    assert (df["customer_recency"] < 0).sum() == 0, "Negative customer recency found"

                                                                
    assert (df["customer_frequency"] < 0).sum() == 0, "Negative customer frequency found"

                                                            
    assert (df["customer_monetary_value"] < 0).sum() == 0, "Negative monetary value found"

                                                                        
    active_custs = df[df["customer_frequency"] > 0]
    calc_aov = active_custs["customer_monetary_value"] / active_custs["customer_frequency"]
    diff_aov = np.abs(active_custs["average_order_value"] - calc_aov)
    assert (diff_aov > 0.05).sum() == 0, "AOV formula mismatch: monetary / frequency"

                                                        
    assert df["peak_hour_frequency"].between(0.0, 1.0).all(), "Peak hour frequency out of [0, 1]"

                                                        
    assert df["weekend_order_ratio"].between(0.0, 1.0).all(), "Weekend order ratio out of [0, 1]"

                                                                              
    assert (active_custs["location_performance"] <= 0).sum() == 0, "Location performance <= 0 found for active accounts"

                                                    
    valid_channels = {"Dine-in", "DINE_IN", "TAKEOUT", "Takeaway", "Takeout", "DELIVERY", "Delivery", "DRIVE_THRU", "Website", "App", "Website/App", "Third-party delivery"}
    assert set(df["channel_preference"].unique()).issubset(valid_channels), "Invalid channel preference"

                                                     
    assert (df["basket_size"] < 0.0).sum() == 0, "Basket size < 0.0 found"
