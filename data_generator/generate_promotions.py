"""
Generate Promotions Table
Per SRS Section 2 & Hint specifications.
Injects:
- Multiple promotion campaigns across the 12-month period
- Misleading promotions (unrealistic minimum spend, excluded core items, poor ROI)
- Standard promotions (Happy Hour, Weekend Flash, Loyalty, Free Item)
- Start / End dates matching 2025 calendar
"""
import os
import random
from datetime import datetime, timedelta
import pandas as pd
from config import RAW_DATA_DIR, RANDOM_SEED, START_DATE, END_DATE, ensure_dir

random.seed(RANDOM_SEED)

PROMOTION_TEMPLATES = [
    {
        "promotion_id": "PROMO-001",
        "promotion_name": "New Year Kickoff 20% Off",
        "discount_type": "PERCENTAGE",
        "discount_value": 20.0,
        "min_order_amount": 25.0,
        "max_discount_amount": 15.0,
        "applicable_category": "ALL",
        "start_date": "2025-01-01",
        "end_date": "2025-01-15",
        "target_segment": "ALL",
        "complexity_tag": "STANDARD_SEASONAL",
        "is_misleading": False,
        "description": "20% off all orders over $25 during the first two weeks of January."
    },
    {
        "promotion_id": "PROMO-002",
        "promotion_name": "Valentine Dinner Date $15 Off",
        "discount_type": "FIXED_AMOUNT",
        "discount_value": 15.0,
        "min_order_amount": 50.0,
        "max_discount_amount": 15.0,
        "applicable_category": "CAT-001",                             
        "start_date": "2025-02-10",
        "end_date": "2025-02-16",
        "target_segment": "REGULAR",
        "complexity_tag": "STANDARD_HOLIDAY",
        "is_misleading": False,
        "description": "$15 off signature entrees on Valentine's week for orders over $50."
    },
    {
        "promotion_id": "PROMO-003",
        "promotion_name": "Mega Feast 50% Off (Fine Print: $120 Min)",
        "discount_type": "PERCENTAGE",
        "discount_value": 50.0,
        "min_order_amount": 120.0,
        "max_discount_amount": 25.0,                                           
        "applicable_category": "CAT-002",
        "start_date": "2025-03-01",
        "end_date": "2025-03-31",
        "target_segment": "ALL",
        "complexity_tag": "MISLEADING_HIGH_MIN_SPEND",
        "is_misleading": True,
        "description": "Advertised as '50% Off Feast' but requires $120 min order and caps maximum savings at $25."
    },
    {
        "promotion_id": "PROMO-004",
        "promotion_name": "Spring Weekend Flash 15%",
        "discount_type": "PERCENTAGE",
        "discount_value": 15.0,
        "min_order_amount": 30.0,
        "max_discount_amount": 20.0,
        "applicable_category": "ALL",
        "start_date": "2025-04-01",
        "end_date": "2025-04-30",
        "target_segment": "ALL",
        "complexity_tag": "WEEKEND_FLASH",
        "is_misleading": False,
        "description": "15% off orders on weekend evenings throughout April."
    },
    {
        "promotion_id": "PROMO-005",
        "promotion_name": "Free Dessert Illusion (Min $80 Order)",
        "discount_type": "FIXED_AMOUNT",
        "discount_value": 8.0,
        "min_order_amount": 80.0,
        "max_discount_amount": 8.0,
        "applicable_category": "CAT-005",            
        "start_date": "2025-05-01",
        "end_date": "2025-05-31",
        "target_segment": "OCCASIONAL",
        "complexity_tag": "MISLEADING_LOW_VALUE_PERK",
        "is_misleading": True,
        "description": "Promoted as 'Free Artisan Dessert' but requires an unrealistic $80 spend on lunch/dinner."
    },
    {
        "promotion_id": "PROMO-006",
        "promotion_name": "Summer Happy Hour BOGO Drink",
        "discount_type": "BOGO",
        "discount_value": 6.5,
        "min_order_amount": 15.0,
        "max_discount_amount": 10.0,
        "applicable_category": "CAT-004",             
        "start_date": "2025-06-01",
        "end_date": "2025-08-31",
        "target_segment": "ALL",
        "complexity_tag": "PEAK_HOUR_HAPPY_HOUR",
        "is_misleading": False,
        "description": "Buy one get one free on all handcrafted beverages from 4 PM to 7 PM."
    },
    {
        "promotion_id": "PROMO-007",
        "promotion_name": "Back-to-School Family Combo $10 Off",
        "discount_type": "FIXED_AMOUNT",
        "discount_value": 10.0,
        "min_order_amount": 40.0,
        "max_discount_amount": 10.0,
        "applicable_category": "ALL",
        "start_date": "2025-09-01",
        "end_date": "2025-09-30",
        "target_segment": "REGULAR",
        "complexity_tag": "STANDARD_SEASONAL",
        "is_misleading": False,
        "description": "$10 discount on family meal combos during September."
    },
    {
        "promotion_id": "PROMO-008",
        "promotion_name": "70% Mega Sale (Hidden Exclusions)",
        "discount_type": "PERCENTAGE",
        "discount_value": 70.0,
        "min_order_amount": 60.0,
        "max_discount_amount": 12.0,                   
        "applicable_category": "CAT-007",                                   
        "start_date": "2025-10-01",
        "end_date": "2025-10-31",
        "target_segment": "ALL",
        "complexity_tag": "MISLEADING_EXCLUDED_CATEGORIES",
        "is_misleading": True,
        "description": "Advertised as 70% storewide sale, but actually only applies to low-cost side dips and capped at $12."
    },
    {
        "promotion_id": "PROMO-009",
        "promotion_name": "VIP Platinum Royalty 25% Off",
        "discount_type": "PERCENTAGE",
        "discount_value": 25.0,
        "min_order_amount": 35.0,
        "max_discount_amount": 40.0,
        "applicable_category": "ALL",
        "start_date": "2025-01-01",
        "end_date": "2025-12-31",
        "target_segment": "HIGH_VALUE",
        "complexity_tag": "VIP_EXCLUSIVE",
        "is_misleading": False,
        "description": "Year-round exclusive 25% discount for Platinum tier loyalty members."
    },
    {
        "promotion_id": "PROMO-010",
        "promotion_name": "Black Friday Week Blowout $20 Off",
        "discount_type": "FIXED_AMOUNT",
        "discount_value": 20.0,
        "min_order_amount": 60.0,
        "max_discount_amount": 20.0,
        "applicable_category": "ALL",
        "start_date": "2025-11-24",
        "end_date": "2025-11-30",
        "target_segment": "ALL",
        "complexity_tag": "PEAK_EVENT",
        "is_misleading": False,
        "description": "$20 flat discount on orders over $60 during Thanksgiving/Black Friday week."
    },
    {
        "promotion_id": "PROMO-011",
        "promotion_name": "Holiday Season Cheers 15%",
        "discount_type": "PERCENTAGE",
        "discount_value": 15.0,
        "min_order_amount": 35.0,
        "max_discount_amount": 25.0,
        "applicable_category": "ALL",
        "start_date": "2025-12-01",
        "end_date": "2025-12-31",
        "target_segment": "ALL",
        "complexity_tag": "STANDARD_SEASONAL",
        "is_misleading": False,
        "description": "15% off throughout December for holiday parties and festive dinners."
    },
    {
        "promotion_id": "PROMO-012",
        "promotion_name": "Late Night Craver 20% Off",
        "discount_type": "PERCENTAGE",
        "discount_value": 20.0,
        "min_order_amount": 20.0,
        "max_discount_amount": 15.0,
        "applicable_category": "CAT-006",                       
        "start_date": "2025-01-01",
        "end_date": "2025-12-31",
        "target_segment": "ALL",
        "complexity_tag": "LATE_NIGHT_TARGETED",
        "is_misleading": False,
        "description": "20% off all late night munchies ordered between 9 PM and 2 AM."
    }
]

def generate_promotions(output_path: str = None) -> pd.DataFrame:
    """Generate promotions table matching SRS specifications."""
    df = pd.DataFrame(PROMOTION_TEMPLATES)

    if output_path is None:
        target_dir = os.path.join(RAW_DATA_DIR, "promotions")
        ensure_dir(target_dir)
        output_path = os.path.join(target_dir, "promotions.csv")

    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[OK] Generated {len(df)} promotions -> {output_path}")
    print("--- Promotion Campaign Summary ---")
    print(df[["promotion_id", "promotion_name", "discount_type", "discount_value", "is_misleading"]].to_string())
    return df

if __name__ == "__main__":
    generate_promotions()
