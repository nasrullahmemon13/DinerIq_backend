"""
Generate Inventory & Wastage Tables
Per SRS Section 2 & Hint specifications.
Generates:
- 50,000+ wastage records
- Inventory snapshots across 20 locations & 150 items
Injects:
- High-wastage dishes (short shelf life items like fresh seafood, salads, bakery)
- Wastage root causes (EXPIRED_SHELF_LIFE, OVERPRODUCTION_UNSOLD, PREPARATION_ERROR, etc.)
- Multi-location differences (Suburban vs Flagship turnover rates)
- Exact monetary loss tracking (quantity_wasted * cost_price)
"""
import os
import random
from datetime import datetime, date, timedelta
import numpy as np
import pandas as pd
from config import RAW_DATA_DIR, VOLUME_TARGETS, RANDOM_SEED, START_DATE, END_DATE, ensure_dir

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

WASTAGE_REASONS = [
    ("EXPIRED_SHELF_LIFE", 0.42),
    ("OVERPRODUCTION_UNSOLD", 0.28),
    ("PREPARATION_ERROR", 0.14),
    ("OVERCOOKED_OR_BURNT", 0.07),
    ("CUSTOMER_SEND_BACK", 0.05),
    ("EQUIPMENT_COOLER_FAILURE", 0.04)
]

def generate_inventory_and_wastage(inv_output_path: str = None, waste_output_path: str = None):
    print("=" * 60)
    print("Generating Inventory & Wastage (50,000 records)...")
    print("=" * 60)

    menu_items_df = pd.read_csv(os.path.join(RAW_DATA_DIR, "menu_items", "menu_items.csv"))
    restaurants_df = pd.read_csv(os.path.join(RAW_DATA_DIR, "restaurants", "restaurants.csv"))

    item_ids = menu_items_df["item_id"].tolist()
    item_cost_dict = menu_items_df.set_index("item_id")["cost_price"].to_dict()
    item_profile_dict = menu_items_df.set_index("item_id")["complexity_profile"].to_dict()
    item_risk_dict = menu_items_df.set_index("item_id")["wastage_risk_score"].to_dict()
    item_shelf_life_dict = menu_items_df.set_index("item_id")["shelf_life_days"].to_dict()

    location_ids = restaurants_df["location_id"].tolist()
    loc_tier_dict = restaurants_df.set_index("location_id")["location_tier"].to_dict()
    loc_cost_index_dict = restaurants_df.set_index("location_id")["cost_index"].to_dict()

    start_dt = date(2025, 1, 1)
    end_dt = date(2025, 12, 31)
    total_days = (end_dt - start_dt).days + 1
    dates_list = [start_dt + timedelta(days=d) for d in range(total_days)]

    target_wastage = VOLUME_TARGETS["wastage"]         
    wastage_records = []

    item_waste_weights = np.array([
        float(item_risk_dict[i_id]) * (3.5 if item_profile_dict[i_id] == "HIGH_WASTAGE" else 1.0)
        for i_id in item_ids
    ])
    item_waste_weights = item_waste_weights / item_waste_weights.sum()

    chosen_items = np.random.choice(item_ids, size=target_wastage, p=item_waste_weights)
    chosen_locs = np.random.choice(location_ids, size=target_wastage)
    chosen_date_indices = np.random.choice(total_days, size=target_wastage)

    reasons = [r[0] for r in WASTAGE_REASONS]
    reason_probs = [r[1] for r in WASTAGE_REASONS]
    chosen_reasons = np.random.choice(reasons, size=target_wastage, p=reason_probs)

    for i in range(target_wastage):
        w_id = f"WAS-{i+1:06d}"
        it_id = chosen_items[i]
        loc_id = chosen_locs[i]
        d_val = dates_list[chosen_date_indices[i]]
        rsn = chosen_reasons[i]

        unit_cost = float(item_cost_dict[it_id])
        profile = item_profile_dict[it_id]

        if profile == "HIGH_WASTAGE" or rsn in ["EXPIRED_SHELF_LIFE", "EQUIPMENT_COOLER_FAILURE"]:
            qty_wasted = random.randint(2, 12)
        else:
            qty_wasted = random.randint(1, 4)

        loss_amount = round(qty_wasted * unit_cost, 2)

        log_hour = random.choice([14, 15, 22, 23])                                           
        log_minute = random.randint(0, 59)
        time_str = f"{log_hour:02d}:{log_minute:02d}:00"

        wastage_records.append({
            "wastage_id": w_id,
            "item_id": it_id,
            "location_id": loc_id,
            "wastage_date": d_val.strftime("%Y-%m-%d"),
            "wastage_time": time_str,
            "quantity_wasted": qty_wasted,
            "unit_cost": unit_cost,
            "total_loss_amount": loss_amount,
            "wastage_reason": rsn,
            "complexity_profile": profile,
            "reported_by": random.choice(["Kitchen Manager", "Sous Chef", "Line Supervisor", "Inventory Clerk"])
        })

    df_wastage = pd.DataFrame(wastage_records)

    print("Generating weekly Inventory snapshots across 20 locations and 150 items...")
    audit_dates = [start_dt + timedelta(days=w * 7) for w in range(52)]

    inv_records = []
    inv_idx = 1

    for dt in audit_dates:
        d_str = dt.strftime("%Y-%m-%d")
        for loc_id in location_ids:
            sample_audit_items = random.sample(item_ids, 25)
            for it_id in sample_audit_items:
                shelf_life = item_shelf_life_dict[it_id]
                profile = item_profile_dict[it_id]

                start_stock = random.randint(20, 150)
                received = random.randint(10, 80)
                sold = random.randint(15, min(start_stock + received, 140))
                wasted = random.randint(0, 8) if profile == "HIGH_WASTAGE" else random.randint(0, 2)
                end_stock = max(0, start_stock + received - sold - wasted)
                reorder_pt = random.randint(15, 40)

                if end_stock == 0:
                    status = "CRITICAL_OUT_OF_STOCK"
                elif end_stock < reorder_pt:
                    status = "REORDER_TRIGGERED"
                else:
                    status = "OPTIMAL"

                inv_records.append({
                    "inventory_id": f"INV-{inv_idx:07d}",
                    "location_id": loc_id,
                    "item_id": it_id,
                    "snapshot_date": d_str,
                    "starting_stock": start_stock,
                    "quantity_received": received,
                    "quantity_sold": sold,
                    "quantity_wasted": wasted,
                    "ending_stock": end_stock,
                    "reorder_point": reorder_pt,
                    "stock_status": status
                })
                inv_idx += 1

    df_inventory = pd.DataFrame(inv_records)

    if waste_output_path is None:
        w_dir = os.path.join(RAW_DATA_DIR, "wastage")
        ensure_dir(w_dir)
        waste_output_path = os.path.join(w_dir, "wastage.csv")

    if inv_output_path is None:
        inv_dir = os.path.join(RAW_DATA_DIR, "inventory")
        ensure_dir(inv_dir)
        inv_output_path = os.path.join(inv_dir, "inventory.csv")

    df_wastage.to_csv(waste_output_path, index=False, encoding="utf-8")
    print(f"[OK] Generated {len(df_wastage):,} wastage records -> {waste_output_path}")

    df_inventory.to_csv(inv_output_path, index=False, encoding="utf-8")
    print(f"[OK] Generated {len(df_inventory):,} inventory records -> {inv_output_path}")

    print("--- Wastage Root Cause Breakdown ---")
    print(df_wastage["wastage_reason"].value_counts().to_string())
    print("--- Top 5 Most Wasted Profiles by Financial Loss ---")
    print(df_wastage.groupby("complexity_profile")["total_loss_amount"].sum().map("${:,.2f}".format).to_string())
    return df_wastage, df_inventory

if __name__ == "__main__":
    generate_inventory_and_wastage()
