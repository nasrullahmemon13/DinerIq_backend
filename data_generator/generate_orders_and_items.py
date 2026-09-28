"""
Generate Orders & Order_Items Tables
Per SRS Section 2 & Hint specifications.
Generates:
- 100,000 unique orders
- 1,000,000 order-line records
Injects:
- 12-month transaction history (2025-01-01 to 2025-12-31)
- Seasonal demand (Summer surge + Q4 Holiday peak)
- Weekend patterns (Fri/Sat/Sun volume spikes)
- Peak-hour patterns (Lunch rush 11:30-14:00, Dinner rush 18:00-21:30)
- Multi-location differences (Flagship vs Express traffic & spend)
- Misleading promotions impact (Higher cancellation / lower spend)
- Injected anomalies (Duplicate records, missing values, invalid transactions)
"""
import os
import random
from datetime import datetime, date, timedelta
import numpy as np
import pandas as pd
from config import RAW_DATA_DIR, VOLUME_TARGETS, RANDOM_SEED, START_DATE, END_DATE, ensure_dir

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

def generate_orders_and_items(orders_output_path: str = None, items_output_path: str = None):
    print("=" * 60)
    print("Starting generation of Orders (100,000) & Order_Items (1,000,000)...")
    print("=" * 60)

    customers_df = pd.read_csv(os.path.join(RAW_DATA_DIR, "customers", "customers.csv"))
    restaurants_df = pd.read_csv(os.path.join(RAW_DATA_DIR, "restaurants", "restaurants.csv"))
    menu_items_df = pd.read_csv(os.path.join(RAW_DATA_DIR, "menu_items", "menu_items.csv"))
    promotions_df = pd.read_csv(os.path.join(RAW_DATA_DIR, "promotions", "promotions.csv"))

    num_orders = VOLUME_TARGETS["orders"]                  
    target_items = VOLUME_TARGETS["order_items"]             

    customer_ids = customers_df["customer_id"].tolist()
    location_ids = restaurants_df["location_id"].tolist()

    loc_weights = (restaurants_df["seating_capacity"] * restaurants_df["cost_index"]).values
    loc_weights = loc_weights / loc_weights.sum()

    start_dt = datetime(2025, 1, 1, 8, 0, 0)
    end_dt = datetime(2025, 12, 31, 23, 0, 0)
    total_days = (end_dt.date() - start_dt.date()).days + 1

    all_dates = [start_dt.date() + timedelta(days=d) for d in range(total_days)]
    date_weights = []
    for d in all_dates:
        month = d.month
        if month in [6, 7, 8]:
            season_mult = 1.30
        elif month in [11, 12]:
            season_mult = 1.45
        elif month in [1, 2]:
            season_mult = 0.85
        else:
            season_mult = 1.05

        weekday = d.weekday()
        if weekday in [4, 5, 6]:
            dow_mult = 1.40
        elif weekday in [1, 2]:
            dow_mult = 0.85
        else:
            dow_mult = 1.0

        date_weights.append(season_mult * dow_mult)

    date_weights = np.array(date_weights) / sum(date_weights)
    chosen_date_indices = np.random.choice(total_days, size=num_orders, p=date_weights)
    order_dates = [all_dates[idx] for idx in chosen_date_indices]

    hour_probs = np.array([
        0.002, 0.001, 0.000, 0.000, 0.000, 0.000,          
        0.005, 0.015, 0.035, 0.045, 0.060, 0.120,                            
        0.160, 0.110, 0.040, 0.030, 0.045, 0.080,                             
        0.135, 0.100, 0.055, 0.030, 0.015, 0.005                               
    ])
    hour_probs = hour_probs / hour_probs.sum()
    chosen_hours = np.random.choice(24, size=num_orders, p=hour_probs)
    chosen_minutes = np.random.randint(0, 60, size=num_orders)
    chosen_seconds = np.random.randint(0, 60, size=num_orders)

    chosen_locations = np.random.choice(location_ids, size=num_orders, p=loc_weights)
    chosen_cust_indices = np.random.choice(len(customer_ids), size=num_orders)

    order_types = np.random.choice(["DINE_IN", "TAKEOUT", "DELIVERY", "DRIVE_THRU"], size=num_orders, p=[0.45, 0.25, 0.20, 0.10])
    payment_methods = np.random.choice(["CREDIT_CARD", "DEBIT_CARD", "MOBILE_PAY", "CASH", "GIFT_CARD"], size=num_orders, p=[0.55, 0.20, 0.15, 0.07, 0.03])

    status_choices = ["COMPLETED", "CANCELLED", "REFUNDED", "PENDING"]
    status_weights = [0.91, 0.05, 0.02, 0.02]
    order_statuses = np.random.choice(status_choices, size=num_orders, p=status_weights)

    promo_ids = promotions_df["promotion_id"].tolist()
    promo_misleading = set(promotions_df[promotions_df["is_misleading"] == True]["promotion_id"].tolist())
    promo_choices = ["NONE"] * 65 + promo_ids * 3
    chosen_promos = [random.choice(promo_choices) for _ in range(num_orders)]

    for i in range(num_orders):
        if chosen_promos[i] in promo_misleading:
            if random.random() < 0.28:                                             
                order_statuses[i] = "CANCELLED"

    raw_item_counts = np.random.poisson(lam=9.0, size=num_orders) + 1
    raw_item_counts = np.clip(raw_item_counts, 2, 25)
    current_sum = raw_item_counts.sum()
    diff = target_items - current_sum
    if diff > 0:
        add_indices = np.random.choice(num_orders, size=diff, replace=True)
        np.add.at(raw_item_counts, add_indices, 1)
    elif diff < 0:
        sub_indices = np.random.choice(num_orders, size=abs(diff), replace=True)
        for idx in sub_indices:
            if raw_item_counts[idx] > 2:
                raw_item_counts[idx] -= 1

    final_diff = target_items - raw_item_counts.sum()
    raw_item_counts[0] += final_diff
    print(f"Allocated {raw_item_counts.sum()} total order items across {num_orders} orders.")

    item_ids = menu_items_df["item_id"].values
    item_prices = menu_items_df.set_index("item_id")["base_price"].to_dict()
    item_pop_weights = menu_items_df["popularity_weight"].values
    item_pop_weights = item_pop_weights / item_pop_weights.sum()

    print("Sampling 1,000,000 menu item selections according to popularity weights...")
    sampled_item_ids = np.random.choice(item_ids, size=target_items, p=item_pop_weights)
    sampled_quantities = np.random.choice([1, 2, 3, 4, 5], size=target_items, p=[0.70, 0.20, 0.06, 0.03, 0.01])

    print("Building order line records...")
    order_items_records = []
    order_subtotals = np.zeros(num_orders, dtype=np.float64)

    item_idx = 0
    order_id_strs = [f"ORD-{i+1:06d}" for i in range(num_orders)]

    for o_idx in range(num_orders):
        o_id = order_id_strs[o_idx]
        count = raw_item_counts[o_idx]
        o_subtotal = 0.0

        for _ in range(count):
            it_id = sampled_item_ids[item_idx]
            qty = int(sampled_quantities[item_idx])
            unit_p = float(item_prices[it_id])
            line_sub = round(qty * unit_p, 2)
            o_subtotal += line_sub

            line_id = f"LINE-{item_idx+1:07d}"
            order_items_records.append((line_id, o_id, it_id, qty, unit_p, line_sub, 0.0, line_sub))
            item_idx += 1

        order_subtotals[o_idx] = round(o_subtotal, 2)
        if (o_idx + 1) % 25000 == 0:
            print(f"  Processed {o_idx + 1}/{num_orders} orders ({item_idx} items)...")

    print("Calculating orders finances, taxes, tips, and discounts...")
    orders_records = []
    promo_dict = promotions_df.set_index("promotion_id").to_dict(orient="index")

    for o_idx in range(num_orders):
        o_id = order_id_strs[o_idx]
        cust_id = customer_ids[chosen_cust_indices[o_idx]]
        loc_id = chosen_locations[o_idx]
        o_dt = order_dates[o_idx]
        hr, mn, sc = int(chosen_hours[o_idx]), int(chosen_minutes[o_idx]), int(chosen_seconds[o_idx])
        timestamp_str = f"{o_dt.strftime('%Y-%m-%d')} {hr:02d}:{mn:02d}:{sc:02d}"

        o_type = order_types[o_idx]
        p_method = payment_methods[o_idx]
        status = order_statuses[o_idx]
        p_id = chosen_promos[o_idx]

        subtotal = float(order_subtotals[o_idx])
        discount = 0.0

        if p_id != "NONE" and p_id in promo_dict:
            p_info = promo_dict[p_id]
            if subtotal >= float(p_info["min_order_amount"]):
                if p_info["discount_type"] == "PERCENTAGE":
                    discount = min(subtotal * (float(p_info["discount_value"]) / 100.0), float(p_info["max_discount_amount"]))
                elif p_info["discount_type"] in ["FIXED_AMOUNT", "BOGO"]:
                    discount = min(float(p_info["discount_value"]), float(p_info["max_discount_amount"]))
            else:
                p_id = "NONE"                              

        discount = round(discount, 2)
        taxable = max(0.0, subtotal - discount)
        tax = round(taxable * 0.0825, 2)            

        if o_type in ["DINE_IN", "DELIVERY"] and status == "COMPLETED":
            tip_pct = random.choices([0.10, 0.15, 0.18, 0.20, 0.25], weights=[0.15, 0.35, 0.30, 0.15, 0.05])[0]
            tip = round(taxable * tip_pct, 2)
        else:
            tip = 0.0

        delivery_fee = 4.99 if o_type == "DELIVERY" else 0.0
        total = round(taxable + tax + tip + delivery_fee, 2)

        if random.random() < 0.04:
            cust_id = None

        table_num = None
        if o_type == "DINE_IN":
            table_num = None if random.random() < 0.06 else random.randint(1, 40)

        if o_idx in range(50, 200, 3):
            total = -abs(total)                  
        elif o_idx in range(300, 450, 3):
            total = 0.00

        orders_records.append({
            "order_id": o_id,
            "customer_id": cust_id,
            "location_id": loc_id,
            "order_date": o_dt.strftime("%Y-%m-%d"),
            "order_time": f"{hr:02d}:{mn:02d}:{sc:02d}",
            "order_timestamp": timestamp_str,
            "order_type": o_type,
            "order_status": status,
            "payment_method": p_method,
            "subtotal_amount": subtotal,
            "discount_amount": discount,
            "tax_amount": tax,
            "tip_amount": tip,
            "delivery_fee": delivery_fee,
            "total_amount": total,
            "promotion_id": None if p_id == "NONE" else p_id,
            "table_number": table_num
        })

    print("Converting to DataFrames...")
    df_orders = pd.DataFrame(orders_records)
    df_order_items = pd.DataFrame(
        order_items_records,
        columns=["order_item_id", "order_id", "item_id", "quantity", "unit_price", "subtotal", "item_discount", "item_total"]
    )

    print("Injecting SRS hint duplicate records for Spark deduplication tests...")
    dup_orders = df_orders.sample(n=200, random_state=RANDOM_SEED)
    df_orders = pd.concat([df_orders, dup_orders], ignore_index=True)

    dup_items = df_order_items.sample(n=1500, random_state=RANDOM_SEED)
    df_order_items = pd.concat([df_order_items, dup_items], ignore_index=True)

    if orders_output_path is None:
        orders_dir = os.path.join(RAW_DATA_DIR, "orders")
        ensure_dir(orders_dir)
        orders_output_path = os.path.join(orders_dir, "orders.csv")

    if items_output_path is None:
        items_dir = os.path.join(RAW_DATA_DIR, "order_items")
        ensure_dir(items_dir)
        items_output_path = os.path.join(items_dir, "order_items.csv")

    print(f"Writing {orders_output_path} ({len(df_orders):,} rows)...")
    df_orders.to_csv(orders_output_path, index=False, encoding="utf-8")

    print(f"Writing {items_output_path} ({len(df_order_items):,} rows)...")
    df_order_items.to_csv(items_output_path, index=False, encoding="utf-8")

    print("=" * 60)
    print(f"[SUCCESS] Orders generated: {len(df_orders):,} (inc. duplicates)")
    print(f"[SUCCESS] Order items generated: {len(df_order_items):,} (inc. duplicates)")
    print("--- Order Status Breakdown ---")
    print(df_orders["order_status"].value_counts().to_string())
    print("--- Order Type Breakdown ---")
    print(df_orders["order_type"].value_counts().to_string())
    print("=" * 60)

if __name__ == "__main__":
    generate_orders_and_items()
