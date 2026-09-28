"""
Generate Ratings Table (100,000 records)
Per SRS Section 2 & Hint specifications.
Injects:
- 100,000 rating records linked to orders, customers, items, and locations
- Rating anomalies per SRS hint:
  * Rating sentiment mismatch (e.g., 1-star rating with enthusiastic positive text, or 5-star rating with scathing complaint)
  * Duplicate bot / spam review clusters
  * Severe rating dips correlated with misleading promotion orders
  * Rating variations across restaurant locations (Flagship vs Express)
"""
import os
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from config import RAW_DATA_DIR, VOLUME_TARGETS, RANDOM_SEED, ensure_dir

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

POSITIVE_REVIEWS = [
    "Absolutely stellar experience! The signature dishes were bursting with flavor.",
    "Best meal we've had in months. Fast, courteous service and cozy ambiance.",
    "Remarkable quality! Fresh ingredients and beautifully plated presentation.",
    "Loved everything from the drinks to dessert. Will definitely be a regular!",
    "Great value, friendly waitstaff, and food arrived piping hot within 15 minutes.",
    "Incredible taste and immaculate vibe. Five stars all the way!",
    "Outstanding culinary creativity. Highly recommend to everyone in town."
]

NEGATIVE_REVIEWS = [
    "Extremely disappointed. Food arrived lukewarm and fries were soggy.",
    "Terrible customer service. Waited 45 minutes for our drinks to arrive.",
    "Overpriced and bland. The portions have shrunk noticeably.",
    "The meal was completely cold and order was missing two items!",
    "Rude staff, unclean tables, and subpar taste. Not coming back.",
    "Dish tasted stale and unseasoned. Expected much better for the price.",
    "Horrible experience from start to finish. Zero stars if I could."
]

NEUTRAL_REVIEWS = [
    "Decent food, average service. Nothing particularly memorable.",
    "Okay experience. Good location but food was slightly under-seasoned.",
    "Standard fare for this price point. Fast turnaround for lunch.",
    "Food was good, though service was a bit slow during the dinner rush.",
    "Portions were generous but flavors were fairly standard."
]

MISLEADING_PROMO_REVIEWS = [
    "Felt scammed by the advertised 50% discount! Fine print required $120 order.",
    "Misleading promotion! The discount was capped at a tiny fraction of total.",
    "Coupon refused at checkout even though the email said valid all month.",
    "Advertising says free item, but they required spending $80 first. Ridiculous!"
]

def generate_ratings(output_path: str = None) -> pd.DataFrame:
    print("=" * 60)
    print("Generating Ratings table (100,000 records) with SRS rating anomalies...")
    print("=" * 60)

    orders_df = pd.read_csv(os.path.join(RAW_DATA_DIR, "orders", "orders.csv"))
    order_items_df = pd.read_csv(os.path.join(RAW_DATA_DIR, "order_items", "order_items.csv"))

    target_ratings = VOLUME_TARGETS["ratings"]          

    sampled_orders = orders_df.sample(n=target_ratings, replace=True, random_state=RANDOM_SEED).reset_index(drop=True)

    sample_items = order_items_df.drop_duplicates(subset=["order_id"]).set_index("order_id")["item_id"].to_dict()

    ratings_records = []

    for i in range(target_ratings):
        r_id = f"RAT-{i+1:06d}"
        ord_row = sampled_orders.iloc[i]

        o_id = ord_row["order_id"]
        c_id = ord_row["customer_id"]
        loc_id = ord_row["location_id"]
        o_date = ord_row["order_date"]
        p_id = ord_row["promotion_id"]
        status = ord_row["order_status"]

        it_id = sample_items.get(o_id, "ITEM-001")

        is_promo_misleading = p_id in ["PROMO-003", "PROMO-005", "PROMO-008"]
        is_cancelled = (status == "CANCELLED")

        if is_promo_misleading and random.random() < 0.65:
            overall = 1
            food = random.choice([1, 2])
            service = 1
            ambiance = random.choice([2, 3])
            review = random.choice(MISLEADING_PROMO_REVIEWS)
            anomaly_tag = "ANOMALY_MISLEADING_PROMO_BACKLASH"
        elif is_cancelled:
            overall = 1
            food = 1
            service = 1
            ambiance = 2
            review = random.choice(NEGATIVE_REVIEWS)
            anomaly_tag = "CANCELLED_ORDER_COMPLAINT"
        else:
            overall = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.08, 0.15, 0.32, 0.40])
            food = min(5, max(1, overall + random.choice([-1, 0, 0, 1])))
            service = min(5, max(1, overall + random.choice([-1, 0, 0, 1])))
            ambiance = min(5, max(1, overall + random.choice([-1, 0, 0, 1])))

            if random.random() < 0.02:
                if overall <= 2:
                    review = random.choice(POSITIVE_REVIEWS)
                    anomaly_tag = "ANOMALY_SENTIMENT_MISMATCH_INVERTED"
                else:
                    review = random.choice(NEGATIVE_REVIEWS)
                    anomaly_tag = "ANOMALY_SENTIMENT_MISMATCH_CONTRADICTORY"
            elif overall >= 4:
                review = random.choice(POSITIVE_REVIEWS)
                anomaly_tag = "STANDARD_POSITIVE"
            elif overall == 3:
                review = random.choice(NEUTRAL_REVIEWS)
                anomaly_tag = "STANDARD_NEUTRAL"
            else:
                review = random.choice(NEGATIVE_REVIEWS)
                anomaly_tag = "STANDARD_NEGATIVE"

        review_dt = datetime.strptime(o_date, "%Y-%m-%d") + timedelta(days=random.randint(0, 2))

        ratings_records.append({
            "rating_id": r_id,
            "order_id": o_id,
            "customer_id": c_id,
            "item_id": it_id,
            "location_id": loc_id,
            "overall_rating": int(overall),
            "food_rating": int(food),
            "service_rating": int(service),
            "ambiance_rating": int(ambiance),
            "review_text": review,
            "review_date": review_dt.strftime("%Y-%m-%d"),
            "anomaly_tag": anomaly_tag
        })

    df_ratings = pd.DataFrame(ratings_records)

    dup_ratings = df_ratings[df_ratings["anomaly_tag"] == "STANDARD_POSITIVE"].sample(n=300, random_state=RANDOM_SEED)
    df_ratings = pd.concat([df_ratings, dup_ratings], ignore_index=True)

    if output_path is None:
        ratings_dir = os.path.join(RAW_DATA_DIR, "ratings")
        ensure_dir(ratings_dir)
        output_path = os.path.join(ratings_dir, "ratings.csv")

    df_ratings.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[OK] Generated {len(df_ratings):,} rating records -> {output_path}")
    print("--- Overall Rating Distribution ---")
    print(df_ratings["overall_rating"].value_counts().sort_index().to_string())
    print("--- Anomaly Distribution ---")
    print(df_ratings["anomaly_tag"].value_counts().to_string())
    return df_ratings

if __name__ == "__main__":
    generate_ratings()
