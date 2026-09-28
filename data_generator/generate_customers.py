"""
Generate Customers Table (50,000 customers)
Per SRS Section 2 & Hint specifications.
Injects:
- High-value customers (VIP, high spend/points)
- Churned customers (inactive, high churn risk score)
- New customers (recent signups)
- Missing values (realistic nulls in email/phone/loyalty)
"""
import os
import random
from datetime import datetime, timedelta
import pandas as pd
from faker import Faker
from config import RAW_DATA_DIR, VOLUME_TARGETS, RANDOM_SEED, START_DATE, END_DATE, ensure_dir

fake = Faker()
Faker.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

def generate_customers(output_path: str = None) -> pd.DataFrame:
    """Generate 50,000 customers with realistic segments and anomaly profiles."""
    count = VOLUME_TARGETS["customers"]
    records = []

    locations = [f"LOC-{i+1:03d}" for i in range(VOLUME_TARGETS["restaurants"])]
    segments = ["HIGH_VALUE", "REGULAR", "OCCASIONAL", "NEW", "CHURNED"]
    segment_weights = [0.10, 0.40, 0.25, 0.12, 0.13]

    start_dt = datetime.combine(START_DATE, datetime.min.time()) - timedelta(days=730)                            
    end_dt = datetime.combine(END_DATE, datetime.min.time())
    total_days = (end_dt - start_dt).days

    for i in range(count):
        cust_id = f"CUST-{i+1:05d}"
        segment = random.choices(segments, weights=segment_weights)[0]

        if segment == "NEW":
            signup_days = random.randint(total_days - 60, total_days)
            loyalty_tier = "BRONZE"
            points = random.randint(0, 250)
            churn_risk = round(random.uniform(0.1, 0.3), 3)
            is_active = True
        elif segment == "HIGH_VALUE":
            signup_days = random.randint(0, total_days - 120)
            loyalty_tier = random.choices(["GOLD", "PLATINUM"], weights=[0.4, 0.6])[0]
            points = random.randint(2500, 15000)
            churn_risk = round(random.uniform(0.02, 0.15), 3)
            is_active = True
        elif segment == "CHURNED":
            signup_days = random.randint(0, total_days - 180)
            loyalty_tier = random.choices(["BRONZE", "SILVER"], weights=[0.7, 0.3])[0]
            points = random.randint(50, 600)
            churn_risk = round(random.uniform(0.80, 0.99), 3)
            is_active = False
        else:                        
            signup_days = random.randint(0, total_days - 30)
            loyalty_tier = random.choices(["BRONZE", "SILVER", "GOLD"], weights=[0.5, 0.35, 0.15])[0]
            points = random.randint(100, 2400)
            churn_risk = round(random.uniform(0.20, 0.65), 3)
            is_active = True

        signup_date = (start_dt + timedelta(days=signup_days)).strftime("%Y-%m-%d")

        email = None if random.random() < 0.03 else fake.email()
        phone = None if random.random() < 0.04 else fake.phone_number()

        record = {
            "customer_id": cust_id,
            "first_name": fake.first_name(),
            "last_name": fake.last_name(),
            "email": email,
            "phone_number": phone,
            "customer_segment": segment,
            "loyalty_tier": loyalty_tier,
            "loyalty_points": points,
            "signup_date": signup_date,
            "preferred_location_id": random.choice(locations),
            "is_active": is_active,
            "churn_risk_score": churn_risk
        }
        records.append(record)

    df = pd.DataFrame(records)

    if output_path is None:
        target_dir = os.path.join(RAW_DATA_DIR, "customers")
        ensure_dir(target_dir)
        output_path = os.path.join(target_dir, "customers.csv")

    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[OK] Generated {len(df)} customers -> {output_path}")
    print("--- Customer Segment Distribution ---")
    print(df["customer_segment"].value_counts().to_string())
    print("--- Missing Values ---")
    print(df.isnull().sum()[df.isnull().sum() > 0].to_string())
    return df

if __name__ == "__main__":
    generate_customers()
