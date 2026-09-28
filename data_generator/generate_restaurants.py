"""
Generate Restaurants Table (20 locations)
Per SRS Section 2 & Hint specifications.
"""
import os
import random
import pandas as pd
from faker import Faker
from config import RAW_DATA_DIR, VOLUME_TARGETS, RANDOM_SEED, ensure_dir

fake = Faker()
Faker.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

RESTAURANT_PRESETS = [
    {"name": "DineIQ Flagship Downtown", "city": "New York", "state": "NY", "tier": "Flagship", "capacity": 220, "cost_index": 1.35, "has_dt": False, "has_patio": True},
    {"name": "DineIQ Financial Center", "city": "New York", "state": "NY", "tier": "Urban Corporate", "capacity": 180, "cost_index": 1.30, "has_dt": False, "has_patio": False},
    {"name": "DineIQ Brooklyn Heights", "city": "Brooklyn", "state": "NY", "tier": "Suburban Upscale", "capacity": 140, "cost_index": 1.15, "has_dt": False, "has_patio": True},
    {"name": "DineIQ Chicago Magnificent Mile", "city": "Chicago", "state": "IL", "tier": "Flagship", "capacity": 200, "cost_index": 1.25, "has_dt": False, "has_patio": True},
    {"name": "DineIQ West Loop Bistro", "city": "Chicago", "state": "IL", "tier": "Urban Dining", "capacity": 160, "cost_index": 1.20, "has_dt": False, "has_patio": True},
    {"name": "DineIQ Austin Downtown", "city": "Austin", "state": "TX", "tier": "Urban Dining", "capacity": 175, "cost_index": 1.10, "has_dt": False, "has_patio": True},
    {"name": "DineIQ Austin Domain Drive-Thru", "city": "Austin", "state": "TX", "tier": "Drive-Thru Express", "capacity": 85, "cost_index": 0.95, "has_dt": True, "has_patio": False},
    {"name": "DineIQ Houston Galleria", "city": "Houston", "state": "TX", "tier": "Suburban Mall", "capacity": 190, "cost_index": 1.05, "has_dt": False, "has_patio": False},
    {"name": "DineIQ Dallas Arts District", "city": "Dallas", "state": "TX", "tier": "Urban Dining", "capacity": 165, "cost_index": 1.12, "has_dt": False, "has_patio": True},
    {"name": "DineIQ San Francisco Embarcadero", "city": "San Francisco", "state": "CA", "tier": "Flagship", "capacity": 195, "cost_index": 1.40, "has_dt": False, "has_patio": True},
    {"name": "DineIQ Silicon Valley Hub", "city": "San Jose", "state": "CA", "tier": "Urban Corporate", "capacity": 150, "cost_index": 1.30, "has_dt": False, "has_patio": True},
    {"name": "DineIQ Los Angeles Beverly", "city": "Los Angeles", "state": "CA", "tier": "Flagship", "capacity": 210, "cost_index": 1.38, "has_dt": False, "has_patio": True},
    {"name": "DineIQ Santa Monica Promenade", "city": "Santa Monica", "state": "CA", "tier": "Coastal Boardwalk", "capacity": 185, "cost_index": 1.32, "has_dt": False, "has_patio": True},
    {"name": "DineIQ Seattle Pike Place", "city": "Seattle", "state": "WA", "tier": "Urban Dining", "capacity": 155, "cost_index": 1.22, "has_dt": False, "has_patio": False},
    {"name": "DineIQ Bellevue Tech Plaza", "city": "Bellevue", "state": "WA", "tier": "Urban Corporate", "capacity": 145, "cost_index": 1.20, "has_dt": False, "has_patio": True},
    {"name": "DineIQ Miami South Beach", "city": "Miami", "state": "FL", "tier": "Coastal Boardwalk", "capacity": 190, "cost_index": 1.28, "has_dt": False, "has_patio": True},
    {"name": "DineIQ Orlando Theme Park Outpost", "city": "Orlando", "state": "FL", "tier": "Family Destination", "capacity": 240, "cost_index": 1.08, "has_dt": True, "has_patio": True},
    {"name": "DineIQ Denver LoDo", "city": "Denver", "state": "CO", "tier": "Urban Dining", "capacity": 150, "cost_index": 1.10, "has_dt": False, "has_patio": True},
    {"name": "DineIQ Atlanta Midtown", "city": "Atlanta", "state": "GA", "tier": "Urban Dining", "capacity": 160, "cost_index": 1.05, "has_dt": False, "has_patio": True},
    {"name": "DineIQ Boston Back Bay", "city": "Boston", "state": "MA", "tier": "Historic Urban", "capacity": 140, "cost_index": 1.25, "has_dt": False, "has_patio": False},
]

def generate_restaurants(output_path: str = None) -> pd.DataFrame:
    """Generate 20 restaurant locations per SRS requirement."""
    records = []
    num_locations = VOLUME_TARGETS["restaurants"]

    for i in range(num_locations):
        loc_id = f"LOC-{i+1:03d}"
        preset = RESTAURANT_PRESETS[i]
        open_year = random.randint(2018, 2024)
        open_month = random.randint(1, 12)
        open_day = random.randint(1, 28)

        record = {
            "restaurant_id": loc_id,
            "location_id": loc_id,
            "name": preset["name"],
            "city": preset["city"],
            "state": preset["state"],
            "country": "USA",
            "postal_code": fake.postcode(),
            "latitude": round(float(fake.latitude()), 6),
            "longitude": round(float(fake.longitude()), 6),
            "location_tier": preset["tier"],
            "seating_capacity": preset["capacity"],
            "cost_index": preset["cost_index"],
            "has_drive_thru": preset["has_dt"],
            "has_outdoor_seating": preset["has_patio"],
            "opened_date": f"{open_year}-{open_month:02d}-{open_day:02d}",
            "manager_name": fake.name(),
            "phone_number": fake.phone_number(),
            "operating_status": "ACTIVE"
        }
        records.append(record)

    df = pd.DataFrame(records)

    if output_path is None:
        target_dir = os.path.join(RAW_DATA_DIR, "restaurants")
        ensure_dir(target_dir)
        output_path = os.path.join(target_dir, "restaurants.csv")

    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[OK] Generated {len(df)} restaurants -> {output_path}")
    return df

if __name__ == "__main__":
    generate_restaurants()
