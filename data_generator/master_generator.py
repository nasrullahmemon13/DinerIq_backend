"""
DineIQ Analytics - Master Data Generator
Executes the generation of all 11 tables in the exact dependency order:
1. Restaurants (20 locations)
2. Menu Categories (10 categories)
3. Menu Items (150 items)
4. Customers (50,000 customers)
5. Promotions (12 promotional campaigns)
6. Pricing History (500+ records)
7. Orders & Order_Items (100,000 orders & 1,000,000 order-lines)
8. Ratings (100,000 ratings)
9. Inventory & Wastage (26,000 inventory & 50,000 wastage records)
"""
import time
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_restaurants import generate_restaurants
from generate_menu_categories import generate_menu_categories
from generate_menu_items import generate_menu_items
from generate_customers import generate_customers
from generate_promotions import generate_promotions
from generate_pricing_history import generate_pricing_history
from generate_orders_and_items import generate_orders_and_items
from generate_ratings import generate_ratings
from generate_inventory_and_wastage import generate_inventory_and_wastage

def run_master_generation():
    start_total = time.time()
    print("=" * 70)
    print("DineIQ Analytics - Master Restaurant Dataset Generator (SRS Step 1)")
    print("=" * 70)

    print("\n[1/7] Generating Restaurants, Categories & Menu Items...")
    generate_restaurants()
    generate_menu_categories()
    generate_menu_items()

    print("\n[2/7] Generating Customers...")
    generate_customers()

    print("\n[3/7] Generating Promotions...")
    generate_promotions()

    print("\n[4/7] Generating Pricing History...")
    generate_pricing_history()

    print("\n[5/7] Generating Orders (100k) & Order Items (1M)...")
    generate_orders_and_items()

    print("\n[6/7] Generating Ratings (100k)...")
    generate_ratings()

    print("\n[7/7] Generating Inventory & Wastage (50k)...")
    generate_inventory_and_wastage()

    elapsed = time.time() - start_total
    print("\n" + "=" * 70)
    print(f"All 11 DineIQ Analytics datasets successfully generated in {elapsed:.1f} seconds!")
    print("=" * 70)

if __name__ == "__main__":
    run_master_generation()
