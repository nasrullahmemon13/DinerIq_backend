"""
Generate Menu Categories Table (10 categories)
Per SRS Section 2 & Hint specifications.
"""
import os
import pandas as pd
from config import RAW_DATA_DIR, VOLUME_TARGETS, ensure_dir

MENU_CATEGORIES = [
    {
        "category_id": "CAT-01",
        "category_name": "Appetizers & Small Plates",
        "description": "Starters, shareables, crispy bites, and dips",
        "target_margin_pct": 65.0,
        "display_order": 1,
        "is_alcohol": False,
        "is_active": True
    },
    {
        "category_id": "CAT-02",
        "category_name": "Artisanal Burgers & Handhelds",
        "description": "Gourmet beef, chicken, and brioche sandwiches",
        "target_margin_pct": 60.0,
        "display_order": 2,
        "is_alcohol": False,
        "is_active": True
    },
    {
        "category_id": "CAT-03",
        "category_name": "Wood-Fired Pizzas & Pastas",
        "description": "Hand-stretched sourdough pizzas and fresh pastas",
        "target_margin_pct": 72.0,
        "display_order": 3,
        "is_alcohol": False,
        "is_active": True
    },
    {
        "category_id": "CAT-04",
        "category_name": "Prime Steaks & Butcher Cuts",
        "description": "Dry-aged steaks, chops, and slow-braised meats",
        "target_margin_pct": 48.0,
        "display_order": 4,
        "is_alcohol": False,
        "is_active": True
    },
    {
        "category_id": "CAT-05",
        "category_name": "Chef Specials & Seafood",
        "description": "Pan-seared fish, seasonal shellfish, and delicacies",
        "target_margin_pct": 52.0,
        "display_order": 5,
        "is_alcohol": False,
        "is_active": True
    },
    {
        "category_id": "CAT-06",
        "category_name": "Farm-Fresh Salads & Grain Bowls",
        "description": "Organic greens, grain bowls, and superfood salads",
        "target_margin_pct": 68.0,
        "display_order": 6,
        "is_alcohol": False,
        "is_active": True
    },
    {
        "category_id": "CAT-07",
        "category_name": "Vegan & Plant-Based Creations",
        "description": "Plant-forward entrees, gluten-free bowls, and tofu grills",
        "target_margin_pct": 66.0,
        "display_order": 7,
        "is_alcohol": False,
        "is_active": True
    },
    {
        "category_id": "CAT-08",
        "category_name": "Handcrafted Desserts & Pastries",
        "description": "Artisanal cakes, tarts, gelatos, and warm puddings",
        "target_margin_pct": 75.0,
        "display_order": 8,
        "is_alcohol": False,
        "is_active": True
    },
    {
        "category_id": "CAT-09",
        "category_name": "Signature Cocktails & Mocktails",
        "description": "Craft mixology, zero-proof botanicals, and house sodas",
        "target_margin_pct": 82.0,
        "display_order": 9,
        "is_alcohol": True,
        "is_active": True
    },
    {
        "category_id": "CAT-10",
        "category_name": "Specialty Coffee & Beverages",
        "description": "Single-origin espresso, loose-leaf teas, and cold brews",
        "target_margin_pct": 85.0,
        "display_order": 10,
        "is_alcohol": False,
        "is_active": True
    }
]

def generate_menu_categories(output_path: str = None) -> pd.DataFrame:
    """Generate 10 menu categories per SRS requirement."""
    assert len(MENU_CATEGORIES) == VOLUME_TARGETS["menu_categories"], "Must equal 10 categories"
    df = pd.DataFrame(MENU_CATEGORIES)

    if output_path is None:
        target_dir = os.path.join(RAW_DATA_DIR, "menu_categories")
        ensure_dir(target_dir)
        output_path = os.path.join(target_dir, "menu_categories.csv")

    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[OK] Generated {len(df)} menu categories -> {output_path}")
    return df

if __name__ == "__main__":
    generate_menu_categories()
