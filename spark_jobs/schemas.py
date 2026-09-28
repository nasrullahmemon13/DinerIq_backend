"""
DineIQ Analytics — Production PySpark Schemas & Contracts
Defines explicit StructType schemas for:
1. Customers
2. Orders
3. Order_Items
4. Menu_Items
5. Menu_Categories
6. Restaurants
7. Pricing_History
8. Promotions
9. Ratings
10. Inventory
11. Wastage
Additional System Schemas:
- Quarantine Records
- Prediction Outputs
- Pipeline Audit / Telemetry
"""

from typing import Dict, Any, List, Optional
try:
    from pyspark.sql.types import (
        StructType,
        StructField,
        StringType,
        IntegerType,
        LongType,
        DoubleType,
        BooleanType,
        DateType,
        TimestampType,
        DecimalType
    )
except ImportError:
    class DataType:
        def __repr__(self): return self.__class__.__name__
    class StringType(DataType): pass
    class IntegerType(DataType): pass
    class LongType(DataType): pass
    class DoubleType(DataType): pass
    class BooleanType(DataType): pass
    class DateType(DataType): pass
    class TimestampType(DataType): pass
    class DecimalType(DataType):
        def __init__(self, p=10, s=2): self.p, self.s = p, s
    class StructField:
        def __init__(self, name: str, dataType: Any, nullable: bool = True, metadata=None):
            self.name = name
            self.dataType = dataType
            self.nullable = nullable
            self.metadata = metadata or {}
    class StructType(DataType):
        def __init__(self, fields=None): self.fields = fields or []
        def __getitem__(self, item):
            if isinstance(item, str):
                for f in self.fields:
                    if f.name == item: return f
                raise KeyError(item)
            return self.fields[item]
        def __len__(self): return len(self.fields)


def get_customers_schema() -> StructType:
    return StructType([
        StructField("customer_id", StringType(), nullable=False),
        StructField("first_name", StringType(), nullable=True),
        StructField("last_name", StringType(), nullable=True),
        StructField("email", StringType(), nullable=True),
        StructField("phone_number", StringType(), nullable=True),
        StructField("customer_segment", StringType(), nullable=False),
        StructField("loyalty_tier", StringType(), nullable=True),
        StructField("loyalty_points", IntegerType(), nullable=True),
        StructField("signup_date", DateType(), nullable=True),
        StructField("preferred_location_id", StringType(), nullable=True),
        StructField("is_active", BooleanType(), nullable=True),
        StructField("churn_risk_score", DoubleType(), nullable=True),
        StructField("_corrupt_record", StringType(), nullable=True)
    ])


def get_orders_schema() -> StructType:
    return StructType([
        StructField("order_id", StringType(), nullable=False),
        StructField("customer_id", StringType(), nullable=True),
        StructField("location_id", StringType(), nullable=False),
        StructField("order_date", DateType(), nullable=False),
        StructField("order_time", StringType(), nullable=True),
        StructField("order_timestamp", TimestampType(), nullable=True),
        StructField("order_type", StringType(), nullable=False),
        StructField("order_status", StringType(), nullable=False),
        StructField("payment_method", StringType(), nullable=True),
        StructField("subtotal_amount", DoubleType(), nullable=False),
        StructField("discount_amount", DoubleType(), nullable=True),
        StructField("tax_amount", DoubleType(), nullable=True),
        StructField("tip_amount", DoubleType(), nullable=True),
        StructField("delivery_fee", DoubleType(), nullable=True),
        StructField("total_amount", DoubleType(), nullable=False),
        StructField("promotion_id", StringType(), nullable=True),
        StructField("table_number", IntegerType(), nullable=True),
        StructField("_corrupt_record", StringType(), nullable=True)
    ])


def get_order_items_schema() -> StructType:
    return StructType([
        StructField("order_item_id", StringType(), nullable=False),
        StructField("order_id", StringType(), nullable=False),
        StructField("item_id", StringType(), nullable=False),
        StructField("quantity", IntegerType(), nullable=False),
        StructField("unit_price", DoubleType(), nullable=False),
        StructField("subtotal", DoubleType(), nullable=False),
        StructField("item_discount", DoubleType(), nullable=True),
        StructField("item_total", DoubleType(), nullable=False),
        StructField("_corrupt_record", StringType(), nullable=True)
    ])


def get_menu_items_schema() -> StructType:
    return StructType([
        StructField("item_id", StringType(), nullable=False),
        StructField("category_id", StringType(), nullable=False),
        StructField("name", StringType(), nullable=False),
        StructField("description", StringType(), nullable=True),
        StructField("base_price", DoubleType(), nullable=False),
        StructField("cost_price", DoubleType(), nullable=False),
        StructField("margin_pct", DoubleType(), nullable=True),
        StructField("is_vegetarian", BooleanType(), nullable=True),
        StructField("is_gluten_free", BooleanType(), nullable=True),
        StructField("is_alcohol", BooleanType(), nullable=True),
        StructField("prep_time_minutes", IntegerType(), nullable=True),
        StructField("shelf_life_days", IntegerType(), nullable=True),
        StructField("is_seasonal", BooleanType(), nullable=True),
        StructField("complexity_profile", StringType(), nullable=True),
        StructField("popularity_weight", DoubleType(), nullable=True),
        StructField("wastage_risk_score", DoubleType(), nullable=True),
        StructField("elasticity_score", DoubleType(), nullable=True),
        StructField("is_active", BooleanType(), nullable=True),
        StructField("_corrupt_record", StringType(), nullable=True)
    ])


def get_menu_categories_schema() -> StructType:
    return StructType([
        StructField("category_id", StringType(), nullable=False),
        StructField("category_name", StringType(), nullable=False),
        StructField("description", StringType(), nullable=True),
        StructField("target_margin_pct", DoubleType(), nullable=True),
        StructField("is_active", BooleanType(), nullable=True),
        StructField("_corrupt_record", StringType(), nullable=True)
    ])


def get_restaurants_schema() -> StructType:
    return StructType([
        StructField("restaurant_id", StringType(), nullable=False),
        StructField("location_id", StringType(), nullable=False),
        StructField("name", StringType(), nullable=False),
        StructField("city", StringType(), nullable=False),
        StructField("state", StringType(), nullable=False),
        StructField("country", StringType(), nullable=True),
        StructField("postal_code", StringType(), nullable=True),
        StructField("latitude", DoubleType(), nullable=True),
        StructField("longitude", DoubleType(), nullable=True),
        StructField("location_tier", StringType(), nullable=True),
        StructField("seating_capacity", IntegerType(), nullable=True),
        StructField("cost_index", DoubleType(), nullable=True),
        StructField("has_drive_thru", BooleanType(), nullable=True),
        StructField("has_outdoor_seating", BooleanType(), nullable=True),
        StructField("opened_date", DateType(), nullable=True),
        StructField("manager_name", StringType(), nullable=True),
        StructField("phone_number", StringType(), nullable=True),
        StructField("operating_status", StringType(), nullable=True),
        StructField("_corrupt_record", StringType(), nullable=True)
    ])


def get_pricing_history_schema() -> StructType:
    return StructType([
        StructField("price_history_id", StringType(), nullable=False),
        StructField("item_id", StringType(), nullable=False),
        StructField("location_id", StringType(), nullable=False),
        StructField("base_price", DoubleType(), nullable=False),
        StructField("cost_price", DoubleType(), nullable=False),
        StructField("effective_start_date", DateType(), nullable=False),
        StructField("effective_end_date", DateType(), nullable=True),
        StructField("change_reason", StringType(), nullable=True),
        StructField("complexity_profile", StringType(), nullable=True),
        StructField("_corrupt_record", StringType(), nullable=True)
    ])


def get_promotions_schema() -> StructType:
    return StructType([
        StructField("promotion_id", StringType(), nullable=False),
        StructField("promotion_name", StringType(), nullable=False),
        StructField("discount_type", StringType(), nullable=False),
        StructField("discount_value", DoubleType(), nullable=False),
        StructField("min_order_amount", DoubleType(), nullable=True),
        StructField("max_discount_amount", DoubleType(), nullable=True),
        StructField("applicable_category", StringType(), nullable=True),
        StructField("start_date", DateType(), nullable=True),
        StructField("end_date", DateType(), nullable=True),
        StructField("target_segment", StringType(), nullable=True),
        StructField("complexity_tag", StringType(), nullable=True),
        StructField("is_misleading", BooleanType(), nullable=True),
        StructField("description", StringType(), nullable=True),
        StructField("_corrupt_record", StringType(), nullable=True)
    ])


def get_ratings_schema() -> StructType:
    return StructType([
        StructField("rating_id", StringType(), nullable=False),
        StructField("order_id", StringType(), nullable=False),
        StructField("customer_id", StringType(), nullable=True),
        StructField("item_id", StringType(), nullable=True),
        StructField("location_id", StringType(), nullable=False),
        StructField("overall_rating", IntegerType(), nullable=False),
        StructField("food_rating", IntegerType(), nullable=True),
        StructField("service_rating", IntegerType(), nullable=True),
        StructField("ambiance_rating", IntegerType(), nullable=True),
        StructField("review_text", StringType(), nullable=True),
        StructField("review_date", DateType(), nullable=True),
        StructField("anomaly_tag", StringType(), nullable=True),
        StructField("_corrupt_record", StringType(), nullable=True)
    ])


def get_inventory_schema() -> StructType:
    return StructType([
        StructField("inventory_id", StringType(), nullable=False),
        StructField("location_id", StringType(), nullable=False),
        StructField("item_id", StringType(), nullable=False),
        StructField("snapshot_date", DateType(), nullable=False),
        StructField("starting_stock", IntegerType(), nullable=True),
        StructField("quantity_received", IntegerType(), nullable=True),
        StructField("quantity_sold", IntegerType(), nullable=True),
        StructField("quantity_wasted", IntegerType(), nullable=True),
        StructField("ending_stock", IntegerType(), nullable=True),
        StructField("reorder_point", IntegerType(), nullable=True),
        StructField("stock_status", StringType(), nullable=True),
        StructField("_corrupt_record", StringType(), nullable=True)
    ])


def get_wastage_schema() -> StructType:
    return StructType([
        StructField("wastage_id", StringType(), nullable=False),
        StructField("item_id", StringType(), nullable=False),
        StructField("location_id", StringType(), nullable=False),
        StructField("wastage_date", DateType(), nullable=False),
        StructField("wastage_time", StringType(), nullable=True),
        StructField("quantity_wasted", IntegerType(), nullable=False),
        StructField("unit_cost", DoubleType(), nullable=False),
        StructField("total_loss_amount", DoubleType(), nullable=False),
        StructField("wastage_reason", StringType(), nullable=True),
        StructField("complexity_profile", StringType(), nullable=True),
        StructField("reported_by", StringType(), nullable=True),
        StructField("_corrupt_record", StringType(), nullable=True)
    ])


def get_quarantine_schema() -> StructType:
    """Schema for isolated corrupt or rule-violating records."""
    return StructType([
        StructField("record_id", StringType(), nullable=False),
        StructField("dataset", StringType(), nullable=False),
        StructField("rule_id", StringType(), nullable=False),
        StructField("rule_name", StringType(), nullable=True),
        StructField("reason", StringType(), nullable=False),
        StructField("original_value", StringType(), nullable=True),
        StructField("action", StringType(), nullable=False),
        StructField("run_id", StringType(), nullable=True),
        StructField("timestamp", StringType(), nullable=False)
    ])


def get_prediction_schema() -> StructType:
    """Standardized schema for model inference outputs."""
    return StructType([
        StructField("record_id", StringType(), nullable=False),
        StructField("actual_value", DoubleType(), nullable=True),
        StructField("spark_prediction", DoubleType(), nullable=False),
        StructField("spark_model", StringType(), nullable=False),
        StructField("spark_model_version", StringType(), nullable=False),
        StructField("prediction_timestamp", StringType(), nullable=False)
    ])


def get_all_schemas() -> Dict[str, StructType]:
    return {
        "customers": get_customers_schema(),
        "orders": get_orders_schema(),
        "order_items": get_order_items_schema(),
        "menu_items": get_menu_items_schema(),
        "menu_categories": get_menu_categories_schema(),
        "restaurants": get_restaurants_schema(),
        "pricing_history": get_pricing_history_schema(),
        "promotions": get_promotions_schema(),
        "ratings": get_ratings_schema(),
        "inventory": get_inventory_schema(),
        "wastage": get_wastage_schema()
    }
