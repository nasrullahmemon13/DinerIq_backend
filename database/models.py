"""
DineIQ Analytics - Relational Database ORM Models (SQLAlchemy)
Defines schema for all 11 tables per SRS Section 2 & Step 2 requirements:
- Restaurants
- Menu Categories
- Menu Items
- Customers
- Promotions
- Pricing History
- Orders
- Order Items
- Ratings
- Inventory
- Wastage
"""
from sqlalchemy import (
    Column, Integer, BigInteger, String, Float, Boolean, Date, DateTime, Time, ForeignKey, Index, Text
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class Role(Base):
    __tablename__ = "roles"

    role_id = Column(String(50), primary_key=True)
    role_name = Column(String(100), nullable=False, unique=True)
    description = Column(Text)

class User(Base):
    __tablename__ = "users"

    user_id = Column(String(50), primary_key=True)
    username = Column(String(100), nullable=False, unique=True, index=True)
    email = Column(String(255), nullable=False, unique=True, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(150), nullable=False)
    role_id = Column(String(50), ForeignKey("roles.role_id"), nullable=False, index=True)
    assigned_location_id = Column(String(50), nullable=True)
    is_active = Column(Boolean, default=True)

class AuthSession(Base):
    __tablename__ = "auth_sessions"
    token_hash = Column(String(64), primary_key=True)
    user_id = Column(String(50), ForeignKey("users.user_id"), nullable=False)
    expires_at = Column(Float, nullable=False)


class Restaurant(Base):
    __tablename__ = "restaurants"

    restaurant_id = Column(String(20), primary_key=True)
    location_id = Column(String(20), index=True, unique=True, nullable=False)
    name = Column(String(100), nullable=False)
    city = Column(String(50), nullable=False)
    state = Column(String(20), nullable=False)
    country = Column(String(20), default="USA")
    postal_code = Column(String(20))
    latitude = Column(Float)
    longitude = Column(Float)
    location_tier = Column(String(50))
    seating_capacity = Column(Integer)
    cost_index = Column(Float)
    has_drive_thru = Column(Boolean, default=False)
    has_outdoor_seating = Column(Boolean, default=False)
    opened_date = Column(Date)
    manager_name = Column(String(100))
    phone_number = Column(String(30))
    operating_status = Column(String(20), default="ACTIVE")

class MenuCategory(Base):
    __tablename__ = "menu_categories"

    category_id = Column(String(20), primary_key=True)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    target_margin_pct = Column(Float)
    is_active = Column(Boolean, default=True)

class MenuItem(Base):
    __tablename__ = "menu_items"

    item_id = Column(String(20), primary_key=True)
    category_id = Column(String(20), ForeignKey("menu_categories.category_id"), nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(Text)
    base_price = Column(Float, nullable=False)
    cost_price = Column(Float, nullable=False)
    margin_pct = Column(Float)
    is_vegetarian = Column(Boolean, default=False)
    is_gluten_free = Column(Boolean, default=False)
    is_alcohol = Column(Boolean, default=False)
    prep_time_minutes = Column(Integer)
    shelf_life_days = Column(Integer)
    is_seasonal = Column(Boolean, default=False)
    complexity_profile = Column(String(50))
    popularity_weight = Column(Float)
    wastage_risk_score = Column(Float)
    elasticity_score = Column(Float)
    is_active = Column(Boolean, default=True)

class Customer(Base):
    __tablename__ = "customers"

    customer_id = Column(String(20), primary_key=True)
    first_name = Column(String(50))
    last_name = Column(String(50))
    email = Column(String(100), index=True)
    phone_number = Column(String(30))
    customer_segment = Column(String(30), index=True)
    loyalty_tier = Column(String(30))
    loyalty_points = Column(Integer, default=0)
    signup_date = Column(Date)
    preferred_location_id = Column(String(20), ForeignKey("restaurants.location_id"))
    is_active = Column(Boolean, default=True)
    churn_risk_score = Column(Float)

class Promotion(Base):
    __tablename__ = "promotions"

    promotion_id = Column(String(20), primary_key=True)
    promotion_name = Column(String(100), nullable=False)
    discount_type = Column(String(30))
    discount_value = Column(Float)
    min_order_amount = Column(Float)
    max_discount_amount = Column(Float)
    applicable_category = Column(String(30))
    start_date = Column(Date)
    end_date = Column(Date)
    target_segment = Column(String(30))
    complexity_tag = Column(String(50))
    is_misleading = Column(Boolean, default=False)
    description = Column(Text)

class PricingHistory(Base):
    __tablename__ = "pricing_history"

    price_history_id = Column(String(20), primary_key=True)
    item_id = Column(String(20), ForeignKey("menu_items.item_id"), nullable=False)
    location_id = Column(String(20), nullable=False)
    base_price = Column(Float, nullable=False)
    cost_price = Column(Float, nullable=False)
    effective_start_date = Column(Date, nullable=False)
    effective_end_date = Column(Date)
    change_reason = Column(String(100))
    complexity_profile = Column(String(50))

class Order(Base):
    __tablename__ = "orders"

    order_id = Column(String(30), primary_key=True)
    customer_id = Column(String(20), ForeignKey("customers.customer_id"), nullable=True)
    location_id = Column(String(20), ForeignKey("restaurants.location_id"), nullable=False)
    order_date = Column(Date, index=True, nullable=False)
    order_time = Column(String(20))
    order_timestamp = Column(String(30), index=True)
    order_type = Column(String(30))
    order_status = Column(String(30), index=True)
    payment_method = Column(String(30))
    subtotal_amount = Column(Float)
    discount_amount = Column(Float)
    tax_amount = Column(Float)
    tip_amount = Column(Float)
    delivery_fee = Column(Float)
    total_amount = Column(Float)
    promotion_id = Column(String(20), ForeignKey("promotions.promotion_id"), nullable=True)
    table_number = Column(Integer, nullable=True)

class OrderItem(Base):
    __tablename__ = "order_items"

    order_item_id = Column(String(30), primary_key=True)
    order_id = Column(String(30), ForeignKey("orders.order_id"), nullable=False)
    item_id = Column(String(20), ForeignKey("menu_items.item_id"), nullable=False)
    quantity = Column(Integer, nullable=False)
    unit_price = Column(Float, nullable=False)
    subtotal = Column(Float, nullable=False)
    item_discount = Column(Float, default=0.0)
    item_total = Column(Float, nullable=False)

class Rating(Base):
    __tablename__ = "ratings"

    rating_id = Column(String(30), primary_key=True)
    order_id = Column(String(30), ForeignKey("orders.order_id"))
    customer_id = Column(String(20), ForeignKey("customers.customer_id"), nullable=True)
    item_id = Column(String(20), ForeignKey("menu_items.item_id"))
    location_id = Column(String(20), ForeignKey("restaurants.location_id"))
    overall_rating = Column(Integer, nullable=False)
    food_rating = Column(Integer)
    service_rating = Column(Integer)
    ambiance_rating = Column(Integer)
    review_text = Column(Text)
    review_date = Column(Date)
    anomaly_tag = Column(String(50))

class Wastage(Base):
    __tablename__ = "wastage"

    wastage_id = Column(String(30), primary_key=True)
    item_id = Column(String(20), ForeignKey("menu_items.item_id"), nullable=False)
    location_id = Column(String(20), ForeignKey("restaurants.location_id"), nullable=False)
    wastage_date = Column(Date, index=True)
    wastage_time = Column(String(20))
    quantity_wasted = Column(Integer, nullable=False)
    unit_cost = Column(Float, nullable=False)
    total_loss_amount = Column(Float, nullable=False)
    wastage_reason = Column(String(50))
    complexity_profile = Column(String(50))
    reported_by = Column(String(50))

class Inventory(Base):
    __tablename__ = "inventory"

    inventory_id = Column(String(30), primary_key=True)
    location_id = Column(String(20), ForeignKey("restaurants.location_id"), nullable=False)
    item_id = Column(String(20), ForeignKey("menu_items.item_id"), nullable=False)
    snapshot_date = Column(Date, index=True)
    starting_stock = Column(Integer)
    quantity_received = Column(Integer)
    quantity_sold = Column(Integer)
    quantity_wasted = Column(Integer)
    ending_stock = Column(Integer)
    reorder_point = Column(Integer)
    stock_status = Column(String(30))

class AuditLog(Base):
    __tablename__ = "audit_log"

    audit_id = Column(String(50), primary_key=True)
    event_type = Column(String(50), nullable=False, index=True)                                                                    
    action = Column(String(100), nullable=False, index=True)
    actor = Column(String(100), nullable=False, index=True)
    resource_id = Column(String(100), nullable=True)
    status = Column(String(30), nullable=False, default="SUCCESS", index=True)
    details = Column(Text, nullable=True)
    ip_address = Column(String(50), nullable=True)
    timestamp = Column(DateTime, nullable=False, index=True)

class ModelVersion(Base):
    __tablename__ = "model_versions"

    version_id = Column(String(50), primary_key=True)
    model_name = Column(String(100), nullable=False, index=True)
    version_tag = Column(String(50), nullable=False, index=True)
    framework = Column(String(50), nullable=False)
    pipeline_type = Column(String(50), nullable=False)
    task_type = Column(String(50), nullable=False, index=True)
    metrics = Column(Text, nullable=True)
    parameters = Column(Text, nullable=True)
    artifact_uri = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True, index=True)
    trained_at = Column(DateTime)
    created_at = Column(DateTime)

class SystemConfig(Base):
    __tablename__ = "system_configs"

    config_key = Column(String(100), primary_key=True)
    config_value = Column(Text, nullable=False)
    category = Column(String(50), nullable=False, index=True)
    description = Column(Text, nullable=True)
    is_secret = Column(Boolean, default=False)
    updated_by = Column(String(100), nullable=True)
    updated_at = Column(DateTime)

class PredictionResult(Base):
    __tablename__ = "prediction_results"

    prediction_id = Column(String(50), primary_key=True)
    model_version_id = Column(String(50), ForeignKey("model_versions.version_id"), nullable=False, index=True)
    task_type = Column(String(50), nullable=False, index=True)
    entity_type = Column(String(50), nullable=False)
    entity_id = Column(String(100), nullable=False, index=True)
    predicted_value = Column(Text, nullable=False)
    actual_value = Column(Text, nullable=True)
    confidence_score = Column(Float, nullable=True)
    metadata_json = Column(Text, nullable=True)
    prediction_timestamp = Column(DateTime, nullable=False)

class SparkJob(Base):
    __tablename__ = "spark_jobs"

    job_id = Column(String(50), primary_key=True)
    job_name = Column(String(150), nullable=False)
    pipeline_type = Column(String(50), nullable=False, default="Spark")
    status = Column(String(30), nullable=False, default="SUBMITTED", index=True)
    stages_completed = Column(Integer, default=0)
    total_stages = Column(Integer, default=1)
    records_processed = Column(BigInteger, default=0)
    duration_seconds = Column(Float, default=0.0)
    metrics = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    start_time = Column(DateTime, nullable=False, index=True)
    end_time = Column(DateTime, nullable=True)
