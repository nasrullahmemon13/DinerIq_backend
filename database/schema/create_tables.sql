                                                                                
                                                    
                                                            
                                               
                                           
                                         
                                                                                         
                                     
                                                         
                                                   
                                                                                         
                            
                               
                             
                                                                                

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

                                                                                
                                                                           
                                                                                
CREATE TABLE IF NOT EXISTS roles (
    role_id VARCHAR(50) PRIMARY KEY,
    role_name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO roles (role_id, role_name, description) VALUES
    ('admin', 'Administrator', 'Full system access and location/user CRUD operations'),
    ('regional_manager', 'Regional Manager', 'Multi-location operations and regional approval access'),
    ('manager', 'Store Manager', 'Local restaurant location and shift management'),
    ('analyst', 'Data Analyst', 'Analytical reporting, what-if simulations, data exports')
ON CONFLICT (role_id) DO NOTHING;

CREATE TABLE IF NOT EXISTS users (
    user_id VARCHAR(50) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    username VARCHAR(100) NOT NULL UNIQUE,
    email VARCHAR(255) NOT NULL UNIQUE,
    hashed_password VARCHAR(255) NOT NULL,
    full_name VARCHAR(150) NOT NULL,
    role_id VARCHAR(50) NOT NULL REFERENCES roles(role_id) ON DELETE RESTRICT,
    assigned_location_id VARCHAR(50),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP WITH TIME ZONE
);

CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role_id);

                                                                                
                                       
                                                                                
CREATE TABLE IF NOT EXISTS restaurants (
    restaurant_id VARCHAR(50) PRIMARY KEY,
    location_id VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(150) NOT NULL,
    city VARCHAR(100) NOT NULL,
    state VARCHAR(50) NOT NULL,
    country VARCHAR(50) DEFAULT 'USA',
    postal_code VARCHAR(30),
    latitude NUMERIC(10, 6),
    longitude NUMERIC(10, 6),
    location_tier VARCHAR(50) DEFAULT 'TIER_1',
    seating_capacity INTEGER DEFAULT 120,
    cost_index NUMERIC(6, 3) DEFAULT 1.000,
    has_drive_thru BOOLEAN DEFAULT FALSE,
    has_outdoor_seating BOOLEAN DEFAULT FALSE,
    opened_date DATE,
    manager_name VARCHAR(150),
    phone_number VARCHAR(50),
    operating_status VARCHAR(30) DEFAULT 'ACTIVE',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_restaurants_location_id ON restaurants(location_id);
CREATE INDEX IF NOT EXISTS idx_restaurants_city_state ON restaurants(city, state);
CREATE INDEX IF NOT EXISTS idx_restaurants_status ON restaurants(operating_status);

                                                                                
                                            
                                                                                
CREATE TABLE IF NOT EXISTS menu_categories (
    category_id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(150) NOT NULL UNIQUE,
    description TEXT,
    target_margin_pct NUMERIC(6, 2) DEFAULT 60.00,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS menu_items (
    item_id VARCHAR(50) PRIMARY KEY,
    category_id VARCHAR(50) NOT NULL REFERENCES menu_categories(category_id) ON DELETE RESTRICT,
    name VARCHAR(150) NOT NULL,
    description TEXT,
    base_price NUMERIC(10, 2) NOT NULL CHECK (base_price >= 0),
    cost_price NUMERIC(10, 2) NOT NULL CHECK (cost_price >= 0),
    margin_pct NUMERIC(6, 2) GENERATED ALWAYS AS (
        CASE WHEN base_price > 0 THEN ROUND(((base_price - cost_price) / base_price) * 100, 2) ELSE 0 END
    ) STORED,
    is_vegetarian BOOLEAN DEFAULT FALSE,
    is_gluten_free BOOLEAN DEFAULT FALSE,
    is_alcohol BOOLEAN DEFAULT FALSE,
    prep_time_minutes INTEGER DEFAULT 15,
    shelf_life_days INTEGER DEFAULT 3,
    is_seasonal BOOLEAN DEFAULT FALSE,
    complexity_profile VARCHAR(50) DEFAULT 'Standard',
    popularity_weight NUMERIC(6, 3) DEFAULT 1.000,
    wastage_risk_score NUMERIC(6, 3) DEFAULT 0.200,
    elasticity_score NUMERIC(6, 3) DEFAULT -1.200,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_menu_items_category ON menu_items(category_id);
CREATE INDEX IF NOT EXISTS idx_menu_items_active ON menu_items(is_active);
CREATE INDEX IF NOT EXISTS idx_menu_items_price ON menu_items(base_price);

                                                                                
                                 
                                                                                
CREATE TABLE IF NOT EXISTS pricing_history (
    price_history_id VARCHAR(50) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    item_id VARCHAR(50) NOT NULL REFERENCES menu_items(item_id) ON DELETE CASCADE,
    location_id VARCHAR(50) NOT NULL REFERENCES restaurants(location_id) ON DELETE CASCADE,
    base_price NUMERIC(10, 2) NOT NULL,
    cost_price NUMERIC(10, 2) NOT NULL,
    effective_start_date DATE NOT NULL,
    effective_end_date DATE,
    change_reason VARCHAR(200),
    complexity_profile VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_pricing_history_item_loc ON pricing_history(item_id, location_id);
CREATE INDEX IF NOT EXISTS idx_pricing_history_dates ON pricing_history(effective_start_date, effective_end_date);

                                                                                
                                                      
                                                                                
CREATE TABLE IF NOT EXISTS customers (
    customer_id VARCHAR(50) PRIMARY KEY,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    email VARCHAR(255),
    phone_number VARCHAR(50),
    customer_segment VARCHAR(50) DEFAULT 'New Customers',
    loyalty_tier VARCHAR(50) DEFAULT 'BRONZE',
    loyalty_points INTEGER DEFAULT 0,
    signup_date DATE DEFAULT CURRENT_DATE,
    preferred_location_id VARCHAR(50) REFERENCES restaurants(location_id) ON DELETE SET NULL,
    is_active BOOLEAN DEFAULT TRUE,
    churn_risk_score NUMERIC(6, 4) DEFAULT 0.0000,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_customers_segment ON customers(customer_segment);
CREATE INDEX IF NOT EXISTS idx_customers_loyalty ON customers(loyalty_tier);
CREATE INDEX IF NOT EXISTS idx_customers_pref_loc ON customers(preferred_location_id);

                                                                                
                              
                                                                                
CREATE TABLE IF NOT EXISTS promotions (
    promotion_id VARCHAR(50) PRIMARY KEY,
    promotion_name VARCHAR(150) NOT NULL,
    discount_type VARCHAR(50) NOT NULL CHECK (discount_type IN ('PERCENTAGE', 'FIXED_AMOUNT', 'BOGO', 'BUNDLE')),
    discount_value NUMERIC(10, 2) NOT NULL CHECK (discount_value >= 0),
    min_order_amount NUMERIC(10, 2) DEFAULT 0.00,
    max_discount_amount NUMERIC(10, 2),
    applicable_category VARCHAR(100),
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    target_segment VARCHAR(100) DEFAULT 'All Segments',
    complexity_tag VARCHAR(100),
    is_misleading BOOLEAN DEFAULT FALSE,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_promotions_dates ON promotions(start_date, end_date);
CREATE INDEX IF NOT EXISTS idx_promotions_discount_type ON promotions(discount_type);

                                                                                
                                                 
                                                                                
CREATE TABLE IF NOT EXISTS orders (
    order_id VARCHAR(60) PRIMARY KEY,
    customer_id VARCHAR(50) REFERENCES customers(customer_id) ON DELETE SET NULL,
    location_id VARCHAR(50) NOT NULL REFERENCES restaurants(location_id) ON DELETE RESTRICT,
    order_date DATE NOT NULL,
    order_time VARCHAR(30),
    order_timestamp TIMESTAMP WITH TIME ZONE,
    order_type VARCHAR(50) NOT NULL CHECK (order_type IN ('Dine-in', 'Takeaway', 'Website/App', 'Third-party delivery', 'Drive-thru')),
    order_status VARCHAR(50) NOT NULL DEFAULT 'COMPLETED',
    payment_method VARCHAR(50) DEFAULT 'CARD',
    subtotal_amount NUMERIC(10, 2) NOT NULL DEFAULT 0.00,
    discount_amount NUMERIC(10, 2) DEFAULT 0.00,
    tax_amount NUMERIC(10, 2) DEFAULT 0.00,
    tip_amount NUMERIC(10, 2) DEFAULT 0.00,
    delivery_fee NUMERIC(10, 2) DEFAULT 0.00,
    total_amount NUMERIC(10, 2) NOT NULL DEFAULT 0.00,
    promotion_id VARCHAR(50) REFERENCES promotions(promotion_id) ON DELETE SET NULL,
    table_number INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_id);
CREATE INDEX IF NOT EXISTS idx_orders_location ON orders(location_id);
CREATE INDEX IF NOT EXISTS idx_orders_date ON orders(order_date);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(order_status);

CREATE TABLE IF NOT EXISTS order_items (
    order_item_id VARCHAR(60) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    order_id VARCHAR(60) NOT NULL REFERENCES orders(order_id) ON DELETE CASCADE,
    item_id VARCHAR(50) NOT NULL REFERENCES menu_items(item_id) ON DELETE RESTRICT,
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    unit_price NUMERIC(10, 2) NOT NULL,
    subtotal NUMERIC(10, 2) NOT NULL,
    item_discount NUMERIC(10, 2) DEFAULT 0.00,
    item_total NUMERIC(10, 2) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_order_items_order ON order_items(order_id);
CREATE INDEX IF NOT EXISTS idx_order_items_item ON order_items(item_id);

                                                                                
                         
                                                                                
CREATE TABLE IF NOT EXISTS ratings (
    rating_id VARCHAR(60) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    order_id VARCHAR(60) REFERENCES orders(order_id) ON DELETE SET NULL,
    customer_id VARCHAR(50) REFERENCES customers(customer_id) ON DELETE SET NULL,
    item_id VARCHAR(50) REFERENCES menu_items(item_id) ON DELETE SET NULL,
    location_id VARCHAR(50) NOT NULL REFERENCES restaurants(location_id) ON DELETE RESTRICT,
    overall_rating INTEGER NOT NULL CHECK (overall_rating BETWEEN 1 AND 5),
    food_rating INTEGER CHECK (food_rating BETWEEN 1 AND 5),
    service_rating INTEGER CHECK (service_rating BETWEEN 1 AND 5),
    ambiance_rating INTEGER CHECK (ambiance_rating BETWEEN 1 AND 5),
    review_text TEXT,
    review_date DATE DEFAULT CURRENT_DATE,
    anomaly_tag VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_ratings_location ON ratings(location_id);
CREATE INDEX IF NOT EXISTS idx_ratings_item ON ratings(item_id);
CREATE INDEX IF NOT EXISTS idx_ratings_customer ON ratings(customer_id);
CREATE INDEX IF NOT EXISTS idx_ratings_score ON ratings(overall_rating);

                                                                                
                           
                                                                                
CREATE TABLE IF NOT EXISTS inventory (
    inventory_id VARCHAR(60) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    location_id VARCHAR(50) NOT NULL REFERENCES restaurants(location_id) ON DELETE RESTRICT,
    item_id VARCHAR(50) NOT NULL REFERENCES menu_items(item_id) ON DELETE RESTRICT,
    snapshot_date DATE NOT NULL,
    starting_stock INTEGER NOT NULL DEFAULT 0,
    quantity_received INTEGER NOT NULL DEFAULT 0,
    quantity_sold INTEGER NOT NULL DEFAULT 0,
    quantity_wasted INTEGER NOT NULL DEFAULT 0,
    ending_stock INTEGER NOT NULL DEFAULT 0,
    reorder_point INTEGER DEFAULT 30,
    stock_status VARCHAR(50) DEFAULT 'ADEQUATE',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_inventory_loc_item ON inventory(location_id, item_id);
CREATE INDEX IF NOT EXISTS idx_inventory_snapshot_date ON inventory(snapshot_date);

                                                                                
                          
                                                                                
CREATE TABLE IF NOT EXISTS wastage (
    wastage_id VARCHAR(60) PRIMARY KEY DEFAULT uuid_generate_v4()::text,
    item_id VARCHAR(50) NOT NULL REFERENCES menu_items(item_id) ON DELETE RESTRICT,
    location_id VARCHAR(50) NOT NULL REFERENCES restaurants(location_id) ON DELETE RESTRICT,
    wastage_date DATE NOT NULL,
    wastage_time VARCHAR(30),
    quantity_wasted INTEGER NOT NULL CHECK (quantity_wasted > 0),
    unit_cost NUMERIC(10, 2) NOT NULL CHECK (unit_cost >= 0),
    total_loss_amount NUMERIC(10, 2) NOT NULL CHECK (total_loss_amount >= 0),
    wastage_reason VARCHAR(100) DEFAULT 'OVERPRODUCTION_UNSOLD',
    complexity_profile VARCHAR(100),
    reported_by VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_wastage_item ON wastage(item_id);
CREATE INDEX IF NOT EXISTS idx_wastage_location ON wastage(location_id);
CREATE INDEX IF NOT EXISTS idx_wastage_date ON wastage(wastage_date);
CREATE INDEX IF NOT EXISTS idx_wastage_reason ON wastage(wastage_reason);
