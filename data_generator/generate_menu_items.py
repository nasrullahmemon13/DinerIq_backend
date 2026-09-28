"""
Generate Menu Items Table (150 items across 10 categories)
Per SRS Section 2 & Hint specifications.
Injects:
- Popular but low-margin dishes
- Profitable but low-selling dishes
- High-wastage dishes
- Price-sensitive items
"""
import os
import random
import pandas as pd
from config import RAW_DATA_DIR, VOLUME_TARGETS, RANDOM_SEED, ensure_dir

random.seed(RANDOM_SEED)

ITEMS_DATA = [
    ("Crispy Parmesan Truffle Fries", "CAT-01", 8.99, 6.75, True, False, False, 8, 14, False, "POPULAR_LOW_MARGIN", 3.5, 0.25, 0.40),
    ("Classic Buffalo Chicken Wings", "CAT-01", 14.50, 11.20, False, True, False, 14, 5, False, "POPULAR_LOW_MARGIN", 3.2, 0.35, 0.50),
    ("Golden Mozzarella Sticks", "CAT-01", 9.50, 7.10, True, False, False, 7, 30, False, "POPULAR_LOW_MARGIN", 2.8, 0.15, 0.35),
    ("Garlic Herb Breadsticks", "CAT-01", 6.99, 4.80, True, False, False, 6, 4, False, "POPULAR_LOW_MARGIN", 3.0, 0.20, 0.30),
    ("Wild Alaskan Salmon Tartare", "CAT-01", 21.00, 11.50, False, True, False, 10, 2, False, "HIGH_WASTAGE", 0.7, 0.90, 0.65),
    ("Soft-Shell Crab Sliders", "CAT-01", 19.50, 10.20, False, False, False, 12, 2, False, "HIGH_WASTAGE", 0.6, 0.85, 0.60),
    ("Crispy Calamari Fritti", "CAT-01", 15.99, 7.50, False, False, False, 9, 3, False, "STANDARD", 2.0, 0.50, 0.50),
    ("Reserve Osetra Caviar Blinis", "CAT-01", 65.00, 16.50, False, False, False, 6, 7, False, "PROFITABLE_LOW_SELLING", 0.15, 0.60, 0.30),
    ("Truffle Whipped Burrata", "CAT-01", 17.50, 7.80, True, True, False, 6, 4, False, "STANDARD", 1.8, 0.55, 0.50),
    ("Smoked Duck Croquettes", "CAT-01", 16.00, 7.00, False, False, False, 11, 4, False, "STANDARD", 1.2, 0.40, 0.45),
    ("Spicy Tuna Crispy Rice", "CAT-01", 18.00, 9.80, False, True, False, 10, 2, False, "HIGH_WASTAGE", 1.5, 0.88, 0.55),
    ("Heirloom Tomato Bruschetta", "CAT-01", 11.50, 4.20, True, False, False, 7, 3, False, "PRICE_SENSITIVE", 2.2, 0.45, 0.75),
    ("Loaded Queso Nachos", "CAT-01", 12.99, 5.50, True, True, False, 8, 10, False, "STANDARD", 2.4, 0.20, 0.45),
    ("Spanakopita Herb Triangles", "CAT-01", 11.00, 4.50, True, False, False, 10, 7, False, "STANDARD", 1.1, 0.30, 0.40),
    ("Bacon Wrapped Stuffed Dates", "CAT-01", 13.50, 5.20, False, True, False, 9, 7, False, "STANDARD", 1.4, 0.25, 0.40),

    ("DineIQ Smash Double Cheeseburger", "CAT-02", 13.99, 10.80, False, False, False, 10, 5, False, "POPULAR_LOW_MARGIN", 3.8, 0.30, 0.70),
    ("Crispy Buttermilk Chicken Sandwich", "CAT-02", 14.50, 11.00, False, False, False, 12, 5, False, "POPULAR_LOW_MARGIN", 3.4, 0.30, 0.65),
    ("Prime Wagyu Truffle Burger", "CAT-02", 28.00, 11.50, False, False, False, 14, 4, False, "PROFITABLE_LOW_SELLING", 0.8, 0.45, 0.45),
    ("BBQ Pulled Pork Brioche", "CAT-02", 15.00, 7.20, False, False, False, 8, 6, False, "STANDARD", 2.1, 0.35, 0.50),
    ("Spicy Nashville Hot Chicken Bun", "CAT-02", 15.50, 7.50, False, False, False, 12, 5, False, "STANDARD", 2.5, 0.30, 0.55),
    ("California Turkey Avocado Club", "CAT-02", 16.00, 8.90, False, False, False, 9, 3, False, "HIGH_WASTAGE", 1.6, 0.75, 0.50),
    ("Black Angus Bacon Cheddar", "CAT-02", 17.50, 8.00, False, False, False, 12, 5, False, "STANDARD", 2.6, 0.35, 0.50),
    ("Smoked Brisket Panini", "CAT-02", 17.00, 8.20, False, False, False, 10, 5, False, "STANDARD", 1.5, 0.30, 0.45),
    ("Portobello Mushroom Swiss Burger", "CAT-02", 15.00, 5.80, True, False, False, 11, 4, False, "STANDARD", 1.2, 0.40, 0.50),
    ("Philly Ribeye Cheesesteak", "CAT-02", 18.00, 9.20, False, False, False, 11, 4, False, "STANDARD", 1.8, 0.40, 0.50),
    ("Classic BLT Sourdough", "CAT-02", 11.99, 4.80, False, False, False, 7, 4, False, "PRICE_SENSITIVE", 2.2, 0.35, 0.80),
    ("Chipotle Fish Tacos (3pcs)", "CAT-02", 16.50, 8.70, False, True, False, 10, 2, False, "HIGH_WASTAGE", 1.4, 0.80, 0.55),
    ("Cubano Roast Pork Sandwich", "CAT-02", 15.50, 7.20, False, False, False, 11, 5, False, "STANDARD", 1.3, 0.25, 0.45),
    ("Truffle Lobster Roll", "CAT-02", 34.00, 14.50, False, False, False, 8, 2, False, "PROFITABLE_LOW_SELLING", 0.4, 0.85, 0.40),
    ("Grilled Halloumi Flatbread", "CAT-02", 14.00, 5.50, True, False, False, 9, 5, False, "STANDARD", 1.0, 0.35, 0.45),

    ("Margherita Sourdough Pizza", "CAT-03", 15.99, 12.00, True, False, False, 11, 5, False, "POPULAR_LOW_MARGIN", 3.7, 0.25, 0.70),
    ("Double Pepperoni Hot Honey Pizza", "CAT-03", 18.50, 13.80, False, False, False, 12, 6, False, "POPULAR_LOW_MARGIN", 3.6, 0.25, 0.65),
    ("Quattro Formaggi Bianca", "CAT-03", 18.00, 7.50, True, False, False, 11, 5, False, "STANDARD", 2.2, 0.30, 0.45),
    ("Prosciutto e Rucola Pizza", "CAT-03", 21.00, 9.80, False, False, False, 12, 3, False, "HIGH_WASTAGE", 1.5, 0.70, 0.50),
    ("Black Truffle Tagliolini Caviar", "CAT-03", 42.00, 11.50, True, False, False, 14, 5, False, "PROFITABLE_LOW_SELLING", 0.35, 0.45, 0.35),
    ("Slow-Braised Wild Boar Pappardelle", "CAT-03", 26.00, 9.50, False, False, False, 15, 5, False, "STANDARD", 1.1, 0.35, 0.40),
    ("Spaghetti All'Amatriciana", "CAT-03", 17.50, 5.50, False, False, False, 12, 6, False, "PRICE_SENSITIVE", 2.4, 0.25, 0.75),
    ("Creamy Fettuccine Alfredo", "CAT-03", 16.99, 5.20, True, False, False, 11, 6, False, "PRICE_SENSITIVE", 2.8, 0.20, 0.80),
    ("Handcrafted Lobster Ravioli", "CAT-03", 29.50, 11.20, False, False, False, 13, 3, False, "HIGH_WASTAGE", 0.9, 0.82, 0.45),
    ("Truffled Wild Mushroom Gnocchi", "CAT-03", 22.00, 7.80, True, False, False, 12, 4, False, "STANDARD", 1.6, 0.40, 0.45),
    ("Calabrian Sausage Rigatoni", "CAT-03", 20.00, 6.80, False, False, False, 13, 5, False, "STANDARD", 2.0, 0.30, 0.50),
    ("Spicy Diavola Pizza", "CAT-03", 19.00, 6.90, False, False, False, 12, 5, False, "STANDARD", 2.3, 0.25, 0.55),
    ("Burrata Caprese Pizza", "CAT-03", 20.50, 8.50, True, False, False, 11, 3, False, "HIGH_WASTAGE", 1.4, 0.75, 0.50),
    ("Lasagna Bolognese Al Forno", "CAT-03", 21.00, 7.90, False, False, False, 16, 5, False, "STANDARD", 1.9, 0.25, 0.45),
    ("Linguine alle Vongole (Clams)", "CAT-03", 25.00, 12.00, False, False, False, 14, 2, False, "HIGH_WASTAGE", 0.8, 0.90, 0.55),

    ("USDA Prime Center-Cut Filet Mignon (8oz)", "CAT-04", 48.00, 24.50, False, True, False, 18, 5, False, "STANDARD", 1.4, 0.40, 0.40),
    ("Dry-Aged Tomahawk Ribeye (36oz)", "CAT-04", 125.00, 38.00, False, True, False, 25, 6, False, "PROFITABLE_LOW_SELLING", 0.18, 0.35, 0.30),
    ("Dry-Aged Bone-in NY Strip (16oz)", "CAT-04", 56.00, 26.00, False, True, False, 20, 5, False, "STANDARD", 0.9, 0.40, 0.38),
    ("Colorado Rosemary Lamb Chops", "CAT-04", 44.00, 19.50, False, True, False, 17, 4, False, "STANDARD", 0.8, 0.50, 0.42),
    ("Slow-Braised Beef Short Ribs", "CAT-04", 36.00, 14.50, False, True, False, 15, 6, False, "STANDARD", 1.6, 0.30, 0.45),
    ("A5 Miyazaki Japanese Wagyu (4oz)", "CAT-04", 95.00, 32.00, False, True, False, 12, 5, False, "PROFITABLE_LOW_SELLING", 0.12, 0.40, 0.25),
    ("Steak Frites Garlic Herb Butter", "CAT-04", 29.99, 18.50, False, True, False, 15, 5, False, "POPULAR_LOW_MARGIN", 2.7, 0.35, 0.65),
    ("Smoked Bone-in Kurobuta Pork Chop", "CAT-04", 34.00, 13.80, False, True, False, 18, 5, False, "STANDARD", 0.9, 0.40, 0.40),
    ("Coffee-Crusted Prime Ribeye (14oz)", "CAT-04", 52.00, 23.00, False, True, False, 19, 5, False, "STANDARD", 1.1, 0.40, 0.40),
    ("Steakhouse Flat Iron Chimichurri", "CAT-04", 26.50, 13.20, False, True, False, 14, 5, False, "PRICE_SENSITIVE", 1.9, 0.35, 0.70),
    ("Pan-Roasted Duck Breast Plum Glaze", "CAT-04", 35.00, 14.00, False, True, False, 16, 4, False, "STANDARD", 0.7, 0.55, 0.40),
    ("Roasted Veal Chop Chanterelles", "CAT-04", 49.00, 22.00, False, True, False, 20, 4, False, "PROFITABLE_LOW_SELLING", 0.25, 0.45, 0.32),
    ("BBQ Glazed St. Louis Ribs (Full)", "CAT-04", 31.00, 14.00, False, True, False, 15, 6, False, "STANDARD", 1.7, 0.30, 0.50),
    ("Bistro Hanger Steak Shallot Jus", "CAT-04", 28.00, 13.50, False, True, False, 15, 5, False, "STANDARD", 1.3, 0.35, 0.55),
    ("Pepper-Crusted Bison Tenderloin", "CAT-04", 58.00, 25.00, False, True, False, 18, 4, False, "PROFITABLE_LOW_SELLING", 0.22, 0.42, 0.35),

    ("Pan-Seared Chilean Sea Bass", "CAT-05", 46.00, 18.50, False, True, False, 18, 3, False, "PROFITABLE_LOW_SELLING", 0.6, 0.75, 0.40),
    ("Crispy Skin Mediterranean Branzino", "CAT-05", 38.00, 16.00, False, True, False, 17, 2, False, "HIGH_WASTAGE", 0.7, 0.88, 0.45),
    ("Wild King Salmon Lemon Herb", "CAT-05", 34.00, 16.20, False, True, False, 15, 3, False, "STANDARD", 1.8, 0.65, 0.48),
    ("Butter Poached Maine Lobster Tail", "CAT-05", 52.00, 23.50, False, True, False, 14, 2, False, "HIGH_WASTAGE", 0.45, 0.92, 0.38),
    ("Jumbo Lump Crab Cakes (2pcs)", "CAT-05", 32.00, 15.00, False, False, False, 12, 3, False, "HIGH_WASTAGE", 1.0, 0.82, 0.50),
    ("Seared Diver Scallops Sweet Corn", "CAT-05", 36.00, 16.80, False, True, False, 12, 2, False, "HIGH_WASTAGE", 0.8, 0.90, 0.45),
    ("Seafood Paella Valenciana", "CAT-05", 35.00, 16.00, False, True, False, 22, 2, False, "HIGH_WASTAGE", 1.1, 0.85, 0.52),
    ("Classic Fish & Chips Tartar", "CAT-05", 19.99, 14.50, False, False, False, 12, 4, False, "POPULAR_LOW_MARGIN", 2.9, 0.30, 0.68),
    ("Grilled Yellowfin Ahi Tuna Steak", "CAT-05", 33.00, 14.80, False, True, False, 12, 2, False, "HIGH_WASTAGE", 0.9, 0.85, 0.46),
    ("Garlic Butter Tiger Prawns Skewer", "CAT-05", 27.00, 11.50, False, True, False, 11, 3, False, "STANDARD", 1.5, 0.50, 0.50),
    ("Steamed Blue Mussels White Wine", "CAT-05", 22.00, 9.50, False, True, False, 12, 2, False, "HIGH_WASTAGE", 1.1, 0.88, 0.52),
    ("Whole Grilled Red Snapper Mojo", "CAT-05", 42.00, 17.50, False, True, False, 20, 2, False, "HIGH_WASTAGE", 0.4, 0.86, 0.40),
    ("Pacific Black Cod Miso Glaze", "CAT-05", 44.00, 18.00, False, True, False, 16, 3, False, "PROFITABLE_LOW_SELLING", 0.5, 0.70, 0.38),
    ("Fried Softshell Crab Basket", "CAT-05", 26.00, 12.00, False, False, False, 11, 2, False, "HIGH_WASTAGE", 0.7, 0.84, 0.50),
    ("Seafood Bouillabaisse Saffron Rouille", "CAT-05", 37.00, 16.50, False, True, False, 18, 2, False, "HIGH_WASTAGE", 0.6, 0.89, 0.45),

    ("DineIQ Signature Caesar Salad", "CAT-06", 12.99, 8.80, False, False, False, 6, 4, False, "POPULAR_LOW_MARGIN", 3.2, 0.35, 0.68),
    ("Mediterranean Quinoa Crunch Bowl", "CAT-06", 14.50, 5.20, True, True, False, 7, 4, False, "STANDARD", 2.2, 0.40, 0.55),
    ("Burrata Caprese Heirloom Salad", "CAT-06", 16.50, 7.80, True, True, False, 6, 3, False, "HIGH_WASTAGE", 1.3, 0.78, 0.50),
    ("Strawberry Gorgonzola Pecan Salad", "CAT-06", 14.00, 5.00, True, True, False, 6, 3, False, "STANDARD", 1.6, 0.60, 0.55),
    ("Roasted Beet & Goat Cheese Bowl", "CAT-06", 15.00, 5.40, True, True, False, 7, 5, False, "STANDARD", 1.4, 0.45, 0.48),
    ("Thai Crunch Peanut Chicken Salad", "CAT-06", 16.00, 6.20, False, True, False, 8, 4, False, "STANDARD", 1.9, 0.45, 0.52),
    ("Wild Rice & Avocado Harvest Bowl", "CAT-06", 15.50, 7.50, True, True, False, 8, 2, False, "HIGH_WASTAGE", 1.5, 0.85, 0.55),
    ("Classic Greek Village Salad", "CAT-06", 13.50, 4.80, True, True, False, 6, 4, False, "PRICE_SENSITIVE", 2.1, 0.40, 0.72),
    ("Kale Tahini Caesar Roasted Chickpeas", "CAT-06", 13.99, 4.90, True, True, False, 7, 5, False, "STANDARD", 1.5, 0.35, 0.55),
    ("Blackened Salmon Cobb Salad", "CAT-06", 21.00, 9.80, False, True, False, 9, 3, False, "STANDARD", 1.7, 0.55, 0.50),
    ("Sesame Ahi Tuna Poke Bowl", "CAT-06", 19.50, 10.20, False, True, False, 8, 2, False, "HIGH_WASTAGE", 1.8, 0.88, 0.58),
    ("Green Goddess Farro Bowl", "CAT-06", 14.50, 4.80, True, False, False, 7, 5, False, "STANDARD", 1.2, 0.35, 0.50),
    ("Simple Garden House Salad", "CAT-06", 8.99, 3.20, True, True, False, 5, 4, False, "PRICE_SENSITIVE", 2.6, 0.35, 0.85),
    ("Southwest Chipotle Chicken Bowl", "CAT-06", 15.50, 5.80, False, True, False, 8, 4, False, "STANDARD", 2.3, 0.35, 0.60),
    ("Warm Roasted Butternut Squash Bowl", "CAT-06", 14.00, 4.50, True, True, False, 9, 6, True, "STANDARD", 0.9, 0.40, 0.45),

    ("Beyond Meat Gourmet Burger", "CAT-07", 16.99, 11.50, True, False, False, 11, 7, False, "POPULAR_LOW_MARGIN", 2.2, 0.25, 0.60),
    ("Truffle Wild Mushroom Risotto", "CAT-07", 23.00, 7.20, True, True, False, 16, 5, False, "STANDARD", 1.5, 0.40, 0.48),
    ("Crispy Cauliflower Buffalo Bites", "CAT-07", 12.00, 4.20, True, True, False, 9, 5, False, "STANDARD", 2.0, 0.30, 0.55),
    ("Creamy Cashew Alfredo Tagliatelle", "CAT-07", 19.00, 5.80, True, False, False, 12, 5, False, "STANDARD", 1.4, 0.35, 0.50),
    ("Spicy Tofu Peanut Noodle Bowl", "CAT-07", 16.50, 4.90, True, False, False, 10, 5, False, "STANDARD", 1.7, 0.35, 0.52),
    ("Roasted Eggplant Harissa Shakshuka", "CAT-07", 15.00, 4.50, True, True, False, 13, 4, False, "STANDARD", 1.1, 0.40, 0.48),
    ("Jackfruit Smoky Carnitas Tacos", "CAT-07", 15.50, 5.10, True, True, False, 9, 5, False, "STANDARD", 1.6, 0.35, 0.54),
    ("Avocado Supergreen Toast Sourdough", "CAT-07", 13.50, 6.80, True, False, False, 7, 2, False, "HIGH_WASTAGE", 1.9, 0.85, 0.60),
    ("Lentil Shepherd's Pie Sweet Potato", "CAT-07", 17.50, 5.20, True, True, False, 15, 6, False, "STANDARD", 1.0, 0.30, 0.45),
    ("Crispy Falafel Mezze Platter", "CAT-07", 16.00, 5.00, True, False, False, 10, 6, False, "STANDARD", 1.8, 0.35, 0.52),
    ("Thai Coconut Green Curry Tofu", "CAT-07", 18.00, 5.50, True, True, False, 12, 5, False, "STANDARD", 1.6, 0.40, 0.50),
    ("Mushroom & Walnut Bolognese", "CAT-07", 18.50, 5.60, True, False, False, 12, 5, False, "STANDARD", 1.3, 0.35, 0.50),
    ("Korean Gochujang Crispy Tofu", "CAT-07", 14.50, 4.40, True, True, False, 10, 6, False, "STANDARD", 1.4, 0.30, 0.55),
    ("Golden Turmeric Lentil Soup Bowl", "CAT-07", 9.99, 2.80, True, True, False, 6, 7, False, "PRICE_SENSITIVE", 2.0, 0.25, 0.78),
    ("Artisanal Vegan Charcuterie Board", "CAT-07", 26.00, 8.50, True, False, False, 8, 8, False, "PROFITABLE_LOW_SELLING", 0.4, 0.40, 0.35),

    ("Molten Chocolate Lava Cake", "CAT-08", 10.99, 7.80, True, False, False, 12, 7, False, "POPULAR_LOW_MARGIN", 3.3, 0.25, 0.55),
    ("Classic Madagascar Vanilla Bean Gelato", "CAT-08", 7.50, 1.80, True, True, False, 3, 30, False, "PRICE_SENSITIVE", 2.8, 0.15, 0.70),
    ("New York Style Cheesecake", "CAT-08", 9.99, 6.90, True, False, False, 4, 6, False, "POPULAR_LOW_MARGIN", 2.9, 0.30, 0.60),
    ("Warm Apple Cinnamon Crisp", "CAT-08", 9.50, 2.80, True, False, False, 10, 5, True, "STANDARD", 1.8, 0.35, 0.55),
    ("Fresh Wild Berry Pavlova", "CAT-08", 13.50, 5.80, True, True, False, 8, 1, False, "HIGH_WASTAGE", 0.9, 0.92, 0.50),
    ("Traditional Italian Tiramisu", "CAT-08", 10.50, 3.20, True, False, False, 4, 4, False, "STANDARD", 2.4, 0.40, 0.55),
    ("Salted Caramel Creme Brulee", "CAT-08", 11.00, 3.10, True, True, False, 7, 4, False, "STANDARD", 2.1, 0.45, 0.50),
    ("Vintage Champagne Berry Float", "CAT-08", 24.00, 5.50, True, True, True, 5, 3, False, "PROFITABLE_LOW_SELLING", 0.25, 0.50, 0.30),
    ("Warm Churros with Spiced Dulce", "CAT-08", 9.00, 2.40, True, False, False, 8, 4, False, "STANDARD", 2.2, 0.25, 0.60),
    ("Pecan Pie Bourbon Whipped Cream", "CAT-08", 10.00, 3.00, True, False, False, 5, 7, True, "STANDARD", 1.2, 0.30, 0.50),
    ("Key Lime Tart Graham Crust", "CAT-08", 9.50, 2.70, True, False, False, 5, 5, False, "STANDARD", 1.6, 0.40, 0.55),
    ("Artisanal Macaron Assortment (6pcs)", "CAT-08", 15.00, 4.20, True, True, False, 4, 10, False, "STANDARD", 1.4, 0.35, 0.45),
    ("Dark Chocolate Souffle Grand Marnier", "CAT-08", 16.50, 4.00, True, False, True, 18, 2, False, "PROFITABLE_LOW_SELLING", 0.35, 0.60, 0.35),
    ("Matcha Green Tea Crepe Cake", "CAT-08", 11.50, 3.20, True, False, False, 5, 3, False, "HIGH_WASTAGE", 1.0, 0.80, 0.50),
    ("Affogato al Caffe Espresso Scoop", "CAT-08", 6.99, 1.50, True, True, False, 3, 20, False, "PRICE_SENSITIVE", 2.3, 0.15, 0.75),

    ("DineIQ House Draft Lager (Pint)", "CAT-09", 6.50, 4.80, True, False, True, 2, 60, False, "POPULAR_LOW_MARGIN", 3.6, 0.10, 0.85),
    ("Signature Smoked Old Fashioned", "CAT-09", 16.00, 3.20, True, True, True, 4, 365, False, "STANDARD", 2.5, 0.10, 0.45),
    ("Spicy Passionfruit Mezcalita", "CAT-09", 15.50, 3.10, True, True, True, 4, 60, False, "STANDARD", 2.2, 0.15, 0.50),
    ("Elderflower French 75 Champagne", "CAT-09", 17.00, 3.60, True, True, True, 4, 180, False, "STANDARD", 1.8, 0.15, 0.42),
    ("Black Truffle Infused Manhattan", "CAT-09", 28.00, 4.50, True, True, True, 5, 180, False, "PROFITABLE_LOW_SELLING", 0.3, 0.10, 0.30),
    ("Espresso Martini Vanilla Foam", "CAT-09", 15.00, 2.90, True, True, True, 4, 30, False, "STANDARD", 2.6, 0.20, 0.50),
    ("Zero-Proof Cucumber Mint Cooler", "CAT-09", 8.50, 1.60, True, True, False, 3, 7, False, "PRICE_SENSITIVE", 2.1, 0.30, 0.72),
    ("Hibiscus Berry Botanical Fizz (Mocktail)", "CAT-09", 8.99, 1.80, True, True, False, 3, 7, False, "STANDARD", 1.9, 0.30, 0.65),
    ("Smoked Rosemary Paloma Tequila", "CAT-09", 14.50, 2.80, True, True, True, 4, 60, False, "STANDARD", 2.0, 0.15, 0.50),
    ("Classic Moscow Mule Copper Mug", "CAT-09", 13.50, 2.50, True, True, True, 3, 180, False, "STANDARD", 2.4, 0.10, 0.55),
    ("Yuzu Lavender Gin Tonic", "CAT-09", 14.00, 2.70, True, True, True, 3, 180, False, "STANDARD", 1.9, 0.15, 0.48),
    ("Aperol Spritz Prosecco Orange", "CAT-09", 13.99, 2.60, True, True, True, 3, 90, False, "STANDARD", 2.7, 0.15, 0.55),
    ("Rare Japanese Whisky Flight (3-pour)", "CAT-09", 55.00, 14.00, True, True, True, 5, 365, False, "PROFITABLE_LOW_SELLING", 0.2, 0.05, 0.28),
    ("Strawberry Basil Smash Lemonade (Mocktail)", "CAT-09", 8.50, 1.70, True, True, False, 3, 5, False, "PRICE_SENSITIVE", 2.0, 0.40, 0.70),
    ("Red Sangria Pitcher (House Wine)", "CAT-09", 29.00, 6.20, True, True, True, 5, 30, False, "STANDARD", 1.5, 0.15, 0.50),

    ("DineIQ House Drip Roast Coffee", "CAT-10", 3.99, 0.60, True, True, False, 2, 60, False, "PRICE_SENSITIVE", 4.2, 0.05, 0.90),
    ("Nitro Cold Brew Sweet Cream", "CAT-10", 5.99, 1.10, True, True, False, 2, 14, False, "STANDARD", 3.0, 0.10, 0.65),
    ("Double Espresso Single Origin", "CAT-10", 3.50, 0.50, True, True, False, 2, 60, False, "PRICE_SENSITIVE", 3.2, 0.05, 0.85),
    ("Vanilla Bean Oat Milk Latte", "CAT-10", 5.75, 1.20, True, True, False, 3, 14, False, "STANDARD", 3.5, 0.10, 0.60),
    ("Caramel Macchiato Sea Salt", "CAT-10", 5.99, 1.25, True, True, False, 3, 14, False, "STANDARD", 3.1, 0.10, 0.62),
    ("Ceremonial Iced Matcha Latte", "CAT-10", 6.50, 1.40, True, True, False, 3, 30, False, "STANDARD", 2.2, 0.15, 0.55),
    ("Spiced Masala Chai Latte", "CAT-10", 5.25, 0.90, True, True, False, 3, 30, False, "STANDARD", 2.0, 0.10, 0.58),
    ("Organic Earl Grey Loose Leaf Tea", "CAT-10", 4.25, 0.45, True, True, False, 3, 180, False, "STANDARD", 1.8, 0.05, 0.70),
    ("Fresh Squeezed Orange Juice", "CAT-10", 5.50, 2.60, True, True, False, 3, 2, False, "HIGH_WASTAGE", 2.3, 0.85, 0.65),
    ("Sparkling San Pellegrino (750ml)", "CAT-10", 6.00, 1.50, True, True, False, 1, 365, False, "STANDARD", 1.9, 0.05, 0.60),
    ("Acai Berry Superfood Smoothie", "CAT-10", 7.99, 3.40, True, True, False, 4, 2, False, "HIGH_WASTAGE", 1.6, 0.80, 0.58),
    ("Belgian Hot Chocolate Marshmallows", "CAT-10", 5.50, 1.10, True, False, False, 3, 60, True, "STANDARD", 1.7, 0.10, 0.60),
    ("Cold Pressed Green Detox Juice", "CAT-10", 8.50, 4.10, True, True, False, 2, 2, False, "HIGH_WASTAGE", 1.1, 0.90, 0.62),
    ("Rare Blue Mountain Jamaican Pour-Over", "CAT-10", 12.00, 2.50, True, True, False, 6, 90, False, "PROFITABLE_LOW_SELLING", 0.3, 0.05, 0.35),
    ("Artisanal Kombucha Peach Ginger", "CAT-10", 6.25, 1.80, True, True, False, 2, 45, False, "STANDARD", 1.5, 0.10, 0.55)
]

def generate_menu_items(output_path: str = None) -> pd.DataFrame:
    """Generate 150 menu items per SRS requirement with business complexity tags."""
    assert len(ITEMS_DATA) == VOLUME_TARGETS["menu_items"], f"Expected 150 items, got {len(ITEMS_DATA)}"
    records = []

    for i, item in enumerate(ITEMS_DATA):
        item_id = f"ITEM-{i+1:03d}"
        (name, cat_id, base_price, cost_price, is_veg, is_gf, is_alc, prep_time, shelf_life, 
         is_seasonal, complexity, pop_weight, waste_risk, elasticity) = item

        margin_pct = round(((base_price - cost_price) / base_price) * 100, 2)

        record = {
            "item_id": item_id,
            "category_id": cat_id,
            "name": name,
            "description": f"DineIQ signature {name.lower()} crafted with premium ingredients.",
            "base_price": round(base_price, 2),
            "cost_price": round(cost_price, 2),
            "margin_pct": margin_pct,
            "is_vegetarian": is_veg,
            "is_gluten_free": is_gf,
            "is_alcohol": is_alc,
            "prep_time_minutes": prep_time,
            "shelf_life_days": shelf_life,
            "is_seasonal": is_seasonal,
            "complexity_profile": complexity,
            "popularity_weight": pop_weight,
            "wastage_risk_score": waste_risk,
            "elasticity_score": elasticity,
            "is_active": True
        }
        records.append(record)

    df = pd.DataFrame(records)

    if output_path is None:
        target_dir = os.path.join(RAW_DATA_DIR, "menu_items")
        ensure_dir(target_dir)
        output_path = os.path.join(target_dir, "menu_items.csv")

    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"[OK] Generated {len(df)} menu items -> {output_path}")

    print("\n--- Menu Items Complexity Breakdown ---")
    print(df["complexity_profile"].value_counts().to_string())
    return df

if __name__ == "__main__":
    generate_menu_items()
