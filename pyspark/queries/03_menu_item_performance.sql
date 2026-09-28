                                                         
                                                                        
SELECT 
    m.item_id,
    m.name AS item_name,
    c.category_name,
    m.base_price,
    m.cost_price,
    SUM(oi.quantity) AS total_quantity_sold,
    ROUND(SUM(oi.item_total), 2) AS total_item_revenue,
    COUNT(DISTINCT oi.order_id) AS distinct_orders_containing_item
FROM order_items oi
JOIN menu_items m ON oi.item_id = m.item_id
JOIN menu_categories c ON m.category_id = c.category_id
GROUP BY m.item_id, m.name, c.category_name, m.base_price, m.cost_price
ORDER BY total_quantity_sold DESC;
