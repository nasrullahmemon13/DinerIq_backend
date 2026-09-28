                                                             
                                                                                  
SELECT 
    c.category_name,
    COUNT(oi.order_item_id) AS total_items_sold,
    ROUND(SUM(oi.item_total), 2) AS total_revenue,
    ROUND(SUM(oi.quantity * m.cost_price), 2) AS total_ingredient_cost,
    ROUND(SUM(oi.item_total) - SUM(oi.quantity * m.cost_price), 2) AS contribution_margin,
    ROUND(((SUM(oi.item_total) - SUM(oi.quantity * m.cost_price)) / SUM(oi.item_total)) * 100, 2) AS margin_percentage
FROM order_items oi
JOIN menu_items m ON oi.item_id = m.item_id
JOIN menu_categories c ON m.category_id = c.category_id
GROUP BY c.category_name
ORDER BY contribution_margin DESC;
