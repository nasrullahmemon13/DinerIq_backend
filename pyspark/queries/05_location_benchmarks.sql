                                                        
                                                                                 
SELECT 
    r.location_id,
    r.name AS restaurant_name,
    r.city,
    COUNT(DISTINCT o.order_id) AS total_orders_fulfilled,
    ROUND(SUM(o.total_amount), 2) AS total_location_revenue,
    ROUND(AVG(o.total_amount), 2) AS average_order_ticket
FROM orders o
JOIN restaurants r ON o.location_id = r.location_id
GROUP BY r.location_id, r.name, r.city
ORDER BY total_location_revenue DESC;
