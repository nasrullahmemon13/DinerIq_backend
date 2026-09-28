                                                               
                                                                                   
SELECT 
    p.promotion_id,
    p.promotion_name,
    p.discount_type,
    p.discount_value,
    COUNT(DISTINCT o.order_id) AS promo_orders_count,
    ROUND(SUM(o.total_amount), 2) AS promo_generated_revenue,
    ROUND(SUM(o.discount_amount), 2) AS promo_discounts_absorbed
FROM orders o
JOIN promotions p ON o.promotion_id = p.promotion_id
GROUP BY p.promotion_id, p.promotion_name, p.discount_type, p.discount_value
ORDER BY promo_generated_revenue DESC;
