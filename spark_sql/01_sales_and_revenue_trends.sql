                                          
                                                                                    
SELECT 
    order_date,
    COUNT(DISTINCT order_id) AS total_orders,
    ROUND(SUM(total_amount), 2) AS gross_revenue,
    ROUND(AVG(total_amount), 2) AS average_order_value,
    ROUND(SUM(discount_amount), 2) AS total_discounts_given
FROM orders
GROUP BY order_date
ORDER BY order_date ASC;
