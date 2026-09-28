                                                                
                                                                   
SELECT 
    c.customer_id,
    c.customer_segment,
    c.loyalty_tier,
    COUNT(DISTINCT o.order_id) AS completed_orders_count,
    ROUND(SUM(o.total_amount), 2) AS lifetime_monetary_spend,
    ROUND(AVG(o.total_amount), 2) AS average_ticket_size,
    MAX(o.order_date) AS most_recent_order_date
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
WHERE UPPER(o.order_status) = 'COMPLETED'
GROUP BY c.customer_id, c.customer_segment, c.loyalty_tier
ORDER BY lifetime_monetary_spend DESC;
