                                                                        
                                                                         
SELECT 
    order_type AS fulfillment_channel,
    COUNT(DISTINCT order_id) AS total_transactions,
    ROUND(SUM(total_amount), 2) AS channel_gross_revenue,
    ROUND(AVG(total_amount), 2) AS average_ticket_size,
    ROUND(SUM(delivery_fee), 2) AS total_delivery_fees_collected
FROM orders
GROUP BY order_type
ORDER BY channel_gross_revenue DESC;
