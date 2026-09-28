                                                          
                                                                           
SELECT 
    wastage_reason,
    COUNT(wastage_id) AS waste_incident_count,
    ROUND(SUM(quantity_wasted), 2) AS total_quantity_wasted_kg,
    ROUND(SUM(total_loss_amount), 2) AS total_financial_loss_dollars
FROM wastage
GROUP BY wastage_reason
ORDER BY total_financial_loss_dollars DESC;
