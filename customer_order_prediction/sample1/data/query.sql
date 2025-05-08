WITH last_order_date AS (
    SELECT MAX(order_date) AS max_date
    FROM orders
),
customer_order_stats AS (
    SELECT
        c.customer_id,
        COUNT(o.order_id) AS total_orders,
        SUM(od.unit_price * od.quantity) AS total_spent,
        AVG(od.unit_price * od.quantity) AS avg_order_value
    FROM orders o
    INNER JOIN customers c ON o.customer_id = c.customer_id
    INNER JOIN order_details od ON od.order_id = o.order_id
    GROUP BY c.customer_id
),
label_date AS (
    SELECT
        c.customer_id,
        CASE
            WHEN EXISTS (
                SELECT 1
                FROM orders o2
                CROSS JOIN last_order_date lod
                WHERE o2.customer_id = c.customer_id
                  AND o2.order_date > (lod.max_date - INTERVAL '6 months')
            )
            THEN 1 ELSE 0
        END AS will_order_again
    FROM customers c
)

SELECT s.customer_id,
s.total_orders,
s.total_spent,
s.avg_order_value,
l.will_order_again
FROM customer_order_stats s
JOIN label_date l
ON s.customer_id = l.customer_id;
