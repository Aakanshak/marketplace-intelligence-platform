WITH monthly AS (
  SELECT date_trunc('month', order_date) AS month, category,
         SUM(price_paid * quantity) AS gmv,
         COUNT(DISTINCT buyer_id) AS active_buyers,
         COUNT(DISTINCT transaction_id) AS orders
  FROM transactions GROUP BY 1, 2
), metrics AS (
  SELECT *, orders * 1.0 / active_buyers AS orders_per_buyer,
         gmv / orders AS aov FROM monthly
)
SELECT *,
  gmv / LAG(gmv) OVER (PARTITION BY category ORDER BY month) - 1 AS gmv_mom,
  active_buyers * 1.0 / LAG(active_buyers) OVER (PARTITION BY category ORDER BY month) - 1 AS buyers_mom,
  orders_per_buyer / LAG(orders_per_buyer) OVER (PARTITION BY category ORDER BY month) - 1 AS frequency_mom,
  aov / LAG(aov) OVER (PARTITION BY category ORDER BY month) - 1 AS aov_mom
FROM metrics ORDER BY month, category;

