WITH seller_sales AS (
 SELECT seller_id, MAX(order_date) AS last_sale, COUNT(*) AS orders,
        SUM(price_paid * quantity) AS gmv FROM transactions GROUP BY 1
), listing_frequency AS (
 SELECT seller_id, COUNT(*) AS listings FROM listings GROUP BY 1
)
SELECT s.seller_id, s.category, s.fulfillment_rate, ss.last_sale,
       COALESCE(ss.orders, 0) AS orders, COALESCE(ss.gmv, 0) AS gmv,
       COALESCE(lf.listings, 0) AS listing_frequency
FROM sellers s LEFT JOIN seller_sales ss USING (seller_id)
LEFT JOIN listing_frequency lf USING (seller_id);
