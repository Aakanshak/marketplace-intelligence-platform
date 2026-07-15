WITH bucketed AS (
  SELECT category,
    CASE WHEN click_rank_position BETWEEN 1 AND 5 THEN '1-5'
         WHEN click_rank_position BETWEEN 6 AND 10 THEN '6-10'
         WHEN click_rank_position BETWEEN 11 AND 20 THEN '11-20'
         WHEN click_rank_position BETWEEN 21 AND 50 THEN '21-50' END AS rank_bucket,
    search_id, clicked_listing_id
  FROM search_events
)
SELECT b.category, b.rank_bucket, COUNT(DISTINCT b.search_id) AS clicks,
       COUNT(DISTINCT CASE WHEN t.transaction_id IS NOT NULL THEN b.search_id END) AS purchases
FROM bucketed b LEFT JOIN transactions t ON b.clicked_listing_id = t.listing_id
WHERE b.rank_bucket IS NOT NULL GROUP BY 1, 2 ORDER BY 1, 2;

