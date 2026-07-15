"""Search funnel, rank CTR, zero-result, and buyer cohort analytics."""
from __future__ import annotations
import numpy as np
import pandas as pd

RANK_LABELS = ["1-5", "6-10", "11-20", "21-50"]


def rank_ctr(searches: pd.DataFrame) -> pd.DataFrame:
    clicked = searches.dropna(subset=["click_rank_position"]).copy()
    counts = clicked.click_rank_position.astype(int).value_counts().reindex(range(1, 51), fill_value=0)
    return pd.DataFrame({"rank": range(1, 51), "ctr": counts.to_numpy() / len(searches)})


def search_funnel(searches: pd.DataFrame, tx: pd.DataFrame) -> pd.DataFrame:
    x = searches.copy()
    x["rank_bucket"] = pd.cut(x.click_rank_position, [0, 5, 10, 20, 50], labels=RANK_LABELS)
    clicks = x.dropna(subset=["rank_bucket"]).groupby(["category", "rank_bucket"], observed=True).size().rename("clicks")
    searches_by_cat = x.groupby("category").size().rename("searches")
    purchased = set(tx.listing_id)
    p = x[x.clicked_listing_id.isin(purchased)].groupby(["category", "rank_bucket"], observed=True).size().rename("purchases")
    out = clicks.to_frame().join(p, how="left").reset_index().merge(searches_by_cat, on="category")
    out["purchases"] = out.purchases.fillna(0).astype(int)
    out["ctr"] = out.clicks / out.searches
    out["purchase_through_rate"] = out.purchases / out.searches
    return out


def zero_result_trend(searches: pd.DataFrame) -> pd.DataFrame:
    x = searches.assign(month=pd.to_datetime(searches.timestamp).dt.to_period("M").dt.to_timestamp(), zero=searches.result_count.eq(0))
    return x.groupby(["month", "category"]).zero.mean().rename("zero_result_rate").reset_index()


def cliff_point(searches: pd.DataFrame) -> int:
    curve = rank_ctr(searches)
    smooth = curve.ctr.rolling(3, center=True, min_periods=1).mean()
    return int(curve.loc[smooth.diff().idxmin(), "rank"])


def cohort_retention(tx: pd.DataFrame, periods: int = 6) -> pd.DataFrame:
    x = tx.copy()
    x["order_month"] = pd.to_datetime(x.order_date).dt.to_period("M")
    first = x.sort_values("order_date").groupby("buyer_id").first()[["category", "order_month"]].rename(columns={"category": "first_category", "order_month": "cohort_month"})
    x = x.join(first, on="buyer_id")
    x["period"] = (x.order_month.astype(int) - x.cohort_month.astype(int)).clip(lower=0)
    base = first.groupby("first_category").size()
    retained = x[x.period < periods].groupby(["first_category", "period"]).buyer_id.nunique().unstack(fill_value=0)
    return retained.div(base, axis=0).reindex(columns=range(periods), fill_value=0)

