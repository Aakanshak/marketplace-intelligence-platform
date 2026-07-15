"""GMV, seller health, elasticity, and slowdown diagnostics."""
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm


def monthly_gmv_decomposition(tx: pd.DataFrame) -> pd.DataFrame:
    x = tx.copy()
    x["month"] = pd.to_datetime(x.order_date).dt.to_period("M").dt.to_timestamp()
    x["gmv"] = x.price_paid * x.quantity
    out = x.groupby(["month", "category"]).agg(gmv=("gmv", "sum"), active_buyers=("buyer_id", "nunique"),
        orders=("transaction_id", "nunique")).reset_index()
    out["orders_per_buyer"] = out.orders / out.active_buyers
    out["aov"] = out.gmv / out.orders
    for col in ["gmv", "active_buyers", "orders_per_buyer", "aov"]:
        out[f"{col}_mom"] = out.groupby("category")[col].pct_change()
    return out


def seller_health(sellers: pd.DataFrame, listings: pd.DataFrame, tx: pd.DataFrame,
                  as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    as_of = as_of or (pd.to_datetime(tx.order_date).max() + pd.Timedelta(days=1))
    t = tx.assign(order_date=pd.to_datetime(tx.order_date), gmv=tx.price_paid * tx.quantity)
    agg = t.groupby("seller_id").agg(last_sale=("order_date", "max"), monetary=("gmv", "sum"), orders=("transaction_id", "nunique"))
    freq = listings.groupby("seller_id").size().rename("listing_frequency")
    out = sellers.merge(agg, on="seller_id", how="left").merge(freq, on="seller_id", how="left")
    out["recency_days"] = (as_of - out.last_sale).dt.days.fillna(999)
    out[["monetary", "orders", "listing_frequency"]] = out[["monetary", "orders", "listing_frequency"]].fillna(0)
    def score(s: pd.Series, high_good: bool = True) -> pd.Series:
        ranked = s.rank(method="average", pct=True)
        return (ranked * 4).clip(1, 4).astype(int) if high_good else ((1 - ranked) * 4 + 1).clip(1, 4).astype(int)
    out["health_score"] = score(out.recency_days, False) + score(out.listing_frequency) + score(out.monetary)
    out["health_segment"] = np.select([out.is_churned | (out.recency_days > 120), out.health_score < 7], ["Churned", "At-Risk"], default="Healthy")
    return out


def price_elasticity(tx: pd.DataFrame) -> pd.DataFrame:
    x = tx.copy()
    x["month"] = pd.to_datetime(x.order_date).dt.to_period("M")
    panel = x.groupby(["category", "month"]).agg(quantity=("quantity", "sum"), avg_price=("price_paid", "mean")).reset_index()
    rows = []
    for cat, g in panel.groupby("category"):
        X = sm.add_constant(np.log(g.avg_price))
        model = sm.OLS(np.log(g.quantity), X).fit()
        rows.append({"category": cat, "elasticity": model.params["avg_price"], "p_value": model.pvalues["avg_price"], "r_squared": model.rsquared})
    return pd.DataFrame(rows).sort_values("elasticity")


def slowdown_diagnosis(decomp: pd.DataFrame) -> dict:
    total = decomp.groupby("month").agg(gmv=("gmv", "sum"), buyers=("active_buyers", "sum"), orders=("orders", "sum")).reset_index()
    total["opb"] = total.orders / total.buyers
    total["aov"] = total.gmv / total.orders
    latest, prior = total.iloc[-1], total.iloc[-4]
    gmv_change = latest.gmv - prior.gmv
    # Shapley-like first-order decomposition around the prior-period baseline.
    buyer_impact = (latest.buyers - prior.buyers) * prior.opb * prior.aov
    frequency_impact = prior.buyers * (latest.opb - prior.opb) * prior.aov
    aov_impact = prior.buyers * prior.opb * (latest.aov - prior.aov)
    return {"latest_month": str(latest.month.date()), "gmv_change": gmv_change,
        "gmv_change_pct": latest.gmv / prior.gmv - 1, "buyer_impact": buyer_impact,
        "frequency_impact": frequency_impact, "aov_impact": aov_impact,
        "latest_gmv": latest.gmv, "prior_gmv": prior.gmv}


def churn_gmv_relationship(sellers: pd.DataFrame, tx: pd.DataFrame) -> dict:
    churn = sellers.groupby("category").is_churned.mean().rename("churn_rate")
    x = tx.assign(month=pd.to_datetime(tx.order_date).dt.to_period("M"), gmv=tx.price_paid * tx.quantity)
    g = x.groupby(["category", "month"]).gmv.sum().unstack()
    decline = (g.iloc[:, -1] / g.iloc[:, -4] - 1).rename("gmv_change")
    frame = pd.concat([churn, decline], axis=1).dropna()
    model = sm.OLS(frame.gmv_change, sm.add_constant(frame.churn_rate)).fit()
    return {"correlation": frame.corr().iloc[0, 1], "coefficient": model.params["churn_rate"], "p_value": model.pvalues["churn_rate"]}

