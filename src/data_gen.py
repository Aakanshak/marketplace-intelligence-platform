"""Deterministic synthetic marketplace data generation.

The generated tables intentionally contain discoverable business signals while
remaining related through valid seller, listing, buyer, and category keys.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

SEED = 42
CATEGORIES = ["Electronics", "Home", "Beauty", "Fashion", "Sports", "Books"]
CATEGORY_BASE_PRICE = dict(zip(CATEGORIES, [115, 62, 31, 48, 72, 22]))
CATEGORY_ELASTICITY = dict(zip(CATEGORIES, [-1.65, -1.20, -0.70, -1.45, -0.95, -0.35]))
POOR_SEARCH_CATEGORIES = {"Electronics", "Fashion"}


@dataclass(frozen=True)
class GenerationConfig:
    n_sellers: int = 800
    n_listings: int = 5_000
    n_searches: int = 320_000
    target_transactions: int = 42_000
    seed: int = SEED
    end_date: str = "2026-06-30"


def _write_table(df: pd.DataFrame, output_dir: Path, name: str) -> None:
    df.to_csv(output_dir / f"{name}.csv", index=False)
    df.to_parquet(output_dir / f"{name}.parquet", index=False)


def generate_sellers(cfg: GenerationConfig, rng: np.random.Generator) -> pd.DataFrame:
    end = pd.Timestamp(cfg.end_date)
    fulfillment = np.clip(rng.beta(18, 2, cfg.n_sellers), 0.55, 0.999)
    low = fulfillment < 0.90
    # A stopped-listing flag is exposed as churned; the probability ratio is 4x.
    churned = rng.random(cfg.n_sellers) < np.where(low, 0.32, 0.08)
    sellers = pd.DataFrame({
        "seller_id": [f"S{i:04d}" for i in range(1, cfg.n_sellers + 1)],
        "join_date": end - pd.to_timedelta(rng.integers(30, 1_400, cfg.n_sellers), unit="D"),
        "category": rng.choice(CATEGORIES, cfg.n_sellers, p=[.18, .18, .14, .22, .14, .14]),
        "fulfillment_rate": fulfillment.round(4),
        "avg_rating": np.clip(2.8 + 2.0 * fulfillment + rng.normal(0, .18, cfg.n_sellers), 2.5, 5).round(2),
        "inventory_count": rng.lognormal(4.1, .75, cfg.n_sellers).astype(int).clip(1, 800),
        "is_churned": churned,
    })
    return sellers


def generate_listings(cfg: GenerationConfig, sellers: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    weights = np.where(sellers["is_churned"], .35, 1.0) * (sellers["inventory_count"] + 20)
    seller_idx = rng.choice(len(sellers), cfg.n_listings, p=weights / weights.sum())
    chosen = sellers.iloc[seller_idx].reset_index(drop=True)
    base_price = chosen["category"].map(CATEGORY_BASE_PRICE).to_numpy()
    price = base_price * rng.lognormal(0, .38, cfg.n_listings)
    created = pd.Timestamp(cfg.end_date) - pd.to_timedelta(rng.integers(1, 730, cfg.n_listings), unit="D")
    active = (~chosen["is_churned"].to_numpy()) & (rng.random(cfg.n_listings) > .06)
    return pd.DataFrame({
        "listing_id": [f"L{i:05d}" for i in range(1, cfg.n_listings + 1)],
        "seller_id": chosen["seller_id"],
        "category": chosen["category"],
        "price": price.round(2),
        "base_demand_score": np.clip(rng.beta(2.3, 3.2, cfg.n_listings) * 100, 1, 99).round(2),
        "created_date": created,
        "is_active": active,
    })


def generate_search_events(cfg: GenerationConfig, listings: pd.DataFrame, sellers: pd.DataFrame,
                           rng: np.random.Generator) -> pd.DataFrame:
    end = pd.Timestamp(cfg.end_date) + pd.Timedelta(hours=23, minutes=59)
    start = end - pd.DateOffset(months=12) + pd.Timedelta(days=1)
    category = rng.choice(CATEGORIES, cfg.n_searches, p=[.21, .18, .13, .23, .13, .12])
    buyer_ids = np.array([f"B{i:05d}" for i in range(1, 24_001)])
    buyer = rng.choice(buyer_ids, cfg.n_searches)
    seconds = rng.integers(0, int((end - start).total_seconds()), cfg.n_searches)
    timestamp = start + pd.to_timedelta(seconds, unit="s")
    listing_seller = listings.merge(sellers[["seller_id", "fulfillment_rate"]], on="seller_id")
    pools: dict[str, np.ndarray] = {}
    for cat in CATEGORIES:
        x = listing_seller[(listing_seller.category == cat) & listing_seller.is_active].copy()
        score = x.base_demand_score * (.55 + .45 * x.fulfillment_rate)
        # Poor-search categories systematically bury useful inventory.
        if cat in POOR_SEARCH_CATEGORIES:
            score = .55 * score + rng.normal(0, 25, len(x))
        pools[cat] = x.iloc[np.argsort(-score.to_numpy())].listing_id.to_numpy()

    zero_prob = {"Electronics": .075, "Fashion": .065, "Home": .035, "Beauty": .025,
                 "Sports": .04, "Books": .02}
    is_zero = rng.random(cfg.n_searches) < np.array([zero_prob[c] for c in category])
    n_results = np.where(is_zero, 0, rng.integers(12, 51, cfg.n_searches))
    # Click propensity is calibrated by observed rank: top 5 ~18%, 6-10 ~10%, below 10 ~2%.
    # Outcomes are four click-rank buckets plus no-click. Thus the bucket
    # probabilities are also CTR contributions over the full search population.
    click_bucket = rng.choice(5, cfg.n_searches, p=[.18, .10, .012, .008, .70])
    clicked = (click_bucket < 4) & ~is_zero
    ranges = [(1, 6), (6, 11), (11, 21), (21, 51)]
    click_rank = np.full(cfg.n_searches, np.nan)
    for b, (lo, hi) in enumerate(ranges):
        mask = clicked & (click_bucket == b)
        click_rank[mask] = rng.integers(lo, hi, mask.sum())

    result_json: list[str] = []
    clicked_ids: list[object] = []
    queries = []
    adjectives = np.array(["best", "new", "budget", "premium", "popular", "sale"])
    for i, cat in enumerate(category):
        queries.append(f"{rng.choice(adjectives)} {cat.lower()}")
        if n_results[i] == 0:
            result_json.append("[]")
            clicked_ids.append(None)
            continue
        pool = pools[cat]
        offset_max = max(1, min(40, len(pool) - int(n_results[i])))
        offset = int(rng.integers(0, offset_max))
        ids = pool[offset: offset + int(n_results[i])].tolist()
        result_json.append(json.dumps(ids))
        rank = click_rank[i]
        clicked_ids.append(ids[min(int(rank) - 1, len(ids) - 1)] if not np.isnan(rank) else None)

    return pd.DataFrame({
        "search_id": [f"Q{i:07d}" for i in range(1, cfg.n_searches + 1)],
        "buyer_id": buyer,
        "timestamp": timestamp,
        "query": queries,
        "category": category,
        "results_returned": result_json,
        "result_count": n_results,
        "clicked_listing_id": clicked_ids,
        "click_rank_position": pd.Series(click_rank, dtype="Int64"),
    })


def generate_transactions(cfg: GenerationConfig, listings: pd.DataFrame, searches: pd.DataFrame,
                          sellers: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    end_month = pd.Timestamp(cfg.end_date).to_period("M")
    months = pd.period_range(end=end_month, periods=12, freq="M")
    base_month_weights = np.array([.88, .91, .94, .98, 1.02, 1.06, 1.10, 1.14, 1.18, 1.16, 1.13, 1.10])
    cat_share = dict(zip(CATEGORIES, [.20, .18, .13, .22, .14, .13]))
    rows = []
    buyer_pool = searches.buyer_id.unique()
    active = listings[listings.is_active].merge(sellers[["seller_id", "is_churned"]], on="seller_id")
    for mi, month in enumerate(months):
        for cat in CATEGORIES:
            decel = (1 - .10 * max(0, mi - 8)) if cat in POOR_SEARCH_CATEGORIES else 1.0
            n = max(50, int(cfg.target_transactions * cat_share[cat] * base_month_weights[mi] * decel / base_month_weights.sum()))
            pool = active[(active.category == cat) & ~active.is_churned].copy()
            rel_price = pool.price / CATEGORY_BASE_PRICE[cat]
            demand_w = pool.base_demand_score.clip(2) * np.power(rel_price, CATEGORY_ELASTICITY[cat])
            chosen = pool.iloc[rng.choice(len(pool), n, p=(demand_w / demand_w.sum()).to_numpy())]
            start = month.start_time
            dates = start + pd.to_timedelta(rng.integers(0, month.days_in_month, n), unit="D")
            buyer = rng.choice(buyer_pool, n)
            qty = rng.choice([1, 2, 3], n, p=[.88, .10, .02])
            paid = chosen.price.to_numpy() * rng.normal(.98, .035, n)
            rows.append(pd.DataFrame({"buyer_id": buyer, "seller_id": chosen.seller_id.to_numpy(),
                "listing_id": chosen.listing_id.to_numpy(), "category": cat,
                "price_paid": paid.round(2), "quantity": qty, "order_date": dates}))
    tx = pd.concat(rows, ignore_index=True)
    tx.insert(0, "transaction_id", [f"T{i:07d}" for i in range(1, len(tx) + 1)])
    tx = tx.sort_values(["buyer_id", "order_date"])
    tx["is_repeat_buyer"] = tx.groupby("buyer_id").cumcount().gt(0)
    return tx.sort_values("transaction_id").reset_index(drop=True)


def validation_metrics(sellers: pd.DataFrame, searches: pd.DataFrame) -> dict:
    bucket = pd.cut(searches.click_rank_position, [0, 5, 10, 20, 50], labels=["1-5", "6-10", "11-20", "21-50"])
    # Funnel CTR uses all searches as denominator; a search contributes a click to one bucket at most.
    counts = bucket.value_counts().reindex(["1-5", "6-10", "11-20", "21-50"]).fillna(0)
    ctr = (counts / len(searches)).to_dict()
    churn = sellers.assign(group=np.where(sellers.fulfillment_rate < .90, "<0.90", ">=0.90")).groupby("group").is_churned.mean()
    return {"search_ctr_by_rank_bucket": {str(k): round(float(v), 4) for k, v in ctr.items()},
            "seller_churn_rate": {str(k): round(float(v), 4) for k, v in churn.items()},
            "churn_rate_ratio": round(float(churn.get("<0.90", np.nan) / churn.get(">=0.90", np.nan)), 2)}


def generate_all(output_dir: str | Path, cfg: GenerationConfig | None = None) -> dict[str, pd.DataFrame]:
    cfg = cfg or GenerationConfig()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    Faker.seed(cfg.seed)
    sellers = generate_sellers(cfg, rng)
    listings = generate_listings(cfg, sellers, rng)
    searches = generate_search_events(cfg, listings, sellers, rng)
    transactions = generate_transactions(cfg, listings, searches, sellers, rng)
    tables = {"sellers": sellers, "listings": listings, "search_events": searches, "transactions": transactions}
    for name, frame in tables.items():
        _write_table(frame, output_dir, name)
    metrics = validation_metrics(sellers, searches)
    (output_dir / "validation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return tables
