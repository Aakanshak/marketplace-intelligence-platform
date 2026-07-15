"""CLI entry point for marketplace data generation."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_gen import generate_all, validation_metrics  # noqa: E402


if __name__ == "__main__":
    tables = generate_all(ROOT / "data")
    metrics = validation_metrics(tables["sellers"], tables["search_events"])
    print("Generated:", {name: len(df) for name, df in tables.items()})
    print("CTR by rank bucket:", metrics["search_ctr_by_rank_bucket"])
    print("Churn by fulfillment threshold:", metrics["seller_churn_rate"])
    print("Churn ratio (low/high):", metrics["churn_rate_ratio"])
