"""
End-to-end demo: Ingestion Agent  →  EDA Agent

Creates a synthetic CSV, runs the Ingestion Agent to produce a
standardized data object, then passes it to the EDA Agent for full
exploratory analysis.
"""

import sys, os, json
import pandas as pd
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(__file__))

from agents.ingestion import IngestionAgent
from agents.eda import EDAAgent


# ---------------------------------------------------------------------------
# 1. Generate a synthetic dataset and save as CSV
# ---------------------------------------------------------------------------
def _create_sample_csv(path: str = "data/sample_sales.csv") -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    np.random.seed(42)
    n = 200

    df = pd.DataFrame({
        "order_id":     range(1, n + 1),
        "order_date":   pd.date_range("2024-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
        "customer_id":  np.random.randint(1000, 1100, size=n),
        "product":      np.random.choice(["Widget A", "Widget B", "Gadget X", "Gadget Y"], n),
        "category":     np.random.choice(["Electronics", "Home", "Office"], n),
        "quantity":     np.random.randint(1, 20, size=n),
        "unit_price":   np.round(np.random.uniform(5.0, 150.0, size=n), 2),
        "discount":     np.round(np.random.choice([0, 0.05, 0.10, 0.15, 0.20], n), 2),
        "region":       np.random.choice(["North", "South", "East", "West"], n),
        "sales_rep":    np.random.choice(["Alice", "Bob", "Charlie", "Diana", None], n),
    })

    # Inject a few NaN values to test missing-value handling
    df.loc[df.sample(frac=0.05, random_state=1).index, "unit_price"] = np.nan
    df.loc[df.sample(frac=0.03, random_state=2).index, "quantity"] = np.nan

    df.to_csv(path, index=False)
    print(f"✅  Sample CSV written → {path}  ({len(df)} rows)\n")
    return path


# ---------------------------------------------------------------------------
# 2. Run Ingestion → EDA pipeline
# ---------------------------------------------------------------------------
def main():
    csv_path = _create_sample_csv()

    # ────── AGENT 1: Ingestion ────── #
    print("=" * 70)
    print("  AGENT 1 — Data Ingestion and Access Agent")
    print("=" * 70)
    ingestion = IngestionAgent(preview_rows=5, verbose=True)
    ingestion_result = ingestion.run(csv_path)

    # Quick sanity check
    required = {"dataframe", "schema", "metadata", "preview"}
    assert required.issubset(ingestion_result.keys()), "Ingestion output incomplete!"
    print("✅  Ingestion output contract verified.\n")

    # ────── AGENT 2: EDA ────── #
    print("=" * 70)
    print("  AGENT 2 — Exploratory Data Analysis Agent")
    print("=" * 70)
    eda = EDAAgent(verbose=True)
    eda_result = eda.run(ingestion_result)

    # ---- Inspect top-level keys ----
    print("\n🔑  EDA result keys:")
    for key in eda_result:
        val = eda_result[key]
        if isinstance(val, pd.DataFrame):
            print(f"  {key:35s} → DataFrame {val.shape}")
        elif isinstance(val, dict):
            print(f"  {key:35s} → dict ({len(val)} entries)")
        elif isinstance(val, list):
            print(f"  {key:35s} → list ({len(val)} items)")
        elif isinstance(val, str):
            print(f"  {key:35s} → str ({len(val)} chars)")
        else:
            print(f"  {key:35s} → {type(val).__name__}")

    # ---- Verify contract ----
    eda_required = {
        "eda_summary", "statistics", "correlations",
        "segments", "visualization_recommendations",
    }
    assert eda_required.issubset(eda_result.keys()), "EDA output contract incomplete!"
    print("\n✅  EDA output contract verified — all required keys present.")

    # ---- Print serialisable output (excluding DataFrames) ----
    serialisable = {
        k: v for k, v in eda_result.items()
        if not isinstance(v, pd.DataFrame)
    }
    # Convert any remaining DataFrames nested inside dicts
    if isinstance(serialisable.get("correlations", {}).get("matrix"), pd.DataFrame):
        serialisable["correlations"]["matrix"] = (
            serialisable["correlations"]["matrix"]
            .round(4)
            .to_dict()
        )

    print("\n📄  Serialised EDA output (JSON):")
    print(json.dumps(serialisable, indent=2, default=str))

    return eda_result


if __name__ == "__main__":
    main()
