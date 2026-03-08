"""
Demo / smoke-test for the Data Ingestion and Access Agent.

Creates a synthetic CSV in-memory, writes it to disk, then runs the
full ingestion pipeline to demonstrate the standardized output.
"""

import sys, os, json
import pandas as pd
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(__file__))

from agents.ingestion import IngestionAgent


# ---------------------------------------------------------------------------
# 1. Generate a small synthetic dataset and save as CSV
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
    print(f"✅  Sample CSV written → {path}  ({len(df)} rows)")
    return path


# ---------------------------------------------------------------------------
# 2. Run the Ingestion Agent
# ---------------------------------------------------------------------------
def main():
    csv_path = _create_sample_csv()

    agent = IngestionAgent(preview_rows=5, verbose=True)
    result = agent.run(csv_path)

    # ---- Inspect the output contract ----
    print("\n\n🔑  Top-level keys in result dict:")
    for key in result:
        val = result[key]
        if isinstance(val, pd.DataFrame):
            print(f"  {key:20s} → DataFrame {val.shape}")
        elif isinstance(val, dict):
            print(f"  {key:20s} → dict with keys: {list(val.keys())}")
        else:
            print(f"  {key:20s} → {type(val).__name__}")

    # ---- Schema ----
    print("\n📐  Schema:")
    print(json.dumps(result["schema"], indent=2))

    # ---- Metadata ----
    print("\n📊  Metadata:")
    print(json.dumps(result["metadata"], indent=2))

    # ---- Verify contract is complete ----
    required_keys = {"dataframe", "schema", "metadata", "preview"}
    assert required_keys.issubset(result.keys()), "Missing keys in output!"
    print("\n✅  Output contract verified — all required keys present.")

    return result


if __name__ == "__main__":
    main()
