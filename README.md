# Multi-Agent Data Analyst (AutoEDA) Platform

A modular, multi-agent system for automated exploratory data analysis.

## Agents

### 1. Data Ingestion and Access Agent

**Location:** `agents/ingestion/`

Transforms raw data sources (CSV, Excel, SQL, API) into a standardized, clean DataFrame with schema and metadata for downstream agents.

#### Quick Start

```python
from agents.ingestion import IngestionAgent

agent = IngestionAgent()

# CSV
result = agent.run("data/sales.csv")

# Excel
result = agent.run("data/report.xlsx", sheet_name="Q4")

# SQL
result = agent.run(
    "sqlite:///data/warehouse.db",
    table="transactions",
    row_limit=10000,
)

# API
result = agent.run("https://api.example.com/v1/records")
```

#### Output Contract

```python
{
    "dataframe":      pd.DataFrame,       # clean, typed DataFrame
    "schema": {
        "columns":           [...],
        "dtypes":            {...},
        "numeric_columns":   [...],
        "categorical_columns": [...],
        "datetime_columns":  [...],
    },
    "metadata": {
        "row_count":         int,
        "column_count":      int,
        "missing_values":    {...},
        "total_missing_cells": int,
        "duplicate_row_count": int,
        "memory_usage_mb":   float,
        "source_type":       "csv | excel | sql | api",
    },
    "preview":         pd.DataFrame,      # first 5 rows
    "column_summary":  pd.DataFrame,      # per-column stats
    "query_plan":      QueryPlan,         # parsed user intent
}
```

## Installation

```bash
pip install -r requirements.txt
```

## Running the Demo

```bash
python demo_ingestion.py
```
