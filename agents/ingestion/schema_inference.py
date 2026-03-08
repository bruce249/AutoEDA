"""
STEP 3 & 4 — Schema Inference + Sample Preview
Analyzes the loaded DataFrame and returns a rich schema dict,
metadata dict, and a sample preview.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


# ---------------------------------------------------------------------------
# Column classification
# ---------------------------------------------------------------------------

def _classify_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Split columns into numeric / categorical / datetime buckets."""

    numeric: List[str] = []
    categorical: List[str] = []
    datetime: List[str] = []

    for col in df.columns:
        dtype = df[col].dtype

        if pd.api.types.is_datetime64_any_dtype(dtype):
            datetime.append(col)
        elif pd.api.types.is_numeric_dtype(dtype):
            # Heuristic: if a numeric column has very few unique values
            # relative to its length it *might* be categorical, but we do
            # NOT reclassify automatically — we just report it as numeric.
            numeric.append(col)
        elif pd.api.types.is_bool_dtype(dtype):
            categorical.append(col)
        else:
            categorical.append(col)

    return {
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "datetime_columns": datetime,
    }


# ---------------------------------------------------------------------------
# Schema builder
# ---------------------------------------------------------------------------

def infer_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """
    STEP 3 — Infer the schema (column names, dtypes, classification).

    Returns
    -------
    dict  with keys:
        columns, dtypes, numeric_columns, categorical_columns, datetime_columns
    """

    classification = _classify_columns(df)

    schema: Dict[str, Any] = {
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        **classification,
    }

    return schema


# ---------------------------------------------------------------------------
# Metadata builder
# ---------------------------------------------------------------------------

def build_metadata(
    df: pd.DataFrame,
    source_type: str,
) -> Dict[str, Any]:
    """
    STEP 3 (cont.) — Build dataset-level metadata.

    Returns
    -------
    dict  with keys:
        row_count, column_count, missing_values, source_type,
        memory_usage_mb, duplicate_row_count
    """

    missing: Dict[str, int] = df.isnull().sum().to_dict()
    # Keep only columns that actually have missing values
    missing = {col: int(cnt) for col, cnt in missing.items() if cnt > 0}

    metadata: Dict[str, Any] = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "missing_values": missing,
        "total_missing_cells": int(df.isnull().sum().sum()),
        "duplicate_row_count": int(df.duplicated().sum()),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1_048_576, 3),
        "source_type": source_type,
    }

    return metadata


# ---------------------------------------------------------------------------
# Column-level summary
# ---------------------------------------------------------------------------

def column_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a per-column summary DataFrame useful for quick profiling.

    Columns returned:
        column, dtype, non_null, null_count, null_pct,
        unique, top_value, mean, min, max
    """

    records = []
    for col in df.columns:
        series = df[col]
        rec = {
            "column": col,
            "dtype": str(series.dtype),
            "non_null": int(series.notna().sum()),
            "null_count": int(series.isna().sum()),
            "null_pct": round(series.isna().mean() * 100, 2),
            "unique": int(series.nunique()),
        }

        if pd.api.types.is_numeric_dtype(series):
            rec["mean"] = round(series.mean(), 4) if series.notna().any() else None
            rec["min"] = series.min() if series.notna().any() else None
            rec["max"] = series.max() if series.notna().any() else None
            rec["top_value"] = None
        elif pd.api.types.is_datetime64_any_dtype(series):
            rec["mean"] = None
            rec["min"] = str(series.min()) if series.notna().any() else None
            rec["max"] = str(series.max()) if series.notna().any() else None
            rec["top_value"] = None
        else:
            mode = series.mode()
            rec["top_value"] = str(mode.iloc[0]) if not mode.empty else None
            rec["mean"] = None
            rec["min"] = None
            rec["max"] = None

        records.append(rec)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Preview helper
# ---------------------------------------------------------------------------

def build_preview(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """STEP 4 — Return the first *n* rows."""
    return df.head(n)
