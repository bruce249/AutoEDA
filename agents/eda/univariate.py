"""
STEP 1 — Univariate Analysis
Produces per-column statistics for numeric, categorical, and datetime columns.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


# ======================================================================== #
# Numeric univariate
# ======================================================================== #

def _outlier_bounds_iqr(series: pd.Series) -> Tuple[float, float, int]:
    """Return (lower_bound, upper_bound, outlier_count) via 1.5×IQR."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outlier_count = int(((series < lower) | (series > upper)).sum())
    return float(lower), float(upper), outlier_count


def numeric_univariate(df: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
    """
    For each numeric column compute:
    mean, median, std, skewness, min, max, Q1, Q3, IQR,
    outlier bounds, outlier count, and coefficient of variation.
    """
    result: Dict[str, Any] = {}

    for col in columns:
        s = df[col].dropna()
        if s.empty:
            result[col] = {"error": "all values are null"}
            continue

        lb, ub, n_outliers = _outlier_bounds_iqr(s)
        q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
        mean = float(s.mean())
        std = float(s.std())

        result[col] = {
            "count": int(s.count()),
            "mean": round(mean, 4),
            "median": round(float(s.median()), 4),
            "std": round(std, 4),
            "skewness": round(float(s.skew()), 4),
            "kurtosis": round(float(s.kurtosis()), 4),
            "min": float(s.min()),
            "max": float(s.max()),
            "q1": round(q1, 4),
            "q3": round(q3, 4),
            "iqr": round(q3 - q1, 4),
            "outlier_lower_bound": round(lb, 4),
            "outlier_upper_bound": round(ub, 4),
            "outlier_count": n_outliers,
            "cv": round(std / mean, 4) if mean != 0 else None,
        }

    return result


# ======================================================================== #
# Categorical univariate
# ======================================================================== #

def categorical_univariate(
    df: pd.DataFrame, columns: List[str], *, top_n: int = 10
) -> Dict[str, Any]:
    """
    For each categorical column compute:
    cardinality, top-N value counts, dominant category %, mode.
    """
    result: Dict[str, Any] = {}

    for col in columns:
        s = df[col].dropna()
        if s.empty:
            result[col] = {"error": "all values are null"}
            continue

        vc = s.value_counts()
        total = len(s)

        result[col] = {
            "count": int(total),
            "cardinality": int(s.nunique()),
            "mode": str(vc.index[0]) if len(vc) > 0 else None,
            "mode_frequency": int(vc.iloc[0]) if len(vc) > 0 else 0,
            "mode_pct": round(vc.iloc[0] / total * 100, 2) if len(vc) > 0 else 0,
            "top_values": {
                str(k): int(v)
                for k, v in vc.head(top_n).items()
            },
        }

    return result


# ======================================================================== #
# Datetime univariate
# ======================================================================== #

def datetime_univariate(df: pd.DataFrame, columns: List[str]) -> Dict[str, Any]:
    """
    For each datetime column compute:
    range, min, max, frequency pattern, and count of missing timestamps.
    """
    result: Dict[str, Any] = {}

    for col in columns:
        s = df[col].dropna()
        if s.empty:
            result[col] = {"error": "all values are null"}
            continue

        sorted_s = s.sort_values()
        diffs = sorted_s.diff().dropna()

        # Infer dominant frequency
        if not diffs.empty:
            mode_diff = diffs.mode()
            freq_str = str(mode_diff.iloc[0]) if not mode_diff.empty else "irregular"
        else:
            freq_str = "single value"

        result[col] = {
            "count": int(s.count()),
            "min": str(s.min()),
            "max": str(s.max()),
            "range_days": (s.max() - s.min()).days,
            "inferred_frequency": freq_str,
            "null_count": int(df[col].isna().sum()),
        }

    return result
