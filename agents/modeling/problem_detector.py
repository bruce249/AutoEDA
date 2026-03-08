"""
problem_detector.py — Auto-detect the ML problem type from data schema.

Supported problem types
-----------------------
- **regression** — numeric target predicted from features
- **classification** — categorical target predicted from features
- **time_series** — temporal data with a numeric measure to forecast
- **clustering** — no obvious target; unsupervised grouping
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── Heuristic thresholds ────────────────────────────────────────────────
_MAX_CLASS_CARDINALITY = 20  # above this, treat as regression not classification
_MIN_ROWS_FOR_MODEL = 30
_MAX_CLASS_UNIQUE_RATIO = 0.2


def detect_problem(
    df: pd.DataFrame,
    schema: Dict[str, Any],
    metadata: Dict[str, Any],
    *,
    target_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Analyse the dataset and return a problem specification dict.

    Returns
    -------
    dict with keys:
        problem_type : str   — regression | classification | time_series | clustering
        target       : str | None — name of the target column (None for clustering)
        features     : list[str]
        time_column  : str | None
        reason       : str  — human-readable explanation
    """
    num_cols: List[str] = schema.get("numeric_columns", [])
    cat_cols: List[str] = schema.get("categorical_columns", [])
    dt_cols: List[str] = schema.get("datetime_columns", [])
    row_count: int = metadata.get("row_count", len(df))

    if row_count < _MIN_ROWS_FOR_MODEL:
        return _result("clustering", None, num_cols + cat_cols, None,
                        f"Only {row_count} rows — too few for supervised learning; defaulting to clustering.")

    # If user specified a target, honour it
    if target_hint and target_hint in df.columns:
        return _from_target(df, target_hint, num_cols, cat_cols, dt_cols)

    # ── 1. Time-series heuristic ─────────────────────────────────────
    if dt_cols:
        ts_target = _pick_ts_target(df, num_cols)
        if ts_target:
            features = [c for c in num_cols + cat_cols if c != ts_target]
            return _result(
                "time_series", ts_target, features, dt_cols[0],
                f"Datetime column '{dt_cols[0]}' detected. Forecasting '{ts_target}'.",
            )

    # ── 2. Guess target as the *last* numeric or categorical column ──
    target = _guess_target(df, num_cols, cat_cols)
    if target:
        return _from_target(df, target, num_cols, cat_cols, dt_cols)

    # ── 3. Fallback → clustering ─────────────────────────────────────
    return _result("clustering", None, num_cols + cat_cols, None,
                    "No clear target column; defaulting to clustering.")


# ── Helpers ──────────────────────────────────────────────────────────────

def _from_target(
    df: pd.DataFrame,
    target: str,
    num_cols: List[str],
    cat_cols: List[str],
    dt_cols: List[str],
) -> Dict[str, Any]:
    """Classify problem based on a known target column."""
    features = [c for c in num_cols + cat_cols if c != target]
    time_col = dt_cols[0] if dt_cols else None
    series = df[target].dropna()

    if series.empty:
        return _result(
            "clustering", None, features, time_col,
            f"Target '{target}' is empty after dropping missing values; defaulting to clustering.",
        )

    if _is_identifier_like(df, target):
        return _result(
            "clustering", None, features + [target], time_col,
            f"Column '{target}' looks identifier-like (high uniqueness / key semantics); defaulting to clustering.",
        )

    nunique = int(series.nunique())
    value_counts = series.value_counts(dropna=True)
    min_class_size = int(value_counts.min()) if not value_counts.empty else 0
    unique_ratio = nunique / max(len(series), 1)

    # If the column is floating-point, treat as regression even if cardinality is low
    is_float = pd.api.types.is_float_dtype(series)
    is_integer = pd.api.types.is_integer_dtype(series) or _looks_integer_like(series)
    is_bool = pd.api.types.is_bool_dtype(series)
    is_categorical_like = target in cat_cols or is_bool or (
        is_integer and nunique <= _MAX_CLASS_CARDINALITY and unique_ratio <= _MAX_CLASS_UNIQUE_RATIO
    )

    if is_categorical_like and min_class_size >= 2:
        return _result(
            "classification", target, features, time_col,
            f"Target '{target}' has {nunique} classes and minimum class size {min_class_size} → classification.",
        )

    if is_categorical_like and min_class_size < 2:
        return _result(
            "clustering", None, features + [target], time_col,
            f"Target '{target}' has sparse classes (minimum class size {min_class_size}); supervised classification would be unstable, so defaulting to clustering.",
        )

    if target in num_cols or is_float or is_integer:
        return _result(
            "regression", target, features, time_col,
            f"Target '{target}' is numeric / continuous-like (unique ratio={unique_ratio:.2f}) → regression.",
        )

    return _result(
        "clustering", None, features + [target], time_col,
        f"Target '{target}' does not fit a stable supervised setup; defaulting to clustering.",
    )


def _pick_ts_target(df: pd.DataFrame, num_cols: List[str]) -> Optional[str]:
    """Pick the best numeric column for time-series forecasting."""
    # Prefer columns named like 'sales', 'revenue', 'value', 'price', 'amount'
    priority_keywords = ["sales", "revenue", "value", "price", "amount", "total", "count"]
    for kw in priority_keywords:
        for col in num_cols:
            if kw in col.lower():
                return col
    # Fall back to last numeric column
    return num_cols[-1] if num_cols else None


def _guess_target(
    df: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
) -> Optional[str]:
    """
    Heuristic: the *last* column in the original frame that is not an ID
    is usually the target / response variable.
    """
    all_cols = list(df.columns)
    # Walk backwards looking for a plausible target
    for col in reversed(all_cols):
        lower = col.lower()
        # Skip obvious IDs/keys
        if any(kw in lower for kw in ("_id", "id_", "index", "unnamed")):
            continue
        if _is_identifier_like(df, col):
            continue
        if col in num_cols or col in cat_cols:
            return col
    return None


def _is_identifier_like(df: pd.DataFrame, col: str) -> bool:
    """Heuristic: skip columns that behave like keys/IDs rather than targets."""
    lower = col.lower()
    if any(kw in lower for kw in ("id", "uuid", "guid", "key", "code", "token")):
        return True

    s = df[col].dropna()
    if s.empty:
        return False

    unique_ratio = s.nunique() / max(len(s), 1)
    if unique_ratio >= 0.95:
        return True

    if pd.api.types.is_numeric_dtype(s):
        try:
            monotonic = s.is_monotonic_increasing or s.is_monotonic_decreasing
            if monotonic and unique_ratio >= 0.8:
                return True
        except Exception:
            pass

    return False


def _looks_integer_like(series: pd.Series) -> bool:
    """True when float values are effectively integer-coded classes."""
    try:
        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.empty:
            return False
        return bool((s.round() == s).all())
    except Exception:
        return False


def _result(problem_type: str, target, features, time_col, reason: str) -> Dict[str, Any]:
    logger.info("Problem detected: %s | target=%s | reason=%s", problem_type, target, reason)
    return {
        "problem_type": problem_type,
        "target": target,
        "features": features,
        "time_column": time_col,
        "reason": reason,
    }
