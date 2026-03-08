"""
feature_engineering.py — Prepare raw data for ML models.

Handles:
- Missing values (median for numeric, mode for categorical)
- Label-encoding for low-cardinality categoricals
- One-hot encoding for remaining categoricals
- Standard-scaling for numeric columns
- Time-series feature extraction (lag, rolling, calendar)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_HIGH_CARDINALITY = 15  # one-hot up to this; label-encode above


def engineer_features(
    df: pd.DataFrame,
    problem: Dict[str, Any],
    schema: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Return a dict containing the ML-ready arrays plus metadata.

    Returns
    -------
    dict with keys:
        X             : pd.DataFrame   — feature matrix
        y             : pd.Series|None  — target (None for clustering)
        feature_names : list[str]
        transformations : list[dict]  — human-readable log of what was done
        time_index    : pd.Series|None — original datetime index (for TS)
    """
    target = problem.get("target")
    features = problem.get("features", [])
    time_col = problem.get("time_column")
    problem_type = problem.get("problem_type", "clustering")
    transformations: List[Dict[str, str]] = []

    work = df.copy()

    # ── 0. Sort by time for TS ───────────────────────────────────────
    time_index = None
    if time_col and time_col in work.columns:
        work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
        work.sort_values(time_col, inplace=True)
        work.reset_index(drop=True, inplace=True)
        time_index = work[time_col].copy()
        transformations.append({"action": "sort_by_time", "column": time_col})

    # ── 1. Extract time features (TS) ────────────────────────────────
    if problem_type == "time_series" and time_col and time_col in work.columns:
        work, ts_feats = _extract_time_features(work, time_col, target)
        features = [f for f in features if f != time_col] + ts_feats
        transformations.append({"action": "time_features", "new_columns": ts_feats})

    # ── 2. Subset to features + target ───────────────────────────────
    keep = [c for c in features if c in work.columns]
    if target and target in work.columns:
        keep.append(target)
    work = work[keep].copy()

    # ── 3. Handle missing values ─────────────────────────────────────
    num_cols = schema.get("numeric_columns", [])
    cat_cols = schema.get("categorical_columns", [])

    for col in work.columns:
        if work[col].isna().sum() == 0:
            continue
        if col in num_cols or pd.api.types.is_numeric_dtype(work[col]):
            med = work[col].median()
            work[col] = work[col].fillna(med)
            transformations.append({"action": "fill_median", "column": col, "value": str(med)})
        else:
            mode_val = work[col].mode().iloc[0] if not work[col].mode().empty else "MISSING"
            work[col] = work[col].fillna(mode_val)
            transformations.append({"action": "fill_mode", "column": col, "value": str(mode_val)})

    # ── 4. Encode categoricals ───────────────────────────────────────
    cat_features = [c for c in keep if c in cat_cols and c != target]
    for col in cat_features:
        n_unique = work[col].nunique()
        if n_unique <= _HIGH_CARDINALITY:
            dummies = pd.get_dummies(work[col], prefix=col, drop_first=True, dtype=np.float64)
            work = pd.concat([work.drop(columns=[col]), dummies], axis=1)
            transformations.append({"action": "one_hot", "column": col, "new_cols": list(dummies.columns)})
        else:
            codes, _ = pd.factorize(work[col])
            work[col] = codes.astype(np.float64)
            transformations.append({"action": "label_encode", "column": col})

    # ── 5. Encode target for classification ──────────────────────────
    target_mapping = None
    if target and target in work.columns and problem_type == "classification":
        if not pd.api.types.is_numeric_dtype(work[target]):
            codes, uniques = pd.factorize(work[target])
            target_mapping = {str(u): int(c) for c, u in enumerate(uniques)}
            work[target] = codes.astype(np.float64)
            transformations.append({"action": "encode_target", "column": target, "mapping": target_mapping})

    # ── 6. Separate X / y ────────────────────────────────────────────
    if target and target in work.columns:
        y = work[target].copy()
        X = work.drop(columns=[target])
    else:
        y = None
        X = work.copy()

    # ── 7. Scale numerics ────────────────────────────────────────────
    scaling_stats: Dict[str, Dict[str, float]] = {}
    for col in X.select_dtypes(include=[np.number]).columns:
        mean, std = X[col].mean(), X[col].std()
        if std and std > 0:
            X[col] = (X[col] - mean) / std
            scaling_stats[col] = {"mean": float(mean), "std": float(std)}
    if scaling_stats:
        transformations.append({"action": "standard_scale", "columns": list(scaling_stats.keys())})

    # Drop any remaining non-numeric columns
    X = X.select_dtypes(include=[np.number])

    # Drop rows with NaN that snuck through
    if y is not None:
        mask = X.notna().all(axis=1) & y.notna()
        X = X.loc[mask].reset_index(drop=True)
        y = y.loc[mask].reset_index(drop=True)
        if time_index is not None:
            time_index = time_index.loc[mask].reset_index(drop=True)
    else:
        mask = X.notna().all(axis=1)
        X = X.loc[mask].reset_index(drop=True)
        if time_index is not None:
            time_index = time_index.loc[mask].reset_index(drop=True)

    logger.info("Feature engineering done: X shape %s, features=%d", X.shape, X.shape[1])

    return {
        "X": X,
        "y": y,
        "feature_names": list(X.columns),
        "transformations": transformations,
        "time_index": time_index,
        "target_mapping": target_mapping,
        "scaling_stats": scaling_stats,
    }


# ── Time feature helpers ─────────────────────────────────────────────────

def _extract_time_features(
    df: pd.DataFrame, time_col: str, target: Optional[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """Add calendar + lag features for time-series problems."""
    new_feats: List[str] = []
    ts = df[time_col]

    # Calendar features
    df["month"] = ts.dt.month.astype(np.float64)
    df["day_of_week"] = ts.dt.dayofweek.astype(np.float64)
    df["day_of_month"] = ts.dt.day.astype(np.float64)
    df["quarter"] = ts.dt.quarter.astype(np.float64)
    new_feats.extend(["month", "day_of_week", "day_of_month", "quarter"])

    # Lag and rolling features on target
    if target and target in df.columns:
        for lag in [1, 3, 7]:
            col_name = f"{target}_lag{lag}"
            df[col_name] = df[target].shift(lag)
            new_feats.append(col_name)
        for window in [3, 7]:
            col_name = f"{target}_roll{window}"
            df[col_name] = df[target].rolling(window, min_periods=1).mean()
            new_feats.append(col_name)

    # Drop the raw datetime column
    df.drop(columns=[time_col], inplace=True)

    return df, new_feats
