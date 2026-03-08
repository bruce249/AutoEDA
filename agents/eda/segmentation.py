"""
STEP 4 — Segmentation
Detects meaningful segments / clusters in the data using:
  • Categorical splits  (groups with distinct numeric profiles)
  • Numeric thresholds  (high / low splits based on distribution)
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


# ======================================================================== #
# Categorical-based segmentation
# ======================================================================== #

def _categorical_segments(
    df: pd.DataFrame,
    categorical_cols: List[str],
    numeric_cols: List[str],
    *,
    max_cardinality: int = 15,
    min_group_size: int = 5,
    effect_threshold: float = 0.3,
) -> List[Dict[str, Any]]:
    """
    For each categorical column, measure how much it separates the
    numeric columns (using coefficient of variation of group means
    normalised by overall std).

    Returns a list of segment descriptors, each describing a
    categorical column that meaningfully splits the data.
    """
    segments: List[Dict[str, Any]] = []

    for cat in categorical_cols:
        nunique = df[cat].nunique()
        if nunique < 2 or nunique > max_cardinality:
            continue

        for num in numeric_cols:
            grouped = df.groupby(cat, observed=True)[num]
            means = grouped.mean().dropna()
            counts = grouped.count()

            # Skip groups that are too small
            if (counts < min_group_size).all():
                continue

            overall_std = df[num].std()
            if overall_std == 0 or np.isnan(overall_std):
                continue

            # Effect size: range of group means / overall std
            effect = float((means.max() - means.min()) / overall_std)

            if effect >= effect_threshold:
                segments.append({
                    "type": "categorical_split",
                    "split_column": cat,
                    "target_column": num,
                    "effect_size": round(effect, 4),
                    "group_means": {str(k): round(float(v), 4) for k, v in means.items()},
                    "group_counts": {str(k): int(v) for k, v in counts.items()},
                    "description": (
                        f"Column '{cat}' creates distinct groups on '{num}' "
                        f"(effect size = {effect:.2f}σ)."
                    ),
                })

    segments.sort(key=lambda s: -s["effect_size"])
    return segments


# ======================================================================== #
# Numeric threshold-based segmentation
# ======================================================================== #

def _numeric_segments(
    df: pd.DataFrame,
    numeric_cols: List[str],
    *,
    skew_threshold: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Identify interesting numeric thresholds:
      • Heavily skewed columns → split at median into low / high
      • Columns with IQR-based outliers → flag outlier segment
    """
    segments: List[Dict[str, Any]] = []

    for col in numeric_cols:
        s = df[col].dropna()
        if len(s) < 10:
            continue

        skew = float(s.skew())
        median = float(s.median())
        q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_outliers = int(((s < lower) | (s > upper)).sum())

        if abs(skew) >= skew_threshold:
            segments.append({
                "type": "numeric_threshold",
                "column": col,
                "threshold": median,
                "skewness": round(skew, 4),
                "below_median_count": int((s <= median).sum()),
                "above_median_count": int((s > median).sum()),
                "description": (
                    f"'{col}' is {'right' if skew > 0 else 'left'}-skewed "
                    f"(skew={skew:.2f}). Splitting at median={median:.2f} "
                    f"reveals asymmetric segments."
                ),
            })

        if n_outliers > 0:
            segments.append({
                "type": "outlier_segment",
                "column": col,
                "outlier_count": n_outliers,
                "outlier_pct": round(n_outliers / len(s) * 100, 2),
                "lower_bound": round(lower, 4),
                "upper_bound": round(upper, 4),
                "description": (
                    f"'{col}' has {n_outliers} IQR-outliers "
                    f"({n_outliers / len(s) * 100:.1f}% of data)."
                ),
            })

    return segments


# ======================================================================== #
# Public API
# ======================================================================== #

def detect_segments(
    df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> List[Dict[str, Any]]:
    """
    Run both categorical-split and numeric-threshold segmentation
    and return a merged, sorted list of segment descriptors.
    """
    segs: List[Dict[str, Any]] = []
    segs.extend(_categorical_segments(df, categorical_cols, numeric_cols))
    segs.extend(_numeric_segments(df, numeric_cols))
    return segs
