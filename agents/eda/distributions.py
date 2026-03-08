"""
STEP 3 — Distribution Analysis
Classifies distribution shapes for numeric columns:
  • Normal / approximately normal
  • Right-skewed (long right tail)
  • Left-skewed (long left tail)
  • Multimodal
  • Uniform
  • Heavy-tailed (leptokurtic)
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Distribution classifier

def _classify_distribution(series: pd.Series) -> Dict[str, Any]:
    """
    Classify a single numeric series by its shape characteristics.

    Returns a dict with:
        distribution_type, skewness, kurtosis, details
    """
    s = series.dropna()
    if len(s) < 8:
        return {
            "distribution_type": "insufficient_data",
            "skewness": None,
            "kurtosis": None,
            "details": "Fewer than 8 non-null values — classification skipped.",
        }

    skew = float(s.skew())
    kurt = float(s.kurtosis())  # Fisher (excess) kurtosis: normal ≈ 0

    is_normal = False
    normality_p = None
    try:
        from scipy.stats import shapiro  # type: ignore[import]
        sample = s.sample(min(len(s), 5000), random_state=0)
        _, p = shapiro(sample)
        normality_p = round(float(p), 6)
        if p > 0.05:
            is_normal = True
    except Exception:
        pass
    # Multimodality heuristic (Hartigan dip approx.)
    # Simple heuristic: if the histogram has >1 prominent peak
    is_multimodal = False
    try:
        counts, _ = np.histogram(s, bins="auto")
        # Find local maxima
        peaks = 0
        for i in range(1, len(counts) - 1):
            if counts[i] > counts[i - 1] and counts[i] > counts[i + 1]:
                peaks += 1
        if peaks >= 2:
            is_multimodal = True
    except Exception:
        pass

    # ------ Classify ------ #
    if is_normal and not is_multimodal:
        dist_type = "normal"
    elif is_multimodal:
        dist_type = "multimodal"
    elif abs(skew) < 0.5:
        if kurt > 2:
            dist_type = "heavy_tailed"
        elif kurt < -1:
            dist_type = "light_tailed"
        else:
            dist_type = "approximately_normal"
    elif skew >= 0.5:
        dist_type = "right_skewed"
    else:
        dist_type = "left_skewed"

    return {
        "distribution_type": dist_type,
        "skewness": round(skew, 4),
        "kurtosis": round(kurt, 4),
        "normality_p_value": normality_p,
        "is_multimodal": is_multimodal,
        "details": _description(dist_type, skew, kurt),
    }


def _description(dtype: str, skew: float, kurt: float) -> str:
    """Human-readable one-liner describing the distribution."""
    msgs = {
        "normal": "Values follow a normal (bell-curve) distribution.",
        "approximately_normal": "Distribution is roughly symmetric and bell-shaped.",
        "right_skewed": f"Long right tail (skew={skew:.2f}); most values cluster on the left.",
        "left_skewed": f"Long left tail (skew={skew:.2f}); most values cluster on the right.",
        "multimodal": "Multiple prominent peaks detected — possible subgroups in the data.",
        "heavy_tailed": f"Symmetric but with heavy tails (kurtosis={kurt:.2f}); more extreme values than a normal distribution.",
        "light_tailed": f"Symmetric with light tails (kurtosis={kurt:.2f}); fewer extreme values than a normal distribution.",
        "insufficient_data": "Not enough data to classify.",
    }
    return msgs.get(dtype, "Unknown distribution shape.")


# ======================================================================== #
# Public API
# ======================================================================== #

def distribution_analysis(
    df: pd.DataFrame,
    numeric_cols: List[str],
) -> Dict[str, Dict[str, Any]]:
    """
    Classify distributions for all *numeric_cols* in *df*.

    Returns
    -------
    dict  keyed by column name → classification dict
    """
    return {col: _classify_distribution(df[col]) for col in numeric_cols}
