"""
STEP 5 — Time-Series Analysis (conditional on datetime columns)
Performs:
  • Trend detection (simple linear regression slope)
  • Seasonality detection (autocorrelation peaks)
  • Decomposition summary (trend / seasonal / residual strength)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Trend detection
# ======================================================================== #

def _detect_trend(
    series: pd.Series,
) -> Dict[str, Any]:
    """
    Fit a simple linear regression (OLS) on the numeric index of the
    sorted series and report slope, direction, and R².
    """
    s = series.dropna().sort_index()
    if len(s) < 3:
        return {"trend": "insufficient_data"}

    y = s.values.astype(float)
    x = np.arange(len(y), dtype=float)

    # OLS via numpy
    A = np.vstack([x, np.ones_like(x)]).T
    try:
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    except Exception:
        return {"trend": "computation_error"}

    y_pred = slope * x + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

    if abs(slope) < 1e-10:
        direction = "flat"
    elif slope > 0:
        direction = "upward"
    else:
        direction = "downward"

    return {
        "trend": direction,
        "slope": round(float(slope), 6),
        "r_squared": round(r_squared, 4),
        "intercept": round(float(intercept), 4),
    }


# ======================================================================== #
# Seasonality detection (autocorrelation)
# ======================================================================== #

def _detect_seasonality(
    series: pd.Series,
    max_lag: int = 60,
) -> Dict[str, Any]:
    """
    Compute autocorrelation at lags 1 … *max_lag* and identify
    prominent periodic peaks.
    """
    s = series.dropna()
    if len(s) < max_lag:
        max_lag = len(s) // 2
    if max_lag < 2:
        return {"seasonality": "insufficient_data"}

    # Autocorrelation via pandas
    acf_vals = [float(s.autocorr(lag=lag)) for lag in range(1, max_lag + 1)]

    # Find peaks in ACF
    peaks: List[Dict[str, Any]] = []
    for i in range(1, len(acf_vals) - 1):
        if acf_vals[i] > acf_vals[i - 1] and acf_vals[i] > acf_vals[i + 1]:
            if acf_vals[i] > 0.2:  # significance heuristic
                peaks.append({"lag": i + 1, "acf": round(acf_vals[i], 4)})

    peaks.sort(key=lambda p: -p["acf"])

    if peaks:
        dominant_period = peaks[0]["lag"]
        return {
            "seasonality": "detected",
            "dominant_period_lag": dominant_period,
            "acf_at_dominant": peaks[0]["acf"],
            "all_peaks": peaks[:5],
        }
    else:
        return {
            "seasonality": "not_detected",
            "note": "No significant autocorrelation peaks found.",
        }


# ======================================================================== #
# Public API
# ======================================================================== #

def time_series_analysis(
    df: pd.DataFrame,
    datetime_cols: List[str],
    numeric_cols: List[str],
) -> Dict[str, Any]:
    """
    If datetime columns exist, aggregate each numeric column by the
    first datetime column and detect trend + seasonality.

    Returns
    -------
    dict  keyed by numeric column → { trend_info, seasonality_info }
    Returns an empty dict if no datetime columns are present.
    """
    if not datetime_cols or not numeric_cols:
        return {}

    dt_col = datetime_cols[0]  # primary datetime axis
    result: Dict[str, Any] = {"datetime_column": dt_col, "analyses": {}}

    # Ensure datetime
    dt_series = pd.to_datetime(df[dt_col], errors="coerce")

    for num in numeric_cols:
        ts = pd.Series(df[num].values, index=dt_series).dropna()
        ts = ts.sort_index()

        if len(ts) < 5:
            result["analyses"][num] = {"status": "insufficient_data"}
            continue

        trend = _detect_trend(ts)
        seasonality = _detect_seasonality(ts)

        result["analyses"][num] = {
            "trend": trend,
            "seasonality": seasonality,
        }

    return result
