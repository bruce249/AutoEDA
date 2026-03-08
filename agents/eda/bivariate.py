"""
STEP 2 — Bivariate Analysis
Computes relationships between pairs of columns:
  • Numeric × Numeric   → correlation matrix, strong pairs
  • Categorical × Numeric → grouped statistics
  • Categorical × Categorical → cross-tabulation + Cramér's V
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

# Numeric × Numeric — Correlation

def correlation_matrix(
    df: pd.DataFrame,
    numeric_cols: List[str],
    method: str = "pearson",
) -> pd.DataFrame:
    """Return the full correlation matrix for numeric columns."""
    if len(numeric_cols) < 2:
        return pd.DataFrame()
    return df[numeric_cols].corr(method=method)


def strong_correlations(
    corr: pd.DataFrame,
    threshold: float = 0.7,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract column pairs whose absolute correlation exceeds *threshold*.
    Returns dict with keys ``strong_positive`` and ``strong_negative``.
    """
    pos: List[Dict[str, Any]] = []
    neg: List[Dict[str, Any]] = []

    if corr.empty:
        return {"strong_positive": pos, "strong_negative": neg}

    cols = corr.columns.tolist()
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            r = corr.loc[c1, c2]
            if np.isnan(r):
                continue
            if r >= threshold:
                pos.append({"columns": [c1, c2], "correlation": round(float(r), 4)})
            elif r <= -threshold:
                neg.append({"columns": [c1, c2], "correlation": round(float(r), 4)})

    # Sort by strength descending
    pos.sort(key=lambda x: -x["correlation"])
    neg.sort(key=lambda x: x["correlation"])

    return {"strong_positive": pos, "strong_negative": neg}


# ======================================================================== #
# Categorical × Numeric — Grouped Statistics
# ======================================================================== #

def grouped_statistics(
    df: pd.DataFrame,
    categorical_cols: List[str],
    numeric_cols: List[str],
    *,
    max_cat_pairs: int = 20,
) -> List[Dict[str, Any]]:
    """
    For each (categorical, numeric) pair compute grouped mean, median, std.
    Limits output to *max_cat_pairs* most interesting pairs (sorted by
    variance of group means, descending).
    """
    records: List[Dict[str, Any]] = []

    for cat in categorical_cols:
        if df[cat].nunique() > 50:
            continue  # skip very high-cardinality categoricals
        for num in numeric_cols:
            grouped = df.groupby(cat, observed=True)[num].agg(
                ["mean", "median", "std", "count"]
            ).dropna()
            if grouped.empty:
                continue

            # Measure how much the group means vary
            mean_var = float(grouped["mean"].var())

            records.append({
                "categorical": cat,
                "numeric": num,
                "mean_variance_across_groups": round(mean_var, 4),
                "group_stats": {
                    str(idx): {
                        "mean": round(float(row["mean"]), 4),
                        "median": round(float(row["median"]), 4),
                        "std": round(float(row["std"]), 4) if not np.isnan(row["std"]) else None,
                        "count": int(row["count"]),
                    }
                    for idx, row in grouped.iterrows()
                },
            })

    # Keep most interesting pairs
    records.sort(key=lambda r: -r["mean_variance_across_groups"])
    return records[:max_cat_pairs]

# Categorical × Categorical — Cross-tabulation + Cramér's V

def _cramers_v(confusion_matrix: pd.DataFrame) -> float:
    """Compute Cramér's V from a contingency table."""
    from scipy.stats import chi2_contingency  # type: ignore[import]

    chi2, _, _, _ = chi2_contingency(confusion_matrix)
    n = confusion_matrix.values.sum()
    r, k = confusion_matrix.shape
    denom = n * (min(r, k) - 1)
    if denom == 0:
        return 0.0
    return float(np.sqrt(chi2 / denom))


def categorical_associations(
    df: pd.DataFrame,
    categorical_cols: List[str],
    *,
    max_pairs: int = 15,
    max_cardinality: int = 30,
) -> List[Dict[str, Any]]:
    """
    For each pair of categorical columns compute a cross-tab and Cramér's V.
    Skips columns with cardinality > *max_cardinality*.
    """
    results: List[Dict[str, Any]] = []

    eligible = [c for c in categorical_cols if df[c].nunique() <= max_cardinality]

    for c1, c2 in combinations(eligible, 2):
        ct = pd.crosstab(df[c1], df[c2])
        if ct.size == 0:
            continue

        try:
            v = _cramers_v(ct)
        except Exception:
            v = None

        results.append({
            "columns": [c1, c2],
            "cramers_v": round(v, 4) if v is not None else None,
            "crosstab_shape": list(ct.shape),
        })

    results.sort(key=lambda r: -(r["cramers_v"] or 0))
    return results[:max_pairs]
