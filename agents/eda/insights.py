"""
STEP 6 — Insight Generation & Visualization Recommendations
Synthesises all prior analysis into:
  • A natural-language EDA summary
  • Structured visualization recommendations
"""

from __future__ import annotations

from typing import Any, Dict, List


# ======================================================================== #
# Visualization recommender
# ======================================================================== #

def recommend_visualizations(
    numeric_cols: List[str],
    categorical_cols: List[str],
    datetime_cols: List[str],
    distributions: Dict[str, Any],
    correlations: Dict[str, Any],
    segments: List[Dict[str, Any]],
    time_series: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Return an ordered list of chart recommendations driven by the
    analysis results.
    """
    recs: List[Dict[str, Any]] = []

    # ---- 1. Histograms for every numeric column ---- #
    for col in numeric_cols:
        dist = distributions.get(col, {})
        dtype = dist.get("distribution_type", "unknown")
        recs.append({
            "chart_type": "histogram",
            "column": col,
            "reason": f"Show distribution shape ({dtype}).",
            "priority": 1,
        })

    # ---- 2. Box-plots for columns with outliers ---- #
    for seg in segments:
        if seg.get("type") == "outlier_segment":
            recs.append({
                "chart_type": "boxplot",
                "column": seg["column"],
                "reason": (
                    f"{seg['outlier_count']} IQR outliers detected "
                    f"({seg['outlier_pct']}%)."
                ),
                "priority": 2,
            })

    # ---- 3. Correlation heatmap ---- #
    if len(numeric_cols) >= 2:
        recs.append({
            "chart_type": "correlation_heatmap",
            "columns": numeric_cols,
            "reason": "Visualise pairwise correlations between numeric columns.",
            "priority": 2,
        })

    # ---- 4. Scatter plots for strong correlations ---- #
    for pair in correlations.get("strong_positive", []):
        recs.append({
            "chart_type": "scatter",
            "columns": pair["columns"],
            "reason": f"Strong positive correlation ({pair['correlation']}).",
            "priority": 3,
        })
    for pair in correlations.get("strong_negative", []):
        recs.append({
            "chart_type": "scatter",
            "columns": pair["columns"],
            "reason": f"Strong negative correlation ({pair['correlation']}).",
            "priority": 3,
        })

    # ---- 5. Bar charts for categoricals ---- #
    for col in categorical_cols:
        recs.append({
            "chart_type": "bar",
            "column": col,
            "reason": "Show category frequency distribution.",
            "priority": 4,
        })

    # ---- 6. Grouped bar / box for categorical splits ---- #
    for seg in segments:
        if seg.get("type") == "categorical_split":
            recs.append({
                "chart_type": "grouped_boxplot",
                "categorical_column": seg["split_column"],
                "numeric_column": seg["target_column"],
                "reason": seg["description"],
                "priority": 3,
            })

    # ---- 7. Time-series line charts ---- #
    if time_series and time_series.get("analyses"):
        dt_col = time_series["datetime_column"]
        for num, info in time_series["analyses"].items():
            if info.get("status") == "insufficient_data":
                continue
            trend_dir = info.get("trend", {}).get("trend", "unknown")
            recs.append({
                "chart_type": "line",
                "x_column": dt_col,
                "y_column": num,
                "reason": f"Time-series with {trend_dir} trend.",
                "priority": 2,
            })

    # Sort by priority (lower = more important)
    recs.sort(key=lambda r: r.get("priority", 99))
    return recs


# ======================================================================== #
# EDA summary generator
# ======================================================================== #

def generate_summary(
    metadata: Dict[str, Any],
    numeric_summary: Dict[str, Any],
    categorical_summary: Dict[str, Any],
    distributions: Dict[str, Any],
    correlations: Dict[str, Any],
    segments: List[Dict[str, Any]],
    time_series: Dict[str, Any],
) -> str:
    """
    Produce a concise, factual, plain-text summary of the EDA findings.
    """
    lines: List[str] = []
    rows = metadata.get("row_count", "?")
    cols = metadata.get("column_count", "?")

    lines.append(f"Dataset contains {rows:,} rows and {cols} columns.")

    # Missing data
    total_missing = metadata.get("total_missing_cells", 0)
    if total_missing:
        lines.append(
            f"There are {total_missing:,} missing cells across the dataset."
        )
    else:
        lines.append("No missing values detected.")

    # Distributions
    skewed = [
        col for col, d in distributions.items()
        if d.get("distribution_type", "").endswith("skewed")
    ]
    if skewed:
        lines.append(
            f"Skewed distributions detected in: {', '.join(skewed)}."
        )

    multimodal = [
        col for col, d in distributions.items()
        if d.get("is_multimodal")
    ]
    if multimodal:
        lines.append(
            f"Multimodal distributions detected in: {', '.join(multimodal)}."
        )

    # Correlations
    sp = correlations.get("strong_positive", [])
    sn = correlations.get("strong_negative", [])
    if sp:
        pairs = [f"{p['columns'][0]} & {p['columns'][1]} ({p['correlation']})" for p in sp]
        lines.append(f"Strong positive correlations: {'; '.join(pairs)}.")
    if sn:
        pairs = [f"{p['columns'][0]} & {p['columns'][1]} ({p['correlation']})" for p in sn]
        lines.append(f"Strong negative correlations: {'; '.join(pairs)}.")
    if not sp and not sn:
        lines.append("No strong linear correlations detected (|r| >= 0.7).")

    # Segments
    cat_splits = [s for s in segments if s.get("type") == "categorical_split"]
    outlier_segs = [s for s in segments if s.get("type") == "outlier_segment"]
    if cat_splits:
        top = cat_splits[0]
        lines.append(
            f"Most impactful categorical split: '{top['split_column']}' on "
            f"'{top['target_column']}' (effect = {top['effect_size']}σ)."
        )
    if outlier_segs:
        names = [s["column"] for s in outlier_segs]
        lines.append(f"Outliers detected in: {', '.join(names)}.")

    # Time series
    if time_series and time_series.get("analyses"):
        for num, info in time_series["analyses"].items():
            trend = info.get("trend", {}).get("trend", "unknown")
            r2 = info.get("trend", {}).get("r_squared", "?")
            lines.append(f"Time-series '{num}': {trend} trend (R²={r2}).")

    return " ".join(lines)


# ======================================================================== #
# Insight bullets (structured)
# ======================================================================== #

def generate_insights(
    numeric_summary: Dict[str, Any],
    categorical_summary: Dict[str, Any],
    distributions: Dict[str, Any],
    correlations: Dict[str, Any],
    segments: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Return a list of insight dicts ``{category, detail}``."""
    insights: List[Dict[str, str]] = []

    # Dominant features
    for col, stats in numeric_summary.items():
        cv = stats.get("cv")
        if cv is not None and cv > 1.0:
            insights.append({
                "category": "high_variability",
                "detail": (
                    f"'{col}' has a coefficient of variation of {cv:.2f}, "
                    f"indicating high relative variability."
                ),
            })

    for col, stats in categorical_summary.items():
        pct = stats.get("mode_pct", 0)
        if pct >= 60:
            insights.append({
                "category": "dominant_category",
                "detail": (
                    f"'{col}' is dominated by '{stats['mode']}' "
                    f"({pct}% of values)."
                ),
            })

    # Unexpected correlations
    for pair in correlations.get("strong_positive", []):
        insights.append({
            "category": "strong_correlation",
            "detail": (
                f"Strong positive correlation ({pair['correlation']}) between "
                f"'{pair['columns'][0]}' and '{pair['columns'][1]}'."
            ),
        })
    for pair in correlations.get("strong_negative", []):
        insights.append({
            "category": "strong_correlation",
            "detail": (
                f"Strong negative correlation ({pair['correlation']}) between "
                f"'{pair['columns'][0]}' and '{pair['columns'][1]}'."
            ),
        })

    # Skewed distributions
    for col, d in distributions.items():
        if d.get("distribution_type", "").endswith("skewed"):
            insights.append({
                "category": "skewed_distribution",
                "detail": (
                    f"'{col}' is {d['distribution_type'].replace('_', '-')} "
                    f"(skew={d['skewness']})."
                ),
            })

    # Potential drivers (top categorical splits)
    for seg in segments:
        if seg.get("type") == "categorical_split" and seg["effect_size"] >= 0.5:
            insights.append({
                "category": "potential_driver",
                "detail": seg["description"],
            })

    return insights
