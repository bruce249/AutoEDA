"""
Exploratory Data Analysis Agent — Orchestrator
================================================
Receives the standardized output from the Ingestion Agent and runs:

    STEP 1  →  Univariate analysis     (univariate)
    STEP 2  →  Bivariate analysis      (bivariate)
    STEP 3  →  Distribution analysis   (distributions)
    STEP 4  →  Segmentation            (segmentation)
    STEP 5  →  Time-series analysis    (time_series)
    STEP 6  →  Insight generation      (insights)

Returns a structured EDA result dict for downstream agents.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from .univariate import (
    numeric_univariate,
    categorical_univariate,
    datetime_univariate,
)
from .bivariate import (
    correlation_matrix,
    strong_correlations,
    grouped_statistics,
    categorical_associations,
)
from .distributions import distribution_analysis
from .segmentation import detect_segments
from .time_series import time_series_analysis
from .insights import (
    recommend_visualizations,
    generate_summary,
    generate_insights,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output type-alias
# ---------------------------------------------------------------------------
EDAResult = Dict[str, Any]
"""
{
    "eda_summary":   str,
    "statistics":    { "numeric_summary": {…}, "categorical_summary": {…},
                       "datetime_summary": {…} },
    "correlations":  { "matrix": DataFrame, "strong_positive": [], "strong_negative": [] },
    "distributions": {…},
    "grouped_stats": [{…}],
    "categorical_associations": [{…}],
    "segments":      [{…}],
    "time_series":   {…},
    "insights":      [{category, detail}],
    "visualization_recommendations": [{…}],
}
"""


class EDAAgent:
    """
    Single entry-point for the Exploratory Data Analysis Agent.

    Usage
    -----
    >>> from agents.eda import EDAAgent
    >>> eda = EDAAgent()
    >>> result = eda.run(ingestion_output)
    """

    def __init__(self, *, verbose: bool = True):
        if verbose:
            logging.basicConfig(
                level=logging.INFO,
                format="[%(levelname)s] %(name)s — %(message)s",
            )

    # ------------------------------------------------------------------ #
    # Main pipeline
    # ------------------------------------------------------------------ #
    def run(self, data: Dict[str, Any]) -> EDAResult:
        """
        Execute the full EDA pipeline.

        Parameters
        ----------
        data : dict
            Standardized output from the Ingestion (or Quality) Agent.
            Must contain keys: ``dataframe``, ``schema``, ``metadata``.

        Returns
        -------
        EDAResult
        """
        df: pd.DataFrame = data["dataframe"]
        schema: Dict = data["schema"]
        metadata: Dict = data["metadata"]

        num_cols: List[str] = schema.get("numeric_columns", [])
        cat_cols: List[str] = schema.get("categorical_columns", [])
        dt_cols: List[str] = schema.get("datetime_columns", [])

        # ====== STEP 1: Univariate ====== #
        logger.info("EDA STEP 1 — Univariate analysis …")
        num_summary = numeric_univariate(df, num_cols)
        cat_summary = categorical_univariate(df, cat_cols)
        dt_summary = datetime_univariate(df, dt_cols)

        # ====== STEP 2: Bivariate ====== #
        logger.info("EDA STEP 2 — Bivariate analysis …")
        corr_matrix = correlation_matrix(df, num_cols)
        corr_pairs = strong_correlations(corr_matrix)
        grp_stats = grouped_statistics(df, cat_cols, num_cols)
        cat_assoc = categorical_associations(df, cat_cols)

        # ====== STEP 3: Distributions ====== #
        logger.info("EDA STEP 3 — Distribution analysis …")
        dist = distribution_analysis(df, num_cols)

        # ====== STEP 4: Segmentation ====== #
        logger.info("EDA STEP 4 — Segmentation …")
        segs = detect_segments(df, num_cols, cat_cols)

        # ====== STEP 5: Time-series ====== #
        logger.info("EDA STEP 5 — Time-series analysis …")
        ts = time_series_analysis(df, dt_cols, num_cols)

        # ====== STEP 6: Insights & viz ====== #
        logger.info("EDA STEP 6 — Generating insights & visualization recs …")
        viz_recs = recommend_visualizations(
            num_cols, cat_cols, dt_cols, dist, corr_pairs, segs, ts,
        )
        summary_text = generate_summary(
            metadata, num_summary, cat_summary, dist, corr_pairs, segs, ts,
        )
        insight_list = generate_insights(
            num_summary, cat_summary, dist, corr_pairs, segs,
        )

        # ====== Assemble output ====== #
        result: EDAResult = {
            "eda_summary": summary_text,
            "statistics": {
                "numeric_summary": num_summary,
                "categorical_summary": cat_summary,
                "datetime_summary": dt_summary,
            },
            "correlations": {
                "matrix": corr_matrix,
                "strong_positive": corr_pairs.get("strong_positive", []),
                "strong_negative": corr_pairs.get("strong_negative", []),
            },
            "distributions": dist,
            "grouped_stats": grp_stats,
            "categorical_associations": cat_assoc,
            "segments": segs,
            "time_series": ts,
            "insights": insight_list,
            "visualization_recommendations": viz_recs,
        }

        self._print_report(result)
        return result

    # ------------------------------------------------------------------ #
    # Pretty-print
    # ------------------------------------------------------------------ #
    @staticmethod
    def _print_report(result: EDAResult) -> None:
        print("\n" + "=" * 70)
        print("  EXPLORATORY DATA ANALYSIS — REPORT")
        print("=" * 70)

        # Summary
        print("\n--- Summary ---")
        print(result["eda_summary"])

        # Distributions
        print("\n--- Distribution Types ---")
        for col, d in result["distributions"].items():
            dtype = d.get("distribution_type", "?")
            skew = d.get("skewness", "?")
            print(f"  {col:25s}  {dtype:25s}  skew={skew}")

        # Correlations
        print("\n--- Strong Correlations ---")
        sp = result["correlations"]["strong_positive"]
        sn = result["correlations"]["strong_negative"]
        if sp:
            for p in sp:
                print(f"  (+) {p['columns'][0]} ↔ {p['columns'][1]}  r={p['correlation']}")
        if sn:
            for p in sn:
                print(f"  (−) {p['columns'][0]} ↔ {p['columns'][1]}  r={p['correlation']}")
        if not sp and not sn:
            print("  No strong correlations (|r| ≥ 0.7).")

        # Segments
        print(f"\n--- Segments ({len(result['segments'])}) ---")
        for seg in result["segments"][:10]:
            print(f"  • {seg.get('description', seg.get('type'))}")

        # Insights
        print(f"\n--- Insights ({len(result['insights'])}) ---")
        for ins in result["insights"]:
            print(f"  [{ins['category']}] {ins['detail']}")

        # Viz recommendations
        recs = result["visualization_recommendations"]
        print(f"\n--- Visualization Recommendations ({len(recs)}) ---")
        for r in recs:
            col_info = r.get("column") or r.get("columns") or r.get("y_column") or ""
            print(f"  {r['chart_type']:25s}  {str(col_info):30s}  {r['reason']}")

        print("=" * 70 + "\n")
