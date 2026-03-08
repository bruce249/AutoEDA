"""
Data Ingestion and Access Agent — Orchestrator
================================================
Wires together the full pipeline:
    STEP 1  →  Parse user query          (query_parser)
    STEP 2  →  Load data                 (data_loader)
    STEP 3  →  Infer schema & metadata   (schema_inference)
    STEP 4  →  Build sample preview      (schema_inference)
    STEP 5  →  Return standardized output object

The returned dict is the contract consumed by the downstream
*Data Understanding & Quality Agent*.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from .query_parser import QueryPlan, parse_query
from .data_loader import load_data
from .schema_inference import (
    infer_schema,
    build_metadata,
    build_preview,
    column_summary,
)

logger = logging.getLogger(__name__)

# Output type-alias for readability
IngestionResult = Dict[str, Any]
"""
{
    "dataframe":  pd.DataFrame,
    "schema":     { columns, dtypes, numeric_columns, … },
    "metadata":   { row_count, column_count, missing_values, source_type },
    "preview":    pd.DataFrame   (first 5 rows),
    "column_summary": pd.DataFrame  (per-column stats),
    "query_plan": QueryPlan,
}
"""


class IngestionAgent:
    """
    Single entry-point for the Data Ingestion & Access Agent.

    Usage
    -----
    >>> agent = IngestionAgent()
    >>> result = agent.run("data/sales.csv")
    >>> result["dataframe"].shape
    (10000, 12)
    """

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(self, *, preview_rows: int = 5, verbose: bool = True):
        self.preview_rows = preview_rows
        if verbose:
            logging.basicConfig(
                level=logging.INFO,
                format="[%(levelname)s] %(name)s — %(message)s",
            )

    # ------------------------------------------------------------------ #
    # Main pipeline
    # ------------------------------------------------------------------ #
    def run(
        self,
        source: str,
        *,
        source_type: Optional[str] = None,
        table: Optional[str] = None,
        columns: Optional[List[str]] = None,
        filters: Optional[Dict] = None,
        row_limit: Optional[int] = None,
        **extra,
    ) -> IngestionResult:
        """
        Execute the full ingestion pipeline and return a standardized
        output dictionary.

        Parameters
        ----------
        source : str
            File path, URL, or SQLAlchemy connection string.
        source_type : str, optional
            Force source type (csv | excel | sql | api).
        table : str, optional
            SQL table name.
        columns : list[str], optional
            Subset of columns to load.
        filters : dict, optional
            Column-level equality filters.
        row_limit : int, optional
            Maximum rows to load.
        **extra
            Additional keyword arguments forwarded to the loader
            (e.g. ``sheet_name``, ``headers``, ``params``).

        Returns
        -------
        IngestionResult
            Standardized dict with keys:
            dataframe, schema, metadata, preview, column_summary, query_plan.
        """

        # ------ STEP 1: Parse query ---------------------------------- #
        logger.info("STEP 1 — Parsing query …")
        plan: QueryPlan = parse_query(
            source,
            source_type=source_type,
            table=table,
            columns=columns,
            filters=filters,
            row_limit=row_limit,
            **extra,
        )
        logger.info("\n%s", plan.summary())

        # ------ STEP 2: Load data ------------------------------------- #
        logger.info("STEP 2 — Loading data …")
        df: pd.DataFrame = load_data(plan)
        logger.info("Loaded DataFrame: %d rows × %d cols", *df.shape)

        # ------ STEP 3: Schema inference ------------------------------ #
        logger.info("STEP 3 — Inferring schema & metadata …")
        schema = infer_schema(df)
        metadata = build_metadata(df, source_type=plan.source_type)

        # ------ STEP 4: Sample preview -------------------------------- #
        logger.info("STEP 4 — Building preview …")
        preview = build_preview(df, n=self.preview_rows)
        col_summary = column_summary(df)

        # ------ STEP 5: Assemble output ------------------------------- #
        logger.info("STEP 5 — Assembling standardized output …")
        result: IngestionResult = {
            "dataframe": df,
            "schema": schema,
            "metadata": metadata,
            "preview": preview,
            "column_summary": col_summary,
            "query_plan": plan,
        }

        self._log_summary(result)
        return result

    # ------------------------------------------------------------------ #
    # Pretty-print helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _log_summary(result: IngestionResult) -> None:
        meta = result["metadata"]
        schema = result["schema"]

        print("\n" + "=" * 60)
        print("  DATA INGESTION — SUMMARY")
        print("=" * 60)
        print(f"  Rows            : {meta['row_count']:,}")
        print(f"  Columns         : {meta['column_count']}")
        print(f"  Source type     : {meta['source_type']}")
        print(f"  Memory          : {meta['memory_usage_mb']:.3f} MB")
        print(f"  Missing cells   : {meta['total_missing_cells']:,}")
        print(f"  Duplicate rows  : {meta['duplicate_row_count']:,}")
        print("-" * 60)
        print(f"  Numeric cols    : {schema['numeric_columns']}")
        print(f"  Categorical cols: {schema['categorical_columns']}")
        print(f"  Datetime cols   : {schema['datetime_columns']}")
        print("=" * 60)

        print("\n📋 Column Summary:")
        print(result["column_summary"].to_string(index=False))

        print("\n📄 Preview (first rows):")
        print(result["preview"].to_string(index=False))
        print()


    # ------------------------------------------------------------------ #
    # Convenience: re-export sub-modules for power users
    # ------------------------------------------------------------------ #
    parse_query = staticmethod(parse_query)
    load_data = staticmethod(load_data)
    infer_schema = staticmethod(infer_schema)
    build_metadata = staticmethod(build_metadata)
    build_preview = staticmethod(build_preview)
    column_summary = staticmethod(column_summary)
