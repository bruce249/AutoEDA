"""
STEP 2 — Data Loader
Loads raw data from CSV / Excel / SQL / API into a pandas DataFrame.
Handles encoding, delimiter detection, connection management, and common errors.
"""

from __future__ import annotations

import io
import logging
from typing import Optional

import pandas as pd
import requests

from .query_parser import QueryPlan

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

_ENCODINGS_TO_TRY = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


def _guess_encoding(raw: bytes) -> str:
    """Try several common encodings and return the first that works."""
    for enc in _ENCODINGS_TO_TRY:
        try:
            raw.decode(enc)
            return enc
        except (UnicodeDecodeError, LookupError):
            continue
    return "utf-8"  # fallback


# ---------------------------------------------------------------------------
# Delimiter detection
# ---------------------------------------------------------------------------

def _guess_delimiter(raw: bytes, encoding: str) -> str:
    """Sniff delimiter from the first few lines."""
    try:
        import csv as _csv
        head = raw.decode(encoding).splitlines()[:20]
        sniffer = _csv.Sniffer()
        dialect = sniffer.sniff("\n".join(head))
        return dialect.delimiter
    except Exception:
        return ","


# ---------------------------------------------------------------------------
# Loaders per source type
# ---------------------------------------------------------------------------

def _load_csv(plan: QueryPlan) -> pd.DataFrame:
    """Load a CSV (or TSV) file with automatic encoding & delimiter detection."""
    raw = _read_bytes(plan.source_path)
    encoding = _guess_encoding(raw)
    delimiter = _guess_delimiter(raw, encoding)

    logger.info("CSV detected — encoding=%s  delimiter=%r", encoding, delimiter)

    kwargs: dict = {
        "filepath_or_buffer": io.BytesIO(raw),
        "encoding": encoding,
        "sep": delimiter,
        "on_bad_lines": "warn",
    }

    if plan.columns:
        kwargs["usecols"] = plan.columns

    if plan.row_limit:
        kwargs["nrows"] = plan.row_limit

    df = pd.read_csv(**kwargs)
    return df


def _load_excel(plan: QueryPlan) -> pd.DataFrame:
    """Load an Excel file (.xlsx / .xls)."""
    kwargs: dict = {
        "io": plan.source_path,
        "engine": "openpyxl",
    }

    sheet = plan.extra.get("sheet_name", 0)
    kwargs["sheet_name"] = sheet

    if plan.columns:
        kwargs["usecols"] = plan.columns

    if plan.row_limit:
        kwargs["nrows"] = plan.row_limit

    df = pd.read_excel(**kwargs)
    return df


def _load_sql(plan: QueryPlan) -> pd.DataFrame:
    """Load data from a SQL database via SQLAlchemy."""
    from sqlalchemy import create_engine, text

    engine = create_engine(plan.source_path)

    # Build query
    table = plan.table_name or plan.extra.get("table")
    if table is None:
        raise ValueError(
            "SQL source requires a table name. "
            "Pass it via `table` parameter."
        )

    cols = ", ".join(plan.columns) if plan.columns else "*"
    query = f"SELECT {cols} FROM {table}"

    # Apply simple equality filters
    if plan.filters:
        clauses = []
        for col, val in plan.filters.items():
            if isinstance(val, str):
                clauses.append(f"{col} = '{val}'")
            else:
                clauses.append(f"{col} = {val}")
        query += " WHERE " + " AND ".join(clauses)

    if plan.row_limit:
        query += f" LIMIT {plan.row_limit}"

    logger.info("SQL query: %s", query)

    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)

    return df


def _load_api(plan: QueryPlan) -> pd.DataFrame:
    """Fetch JSON from an HTTP endpoint and convert to DataFrame."""
    headers = plan.extra.get("headers", {})
    params = plan.extra.get("params", {})
    timeout = plan.extra.get("timeout", 30)

    logger.info("Fetching API: %s", plan.source_path)

    # If the source is a local .json file, read directly
    if not plan.source_path.startswith("http"):
        df = pd.read_json(plan.source_path)
    else:
        response = requests.get(
            plan.source_path,
            headers=headers,
            params=params,
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()

        # Normalize nested JSON into flat table
        if isinstance(data, list):
            df = pd.json_normalize(data)
        elif isinstance(data, dict):
            # Check common wrapper keys
            for key in ("data", "results", "items", "records", "rows"):
                if key in data and isinstance(data[key], list):
                    df = pd.json_normalize(data[key])
                    break
            else:
                df = pd.json_normalize(data)
        else:
            raise ValueError(f"Unexpected JSON root type: {type(data)}")

    if plan.columns:
        df = df[[c for c in plan.columns if c in df.columns]]

    if plan.row_limit:
        df = df.head(plan.row_limit)

    return df


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_LOADERS = {
    "csv":   _load_csv,
    "excel": _load_excel,
    "sql":   _load_sql,
    "api":   _load_api,
}


def load_data(plan: QueryPlan) -> pd.DataFrame:
    """
    Main entry point — dispatch to the correct loader based on the QueryPlan.

    Returns
    -------
    pd.DataFrame
        Raw (but decoded) DataFrame ready for schema inference.

    Raises
    ------
    ValueError
        If the source type is not supported or the loaded data is empty.
    """

    loader = _LOADERS.get(plan.source_type)
    if loader is None:
        raise ValueError(f"Unsupported source type: {plan.source_type!r}")

    df = loader(plan)

    if df.empty:
        raise ValueError(
            "Loaded dataset is empty. Check your source path / query."
        )

    # ---------- Auto-convert datetime columns ----------
    for col in df.columns:
        if df[col].dtype == object:
            try:
                converted = pd.to_datetime(df[col], infer_datetime_format=True)
                # Only accept if most values parsed successfully
                if converted.notna().sum() / max(len(df), 1) > 0.5:
                    df[col] = converted
            except (ValueError, TypeError):
                pass

    logger.info("Loaded %d rows × %d columns from %s source.",
                len(df), len(df.columns), plan.source_type)

    return df
