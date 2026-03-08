"""
STEP 1 — Query Parser
Parses the user's natural-language (or structured) request and produces a
QueryPlan that downstream components consume.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class QueryPlan:
    """Structured representation of a parsed user query."""

    source_type: str                        # csv | excel | sql | api
    source_path: str                        # file path, connection string, or URL
    table_name: Optional[str] = None        # for SQL sources
    columns: Optional[List[str]] = None     # subset of columns requested
    filters: Optional[Dict] = None          # column-level filters
    row_limit: Optional[int] = None         # max rows to load
    extra: Dict = field(default_factory=dict)  # any extra params

    def summary(self) -> str:
        parts = [
            f"Source type : {self.source_type}",
            f"Source path : {self.source_path}",
        ]
        if self.table_name:
            parts.append(f"Table       : {self.table_name}")
        if self.columns:
            parts.append(f"Columns     : {', '.join(self.columns)}")
        if self.filters:
            parts.append(f"Filters     : {self.filters}")
        if self.row_limit:
            parts.append(f"Row limit   : {self.row_limit}")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Extension / MIME heuristics
# ---------------------------------------------------------------------------

_EXT_MAP = {
    ".csv":   "csv",
    ".tsv":   "csv",       # TSV is handled as CSV with tab delimiter
    ".xls":   "excel",
    ".xlsx":  "excel",
    ".xlsm":  "excel",
    ".xlsb":  "excel",
    ".json":  "api",       # local JSON treated like an API response
    ".db":    "sql",
    ".sqlite": "sql",
    ".sqlite3": "sql",
}


def _detect_source_type(path_or_url: str) -> str:
    """Heuristically detect the data-source type."""

    # URL → API
    if re.match(r"^https?://", path_or_url, re.IGNORECASE):
        return "api"

    # SQLAlchemy connection string
    if any(path_or_url.startswith(prefix) for prefix in (
        "sqlite:///", "postgresql://", "mysql://",
        "mssql://", "oracle://", "postgres://",
    )):
        return "sql"

    # File extension
    _, ext = os.path.splitext(path_or_url.lower())
    return _EXT_MAP.get(ext, "csv")          # default → csv


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_query(
    source: str,
    *,
    source_type: Optional[str] = None,
    table: Optional[str] = None,
    columns: Optional[List[str]] = None,
    filters: Optional[Dict] = None,
    row_limit: Optional[int] = None,
    **extra,
) -> QueryPlan:
    """
    Build a QueryPlan from user-supplied arguments.

    Parameters
    ----------
    source : str
        File path, URL, or SQLAlchemy connection string.
    source_type : str, optional
        Force a specific source type (csv | excel | sql | api).
        If *None*, the type is auto-detected from *source*.
    table : str, optional
        SQL table name (used only when source_type == 'sql').
    columns : list[str], optional
        Subset of columns to load.
    filters : dict, optional
        Column-level filters (key = column, value = condition).
    row_limit : int, optional
        Maximum number of rows to load.

    Returns
    -------
    QueryPlan
    """

    detected = source_type or _detect_source_type(source)

    return QueryPlan(
        source_type=detected,
        source_path=source,
        table_name=table,
        columns=columns,
        filters=filters,
        row_limit=row_limit,
        extra=extra,
    )
