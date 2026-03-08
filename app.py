"""
FastAPI backend for the Multi-Agent Data Analyst (AutoEDA) platform.
Serves the frontend and exposes /api/analyze for file uploads.
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

# ── project imports ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from agents.ingestion import IngestionAgent
from agents.eda import EDAAgent

logger = logging.getLogger(__name__)

# ── FastAPI app ──────────────────────────────────────────────────────────
app = FastAPI(title="AutoEDA — Multi-Agent Data Analyst")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ── Helpers ──────────────────────────────────────────────────────────────

class _NumpyEncoder(json.JSONEncoder):
    """Handle numpy / pandas types that default json can't serialise."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            return obj.to_dict()
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        return super().default(obj)


def _sanitise(obj: Any) -> Any:
    """Recursively replace NaN / Inf with None for JSON safety."""
    if isinstance(obj, dict):
        return {k: _sanitise(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitise(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _build_chart_data(
    df: pd.DataFrame,
    eda: Dict[str, Any],
    schema: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Build Plotly-compatible trace objects for every recommended
    visualisation (and a few auto-generated ones).
    """
    charts: List[Dict[str, Any]] = []
    num_cols = schema.get("numeric_columns", [])
    cat_cols = schema.get("categorical_columns", [])

    # ── 1. Histograms for numeric columns ────────────────────────────
    for col in num_cols:
        dist = eda.get("distributions", {}).get(col, {})
        dtype = dist.get("distribution_type", "")
        charts.append({
            "id": f"hist_{col}",
            "title": f"Distribution of {col}",
            "subtitle": dtype.replace("_", " ").title() if dtype else "",
            "type": "histogram",
            "traces": [{
                "x": df[col].dropna().tolist(),
                "type": "histogram",
                "marker": {"color": "rgba(99,102,241,0.7)", "line": {"color": "rgba(99,102,241,1)", "width": 1}},
                "nbinsx": 30,
            }],
            "layout": {
                "xaxis": {"title": col},
                "yaxis": {"title": "Count"},
            },
        })

    # ── 2. Box-plots for numeric columns ─────────────────────────────
    if num_cols:
        traces = []
        for col in num_cols:
            traces.append({
                "y": df[col].dropna().tolist(),
                "type": "box",
                "name": col,
                "boxpoints": "outliers",
            })
        charts.append({
            "id": "boxplots_all",
            "title": "Box Plots — Numeric Columns",
            "subtitle": "Outlier detection at a glance",
            "type": "box",
            "traces": traces,
            "layout": {"yaxis": {"title": "Value"}},
        })

    # ── 3. Correlation heatmap ───────────────────────────────────────
    corr_matrix = eda.get("correlations", {}).get("matrix")
    if isinstance(corr_matrix, pd.DataFrame) and not corr_matrix.empty:
        cols_list = corr_matrix.columns.tolist()
        z = corr_matrix.round(3).values.tolist()
        # Build annotation text
        annotations = []
        for i, row in enumerate(z):
            for j, val in enumerate(row):
                annotations.append({
                    "x": cols_list[j], "y": cols_list[i],
                    "text": str(round(val, 2)) if val is not None else "",
                    "showarrow": False,
                    "font": {"color": "white", "size": 11},
                })
        charts.append({
            "id": "correlation_heatmap",
            "title": "Correlation Heatmap",
            "subtitle": "Pearson correlation between numeric features",
            "type": "heatmap",
            "traces": [{
                "z": z,
                "x": cols_list,
                "y": cols_list,
                "type": "heatmap",
                "colorscale": "RdBu",
                "reversescale": True,
                "zmin": -1, "zmax": 1,
            }],
            "layout": {
                "annotations": annotations,
                "xaxis": {"side": "bottom"},
            },
        })

    # ── 4. Bar charts for categorical columns ────────────────────────
    cat_summary = eda.get("statistics", {}).get("categorical_summary", {})
    for col in cat_cols:
        summary = cat_summary.get(col, {})
        tv = summary.get("top_values", {})
        if not tv:
            continue
        labels = list(tv.keys())[:15]
        values = [tv[l] for l in labels]
        charts.append({
            "id": f"bar_{col}",
            "title": f"Value Counts — {col}",
            "subtitle": f"Cardinality: {summary.get('cardinality', '?')}",
            "type": "bar",
            "traces": [{
                "x": labels,
                "y": values,
                "type": "bar",
                "marker": {
                    "color": values,
                    "colorscale": [[0, "rgba(99,102,241,0.5)"], [1, "rgba(168,85,247,0.9)"]],
                },
            }],
            "layout": {
                "xaxis": {"title": col, "tickangle": -35},
                "yaxis": {"title": "Count"},
            },
        })

    # ── 5. Scatter for strong correlations ───────────────────────────
    for pair in eda.get("correlations", {}).get("strong_positive", []):
        c1, c2 = pair["columns"]
        charts.append({
            "id": f"scatter_{c1}_{c2}",
            "title": f"{c1} vs {c2}",
            "subtitle": f"r = {pair['correlation']}",
            "type": "scatter",
            "traces": [{
                "x": df[c1].tolist(),
                "y": df[c2].tolist(),
                "mode": "markers",
                "type": "scatter",
                "marker": {"color": "rgba(99,102,241,0.6)", "size": 6},
            }],
            "layout": {
                "xaxis": {"title": c1},
                "yaxis": {"title": c2},
            },
        })
    for pair in eda.get("correlations", {}).get("strong_negative", []):
        c1, c2 = pair["columns"]
        charts.append({
            "id": f"scatter_{c1}_{c2}",
            "title": f"{c1} vs {c2}",
            "subtitle": f"r = {pair['correlation']}",
            "type": "scatter",
            "traces": [{
                "x": df[c1].tolist(),
                "y": df[c2].tolist(),
                "mode": "markers",
                "type": "scatter",
                "marker": {"color": "rgba(244,63,94,0.6)", "size": 6},
            }],
            "layout": {
                "xaxis": {"title": c1},
                "yaxis": {"title": c2},
            },
        })

    # ── 6. Grouped box-plots for top categorical splits ──────────────
    for seg in eda.get("segments", []):
        if seg.get("type") != "categorical_split":
            continue
        cat_c = seg["split_column"]
        num_c = seg["target_column"]
        groups = df.groupby(cat_c, observed=True)[num_c]
        traces = []
        for name, group in groups:
            traces.append({
                "y": group.dropna().tolist(),
                "type": "box",
                "name": str(name),
            })
        charts.append({
            "id": f"gbox_{cat_c}_{num_c}",
            "title": f"{num_c} by {cat_c}",
            "subtitle": f"Effect size: {seg['effect_size']}σ",
            "type": "grouped_box",
            "traces": traces,
            "layout": {"yaxis": {"title": num_c}},
        })
        if len(charts) > 35:
            break  # cap total charts

    # ── 7. Pie chart for categories with low cardinality ─────────────
    for col in cat_cols:
        summary = cat_summary.get(col, {})
        card = summary.get("cardinality", 999)
        if card > 8 or card < 2:
            continue
        tv = summary.get("top_values", {})
        if not tv:
            continue
        charts.append({
            "id": f"pie_{col}",
            "title": f"Proportion — {col}",
            "subtitle": "",
            "type": "pie",
            "traces": [{
                "labels": list(tv.keys()),
                "values": list(tv.values()),
                "type": "pie",
                "hole": 0.45,
                "marker": {"colors": [
                    "rgba(99,102,241,0.8)", "rgba(168,85,247,0.8)",
                    "rgba(236,72,153,0.8)", "rgba(34,211,238,0.8)",
                    "rgba(250,204,21,0.8)", "rgba(74,222,128,0.8)",
                    "rgba(251,146,60,0.8)", "rgba(148,163,184,0.8)",
                ]},
            }],
            "layout": {},
        })

    return _sanitise(charts)


# ── Routes ───────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    index = FRONTEND_DIR / "index.html"
    return FileResponse(str(index))


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    """Run ingestion + EDA on an uploaded file and return full results."""

    # Save upload to temp file
    suffix = Path(file.filename or "data.csv").suffix
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(400, f"Failed to read uploaded file: {e}")

    try:
        # Agent 1 — Ingestion
        ingestion = IngestionAgent(verbose=False)
        ingestion_result = ingestion.run(tmp_path)

        # Agent 2 — EDA
        eda = EDAAgent(verbose=False)
        eda_result = eda.run(ingestion_result)

        df = ingestion_result["dataframe"]

        # Build chart data
        chart_data = _build_chart_data(df, eda_result, ingestion_result["schema"])

        # Build the preview table
        preview_records = ingestion_result["preview"].fillna("").to_dict(orient="records")
        columns = list(ingestion_result["preview"].columns)

        # Serialisable EDA output
        eda_out = {
            k: v for k, v in eda_result.items()
            if not isinstance(v, pd.DataFrame)
        }
        # Convert matrix if present
        cm = eda_out.get("correlations", {}).get("matrix")
        if isinstance(cm, pd.DataFrame):
            eda_out["correlations"]["matrix"] = cm.round(4).to_dict()

        payload = {
            "status": "ok",
            "filename": file.filename,
            "metadata": ingestion_result["metadata"],
            "schema": ingestion_result["schema"],
            "preview": {"columns": columns, "rows": preview_records},
            "eda_summary": eda_result.get("eda_summary", ""),
            "insights": eda_result.get("insights", []),
            "segments": eda_result.get("segments", [])[:15],
            "distributions": eda_result.get("distributions", {}),
            "visualization_recommendations": eda_result.get("visualization_recommendations", []),
            "charts": chart_data,
        }

        return _sanitise(payload)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, detail=str(e))
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ── Dev entry-point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
