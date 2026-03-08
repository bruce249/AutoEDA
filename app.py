"""
FastAPI backend for the Multi-Agent Data Analyst (AutoEDA) platform.
Serves the frontend and exposes /api/analyze for file uploads.
"""

from __future__ import annotations

import io
import json
import logging
import os

from dotenv import load_dotenv
load_dotenv()  # reads .env into os.environ before anything else
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
from agents.modeling import ModelingAgent
from agents.chat import ChatAgent
from agents.chat.chat_agent import build_chat_context

logger = logging.getLogger(__name__)

# ── In-memory session store (single-user; swap for Redis in prod) ────────
_active_chat: dict = {}  # {"agent": ChatAgent | None}

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


def _build_modeling_charts(modeling_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build Plotly-compatible charts for the Modeling Agent results:
    - Model comparison bar chart
    - Feature importance horizontal bar
    - Actual vs Predicted scatter / line
    """
    charts: List[Dict[str, Any]] = []
    evaluation = modeling_result.get("evaluation", {})
    problem = modeling_result.get("problem", {})
    problem_type = problem.get("problem_type", "")

    # ── 1. Model Comparison ──────────────────────────────────────────
    comparison = modeling_result.get("model_comparison", [])
    if comparison:
        names = [c["name"] for c in comparison]
        # Pick primary metric
        if problem_type in ("regression", "time_series"):
            metric_key = "R²"
        elif problem_type == "classification":
            metric_key = "F1"
        else:
            metric_key = "Silhouette"
        values = [c["metrics"].get(metric_key, 0) for c in comparison]
        best = evaluation.get("best_model", "")
        colors = ["rgba(74,222,128,0.85)" if n == best else "rgba(99,102,241,0.65)" for n in names]

        charts.append({
            "id": "model_comparison",
            "title": "Model Comparison",
            "subtitle": f"Primary metric: {metric_key}",
            "type": "model_bar",
            "traces": [{
                "x": names,
                "y": values,
                "type": "bar",
                "marker": {"color": colors, "line": {"color": "rgba(255,255,255,0.1)", "width": 1}},
                "text": [f"{v:.4f}" for v in values],
                "textposition": "outside",
                "textfont": {"color": "#94a3b8", "size": 12},
            }],
            "layout": {
                "xaxis": {"title": "Model"},
                "yaxis": {"title": metric_key},
            },
        })

        # If regression, show additional metrics (RMSE, MAE) grouped
        if problem_type in ("regression", "time_series") and len(comparison) > 1:
            for extra_metric in ("RMSE", "MAE"):
                extra_vals = [c["metrics"].get(extra_metric, 0) for c in comparison]
                if any(v > 0 for v in extra_vals):
                    charts.append({
                        "id": f"model_{extra_metric.lower()}",
                        "title": f"Model {extra_metric}",
                        "subtitle": "Lower is better",
                        "type": "model_bar",
                        "traces": [{
                            "x": names,
                            "y": extra_vals,
                            "type": "bar",
                            "marker": {"color": [
                                "rgba(34,211,238,0.85)" if n == best else "rgba(99,102,241,0.5)"
                                for n in names
                            ]},
                            "text": [f"{v:.3f}" for v in extra_vals],
                            "textposition": "outside",
                            "textfont": {"color": "#94a3b8", "size": 12},
                        }],
                        "layout": {"xaxis": {"title": "Model"}, "yaxis": {"title": extra_metric}},
                    })

    # ── 2. Feature Importance ────────────────────────────────────────
    importance = evaluation.get("feature_importance", {})
    if importance:
        feats = list(importance.keys())[:15]
        scores = [importance[f] for f in feats]
        # Reverse for horizontal bar (top at top)
        feats = feats[::-1]
        scores = scores[::-1]
        charts.append({
            "id": "feature_importance",
            "title": "Feature Importance",
            "subtitle": f"From {evaluation.get('best_model', 'best model')}",
            "type": "feature_importance",
            "traces": [{
                "y": feats,
                "x": scores,
                "type": "bar",
                "orientation": "h",
                "marker": {
                    "color": scores,
                    "colorscale": [[0, "rgba(99,102,241,0.4)"], [1, "rgba(168,85,247,0.95)"]],
                },
            }],
            "layout": {
                "xaxis": {"title": "Relative Importance"},
                "margin": {"l": 160},
            },
        })

    # ── 3. Actual vs Predicted ───────────────────────────────────────
    preview = modeling_result.get("predictions_preview", [])
    if preview and "actual" in preview[0]:
        actuals = [p["actual"] for p in preview if p["actual"] is not None]
        preds = [p["predicted"] for p in preview if p["predicted"] is not None]

        if problem_type in ("regression", "time_series"):
            # Scatter plot: actual vs predicted
            min_v = min(actuals + preds) if actuals else 0
            max_v = max(actuals + preds) if actuals else 1
            charts.append({
                "id": "actual_vs_predicted",
                "title": "Actual vs Predicted",
                "subtitle": "Perfect predictions lie on the diagonal",
                "type": "predictions",
                "traces": [
                    {
                        "x": actuals,
                        "y": preds,
                        "mode": "markers",
                        "type": "scatter",
                        "name": "Predictions",
                        "marker": {"color": "rgba(99,102,241,0.6)", "size": 6},
                    },
                    {
                        "x": [min_v, max_v],
                        "y": [min_v, max_v],
                        "mode": "lines",
                        "type": "scatter",
                        "name": "Perfect",
                        "line": {"color": "rgba(74,222,128,0.5)", "dash": "dash", "width": 2},
                    },
                ],
                "layout": {
                    "xaxis": {"title": "Actual"},
                    "yaxis": {"title": "Predicted"},
                },
            })

            # Residual plot
            residuals = [p - a for a, p in zip(actuals, preds)]
            charts.append({
                "id": "residuals",
                "title": "Residual Distribution",
                "subtitle": "Should be centred around 0",
                "type": "residuals",
                "traces": [{
                    "x": residuals,
                    "type": "histogram",
                    "marker": {"color": "rgba(168,85,247,0.6)", "line": {"color": "rgba(168,85,247,1)", "width": 1}},
                    "nbinsx": 30,
                }],
                "layout": {
                    "xaxis": {"title": "Residual (Predicted − Actual)"},
                    "yaxis": {"title": "Count"},
                },
            })

        elif problem_type == "time_series":
            # Line chart: actual and predicted over index
            idx = list(range(len(actuals)))
            charts.append({
                "id": "ts_predictions",
                "title": "Time-Series Predictions",
                "subtitle": "Test set: actual vs predicted",
                "type": "predictions",
                "traces": [
                    {"x": idx, "y": actuals, "mode": "lines", "type": "scatter", "name": "Actual",
                     "line": {"color": "rgba(99,102,241,0.8)", "width": 2}},
                    {"x": idx, "y": preds, "mode": "lines", "type": "scatter", "name": "Predicted",
                     "line": {"color": "rgba(74,222,128,0.8)", "width": 2, "dash": "dot"}},
                ],
                "layout": {"xaxis": {"title": "Index"}, "yaxis": {"title": problem.get("target", "Value")}},
            })

    # ── 4. Cluster scatter (2D PCA) for clustering ───────────────────
    if problem_type == "clustering" and preview:
        clusters = [p.get("cluster", 0) for p in preview]
        color_map = [
            "rgba(99,102,241,0.8)", "rgba(168,85,247,0.8)", "rgba(236,72,153,0.8)",
            "rgba(34,211,238,0.8)", "rgba(74,222,128,0.8)", "rgba(250,204,21,0.8)",
            "rgba(244,63,94,0.8)", "rgba(148,163,184,0.8)",
        ]
        colors = [color_map[c % len(color_map)] for c in clusters]
        charts.append({
            "id": "cluster_distribution",
            "title": "Cluster Distribution",
            "subtitle": f"{len(set(clusters))} clusters",
            "type": "cluster",
            "traces": [{
                "x": list(range(len(clusters))),
                "y": clusters,
                "mode": "markers",
                "type": "scatter",
                "marker": {"color": colors, "size": 6},
            }],
            "layout": {"xaxis": {"title": "Sample Index"}, "yaxis": {"title": "Cluster"}},
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

        # Agent 3 — Modeling & Insights
        modeling = ModelingAgent(verbose=False)
        modeling_result = modeling.run(ingestion_result, eda_result)

        df = ingestion_result["dataframe"]

        # Build chart data
        chart_data = _build_chart_data(df, eda_result, ingestion_result["schema"])
        modeling_charts = _build_modeling_charts(modeling_result)

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
            # ── Modeling Agent output ──
            "modeling": {
                "problem": modeling_result.get("problem", {}),
                "feature_engineering": modeling_result.get("feature_engineering", {}),
                "training": modeling_result.get("training", {}),
                "evaluation": modeling_result.get("evaluation", {}),
                "insights": modeling_result.get("insights", {}),
                "predictions_preview": modeling_result.get("predictions_preview", []),
                "model_comparison": modeling_result.get("model_comparison", []),
            },
            "modeling_charts": modeling_charts,
        }

        # Build chat context and store a ChatAgent for this session
        chat_ctx = build_chat_context(ingestion_result, eda_result, modeling_result)
        _active_chat["agent"] = ChatAgent(chat_ctx)

        return _sanitise(payload)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, detail=str(e))
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Answer a user question about the currently-loaded dataset."""
    agent: ChatAgent | None = _active_chat.get("agent")
    if agent is None:
        raise HTTPException(400, "No dataset loaded yet. Upload a file first.")
    answer = agent.ask(req.message)
    return {"answer": answer}


# ── Dev entry-point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
