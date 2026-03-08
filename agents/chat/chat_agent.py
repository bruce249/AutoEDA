"""
chat_agent.py — Conversational Q&A over uploaded datasets using Hugging Face
Inference API.

Uses ``huggingface_hub.InferenceClient`` so answers stream from HF's free
serverless inference endpoints — no local GPU needed.

Model: **moonshotai/Kimi-K2.5** — fast, strong reasoning on
structured/tabular data, free-tier available.
"""

from __future__ import annotations

import json
import logging
import os
import textwrap
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Model config ─────────────────────────────────────────────────────────
DEFAULT_MODEL = "moonshotai/Kimi-K2.5"
MAX_CONTEXT_ROWS = 50          # sample rows sent to the model
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.3


class ChatAgent:
    """
    Stateful chat agent that holds a reference to the analysed dataset
    context and answers user questions via a Hugging Face LLM.

    Usage
    -----
    >>> agent = ChatAgent(data_context)
    >>> answer = agent.ask("What is the average sales by region?")
    """

    def __init__(
        self,
        data_context: Dict[str, Any],
        *,
        model: str = DEFAULT_MODEL,
        hf_token: Optional[str] = None,
    ):
        self.model = model
        self.token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        self.data_context = data_context
        self._system_prompt = self._build_system_prompt(data_context)
        self._history: List[Dict[str, str]] = []

    # ── Public API ───────────────────────────────────────────────────

    def ask(self, question: str) -> str:
        """Send a question and return the model's answer."""
        from huggingface_hub import InferenceClient

        client = InferenceClient(model=self.model, token=self.token)

        messages = [{"role": "system", "content": self._system_prompt}]
        # Append recent history (keep last 6 turns to stay within context)
        messages.extend(self._history[-6:])
        messages.append({"role": "user", "content": question})

        try:
            response = client.chat_completion(
                messages=messages,
                max_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
            )
            answer = response.choices[0].message.content.strip()
        except Exception as exc:
            logger.error("HF Inference error: %s", exc)
            answer = f"Sorry, I couldn't generate an answer. Error: {exc}"

        # Update history
        self._history.append({"role": "user", "content": question})
        self._history.append({"role": "assistant", "content": answer})

        return answer

    def clear_history(self) -> None:
        self._history.clear()

    # ── Context builder ──────────────────────────────────────────────

    @staticmethod
    def _build_system_prompt(ctx: Dict[str, Any]) -> str:
        """Construct an exhaustive system prompt with every piece of analysis data."""

        metadata = ctx.get("metadata", {})
        schema = ctx.get("schema", {})

        num_cols = schema.get("numeric_columns", [])
        cat_cols = schema.get("categorical_columns", [])
        dt_cols = schema.get("datetime_columns", [])

        # ── Column details ──
        col_info_lines = []
        for col, info in ctx.get("column_details", {}).items():
            parts = [f"  - {col} ({info.get('dtype', '?')})"]
            for k in ("mean", "std", "min", "25%", "50%", "75%", "max",
                       "unique", "null_pct", "top_value", "skewness", "kurtosis"):
                v = info.get(k)
                if v is not None:
                    if isinstance(v, float):
                        parts.append(f"{k}={v:.4g}")
                    else:
                        parts.append(f"{k}={v}")
            col_info_lines.append(", ".join(parts))
        col_info_str = "\n".join(col_info_lines) or "N/A"

        # ── Correlation matrix ──
        corr_str = ctx.get("correlation_matrix_csv", "N/A")

        # ── Strong correlations ──
        strong_pos = ctx.get("strong_positive_correlations", [])
        strong_neg = ctx.get("strong_negative_correlations", [])
        corr_pairs_str = ""
        if strong_pos:
            corr_pairs_str += "Strong positive correlations:\n"
            for p in strong_pos:
                corr_pairs_str += f"  {p[0]} ↔ {p[1]}: {p[2]:.4f}\n"
        if strong_neg:
            corr_pairs_str += "Strong negative correlations:\n"
            for p in strong_neg:
                corr_pairs_str += f"  {p[0]} ↔ {p[1]}: {p[2]:.4f}\n"
        if not corr_pairs_str:
            corr_pairs_str = "None detected."

        # ── Distribution analysis ──
        dist_str = _dict_to_str(ctx.get("distributions", {}), max_depth=3)

        # ── Grouped statistics ──
        grouped_str = _list_to_str(ctx.get("grouped_stats", []))

        # ── Categorical associations ──
        cat_assoc_str = _list_to_str(ctx.get("categorical_associations", []))

        # ── Segments ──
        segments = ctx.get("segments", [])
        seg_str = "\n".join(
            f"  - [{s.get('type','')}] {s.get('description','')}"
            for s in segments
        ) if segments else "None."

        # ── Time series ──
        ts_str = _dict_to_str(ctx.get("time_series", {}), max_depth=3)

        # ── All insights ──
        insights = ctx.get("insights", [])
        insight_str = "\n".join(
            f"  - [{i.get('category','')}] {i.get('detail','')}" for i in insights
        ) if insights else "None."

        # ── Visualization recommendations ──
        viz_recs = ctx.get("visualization_recommendations", [])
        viz_str = "\n".join(
            f"  - {v.get('chart_type','?')} on {v.get('columns','?')}: {v.get('reason','')}"
            for v in viz_recs[:30]
        ) if viz_recs else "None."

        # ── Full modeling data ──
        modeling = ctx.get("modeling", {})
        problem = modeling.get("problem", {})
        fe = modeling.get("feature_engineering", {})
        training = modeling.get("training", {})
        evaluation = modeling.get("evaluation", {})
        model_insights = modeling.get("insights", {})
        model_comparison = ctx.get("model_comparison", [])
        predictions_preview = ctx.get("predictions_preview", [])

        model_scores_str = ""
        scores = evaluation.get("model_scores", {})
        if scores:
            model_scores_str = "Model comparison (all candidates):\n"
            for name, metrics in scores.items():
                model_scores_str += f"  {name}: {metrics}\n"

        feat_importance_str = ""
        fi = evaluation.get("feature_importance", {})
        if fi:
            feat_importance_str = "Feature importance (sorted):\n"
            for feat, imp in list(fi.items())[:30]:
                feat_importance_str += f"  {feat}: {imp:.4f}\n"

        takeaways_str = "\n".join(
            f"  - {t}" for t in model_insights.get("key_takeaways", [])
        ) or "None."

        recommendations_str = "\n".join(
            f"  - [{r.get('priority','')}] {r.get('title','')}: {r.get('detail','')}"
            for r in model_insights.get("recommendations", [])
        ) or "None."

        preds_str = ""
        if predictions_preview:
            preds_str = "Predictions preview (first 20 test samples):\n"
            for p in predictions_preview[:20]:
                preds_str += f"  actual={p.get('actual')}, predicted={p.get('predicted')}\n"

        # ── Sample rows ──
        sample_csv = ctx.get("sample_csv", "")

        prompt = textwrap.dedent(f"""\
        You are AutoEDA Chat — an expert data analyst assistant.
        The user uploaded a dataset and ran a full automated analysis pipeline.
        You have access to EVERY piece of information from that analysis below.
        Be concise, factual, and cite specific numbers. If unsure, say so.
        Format answers in Markdown. Use tables when comparing values.

        ════════════════════════════════════════════════════════════
        1. DATASET METADATA
        ════════════════════════════════════════════════════════════
        Rows: {metadata.get('row_count', '?')}
        Columns: {metadata.get('column_count', '?')}
        Missing cells: {metadata.get('total_missing_cells', 0)}
        Duplicate rows: {metadata.get('duplicate_row_count', 0)}
        Memory: {metadata.get('memory_usage_mb', '?')} MB
        Source: {metadata.get('source_type', '?')}

        ════════════════════════════════════════════════════════════
        2. COLUMN SCHEMA
        ════════════════════════════════════════════════════════════
        Numeric columns: {num_cols}
        Categorical columns: {cat_cols}
        Datetime columns: {dt_cols}

        ════════════════════════════════════════════════════════════
        3. COLUMN DETAILS (stats per column)
        ════════════════════════════════════════════════════════════
        {col_info_str}

        ════════════════════════════════════════════════════════════
        4. CORRELATION MATRIX
        ════════════════════════════════════════════════════════════
        {corr_str}

        ════════════════════════════════════════════════════════════
        5. STRONG CORRELATIONS
        ════════════════════════════════════════════════════════════
        {corr_pairs_str}

        ════════════════════════════════════════════════════════════
        6. DISTRIBUTION ANALYSIS
        ════════════════════════════════════════════════════════════
        {dist_str}

        ════════════════════════════════════════════════════════════
        7. GROUPED STATISTICS (cat vs num)
        ════════════════════════════════════════════════════════════
        {grouped_str}

        ════════════════════════════════════════════════════════════
        8. CATEGORICAL ASSOCIATIONS
        ════════════════════════════════════════════════════════════
        {cat_assoc_str}

        ════════════════════════════════════════════════════════════
        9. DATA SEGMENTS
        ════════════════════════════════════════════════════════════
        {seg_str}

        ════════════════════════════════════════════════════════════
        10. TIME-SERIES ANALYSIS
        ════════════════════════════════════════════════════════════
        {ts_str}

        ════════════════════════════════════════════════════════════
        11. KEY INSIGHTS
        ════════════════════════════════════════════════════════════
        {insight_str}

        ════════════════════════════════════════════════════════════
        12. VISUALIZATION RECOMMENDATIONS
        ════════════════════════════════════════════════════════════
        {viz_str}

        ════════════════════════════════════════════════════════════
        13. EDA SUMMARY
        ════════════════════════════════════════════════════════════
        {ctx.get('eda_summary', 'N/A')}

        ════════════════════════════════════════════════════════════
        14. ML PROBLEM DETECTION
        ════════════════════════════════════════════════════════════
        Problem type: {problem.get('problem_type', 'N/A')}
        Target column: {problem.get('target', 'N/A')}
        Features used: {problem.get('features', [])}
        Reason: {problem.get('reason', 'N/A')}

        ════════════════════════════════════════════════════════════
        15. FEATURE ENGINEERING
        ════════════════════════════════════════════════════════════
        Num features: {fe.get('n_features', '?')}
        Num samples: {fe.get('n_samples', '?')}
        Transformations: {fe.get('transformations', [])}
        Target mapping: {fe.get('target_mapping', 'N/A')}

        ════════════════════════════════════════════════════════════
        16. MODEL TRAINING
        ════════════════════════════════════════════════════════════
        Models trained: {training.get('models_trained', [])}
        Train/test split: {training.get('split_info', 'N/A')}

        ════════════════════════════════════════════════════════════
        17. MODEL SCORES (ALL CANDIDATES)
        ════════════════════════════════════════════════════════════
        Best model: {evaluation.get('best_model', 'N/A')}
        Best metrics: {evaluation.get('best_metrics', 'N/A')}

        {model_scores_str}

        ════════════════════════════════════════════════════════════
        18. FEATURE IMPORTANCE
        ════════════════════════════════════════════════════════════
        {feat_importance_str or 'N/A'}

        ════════════════════════════════════════════════════════════
        19. MODEL INSIGHT REPORT
        ════════════════════════════════════════════════════════════
        {model_insights.get('report_text', 'N/A')}

        ════════════════════════════════════════════════════════════
        20. KEY TAKEAWAYS
        ════════════════════════════════════════════════════════════
        {takeaways_str}

        ════════════════════════════════════════════════════════════
        21. RECOMMENDATIONS
        ════════════════════════════════════════════════════════════
        {recommendations_str}

        ════════════════════════════════════════════════════════════
        22. PREDICTIONS PREVIEW (test set)
        ════════════════════════════════════════════════════════════
        {preds_str or 'N/A'}

        ════════════════════════════════════════════════════════════
        23. SAMPLE DATA ROWS (first {MAX_CONTEXT_ROWS} rows, CSV)
        ════════════════════════════════════════════════════════════
        {sample_csv}

        Answer the user's questions using ALL the data above.
        If they ask about correlations, use the full matrix.
        If they ask about model comparison, cite every model's scores.
        If they ask for calculations you can't compute exactly, estimate from the stats.
        """)

        return prompt


# ═══════════════════════════════════════════════════════════════════════
# Context builder — called once after pipeline completes
# ═══════════════════════════════════════════════════════════════════════

def build_chat_context(
    ingestion_result: Dict[str, Any],
    eda_result: Dict[str, Any],
    modeling_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build the FULL context dict that ChatAgent needs from all pipeline results.
    Includes every bit of analysis data so the LLM can answer any question.
    """
    df: pd.DataFrame = ingestion_result["dataframe"]
    schema = ingestion_result["schema"]
    metadata = ingestion_result["metadata"]
    num_cols = schema.get("numeric_columns", [])

    # ── Column-level details (exhaustive) ──
    col_details = _build_column_details(df, ingestion_result)

    # ── Correlation matrix as CSV ──
    corr_data = eda_result.get("correlations", {})
    corr_matrix = corr_data.get("matrix")
    corr_csv = ""
    if isinstance(corr_matrix, pd.DataFrame) and not corr_matrix.empty:
        corr_csv = corr_matrix.round(4).to_csv()
    elif isinstance(corr_matrix, dict):
        try:
            corr_csv = pd.DataFrame(corr_matrix).round(4).to_csv()
        except Exception:
            corr_csv = str(corr_matrix)

    # Strong correlations — normalise to list of tuples
    strong_pos = _normalise_corr_pairs(corr_data.get("strong_positive", []))
    strong_neg = _normalise_corr_pairs(corr_data.get("strong_negative", []))

    # ── Distributions ──
    dist = eda_result.get("distributions", {})
    safe_dist = _make_json_safe(dist)

    # ── Grouped stats ──
    grouped = eda_result.get("grouped_stats", [])
    safe_grouped = _make_json_safe(grouped)

    # ── Categorical associations ──
    cat_assoc = eda_result.get("categorical_associations", [])
    safe_cat_assoc = _make_json_safe(cat_assoc)

    # ── Time-series ──
    ts = eda_result.get("time_series", {})
    safe_ts = _make_json_safe(ts)

    # ── Sample CSV (first N rows) ──
    sample_df = df.head(MAX_CONTEXT_ROWS)
    sample_csv = sample_df.to_csv(index=False)

    # ── Modeling data ──
    modeling = modeling_result or {}

    ctx = {
        "metadata": metadata,
        "schema": schema,
        "column_details": col_details,
        "eda_summary": eda_result.get("eda_summary", ""),
        "insights": eda_result.get("insights", []),
        "segments": eda_result.get("segments", []),
        "visualization_recommendations": eda_result.get("visualization_recommendations", []),
        "correlation_matrix_csv": corr_csv,
        "strong_positive_correlations": strong_pos,
        "strong_negative_correlations": strong_neg,
        "distributions": safe_dist,
        "grouped_stats": safe_grouped,
        "categorical_associations": safe_cat_assoc,
        "time_series": safe_ts,
        "sample_csv": sample_csv,
        "modeling": modeling,
        "model_comparison": modeling.get("model_comparison", []),
        "predictions_preview": modeling.get("predictions_preview", []),
    }

    return ctx


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _build_column_details(df: pd.DataFrame, ingestion_result: Dict) -> Dict:
    """Extract exhaustive per-column stats."""
    col_details = {}

    # From ingestion column_summary if available
    col_summary = ingestion_result.get("column_summary", {})
    if isinstance(col_summary, pd.DataFrame):
        for _, row in col_summary.iterrows():
            name = row.get("column", "")
            col_details[name] = {
                "dtype": str(row.get("dtype", "")),
                "mean": _safe_float(row.get("mean")),
                "std": _safe_float(row.get("std")),
                "min": _safe_val(row.get("min")),
                "25%": _safe_float(row.get("25%")),
                "50%": _safe_float(row.get("50%")),
                "75%": _safe_float(row.get("75%")),
                "max": _safe_val(row.get("max")),
                "unique": _safe_int(row.get("unique")),
                "null_pct": _safe_float(row.get("null_pct")),
                "top_value": _safe_val(row.get("top_value")),
                "skewness": _safe_float(row.get("skewness")),
                "kurtosis": _safe_float(row.get("kurtosis")),
            }

    # Fill in from df.describe() for any columns we missed
    try:
        desc = df.describe(include="all").T
        for col in df.columns:
            if col not in col_details:
                col_details[col] = {"dtype": str(df[col].dtype)}
            info = col_details[col]
            if col in desc.index:
                for stat in ("mean", "std", "min", "25%", "50%", "75%", "max", "count"):
                    if stat not in info or info.get(stat) is None:
                        info[stat] = _safe_float(desc.at[col, stat] if stat in desc.columns else None)
                if info.get("unique") is None:
                    info["unique"] = _safe_int(desc.at[col, "unique"] if "unique" in desc.columns else df[col].nunique())
                if info.get("top_value") is None and "top" in desc.columns:
                    info["top_value"] = _safe_val(desc.at[col, "top"])
            # Null percentage
            if info.get("null_pct") is None:
                null_ct = int(df[col].isnull().sum())
                info["null_pct"] = round(null_ct / len(df) * 100, 2) if len(df) > 0 else 0
    except Exception as exc:
        logger.warning("Error enriching column details: %s", exc)

    return col_details


def _normalise_corr_pairs(pairs) -> List:
    """Ensure correlation pairs are [col_a, col_b, value] tuples."""
    result = []
    for p in (pairs or []):
        if isinstance(p, (list, tuple)) and len(p) >= 3:
            result.append([str(p[0]), str(p[1]), _safe_float(p[2]) or 0.0])
        elif isinstance(p, dict):
            result.append([
                str(p.get("col_a", p.get("column_a", "?"))),
                str(p.get("col_b", p.get("column_b", "?"))),
                _safe_float(p.get("correlation", p.get("value", 0))) or 0.0,
            ])
    return result


def _make_json_safe(obj, max_depth=5, _depth=0):
    """Recursively convert an object to JSON-safe primitives."""
    if _depth > max_depth:
        return str(obj)
    if isinstance(obj, pd.DataFrame):
        return obj.head(100).to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.head(100).to_dict()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v, max_depth, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(v, max_depth, _depth + 1) for v in obj]
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    return obj


def _dict_to_str(d, max_depth=3, _depth=0, indent=2) -> str:
    """Pretty-print a dict/list for the system prompt."""
    if _depth > max_depth:
        return str(d)[:200]
    if isinstance(d, dict):
        if not d:
            return "N/A"
        lines = []
        for k, v in list(d.items())[:50]:
            spacer = " " * indent * _depth
            val_str = _dict_to_str(v, max_depth, _depth + 1, indent)
            lines.append(f"{spacer}{k}: {val_str}")
        return "\n".join(lines)
    if isinstance(d, (list, tuple)):
        if not d:
            return "[]"
        items = []
        for item in d[:30]:
            items.append(_dict_to_str(item, max_depth, _depth + 1, indent))
        return "\n".join(items)
    if isinstance(d, float):
        return f"{d:.4g}" if not (np.isnan(d) or np.isinf(d)) else "null"
    return str(d)[:300]


def _list_to_str(lst) -> str:
    """Convert a list of dicts to a readable string."""
    if not lst:
        return "None."
    lines = []
    for item in lst[:40]:
        if isinstance(item, dict):
            parts = ", ".join(f"{k}={v}" for k, v in item.items() if not isinstance(v, (dict, list, pd.DataFrame)))
            lines.append(f"  - {parts}")
        else:
            lines.append(f"  - {item}")
    return "\n".join(lines) or "None."


def _safe_float(v):
    try:
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else f
    except (TypeError, ValueError):
        return None

def _safe_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None

def _safe_val(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return v
