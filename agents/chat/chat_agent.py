"""
chat_agent.py — Conversational Q&A over uploaded datasets using Hugging Face
Inference API.

Uses ``huggingface_hub.InferenceClient`` so answers stream from HF's free
serverless inference endpoints — no local GPU needed.

Model: **Qwen/Qwen2.5-Coder-32B-Instruct** — fast, strong reasoning on
structured/tabular data, free-tier available.
"""

from __future__ import annotations

import logging
import os
import textwrap
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Model config ─────────────────────────────────────────────────────────
DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"
MAX_CONTEXT_ROWS = 30          # sample rows sent to the model
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
        """Construct a rich system prompt with dataset metadata & samples."""

        metadata = ctx.get("metadata", {})
        schema = ctx.get("schema", {})
        eda_summary = ctx.get("eda_summary", "")
        column_summary = ctx.get("column_summary", "")
        insights = ctx.get("insights", [])
        modeling = ctx.get("modeling", {})

        # Schema summary
        num_cols = schema.get("numeric_columns", [])
        cat_cols = schema.get("categorical_columns", [])
        dt_cols = schema.get("datetime_columns", [])

        col_info_lines = []
        col_details = ctx.get("column_details", {})
        for col, info in col_details.items():
            parts = [f"  - {col} ({info.get('dtype', '?')})"]
            if info.get("mean") is not None:
                parts.append(f"mean={info['mean']:.2f}")
            if info.get("min") is not None:
                parts.append(f"min={info['min']}")
            if info.get("max") is not None:
                parts.append(f"max={info['max']}")
            if info.get("unique") is not None:
                parts.append(f"unique={info['unique']}")
            if info.get("null_pct") is not None and info["null_pct"] > 0:
                parts.append(f"null={info['null_pct']:.1f}%")
            if info.get("top_value") is not None:
                parts.append(f"top='{info['top_value']}'")
            col_info_lines.append(", ".join(parts))

        col_info_str = "\n".join(col_info_lines) if col_info_lines else "No detailed column info."

        # Sample rows
        sample_csv = ctx.get("sample_csv", "")

        # Insight bullets
        insight_str = ""
        if insights:
            insight_str = "\n".join(f"- [{i.get('category','')}] {i.get('detail','')}" for i in insights[:8])

        # Modeling summary
        model_str = ""
        if modeling:
            ev = modeling.get("evaluation", {})
            prob = modeling.get("problem", {})
            if ev.get("best_model"):
                model_str = (
                    f"ML problem type: {prob.get('problem_type', '?')}, "
                    f"target: {prob.get('target', '?')}, "
                    f"best model: {ev.get('best_model', '?')}, "
                    f"metrics: {ev.get('best_metrics', {})}"
                )

        prompt = textwrap.dedent(f"""\
        You are AutoEDA Chat — an expert data analyst assistant.
        The user has uploaded a dataset and you must answer questions about it.
        Be concise, factual, and use numbers from the data. If you're unsure, say so.
        Format answers in Markdown. Use tables when comparing values.

        ── DATASET INFO ──
        Rows: {metadata.get('row_count', '?')}
        Columns: {metadata.get('column_count', '?')}
        Missing cells: {metadata.get('total_missing_cells', 0)}
        Memory: {metadata.get('memory_usage_mb', '?')} MB

        ── COLUMN SCHEMA ──
        Numeric: {num_cols}
        Categorical: {cat_cols}
        Datetime: {dt_cols}

        ── COLUMN DETAILS ──
        {col_info_str}

        ── SAMPLE ROWS (CSV) ──
        {sample_csv}

        ── EDA SUMMARY ──
        {eda_summary}

        ── KEY INSIGHTS ──
        {insight_str}

        ── MODELING ──
        {model_str}

        Answer the user's questions about this data. If they ask for calculations
        you cannot perform exactly, give your best estimate from the stats above.
        """)

        return prompt


def build_chat_context(
    ingestion_result: Dict[str, Any],
    eda_result: Dict[str, Any],
    modeling_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build the context dict that ChatAgent needs from pipeline results.
    Called once after analysis completes; stored server-side per session.
    """
    df: pd.DataFrame = ingestion_result["dataframe"]
    schema = ingestion_result["schema"]
    metadata = ingestion_result["metadata"]

    # Column-level details
    col_details = {}
    col_summary = ingestion_result.get("column_summary", {})
    if isinstance(col_summary, pd.DataFrame):
        for _, row in col_summary.iterrows():
            name = row.get("column", "")
            col_details[name] = {
                "dtype": str(row.get("dtype", "")),
                "mean": _safe_float(row.get("mean")),
                "min": _safe_val(row.get("min")),
                "max": _safe_val(row.get("max")),
                "unique": _safe_int(row.get("unique")),
                "null_pct": _safe_float(row.get("null_pct")),
                "top_value": _safe_val(row.get("top_value")),
            }

    # Sample CSV (first N rows)
    sample_df = df.head(MAX_CONTEXT_ROWS)
    sample_csv = sample_df.to_csv(index=False)

    return {
        "metadata": metadata,
        "schema": schema,
        "eda_summary": eda_result.get("eda_summary", ""),
        "insights": eda_result.get("insights", []),
        "column_details": col_details,
        "sample_csv": sample_csv,
        "modeling": modeling_result or {},
    }


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
