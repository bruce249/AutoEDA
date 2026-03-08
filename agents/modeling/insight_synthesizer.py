"""
insight_synthesizer.py — Combine EDA + modeling results into a coherent
narrative report with actionable recommendations.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def synthesize_insights(
    metadata: Dict[str, Any],
    schema: Dict[str, Any],
    eda_result: Dict[str, Any],
    problem: Dict[str, Any],
    evaluation: Dict[str, Any],
    feature_eng: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Produce a rich insight report.

    Returns
    -------
    dict with keys:
        report_text        : str   — multi-paragraph narrative
        key_takeaways      : list[str]
        recommendations    : list[dict]  — {title, detail, priority}
        drivers            : list[dict]  — top features with context
    """
    problem_type = problem.get("problem_type", "unknown")
    target = problem.get("target")
    best_model = evaluation.get("best_model", "N/A")
    best_metrics = evaluation.get("best_metrics", {})
    importance = evaluation.get("feature_importance", {})
    transformations = feature_eng.get("transformations", [])

    # ── 1. Build narrative paragraphs ────────────────────────────────
    paragraphs: List[str] = []

    # Dataset overview
    rows = metadata.get("row_count", "?")
    cols = metadata.get("column_count", "?")
    n_num = len(schema.get("numeric_columns", []))
    n_cat = len(schema.get("categorical_columns", []))
    missing = metadata.get("total_missing_cells", 0)
    paragraphs.append(
        f"The dataset contains {rows:,} rows and {cols} columns "
        f"({n_num} numeric, {n_cat} categorical). "
        f"{'There are ' + str(missing) + ' missing cells that were imputed during preprocessing.' if missing else 'No missing values were found.'}"
    )

    # Problem & model
    if problem_type in ("regression", "time_series"):
        metric_str = ", ".join(f"{k}={v}" for k, v in best_metrics.items())
        paragraphs.append(
            f"Problem type: **{problem_type}** — predicting '{target}'. "
            f"The best-performing model is **{best_model}** ({metric_str}). "
            + _regression_context(best_metrics)
        )
    elif problem_type == "classification":
        metric_str = ", ".join(f"{k}={v}" for k, v in best_metrics.items())
        paragraphs.append(
            f"Problem type: **classification** — predicting '{target}'. "
            f"The best model is **{best_model}** ({metric_str})."
        )
    else:
        paragraphs.append(
            f"Problem type: **clustering** (unsupervised). "
            f"Best configuration: {best_model} "
            f"(Silhouette={best_metrics.get('Silhouette', '?')})."
        )

    # Feature importance
    top_features = list(importance.items())[:5]
    if top_features:
        feat_lines = ", ".join(f"'{f}' ({round(s*100)}%)" for f, s in top_features)
        paragraphs.append(
            f"Top drivers: {feat_lines}. "
            "These features have the strongest influence on model predictions."
        )

    # EDA highlights
    eda_summary = eda_result.get("eda_summary", "")
    if eda_summary:
        paragraphs.append(f"EDA highlights: {eda_summary}")

    # Correlations
    corr = eda_result.get("correlations", {})
    sp = corr.get("strong_positive", [])
    sn = corr.get("strong_negative", [])
    if sp or sn:
        corr_lines = []
        for p in sp[:3]:
            corr_lines.append(f"{p['columns'][0]} ↔ {p['columns'][1]} (r={p['correlation']})")
        for p in sn[:3]:
            corr_lines.append(f"{p['columns'][0]} ↔ {p['columns'][1]} (r={p['correlation']})")
        paragraphs.append(f"Notable correlations: {'; '.join(corr_lines)}.")

    report_text = "\n\n".join(paragraphs)

    # ── 2. Key takeaways ────────────────────────────────────────────
    takeaways = _build_takeaways(problem, evaluation, eda_result, importance)

    # ── 3. Recommendations ──────────────────────────────────────────
    recommendations = _build_recommendations(problem, evaluation, eda_result, metadata)

    # ── 4. Drivers ──────────────────────────────────────────────────
    drivers = []
    for feat, score in top_features:
        drivers.append({
            "feature": feat,
            "importance": score,
            "description": f"Contributes {round(score*100)}% relative importance to the model.",
        })

    return {
        "report_text": report_text,
        "key_takeaways": takeaways,
        "recommendations": recommendations,
        "drivers": drivers,
    }


# ── Helpers ──────────────────────────────────────────────────────────────

def _regression_context(metrics: Dict[str, Any]) -> str:
    r2 = metrics.get("R²", 0)
    if r2 >= 0.9:
        return "The model explains the vast majority of variance — excellent fit."
    elif r2 >= 0.7:
        return "The model has a good fit, capturing most of the variance."
    elif r2 >= 0.4:
        return "The model has a moderate fit; additional features or engineering may help."
    else:
        return "The model explains limited variance; consider more data or different approaches."


def _build_takeaways(problem, evaluation, eda, importance) -> List[str]:
    t = []
    pt = problem.get("problem_type")
    best = evaluation.get("best_model", "")
    metrics = evaluation.get("best_metrics", {})
    target = problem.get("target")

    if pt in ("regression", "time_series"):
        r2 = metrics.get("R²", 0)
        t.append(f"{best} achieves R²={r2} on the test set for predicting '{target}'.")
    elif pt == "classification":
        f1 = metrics.get("F1", 0)
        acc = metrics.get("Accuracy", 0)
        t.append(f"{best} achieves F1={f1}, Accuracy={acc} for classifying '{target}'.")
    else:
        sil = metrics.get("Silhouette", 0)
        t.append(f"Optimal clustering found at {best} (Silhouette={sil}).")

    top_feats = list(importance.keys())[:3]
    if top_feats:
        t.append(f"The strongest predictors are: {', '.join(top_feats)}.")

    # From EDA
    insights = eda.get("insights", [])
    for ins in insights[:2]:
        t.append(ins.get("detail", ""))

    segs = eda.get("segments", [])
    if segs:
        t.append(f"Found {len(segs)} meaningful data segments that affect model performance.")

    return [x for x in t if x]


def _build_recommendations(problem, evaluation, eda, metadata) -> List[Dict[str, str]]:
    recs: List[Dict[str, str]] = []
    pt = problem.get("problem_type")
    metrics = evaluation.get("best_metrics", {})
    missing = metadata.get("total_missing_cells", 0)

    # Data quality
    if missing > 0:
        recs.append({
            "title": "Improve Data Quality",
            "detail": f"{missing} missing cells were imputed. Investigate root causes to improve model reliability.",
            "priority": "high",
        })

    # Model performance
    if pt in ("regression", "time_series"):
        r2 = metrics.get("R²", 0)
        if r2 < 0.7:
            recs.append({
                "title": "Engineer More Features",
                "detail": f"Current R²={r2}. Consider adding domain-specific features, polynomial terms, or external data.",
                "priority": "high",
            })
        else:
            recs.append({
                "title": "Hypertune Best Model",
                "detail": "Good baseline established. Run grid/random search to squeeze extra performance.",
                "priority": "medium",
            })
    elif pt == "classification":
        f1 = metrics.get("F1", 0)
        if f1 < 0.7:
            recs.append({
                "title": "Address Class Imbalance",
                "detail": f"F1={f1} is moderate. Try SMOTE, class weights, or collect more samples of minority classes.",
                "priority": "high",
            })

    # General
    recs.append({
        "title": "Cross-Validate",
        "detail": "Current evaluation uses a single train/test split. Use k-fold cross-validation for more robust estimates.",
        "priority": "medium",
    })

    recs.append({
        "title": "Monitor Data Drift",
        "detail": "Deploy model monitoring to detect when input distribution shifts from training data.",
        "priority": "low",
    })

    return recs
