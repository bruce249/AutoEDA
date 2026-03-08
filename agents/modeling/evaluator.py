"""
evaluator.py — Evaluate trained models, select the best, extract feature importance.

Metrics:
- Regression / TS:  MAE, RMSE, R², MAPE
- Classification:   Accuracy, Precision, Recall, F1 (macro)
- Clustering:       Silhouette score
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def evaluate_models(
    training_result: Dict[str, Any],
    problem_type: str,
) -> Dict[str, Any]:
    """
    Evaluate all trained models and pick the best.

    Returns
    -------
    dict with keys:
        model_scores  : dict[model_name → dict[metric → float]]
        best_model    : str   — name of top model
        best_metrics  : dict  — its metric dict
        feature_importance : dict[feature → float]  (sorted desc)
        evaluation_summary : str — one-paragraph human summary
    """
    models = training_result["models"]
    predictions = training_result["predictions"]
    y_test = training_result["y_test"]

    if problem_type == "clustering":
        return _evaluate_clustering(training_result)

    if problem_type in ("regression", "time_series"):
        return _evaluate_regression(models, predictions, y_test, training_result)
    else:
        return _evaluate_classification(models, predictions, y_test, training_result)


def _empty_result(reason: str) -> Dict[str, Any]:
    """Return a safe fallback when no models could be evaluated."""
    logger.warning(reason)
    return {
        "model_scores": {},
        "best_model": "N/A",
        "best_metrics": {},
        "feature_importance": {},
        "evaluation_summary": reason,
    }


# ── Regression / TS ──────────────────────────────────────────────────────

def _evaluate_regression(models, predictions, y_test, training_result):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    scores: Dict[str, Dict[str, float]] = {}
    for name, preds in predictions.items():
        y_true = np.array(y_test)
        y_pred = np.array(preds)
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))
        # MAPE — guard against zero
        mask = y_true != 0
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.sum() > 0 else None
        scores[name] = {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R²": round(r2, 4)}
        if mape is not None:
            scores[name]["MAPE"] = round(mape, 2)

    # Best by R²
    if not scores:
        return _empty_result("No regression models trained successfully.")
    best = max(scores, key=lambda n: scores[n]["R²"])
    importance = _extract_importance(models[best], training_result["X_train"])

    summary = (
        f"Best model: {best} (R²={scores[best]['R²']}, RMSE={scores[best]['RMSE']}). "
        f"Evaluated {len(scores)} candidates on {len(y_test)} test samples."
    )

    return {
        "model_scores": scores,
        "best_model": best,
        "best_metrics": scores[best],
        "feature_importance": importance,
        "evaluation_summary": summary,
    }


# ── Classification ───────────────────────────────────────────────────────

def _evaluate_classification(models, predictions, y_test, training_result):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    scores: Dict[str, Dict[str, float]] = {}
    for name, preds in predictions.items():
        avg = "binary" if len(set(y_test)) <= 2 else "macro"
        acc = float(accuracy_score(y_test, preds))
        prec = float(precision_score(y_test, preds, average=avg, zero_division=0))
        rec = float(recall_score(y_test, preds, average=avg, zero_division=0))
        f1 = float(f1_score(y_test, preds, average=avg, zero_division=0))
        scores[name] = {
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1, 4),
        }

    if not scores:
        return _empty_result("No classification models trained successfully.")
    best = max(scores, key=lambda n: scores[n]["F1"])
    importance = _extract_importance(models[best], training_result["X_train"])

    summary = (
        f"Best model: {best} (F1={scores[best]['F1']}, Accuracy={scores[best]['Accuracy']}). "
        f"Evaluated {len(scores)} candidates on {len(y_test)} test samples."
    )

    return {
        "model_scores": scores,
        "best_model": best,
        "best_metrics": scores[best],
        "feature_importance": importance,
        "evaluation_summary": summary,
    }


# ── Clustering ───────────────────────────────────────────────────────────

def _evaluate_clustering(training_result):
    split_info = training_result.get("split_info", {})
    sil_scores = split_info.get("silhouette_scores", {})
    best_k = split_info.get("best_k", 2)
    best_sil = sil_scores.get(best_k, 0)

    scores = {f"KMeans (k={best_k})": {"Silhouette": round(best_sil, 4)}}

    # Try to get feature importance from cluster centres distance
    models = training_result.get("models", {})
    importance = {}
    for name, model in models.items():
        if hasattr(model, "cluster_centers_"):
            centres = model.cluster_centers_
            spread = np.std(centres, axis=0)
            X_train = training_result["X_train"]
            feat_names = list(X_train.columns) if hasattr(X_train, "columns") else [f"f{i}" for i in range(spread.shape[0])]
            for i, f in enumerate(feat_names):
                if i < len(spread):
                    importance[f] = round(float(spread[i]), 4)
            importance = dict(sorted(importance.items(), key=lambda kv: kv[1], reverse=True))
            break

    summary = (
        f"Best clustering: k={best_k} (Silhouette={round(best_sil, 4)}). "
        f"Tested k=2..{max(sil_scores.keys()) if sil_scores else 2}."
    )

    return {
        "model_scores": scores,
        "best_model": f"KMeans (k={best_k})",
        "best_metrics": scores[f"KMeans (k={best_k})"],
        "feature_importance": importance,
        "evaluation_summary": summary,
    }


# ── Feature Importance ───────────────────────────────────────────────────

def _extract_importance(model, X_train: pd.DataFrame) -> Dict[str, float]:
    """Extract feature importance from any sklearn-compatible model."""
    feats = list(X_train.columns)
    importance: Dict[str, float] = {}

    # Tree-based models
    if hasattr(model, "feature_importances_"):
        raw = model.feature_importances_
        for i, f in enumerate(feats):
            if i < len(raw):
                importance[f] = round(float(raw[i]), 4)

    # Linear models — absolute coefficient
    elif hasattr(model, "coef_"):
        coefs = np.array(model.coef_).flatten()
        for i, f in enumerate(feats):
            if i < len(coefs):
                importance[f] = round(float(abs(coefs[i])), 4)

    # Normalise to 0-1
    if importance:
        max_v = max(importance.values()) or 1
        importance = {k: round(v / max_v, 4) for k, v in importance.items()}

    # Sort descending
    importance = dict(sorted(importance.items(), key=lambda kv: kv[1], reverse=True))
    return importance
