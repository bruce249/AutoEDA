"""
agents/modeling/agent.py — Modeling & Insight Agent orchestrator.

Pipeline
--------
1. Problem detection   → regression / classification / time_series / clustering
2. Feature engineering → encode, scale, create features
3. Model training      → multiple candidates per problem type
4. Evaluation          → metrics, best-model selection, feature importance
5. Insight synthesis   → narrative report, takeaways, recommendations

Usage
-----
>>> from agents.modeling import ModelingAgent
>>> result = ModelingAgent().run(ingestion_result, eda_result)
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from .problem_detector import detect_problem
from .feature_engineering import engineer_features
from .model_trainer import train_models
from .evaluator import evaluate_models
from .insight_synthesizer import synthesize_insights

logger = logging.getLogger(__name__)

# Type alias
ModelingResult = Dict[str, Any]

"""
Output contract
---------------
{
    "problem":              {problem_type, target, features, time_column, reason},
    "feature_engineering":  {X, y, feature_names, transformations, …},
    "training":             {models, predictions, X_train, X_test, …},
    "evaluation":           {model_scores, best_model, best_metrics, feature_importance, …},
    "insights":             {report_text, key_takeaways, recommendations, drivers},
    "predictions_preview":  [{actual, predicted}]  — first 200 rows of test set
    "model_comparison":     [{name, metrics}]       — sorted best-first
}
"""


class ModelingAgent:
    """Single entry-point for the Modeling & Insight Agent."""

    def __init__(self, *, verbose: bool = True):
        if verbose:
            logging.basicConfig(
                level=logging.INFO,
                format="[%(levelname)s] %(name)s — %(message)s",
            )

    def run(
        self,
        ingestion_result: Dict[str, Any],
        eda_result: Dict[str, Any],
    ) -> ModelingResult:
        """
        Execute the full modeling pipeline.

        Parameters
        ----------
        ingestion_result : dict — output of IngestionAgent.run()
        eda_result       : dict — output of EDAAgent.run()

        Returns
        -------
        ModelingResult
        """
        df: pd.DataFrame = ingestion_result["dataframe"]
        schema: Dict = ingestion_result["schema"]
        metadata: Dict = ingestion_result["metadata"]

        # ====== STEP 1: Problem Detection ====== #
        logger.info("MODELING STEP 1 — Detecting problem type …")
        problem = detect_problem(df, schema, metadata)
        logger.info("  → %s  target=%s", problem["problem_type"], problem.get("target"))

        # ====== STEP 2: Feature Engineering ====== #
        logger.info("MODELING STEP 2 — Feature engineering …")
        feat = engineer_features(df, problem, schema)
        logger.info("  → X shape %s, %d features", feat["X"].shape, len(feat["feature_names"]))

        # Guard: need enough data
        if feat["X"].shape[0] < 10 or feat["X"].shape[1] == 0:
            logger.warning("Insufficient data for modeling, returning early.")
            return self._empty_result(problem, feat)

        # ====== STEP 3: Model Training ====== #
        logger.info("MODELING STEP 3 — Training models …")
        training = train_models(
            feat["X"], feat["y"],
            problem["problem_type"],
        )
        logger.info("  → Trained %d models", len(training["models"]))

        # ====== STEP 4: Evaluation ====== #
        logger.info("MODELING STEP 4 — Evaluating models …")
        evaluation = evaluate_models(training, problem["problem_type"])
        logger.info("  → Best: %s", evaluation["best_model"])

        # ====== STEP 5: Insight Synthesis ====== #
        logger.info("MODELING STEP 5 — Synthesising insights …")
        insights = synthesize_insights(
            metadata, schema, eda_result, problem, evaluation, feat,
        )

        # ====== Build output ====== #
        predictions_preview = self._build_predictions_preview(training, problem)
        model_comparison = self._build_comparison(evaluation)

        result: ModelingResult = {
            "problem": problem,
            "feature_engineering": {
                "feature_names": feat["feature_names"],
                "transformations": feat["transformations"],
                "target_mapping": feat.get("target_mapping"),
                "n_features": len(feat["feature_names"]),
                "n_samples": feat["X"].shape[0],
            },
            "training": {
                "split_info": training["split_info"],
                "models_trained": list(training["models"].keys()),
            },
            "evaluation": evaluation,
            "insights": insights,
            "predictions_preview": predictions_preview,
            "model_comparison": model_comparison,
        }

        self._print_report(result)
        return result

    # ── Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _build_predictions_preview(training, problem) -> list:
        """First 200 actual vs predicted rows for the best model."""
        y_test = training.get("y_test")
        predictions = training.get("predictions", {})
        if y_test is None or not predictions:
            # Clustering — show cluster labels
            labels = training.get("cluster_labels")
            if labels is not None:
                return [{"index": int(i), "cluster": int(l)} for i, l in enumerate(labels[:200])]
            return []

        # Pick the first model's predictions (sorted by score later anyway)
        first_name = list(predictions.keys())[0]
        preds = predictions[first_name]
        preview = []
        for i in range(min(200, len(y_test))):
            row = {"actual": _safe(y_test.iloc[i]), "predicted": _safe(preds[i])}
            preview.append(row)
        return preview

    @staticmethod
    def _build_comparison(evaluation: Dict) -> list:
        """Sorted list of [{name, metrics}]."""
        scores = evaluation.get("model_scores", {})
        items = [{"name": n, "metrics": m} for n, m in scores.items()]
        # Sort by primary metric descending
        def _sort_key(item):
            m = item["metrics"]
            return m.get("R²", m.get("F1", m.get("Silhouette", 0)))
        items.sort(key=_sort_key, reverse=True)
        return items

    @staticmethod
    def _empty_result(problem, feat) -> ModelingResult:
        return {
            "problem": problem,
            "feature_engineering": {
                "feature_names": feat["feature_names"],
                "transformations": feat["transformations"],
                "target_mapping": feat.get("target_mapping"),
                "n_features": len(feat["feature_names"]),
                "n_samples": feat["X"].shape[0],
            },
            "training": {"split_info": {}, "models_trained": []},
            "evaluation": {
                "model_scores": {},
                "best_model": "N/A",
                "best_metrics": {},
                "feature_importance": {},
                "evaluation_summary": "Insufficient data for modeling.",
            },
            "insights": {
                "report_text": "Not enough data to train models.",
                "key_takeaways": ["Dataset too small or has no usable features."],
                "recommendations": [{"title": "Collect More Data", "detail": "Need at least 30 rows with numeric or categorical features.", "priority": "high"}],
                "drivers": [],
            },
            "predictions_preview": [],
            "model_comparison": [],
        }

    @staticmethod
    def _print_report(result: ModelingResult) -> None:
        print("\n" + "=" * 70)
        print("  MODELING & INSIGHT AGENT — REPORT")
        print("=" * 70)

        p = result["problem"]
        print(f"\n--- Problem ---")
        print(f"  Type:    {p['problem_type']}")
        print(f"  Target:  {p.get('target', 'N/A')}")
        print(f"  Reason:  {p['reason']}")

        fe = result["feature_engineering"]
        print(f"\n--- Feature Engineering ---")
        print(f"  Features: {fe['n_features']}  |  Samples: {fe['n_samples']}")
        for t in fe["transformations"]:
            print(f"  • {t['action']} — {t.get('column', t.get('columns', ''))}")

        ev = result["evaluation"]
        print(f"\n--- Model Comparison ---")
        for mc in result["model_comparison"]:
            metric_str = "  ".join(f"{k}={v}" for k, v in mc["metrics"].items())
            star = " ★" if mc["name"] == ev["best_model"] else ""
            print(f"  {mc['name']:30s} {metric_str}{star}")

        print(f"\n--- Feature Importance (top 10) ---")
        for feat, score in list(ev["feature_importance"].items())[:10]:
            bar = "█" * int(score * 30)
            print(f"  {feat:30s} {score:.3f}  {bar}")

        ins = result["insights"]
        print(f"\n--- Key Takeaways ---")
        for t in ins["key_takeaways"]:
            print(f"  • {t}")

        print(f"\n--- Recommendations ---")
        for r in ins["recommendations"]:
            print(f"  [{r['priority'].upper()}] {r['title']}: {r['detail']}")

        print("=" * 70 + "\n")


def _safe(val):
    """Make a value JSON-serialisable."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        v = float(val)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    if isinstance(val, np.ndarray):
        return val.tolist()
    return val
