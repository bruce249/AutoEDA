"""
model_trainer.py — Train multiple candidate models and return results.

Supports:
- Regression:      LinearRegression, Ridge, RandomForestRegressor, GradientBoostingRegressor
- Classification:  LogisticRegression, RandomForestClassifier, GradientBoostingClassifier
- Time-series:     treats as regression with chronological split
- Clustering:      KMeans (2–8 clusters), picks best silhouette
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Suppress convergence warnings for quick fits
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Lazy imports (keeps startup fast) ────────────────────────────────────

def _sklearn():
    from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
    from sklearn.ensemble import (
        RandomForestRegressor, RandomForestClassifier,
        GradientBoostingRegressor, GradientBoostingClassifier,
    )
    from sklearn.cluster import KMeans
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import silhouette_score
    return {
        "LinearRegression": LinearRegression,
        "Ridge": Ridge,
        "LogisticRegression": LogisticRegression,
        "RandomForestRegressor": RandomForestRegressor,
        "RandomForestClassifier": RandomForestClassifier,
        "GradientBoostingRegressor": GradientBoostingRegressor,
        "GradientBoostingClassifier": GradientBoostingClassifier,
        "KMeans": KMeans,
        "train_test_split": train_test_split,
        "silhouette_score": silhouette_score,
    }


# ── Public API ───────────────────────────────────────────────────────────

def train_models(
    X: pd.DataFrame,
    y: Optional[pd.Series],
    problem_type: str,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train candidate models and return everything needed for evaluation.

    Returns
    -------
    dict with keys:
        models          : dict[str, fitted_model]
        predictions     : dict[str, np.ndarray]  — predictions on test set
        X_train, X_test : pd.DataFrame
        y_train, y_test : pd.Series | None
        split_info      : dict
        cluster_labels  : np.ndarray | None  (clustering only)
    """
    sk = _sklearn()

    if problem_type in ("regression", "time_series"):
        return _train_regression(sk, X, y, problem_type, test_size, random_state)
    elif problem_type == "classification":
        return _train_classification(sk, X, y, test_size, random_state)
    else:
        return _train_clustering(sk, X)


# ── Regression / Time-series ─────────────────────────────────────────────

def _train_regression(sk, X, y, problem_type, test_size, random_state):
    # Chronological split for time-series, random otherwise
    if problem_type == "time_series":
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    else:
        X_train, X_test, y_train, y_test = sk["train_test_split"](
            X, y, test_size=test_size, random_state=random_state,
        )

    candidates = {
        "Linear Regression": sk["LinearRegression"](),
        "Ridge Regression": sk["Ridge"](alpha=1.0),
        "Random Forest": sk["RandomForestRegressor"](
            n_estimators=100, max_depth=10, random_state=random_state, n_jobs=-1,
        ),
        "Gradient Boosting": sk["GradientBoostingRegressor"](
            n_estimators=150, max_depth=5, learning_rate=0.1, random_state=random_state,
        ),
    }

    models = {}
    predictions = {}
    for name, model in candidates.items():
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            models[name] = model
            predictions[name] = preds
            logger.info("Trained %s", name)
        except Exception as exc:
            logger.warning("Failed to train %s: %s", name, exc)

    return {
        "models": models,
        "predictions": predictions,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "split_info": {
            "method": "chronological" if problem_type == "time_series" else "random",
            "test_size": test_size,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
        },
        "cluster_labels": None,
    }


# ── Classification ───────────────────────────────────────────────────────

def _train_classification(sk, X, y, test_size, random_state):
    y_non_null = y.dropna()
    class_counts = y_non_null.value_counts()
    min_class_size = int(class_counts.min()) if not class_counts.empty else 0
    use_stratify = min_class_size >= 2 and y_non_null.nunique() > 1

    if use_stratify:
        X_train, X_test, y_train, y_test = sk["train_test_split"](
            X, y, test_size=test_size, random_state=random_state, stratify=y,
        )
        split_method = "stratified_random"
    else:
        logger.warning(
            "Classification target is too sparse for stratified split (classes=%d, min_class_size=%d); falling back to random split.",
            int(y_non_null.nunique()),
            min_class_size,
        )
        X_train, X_test, y_train, y_test = sk["train_test_split"](
            X, y, test_size=test_size, random_state=random_state,
        )
        split_method = "random"

    n_classes = int(y.nunique())
    candidates = {
        "Logistic Regression": sk["LogisticRegression"](
            max_iter=500, random_state=random_state,
        ),
        "Random Forest": sk["RandomForestClassifier"](
            n_estimators=100, max_depth=10, random_state=random_state, n_jobs=-1,
        ),
        "Gradient Boosting": sk["GradientBoostingClassifier"](
            n_estimators=150, max_depth=5, learning_rate=0.1, random_state=random_state,
        ),
    }

    models = {}
    predictions = {}
    for name, model in candidates.items():
        try:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            models[name] = model
            predictions[name] = preds
            logger.info("Trained %s", name)
        except Exception as exc:
            logger.warning("Failed to train %s: %s", name, exc)

    return {
        "models": models,
        "predictions": predictions,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "split_info": {
            "method": split_method,
            "test_size": test_size,
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "n_classes": int(y_non_null.nunique()),
            "min_class_size": min_class_size,
        },
        "cluster_labels": None,
    }


# ── Clustering ───────────────────────────────────────────────────────────

def _train_clustering(sk, X):
    best_k = 2
    best_score = -1
    best_labels = None
    scores = {}

    for k in range(2, min(9, len(X))):
        try:
            km = sk["KMeans"](n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(X)
            score = sk["silhouette_score"](X, labels)
            scores[k] = float(score)
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels
                best_model = km
            logger.info("KMeans k=%d silhouette=%.3f", k, score)
        except Exception:
            pass

    models = {f"KMeans (k={best_k})": best_model}
    return {
        "models": models,
        "predictions": {},
        "X_train": X,
        "X_test": pd.DataFrame(),
        "y_train": None,
        "y_test": None,
        "split_info": {
            "method": "full_dataset",
            "best_k": best_k,
            "silhouette_scores": scores,
        },
        "cluster_labels": best_labels,
    }
