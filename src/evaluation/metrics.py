"""Per-fold classification metrics."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import roc_auc_score


def fold_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Compute ROC-AUC, balanced accuracy, sensitivity and specificity.

    Args:
        y_true: Binary (0/1) ground-truth labels.
        y_score: Predicted P(y=1) scores.
        threshold: Decision threshold applied to ``y_score`` for the
            threshold-dependent metrics.

    Returns:
        Dict with keys ``roc_auc``, ``balanced_accuracy``, ``sensitivity``,
        ``specificity``.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = (y_score >= threshold).astype(int)

    positive = y_true == 1
    negative = y_true == 0
    sensitivity = float((y_pred[positive] == 1).mean()) if positive.any() else float("nan")
    specificity = float((y_pred[negative] == 0).mean()) if negative.any() else float("nan")

    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }
