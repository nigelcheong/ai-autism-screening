"""Repeated stratified cross-validation driver for probe-style experiments."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from src.evaluation.metrics import fold_metrics


def repeated_stratified_cv(
    X: np.ndarray,
    y: np.ndarray,
    model,
    n_splits: int = 5,
    n_repeats: int = 10,
    seed: int = 42,
    groups: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Fit and score ``model`` across repeated stratified folds.

    ``groups`` is accepted but currently ignored: the dataset has 11 known
    near-duplicate image pairs (5 crossing published-split boundaries), and
    pooling all images for cross-validation means a duplicate can land in
    both the train and test side of a fold, mildly inflating the result.
    Fixing this properly needs perceptual-hash grouping that hasn't been
    built yet. This parameter is the seam where that fix goes later, so
    grouped splitting can be switched on without changing any call site.

    Args:
        X: Feature matrix.
        y: Binary (0/1) labels aligned with ``X``.
        model: An unfitted scikit-learn-compatible estimator, cloned fresh
            for every fold.
        n_splits: Folds per repeat.
        n_repeats: Number of repeats, each with an independently derived
            shuffle seed.
        seed: Base seed; each repeat's fold shuffle is derived from it.
        groups: Currently ignored -- see above.

    Returns:
        DataFrame with one row per fold: ``repeat``, ``fold``, and every
        metric from :func:`src.evaluation.metrics.fold_metrics`.
    """
    del groups  # not yet used -- see docstring.

    rng = np.random.default_rng(seed)
    repeat_seeds = rng.integers(0, 2**31 - 1, size=n_repeats)

    rows = []
    for repeat, repeat_seed in enumerate(repeat_seeds):
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(repeat_seed))
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            fitted = clone(model)
            fitted.fit(X[train_idx], y[train_idx])
            y_score = fitted.predict_proba(X[test_idx])[:, 1]
            metrics = fold_metrics(y[test_idx], y_score)
            rows.append({"repeat": repeat, "fold": fold, **metrics})

    return pd.DataFrame(rows)
