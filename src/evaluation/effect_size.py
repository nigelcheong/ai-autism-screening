"""Rank-based effect size, shared with the `facial-images.ipynb` corpus audit.

Kept identical to the audit's inline `cliffs_delta` (`notebooks/facial-images.ipynb`)
so a value computed here means the same thing as one already reported there.
"""

from __future__ import annotations

import numpy as np


def benjamini_hochberg(p_values) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted p-values (q-values).

    Standard step-up procedure: sort ascending, form
    `q_i = p_i * m / rank_i`, then enforce monotonicity by taking a
    running minimum from the largest rank down to the smallest, and clip
    to `[0, 1]`. Returned in the original (unsorted) order.

    Args:
        p_values: Raw p-values from `m` simultaneous tests.

    Returns:
        `(m,)` array of BH-adjusted p-values, same order as `p_values`.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = np.clip(ranked, 0.0, 1.0)
    return adjusted


def cliffs_delta(x, y) -> float:
    """P(X > Y) - P(X < Y): a rank-based effect size, unaffected by scale or
    distributional shape, unlike a standardised mean difference.

    Args:
        x: First group's values.
        y: Second group's values.

    Returns:
        A value in `[-1, 1]`. `0` is no separation; `+1` means every value
        in `x` exceeds every value in `y`, `-1` the reverse.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    y_sorted = np.sort(y)
    n_y_less = np.searchsorted(y_sorted, x, side="left")
    n_y_greater = len(y) - np.searchsorted(y_sorted, x, side="right")
    return (n_y_less.sum() - n_y_greater.sum()) / (len(x) * len(y))
