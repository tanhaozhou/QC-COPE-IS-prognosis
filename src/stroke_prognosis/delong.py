"""
DeLong test implementation for comparing correlated ROC AUCs.

This module is adapted into a reusable form for script-based execution.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Midrank computation for DeLong."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(predictions_sorted_transposed: np.ndarray, label_1_count: int):
    """Fast DeLong for multiple AUCs."""
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]

    k = predictions_sorted_transposed.shape[0]
    tx = np.empty([k, m])
    ty = np.empty([k, n])
    tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = _compute_midrank(positive_examples[r, :])
        ty[r, :] = _compute_midrank(negative_examples[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])

    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2.0) / n

    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m

    sx = np.cov(v01)
    sy = np.cov(v10)
    delong_cov = sx / m + sy / n
    return aucs, delong_cov


def delong_roc_test(y_true: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray):
    """
    Return (auc_a, auc_b, z, p_value) comparing two correlated AUCs.
    """
    y_true = np.asarray(y_true).astype(int)
    order = np.argsort(-y_true)  # positives first
    label_1_count = int(y_true.sum())

    preds = np.vstack([prob_a, prob_b])[:, order]
    aucs, cov = _fast_delong(preds, label_1_count)

    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = diff / np.sqrt(var) if var > 0 else np.nan
    p = 2 * (1 - norm.cdf(abs(z))) if np.isfinite(z) else np.nan
    return float(aucs[0]), float(aucs[1]), float(z), float(p)
