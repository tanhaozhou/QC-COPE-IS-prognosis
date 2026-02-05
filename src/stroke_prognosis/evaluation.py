"""
Evaluation utilities: AUC, bootstrap CI, calibration, and probability export.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.utils import resample


@dataclass
class MetricResult:
    auc: float
    brier: float


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> MetricResult:
    return MetricResult(
        auc=float(roc_auc_score(y_true, y_prob)),
        brier=float(brier_score_loss(y_true, y_prob)),
    )


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Bootstrap CI for AUC."""
    rng = np.random.RandomState(seed)
    aucs: List[float] = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    if not aucs:
        return float("nan"), float("nan")
    lower = np.percentile(aucs, (1 - alpha) / 2 * 100)
    upper = np.percentile(aucs, (1 + alpha) / 2 * 100)
    return float(lower), float(upper)


def save_predictions_for_delong(
    y_true: np.ndarray,
    prob_dict: Dict[str, np.ndarray],
    out_path: str,
) -> str:
    """Save y_true and predicted probabilities (per model) for downstream DeLong tests."""
    df = pd.DataFrame({"True_Label": y_true})
    for name, probs in prob_dict.items():
        df[name] = probs
    out_path = str(out_path)
    df.to_excel(out_path, index=False)
    return out_path
