"""Ranking, calibration and confusion metrics for imbalanced fraud data."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def recall_at_fpr(y: np.ndarray, s: np.ndarray, max_fpr: float = 0.01) -> float:
    """Highest TPR reachable while keeping FPR <= max_fpr."""
    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= max_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0


def precision_at_recall(y: np.ndarray, s: np.ndarray, min_recall: float = 0.80) -> float:
    """Highest precision reachable while keeping recall >= min_recall."""
    precision, recall, _ = precision_recall_curve(y, s)
    ok = recall >= min_recall
    return float(precision[ok].max()) if ok.any() else 0.0


def confusion(y: np.ndarray, flagged: np.ndarray) -> dict[str, int]:
    y = np.asarray(y).astype(bool)
    f = np.asarray(flagged).astype(bool)
    return {
        "tp": int((y & f).sum()),
        "fp": int((~y & f).sum()),
        "fn": int((y & ~f).sum()),
        "tn": int((~y & ~f).sum()),
    }


def ranking_metrics(y: np.ndarray, s: np.ndarray) -> dict[str, float]:
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    return {
        "pr_auc": float(average_precision_score(y, s)),
        "roc_auc": float(roc_auc_score(y, s)),
        "recall_at_1pct_fpr": recall_at_fpr(y, s, 0.01),
        "precision_at_80pct_recall": precision_at_recall(y, s, 0.80),
    }


def calibration_metrics(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict:
    """Brier score plus a quantile-binned reliability curve."""
    p = np.clip(np.asarray(p, dtype=float), 0, 1)
    y = np.asarray(y)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    mean_pred, frac_pos = [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            mean_pred.append(float(p[m].mean()))
            frac_pos.append(float(y[m].mean()))
    return {
        "brier": float(brier_score_loss(y, p)),
        "curve": {"mean_pred": mean_pred, "frac_pos": frac_pos},
    }
