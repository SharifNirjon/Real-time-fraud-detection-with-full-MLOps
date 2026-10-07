"""The deployable model bundle: features pipeline + LightGBM + calibrator + thresholds.

Pickled as one file, logged to MLflow, loaded by the API. Imports only
numpy / pandas / lightgbm so the serving image stays small.
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from fraud.evaluate.cost import DECISION_NAMES, Thresholds, decide
from fraud.explain import PathExplainer
from fraud.features.pipeline import FeaturePipeline


@dataclass
class IsotonicCalibrator:
    """Piecewise-linear isotonic map stored as plain arrays (fit with sklearn at train time).

    Isotonic regression is flat on long stretches, which ties many scores and
    throws away ranking (PR-AUC). A tie-breaker `eps * raw` (eps=1e-6) keeps
    the map strictly increasing, so ranking is exactly that of the raw model
    while probabilities move by at most 1e-6.
    """

    x: np.ndarray
    y: np.ndarray
    eps: float = 1e-6

    @classmethod
    def fit(cls, raw: np.ndarray, y: np.ndarray) -> IsotonicCalibrator:
        from sklearn.isotonic import IsotonicRegression

        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw, y)
        return cls(np.asarray(iso.X_thresholds_), np.asarray(iso.y_thresholds_))

    def __call__(self, raw: np.ndarray) -> np.ndarray:
        return np.interp(raw, self.x, self.y) * (1 - self.eps) + self.eps * np.asarray(raw)


@dataclass
class ModelBundle:
    booster: lgb.Booster
    pipeline: FeaturePipeline
    calibrator: IsotonicCalibrator
    thresholds: Thresholds
    metadata: dict[str, Any] = field(default_factory=dict)
    explainer: PathExplainer | None = None

    def __post_init__(self) -> None:
        if self.explainer is None:
            self.explainer = PathExplainer.from_booster(self.booster)

    @property
    def feature_names(self) -> list[str]:
        return self.pipeline.feature_names

    def matrix(self, feats: pd.DataFrame) -> np.ndarray:
        return self.pipeline.transform_numpy(feats)

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        return self.booster.predict(X, num_threads=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.calibrator(self.predict_raw(X))

    def decide(self, proba: np.ndarray) -> list[str]:
        return [DECISION_NAMES[int(d)] for d in decide(proba, self.thresholds)]

    def reason_codes(self, X: np.ndarray, k: int = 3) -> list[list[dict[str, Any]]]:
        """Top-k features pushing the score UP (decision-path attribution, log-odds)."""
        if getattr(self, "explainer", None) is None:  # bundles pickled before the explainer existed
            self.explainer = PathExplainer.from_booster(self.booster)
        assert self.explainer is not None
        out = []
        for row_x in X:
            contrib, _ = self.explainer.contributions(row_x)
            top = np.argsort(-contrib)[:k]
            out.append(
                [
                    {
                        "feature": self.feature_names[i],
                        "value": None if np.isnan(row_x[i]) else float(row_x[i]),
                        "contribution": round(float(contrib[i]), 4),
                    }
                    for i in top
                    if contrib[i] > 0
                ]
            )
        return out

    def latency_ms(self, X: np.ndarray, n: int = 500) -> dict[str, float]:
        """Single-row model latency (predict + calibrate + reason codes)."""
        times = []
        for i in range(min(n, len(X))):
            row = X[i : i + 1]
            t = time.perf_counter()
            self.predict_proba(row)
            self.reason_codes(row)
            times.append((time.perf_counter() - t) * 1000)
        a = np.array(times)
        return {f"p{q}": float(np.percentile(a, q)) for q in (50, 95, 99)}

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        return p

    @staticmethod
    def load(path: str | Path) -> ModelBundle:
        with open(path, "rb") as f:
            return pickle.load(f)
