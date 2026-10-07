"""Tiny model bundle trained on synthetic data (no Kaggle data, no MLflow)."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from fraud.evaluate.cost import Thresholds
from fraud.features.history import build_offline_features
from fraud.features.pipeline import FeaturePipeline
from fraud.model import IsotonicCalibrator, ModelBundle


def make_bundle(df: pd.DataFrame, rounds: int = 30, seed: int = 0) -> ModelBundle:
    feats = build_offline_features(df)
    pipe = FeaturePipeline()
    X = pipe.to_numpy(pipe.fit_transform(feats, feats["isFraud"]))
    y = feats["isFraud"].to_numpy()
    params = {
        "objective": "binary",
        "learning_rate": 0.1,
        "num_leaves": 15,
        "min_child_samples": 10,
        "verbosity": -1,
        "seed": seed,
        "deterministic": True,
    }
    booster = lgb.train(params, lgb.Dataset(X, y, feature_name=pipe.feature_names), rounds)
    raw = booster.predict(X)
    cal = IsotonicCalibrator.fit(raw, y)
    p = cal(raw)
    return ModelBundle(
        booster=booster,
        pipeline=pipe,
        calibrator=cal,
        thresholds=Thresholds(float(np.quantile(p, 0.80)), float(np.quantile(p, 0.97))),
        metadata={"feature_version": pipe.version, "trained_at": "test", "test_metrics": {}},
    )
