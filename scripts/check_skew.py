"""Training/serving skew check on REAL traffic.

Compares the feature vectors and scores the API logged (computed online from
Redis state) with the same transactions' features computed offline by the
training pipeline. Only meaningful for a non-drifted replay.

Usage: python scripts/check_skew.py [--model-path artifacts/model_bundle.pkl]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fraud.config import path  # noqa: E402
from fraud.model import ModelBundle  # noqa: E402
from fraud.serve.prediction_log import read_log  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=str(path("artifacts_dir") / "model_bundle.pkl"))
    ap.add_argument("--out", default=str(path("reports_dir") / "skew_check.json"))
    args = ap.parse_args()
    bundle = ModelBundle.load(args.model_path)
    log = read_log(path("prediction_log_dir"))
    log = log.sort_values("logged_at").drop_duplicates("TransactionID", keep="last")
    feats = pd.read_parquet(path("processed_dir") / "features.parquet").set_index("TransactionID")
    log = log[log["TransactionID"].isin(feats.index)]
    off = feats.loc[log["TransactionID"]].reset_index()
    X_off = pd.DataFrame(bundle.matrix(off), columns=bundle.feature_names)
    X_on = log[bundle.feature_names].reset_index(drop=True).astype("float32")
    close = np.isclose(X_off.to_numpy(), X_on.to_numpy(), rtol=1e-4, atol=1e-4, equal_nan=True)
    per_feature = (~close).sum(axis=0)
    s_off = bundle.predict_proba(X_off.to_numpy(np.float32))
    s_diff = np.abs(s_off - log["score"].to_numpy())
    res = {
        "rows": len(log),
        "feature_cells_compared": int(close.size),
        "mismatched_cells": int((~close).sum()),
        "mismatch_rate": float((~close).mean()),
        "features_with_mismatch": {bundle.feature_names[i]: int(c) for i, c in enumerate(per_feature) if c},
        "max_abs_score_diff": float(s_diff.max()),
        "p99_abs_score_diff": float(np.percentile(s_diff, 99)),
        "decision_agreement": float(np.mean(np.array(bundle.decide(s_off)) == log["decision"].to_numpy())),
    }
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
