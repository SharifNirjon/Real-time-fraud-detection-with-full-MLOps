"""Scoring service: online features from Redis -> model -> decision -> state update -> log."""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from fraud.features.definitions import DISTINCT_COLS, add_base_features, derive
from fraud.features.online import OnlineFeatureStore
from fraud.features.pipeline import raw_input_columns
from fraud.model import ModelBundle
from fraud.serve import metrics as m
from fraud.serve.prediction_log import ParquetLog

log = logging.getLogger("fraud.serve")

# fields the online store needs to update its state (besides the derived uid)
STATE_COLS = ["TransactionID", "TransactionDT", "TransactionAmt", "card1", *DISTINCT_COLS.values()]


@dataclass
class LoadedModel:
    bundle: ModelBundle
    version: str


class Scorer:
    def __init__(self, model: LoadedModel, store: OnlineFeatureStore, pred_log: ParquetLog | None):
        self._model = model
        self.store = store
        self.pred_log = pred_log
        self._lock = threading.Lock()  # serialises read->score->write on the feature state
        numeric, categorical = raw_input_columns(model.bundle.pipeline.config)
        self.numeric = numeric
        self.categorical = categorical

    @property
    def model(self) -> LoadedModel:
        return self._model

    def swap_model(self, model: LoadedModel) -> None:
        self._model = model  # atomic reference swap; in-flight requests keep the old one

    def _frame(self, txns: list[dict[str, Any]]) -> pd.DataFrame:
        """Typed frame built from two 2-D blocks (per-column construction is ~50x slower)."""
        num = np.array(
            [[np.nan if t.get(c) is None else t[c] for c in self.numeric] for t in txns],
            dtype="float64",
        )
        cat = np.array([[t.get(c) for c in self.categorical] for t in txns], dtype=object)
        ids = pd.DataFrame(
            {
                "TransactionID": np.array([t["TransactionID"] for t in txns], dtype="int64"),
                "TransactionDT": np.array([t["TransactionDT"] for t in txns], dtype="int64"),
            }
        )
        return pd.concat(
            [
                ids,
                pd.DataFrame(num, columns=self.numeric),
                pd.DataFrame(cat, columns=self.categorical, dtype=object),
            ],
            axis=1,
        )

    def score(self, txns: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Score transactions in time order; each one sees the state left by the previous."""
        model = self._model
        txns = sorted(txns, key=lambda t: (t["TransactionDT"], t["TransactionID"]))
        results = []
        log_records = []
        for txn in txns:
            t0 = time.perf_counter()
            base = add_base_features(self._frame([txn]))
            row = {c: txn.get(c) for c in STATE_COLS} | {"uid": base["uid"].iat[0]}
            with self._lock:
                agg = pd.DataFrame([self.store.read(row)], index=base.index)
                feats = pd.concat([base, derive(base, agg)], axis=1)
                X = model.bundle.matrix(feats)
                proba = float(model.bundle.predict_proba(X)[0])
                self.store.write([row])
            decision = model.bundle.decide(np.array([proba]))[0]
            reasons = model.bundle.reason_codes(X)[0]
            latency_ms = (time.perf_counter() - t0) * 1000
            m.LATENCY.observe(latency_ms / 1000)
            m.SCORE.observe(proba)
            m.DECISIONS.labels(decision=decision).inc()
            m.TRANSACTIONS.inc()
            results.append(
                {
                    "transaction_id": int(txn["TransactionID"]),
                    "fraud_probability": proba,
                    "decision": decision,
                    "reason_codes": reasons,
                    "model_version": model.version,
                    "latency_ms": round(latency_ms, 3),
                }
            )
            rec: dict[str, Any] = {
                "TransactionID": int(txn["TransactionID"]),
                "TransactionDT": int(txn["TransactionDT"]),
                "logged_at": pd.Timestamp.now(tz="UTC"),
                "model_version": model.version,
                "score": proba,
                "decision": decision,
                "latency_ms": latency_ms,
                "raw": json.dumps(txn, default=float),
            }
            rec |= dict(zip(model.bundle.feature_names, X[0].tolist(), strict=True))
            log_records.append(rec)
            log.info(
                "scored",
                extra={
                    "transaction_id": rec["TransactionID"],
                    "score": round(proba, 5),
                    "decision": decision,
                    "model_version": model.version,
                    "latency_ms": round(latency_ms, 2),
                },
            )
        if self.pred_log is not None:
            self.pred_log.append(log_records)
        return results
