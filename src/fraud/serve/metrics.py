"""Prometheus metrics exposed on /metrics."""

from __future__ import annotations

import json
from pathlib import Path

from prometheus_client import Counter, Gauge, Histogram
from prometheus_client.core import GaugeMetricFamily
from prometheus_client.registry import REGISTRY, Collector

REQUESTS = Counter("fraud_requests_total", "Scoring requests", ["endpoint", "status"])
TRANSACTIONS = Counter("fraud_transactions_scored_total", "Transactions scored")
LATENCY = Histogram(
    "fraud_request_latency_seconds",
    "End-to-end latency per scored transaction",
    buckets=(0.002, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0),
)
SCORE = Histogram(
    "fraud_score",
    "Calibrated fraud probability",
    buckets=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0),
)
DECISIONS = Counter("fraud_decisions_total", "Decisions taken", ["decision"])
ERRORS = Counter("fraud_errors_total", "Errors while scoring", ["type"])
MODEL_INFO = Gauge("fraud_model_info", "Currently loaded model (value is always 1)", ["version"])
MODEL_VERSION = Gauge("fraud_model_version", "Currently loaded registry version")
RELOADS = Counter("fraud_model_reloads_total", "Model reloads", ["status"])
LABELS = Counter("fraud_labels_received_total", "Delayed labels received", ["label"])


def set_model_version(version: str) -> None:
    MODEL_INFO.clear()
    MODEL_INFO.labels(version=version).set(1)
    try:
        MODEL_VERSION.set(float(version))
    except ValueError:
        MODEL_VERSION.set(-1)


class DriftCollector(Collector):
    """Expose the latest monitoring summary (written by the drift job) to Prometheus."""

    def __init__(self, summary_path: Path):
        self.summary_path = summary_path

    def collect(self):  # type: ignore[override]
        g = GaugeMetricFamily(
            "fraud_drift", "Latest drift / live-performance monitor values", labels=["metric"]
        )
        try:
            s = json.loads(self.summary_path.read_text())
        except (OSError, ValueError):
            yield g
            return
        for key in [
            "share_drifted_columns",
            "prediction_drift_score",
            "live_pr_auc",
            "live_recall",
            "live_precision",
            "labelled_rows",
            "drift_detected",
        ]:
            v = s.get(key)
            if isinstance(v, bool):
                v = float(v)
            if isinstance(v, int | float):
                g.add_metric([key], float(v))
        yield g


_registered: set[str] = set()


def register_drift_collector(summary_path: Path) -> None:
    if str(summary_path) not in _registered:
        REGISTRY.register(DriftCollector(summary_path))
        _registered.add(str(summary_path))
