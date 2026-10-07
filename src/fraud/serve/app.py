"""FastAPI scoring service.

Env:
  REDIS_URL           redis://redis:6379/0
  MLFLOW_TRACKING_URI http://mlflow:5000 (model loaded from alias MODEL_ALIAS, default champion)
  MODEL_PATH          optional local bundle (fallback when the registry is unreachable; used in CI)
  ADMIN_API_KEY       required for POST /admin/reload
  PREDICTION_LOG_DIR / LABEL_LOG_DIR / DRIFT_SUMMARY_PATH
"""

from __future__ import annotations

import hmac
import logging
import os
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from fraud.config import path as cfg_path
from fraud.features.online import OnlineFeatureStore
from fraud.logging_utils import setup_json_logging
from fraud.model import ModelBundle
from fraud.serve import metrics as m
from fraud.serve.prediction_log import ParquetLog
from fraud.serve.schemas import (
    BatchRequest,
    BatchResponse,
    LabelBatch,
    ScoreResponse,
    Transaction,
)
from fraud.serve.service import LoadedModel, Scorer

log = logging.getLogger("fraud.serve")

ModelLoader = Callable[[], LoadedModel]


def default_model_loader() -> LoadedModel:
    """Champion from the MLflow registry, falling back to MODEL_PATH."""
    alias = os.environ.get("MODEL_ALIAS", "champion")
    err: Exception | None = None
    for attempt in range(int(os.environ.get("MODEL_LOAD_RETRIES", "5"))):
        try:
            from fraud.train.registry import load_bundle

            bundle, version = load_bundle(alias)
            return LoadedModel(bundle, version)
        except Exception as e:  # registry not up yet / no champion
            err = e
            log.warning("registry load failed (attempt %d): %s", attempt + 1, e)
            time.sleep(min(2**attempt, 10))
    local = os.environ.get("MODEL_PATH")
    if local and Path(local).exists():
        return LoadedModel(ModelBundle.load(local), os.environ.get("MODEL_VERSION", "local"))
    raise RuntimeError(f"no model available: {err}")


def default_redis() -> Any:
    import redis

    return redis.Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))


def create_app(
    model_loader: ModelLoader = default_model_loader,
    redis_factory: Callable[[], Any] = default_redis,
    prediction_log_dir: Path | None = None,
    label_log_dir: Path | None = None,
) -> FastAPI:
    pred_dir = prediction_log_dir or Path(
        os.environ.get("PREDICTION_LOG_DIR", cfg_path("prediction_log_dir"))
    )
    label_dir = label_log_dir or Path(os.environ.get("LABEL_LOG_DIR", cfg_path("label_log_dir")))
    drift_summary = Path(
        os.environ.get("DRIFT_SUMMARY_PATH", cfg_path("reports_dir") / "drift" / "latest.json")
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.redis = redis_factory()
        app.state.pred_log = ParquetLog(pred_dir, "pred")
        app.state.label_log = ParquetLog(label_dir, "labels")
        loaded = model_loader()
        app.state.scorer = Scorer(loaded, OnlineFeatureStore(app.state.redis), app.state.pred_log)
        m.set_model_version(loaded.version)
        m.register_drift_collector(drift_summary)
        log.info("model loaded", extra={"model_version": loaded.version})
        yield
        app.state.pred_log.close()
        app.state.label_log.close()

    app = FastAPI(title="Fraud scoring API", version="1.0.0", lifespan=lifespan)
    app.mount("/metrics", make_asgi_app())

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        m.REQUESTS.labels(endpoint=request.url.path, status="422").inc()
        m.ERRORS.labels(type="validation").inc()
        errors = [
            {
                "field": ".".join(str(p) for p in e["loc"] if p != "body"),
                "message": e["msg"],
                "type": e["type"],
            }
            for e in exc.errors()
        ]
        return JSONResponse(
            status_code=422, content={"detail": "invalid transaction", "errors": errors}
        )

    def scorer(request: Request) -> Scorer:
        return request.app.state.scorer

    def _score(s: Scorer, txns: list[dict[str, Any]], endpoint: str) -> list[dict[str, Any]]:
        try:
            out = s.score(txns)
        except Exception as e:
            m.ERRORS.labels(type=type(e).__name__).inc()
            m.REQUESTS.labels(endpoint=endpoint, status="500").inc()
            log.exception("scoring failed")
            raise HTTPException(status_code=500, detail="scoring failed") from e
        m.REQUESTS.labels(endpoint=endpoint, status="200").inc()
        return out

    @app.post("/score", response_model=ScoreResponse)
    def score(txn: Transaction, s: Scorer = Depends(scorer)) -> dict[str, Any]:  # type: ignore[valid-type]
        return _score(s, [txn.model_dump()], "/score")[0]

    @app.post("/score/batch", response_model=BatchResponse)
    def score_batch(req: BatchRequest, s: Scorer = Depends(scorer)) -> dict[str, Any]:
        t0 = time.perf_counter()
        results = _score(s, [t.model_dump() for t in req.transactions], "/score/batch")
        return {"results": results, "latency_ms": round((time.perf_counter() - t0) * 1000, 3)}

    @app.post("/labels")
    def labels(batch: LabelBatch, request: Request) -> dict[str, int]:
        now = pd.Timestamp.now(tz="UTC")
        recs = [lb.model_dump() | {"received_at": now} for lb in batch.labels]
        request.app.state.label_log.append(recs)
        for lb in batch.labels:
            m.LABELS.labels(label=str(lb.isFraud)).inc()
        return {"received": len(recs)}

    @app.get("/health")
    def health(request: Request) -> JSONResponse:
        try:
            redis_ok = bool(request.app.state.redis.ping())
        except Exception:
            redis_ok = False
        s: Scorer = request.app.state.scorer
        body = {
            "status": "ok" if redis_ok else "degraded",
            "redis": redis_ok,
            "model_version": s.model.version,
        }
        return JSONResponse(status_code=200 if redis_ok else 503, content=body)

    @app.get("/metadata")
    def metadata(s: Scorer = Depends(scorer)) -> dict[str, Any]:
        b = s.model.bundle
        md = b.metadata
        return {
            "model_version": s.model.version,
            "feature_version": md.get("feature_version"),
            "trained_at": md.get("trained_at"),
            "n_features": len(b.feature_names),
            "thresholds": {"review": b.thresholds.review, "block": b.thresholds.block},
            "test_metrics": md.get("test_metrics"),
            "split_dates": (md.get("split") or {}).get("dates"),
            "input_fields": list(Transaction.model_fields),
            "required_fields": [k for k, f in Transaction.model_fields.items() if f.is_required()],
        }

    @app.post("/admin/reload")
    def reload(request: Request, x_api_key: str | None = Header(default=None)) -> dict[str, str]:
        expected = os.environ.get("ADMIN_API_KEY")
        if not expected or not x_api_key or not hmac.compare_digest(x_api_key, expected):
            m.RELOADS.labels(status="unauthorized").inc()
            raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")
        s: Scorer = request.app.state.scorer
        old = s.model.version
        try:
            loaded = model_loader()
        except Exception as e:
            m.RELOADS.labels(status="error").inc()
            raise HTTPException(status_code=503, detail=f"reload failed: {e}") from e
        s.swap_model(loaded)
        m.set_model_version(loaded.version)
        m.RELOADS.labels(status="ok").inc()
        log.info("model reloaded", extra={"old_version": old, "new_version": loaded.version})
        return {"previous_version": old, "model_version": loaded.version}

    return app


def build() -> FastAPI:
    """Entry point for uvicorn --factory."""
    setup_json_logging(os.environ.get("LOG_LEVEL", "INFO"))
    return create_app()
