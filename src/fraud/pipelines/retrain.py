"""Champion / challenger retraining logic (plain functions; Prefect wiring is in flows.py).

1. Rebuild the training set: all labelled history before the live stream +
   transactions the API logged that now have (delayed) labels.
2. Fresh time-based holdout = the most recent `holdout_frac` of the newly
   labelled live rows. Nothing at or after the holdout start is used for training.
3. Challenger = champion hyper-parameters, trained in two stages:
   A) fit on the pool minus its most recent 15% (time), early-stop / calibrate /
      choose thresholds on that recent slice (out-of-sample);
   B) refit on the whole pool with A's number of rounds (scaled to the larger
      pool), recent live rows up-weighted, reusing A's calibrator and thresholds.
4. Gate: promote only if challenger PR-AUC on the holdout beats the champion
   by >= `min_pr_auc_gain` AND its p95 model latency is within budget.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from fraud.config import load_config, path
from fraud.data.split import load_bounds
from fraud.evaluate.cost import CostParams, Thresholds, decide, optimize_thresholds, total_cost
from fraud.evaluate.metrics import ranking_metrics
from fraud.features.history import build_offline_features
from fraud.features.pipeline import FeaturePipeline
from fraud.model import IsotonicCalibrator, ModelBundle
from fraud.serve.prediction_log import read_log

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class GateDecision:
    promote: bool
    reason: str


def promotion_gate(
    champion_pr_auc: float,
    challenger_pr_auc: float,
    challenger_latency_p95_ms: float,
    min_gain: float,
    latency_budget_ms: float,
) -> GateDecision:
    gain = challenger_pr_auc - champion_pr_auc
    if challenger_latency_p95_ms > latency_budget_ms:
        return GateDecision(
            False,
            f"latency p95 {challenger_latency_p95_ms:.1f}ms exceeds budget "
            f"{latency_budget_ms:.0f}ms",
        )
    if gain < min_gain:
        return GateDecision(
            False,
            f"PR-AUC gain {gain:+.4f} below required +{min_gain:.4f} "
            f"(challenger {challenger_pr_auc:.4f} vs champion {champion_pr_auc:.4f})",
        )
    return GateDecision(
        True,
        f"PR-AUC gain {gain:+.4f} >= +{min_gain:.4f} "
        f"(challenger {challenger_pr_auc:.4f} vs champion {champion_pr_auc:.4f}), "
        f"latency p95 {challenger_latency_p95_ms:.1f}ms within budget",
    )


def logged_live_transactions(pred_dir: Path, label_dir: Path) -> pd.DataFrame:
    """Raw transactions the API scored, joined with labels that have arrived (NaN = not yet)."""
    preds = read_log(pred_dir)
    if preds.empty:
        return pd.DataFrame()
    preds = preds.sort_values("logged_at").drop_duplicates("TransactionID", keep="last")
    raw = pd.DataFrame([json.loads(r) for r in preds["raw"]])
    labels = read_log(label_dir)
    if labels.empty:
        raw["isFraud"] = np.nan
    else:
        lab = labels.drop_duplicates("TransactionID", keep="last")[["TransactionID", "isFraud"]]
        raw = raw.merge(lab, on="TransactionID", how="left")
    return raw


def build_frame(pred_dir: Path, label_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Offline features over history + logged live traffic; returns (features, is_live)."""
    bounds = load_bounds()
    hist = pd.read_parquet(path("processed_dir") / "transactions.parquet")
    hist = hist[hist["TransactionDT"] < bounds.live_start]
    live = logged_live_transactions(pred_dir, label_dir)
    if live.empty:
        raise RuntimeError("prediction log is empty: nothing to retrain on")
    live = live[~live["TransactionID"].isin(hist["TransactionID"])].reindex(columns=hist.columns)
    for c in hist.columns:
        if hist[c].dtype.kind in "biuf":
            live[c] = pd.to_numeric(live[c], errors="coerce")
        else:
            live[c] = live[c].astype(object)
    full = pd.concat([hist, live], ignore_index=True)
    full = full.sort_values(["TransactionDT", "TransactionID"], kind="mergesort").reset_index(
        drop=True
    )
    for c in ["TransactionDT", "TransactionID", "card1"]:
        full[c] = full[c].astype("int64")
    feats = build_offline_features(full)
    is_live = feats["TransactionDT"] >= bounds.live_start
    return feats, is_live


def _score(bundle: ModelBundle, feats: pd.DataFrame) -> np.ndarray:
    return bundle.predict_proba(bundle.matrix(feats))


def evaluate_on(bundle: ModelBundle, feats: pd.DataFrame, cost: CostParams) -> dict[str, Any]:
    y = feats["isFraud"].to_numpy().astype(int)
    p = _score(bundle, feats)
    res = ranking_metrics(y, p)
    res["cost"] = total_cost(
        y, feats["TransactionAmt"].to_numpy(), decide(p, bundle.thresholds), cost
    )
    return res


def train_challenger(
    feats: pd.DataFrame,
    is_live: pd.Series,
    champion: ModelBundle,
    cfg: dict[str, Any],
) -> tuple[ModelBundle, dict[str, Any], pd.DataFrame]:
    rc = cfg["retrain"]
    cost = CostParams.from_config(cfg)
    labelled = feats["isFraud"].notna()
    live_lab = feats[is_live & labelled]
    if len(live_lab) < 1000 or live_lab["isFraud"].sum() < 20:
        raise RuntimeError(f"not enough newly labelled data ({len(live_lab)} rows)")
    holdout_start = int(live_lab["TransactionDT"].quantile(1 - rc["holdout_frac"]))
    holdout = live_lab[live_lab["TransactionDT"] >= holdout_start]
    pool = feats[labelled & (feats["TransactionDT"] < holdout_start)]
    weight = np.where(is_live[pool.index], rc["recent_weight"], 1.0)

    recent_start = int(pool["TransactionDT"].quantile(0.85))
    a_mask = (pool["TransactionDT"] < recent_start).to_numpy()
    part_a, recent = pool[a_mask], pool[~a_mask]

    params = dict(champion.metadata["params"])
    tc = cfg["train"]
    pipe_a = FeaturePipeline()
    Xa = pipe_a.to_numpy(pipe_a.fit_transform(part_a, part_a["isFraud"]))
    Xr = pipe_a.transform_numpy(recent)
    da = lgb.Dataset(Xa, part_a["isFraud"].to_numpy(), weight=weight[a_mask], free_raw_data=False)
    dr = lgb.Dataset(Xr, recent["isFraud"].to_numpy(), reference=da)
    booster_a = lgb.train(
        params,
        da,
        tc["num_boost_round"],
        valid_sets=[dr],
        callbacks=[lgb.early_stopping(tc["early_stopping_rounds"], verbose=False)],
    )
    raw_r = booster_a.predict(Xr, num_iteration=booster_a.best_iteration)
    calibrator = IsotonicCalibrator.fit(raw_r, recent["isFraud"].to_numpy())
    cal_r = calibrator(raw_r)
    thresholds, _ = optimize_thresholds(
        recent["isFraud"].to_numpy(), recent["TransactionAmt"].to_numpy(), cal_r, cost
    )

    rounds = max(int(booster_a.best_iteration / 0.85), 50)
    pipe = FeaturePipeline()
    X = pipe.to_numpy(pipe.fit_transform(pool, pool["isFraud"]))
    booster = lgb.train(params, lgb.Dataset(X, pool["isFraud"].to_numpy(), weight=weight), rounds)
    challenger = ModelBundle(
        booster=booster,
        pipeline=pipe,
        calibrator=calibrator,
        thresholds=Thresholds(thresholds.review, thresholds.block),
        metadata={
            "trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "feature_version": pipe.version,
            "n_features": len(pipe.feature_names),
            "params": params,
            "best_iteration": rounds,
            "split": champion.metadata.get("split"),
            "retrain": {
                "pool_rows": len(pool),
                "pool_live_rows": int(is_live[pool.index].sum()),
                "holdout_rows": len(holdout),
                "holdout_frauds": int(holdout["isFraud"].sum()),
                "holdout_start_dt": holdout_start,
                "stage_a_best_iteration": booster_a.best_iteration,
                "recent_weight": rc["recent_weight"],
            },
            "monitor_columns": champion.metadata.get("monitor_columns", []),
            "cost_params": cost.__dict__,
        },
    )
    challenger.metadata["latency_ms"] = challenger.latency_ms(
        challenger.matrix(holdout), rc["latency_rows"]
    )
    ref_n = min(cfg["monitor"]["reference_rows"], len(recent))
    ref_idx = np.sort(np.random.default_rng(cfg["seed"]).choice(len(recent), ref_n, replace=False))
    ref = pd.DataFrame(pipe.transform_numpy(recent.iloc[ref_idx]), columns=pipe.feature_names)
    ref["score"] = cal_r[ref_idx]
    ref["isFraud"] = recent["isFraud"].to_numpy()[ref_idx]
    return challenger, {"holdout": holdout}, ref


def append_retraining_log(entry: dict[str, Any]) -> Path:
    p = path("reports_dir") / "retraining_log.md"
    if not p.exists():
        p.write_text(
            "# Retraining log\n\nEvery retraining run and its decision (newest last).\n\n"
            "| Run (UTC) | Trigger | Champion | Challenger | Holdout rows (frauds) | Champion PR-AUC "
            "| Challenger PR-AUC | Challenger p95 latency | Decision | Reason |\n"
            "|---|---|---|---|---|---:|---:|---:|---|---|\n"
        )
    with open(p, "a") as f:
        f.write(
            f"| {entry['run_at']} | {entry['trigger']} | v{entry['champion_version']} | "
            f"v{entry['challenger_version']} | {entry['holdout_rows']:,} ({entry['holdout_frauds']}) | "
            f"{entry['champion_pr_auc']:.4f} | {entry['challenger_pr_auc']:.4f} | "
            f"{entry['challenger_latency_p95_ms']:.1f} ms | **{entry['decision']}** | {entry['reason']} |\n"
        )
    return p


def reload_api() -> str:
    import httpx

    url = os.environ.get("API_URL", "http://localhost:8000")
    key = os.environ.get("ADMIN_API_KEY", "")
    try:
        r = httpx.post(f"{url}/admin/reload", headers={"X-API-Key": key}, timeout=60)
        return f"{r.status_code} {r.text[:200]}"
    except httpx.HTTPError as e:
        return f"reload failed: {e}"


def gate_from_config(
    champ: dict[str, Any], chall: dict[str, Any], latency_p95: float
) -> GateDecision:
    cfg = load_config()
    return promotion_gate(
        champ["pr_auc"],
        chall["pr_auc"],
        latency_p95,
        cfg["retrain"]["min_pr_auc_gain"],
        cfg["serve"]["latency_budget_ms"],
    )


__all__ = [
    "GateDecision",
    "promotion_gate",
    "build_frame",
    "train_challenger",
    "evaluate_on",
    "append_retraining_log",
    "reload_api",
    "gate_from_config",
    "asdict",
]
