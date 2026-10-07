"""Prefect flows: drift monitor (may trigger retraining) and champion/challenger retraining.

Usage:
  python -m fraud.pipelines.flows monitor [--auto-retrain]
  python -m fraud.pipelines.flows retrain [--trigger manual]
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
from prefect import flow, get_run_logger, task

from fraud.config import load_config, path
from fraud.evaluate.cost import CostParams
from fraud.model import ModelBundle
from fraud.monitor.drift import run_drift_check, write_reference
from fraud.pipelines import retrain as R

log = logging.getLogger(__name__)


@task(name="check-drift")
def check_drift() -> dict[str, Any]:
    return run_drift_check()


@task(name="load-champion")
def load_champion() -> tuple[ModelBundle, str]:
    from fraud.train.registry import load_bundle

    return load_bundle("champion")


@task(name="rebuild-training-set")
def rebuild_training_set() -> tuple[pd.DataFrame, pd.Series]:
    return R.build_frame(path("prediction_log_dir"), path("label_log_dir"))


@task(name="train-challenger")
def train_challenger(feats: pd.DataFrame, is_live: pd.Series, champion: ModelBundle) -> Any:
    return R.train_challenger(feats, is_live, champion, load_config())


@task(name="evaluate-on-holdout")
def evaluate(
    champion: ModelBundle, challenger: ModelBundle, holdout: pd.DataFrame
) -> dict[str, Any]:
    cost = CostParams.from_config(load_config())
    return {
        "champion": R.evaluate_on(champion, holdout, cost),
        "challenger": R.evaluate_on(challenger, holdout, cost),
    }


@task(name="register-and-decide")
def register_and_decide(
    champion: ModelBundle,
    champion_version: str,
    challenger: ModelBundle,
    ref: pd.DataFrame,
    holdout: pd.DataFrame,
    metrics: dict[str, Any],
    trigger: str,
    drift: dict[str, Any] | None,
) -> dict[str, Any]:
    import mlflow
    from mlflow import MlflowClient

    from fraud.train.registry import (
        CHAMPION,
        git_commit,
        log_bundle,
        model_name,
        register,
        tracking_uri,
    )

    cfg = load_config()
    lat = challenger.metadata["latency_ms"]["p95"]
    gate = R.gate_from_config(metrics["champion"], metrics["challenger"], lat)
    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment("fraud-retraining")
    run_at = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S")
    with (
        mlflow.start_run(run_name=f"retrain-{run_at}") as run,
        tempfile.TemporaryDirectory() as tmp,
    ):
        rt = challenger.metadata["retrain"]
        mlflow.set_tags(
            {
                "trigger": trigger,
                "decision": "promoted" if gate.promote else "rejected",
                "reason": gate.reason,
                "git_commit": git_commit(),
                "champion_version": champion_version,
                "kind": "retraining",
            }
        )
        mlflow.log_params(
            {f"retrain.{k}": v for k, v in rt.items()}
            | {
                "min_pr_auc_gain": cfg["retrain"]["min_pr_auc_gain"],
                "latency_budget_ms": cfg["serve"]["latency_budget_ms"],
            }
        )
        mlflow.log_metrics(
            {f"{who}.{k}": float(v) for who, m in metrics.items() for k, v in m.items()}
            | {"challenger.latency_p95_ms": lat}
        )
        if drift:
            mlflow.log_dict(
                {k: v for k, v in drift.items() if k != "column_drift_scores"}, "drift_summary.json"
            )
        bundle_path = challenger.save(Path(tmp) / "model_bundle.pkl")
        info = log_bundle(bundle_path, {"feature_version": challenger.metadata["feature_version"]})
        version = register(
            info.model_uri,
            {"git_commit": git_commit(), "run_id": run.info.run_id, "kind": "retraining"},
        )
        client = MlflowClient()
        reload_status = None
        if gate.promote:
            client.set_registered_model_alias(model_name(), "previous-champion", champion_version)
            client.set_registered_model_alias(model_name(), CHAMPION, version)
            write_reference(
                ref,
                {
                    "model_version": version,
                    "monitor_columns": challenger.metadata["monitor_columns"],
                    "test_pr_auc": metrics["challenger"]["pr_auc"],
                    "source": "recent pre-holdout slice (out-of-sample stage-A scores)",
                },
            )
            reload_status = R.reload_api()
            mlflow.set_tag("api_reload", reload_status)
    entry = {
        "run_at": run_at,
        "trigger": trigger,
        "champion_version": champion_version,
        "challenger_version": version,
        "holdout_rows": rt["holdout_rows"],
        "holdout_frauds": rt["holdout_frauds"],
        "champion_pr_auc": metrics["champion"]["pr_auc"],
        "challenger_pr_auc": metrics["challenger"]["pr_auc"],
        "challenger_latency_p95_ms": lat,
        "decision": "PROMOTED" if gate.promote else "REJECTED",
        "reason": gate.reason,
        "api_reload": reload_status,
        "metrics": metrics,
    }
    R.append_retraining_log(entry)
    (path("reports_dir") / "retraining_last.json").write_text(
        json.dumps(entry, indent=2, default=float)
    )
    return entry


@flow(name="retrain-champion-challenger")
def retrain_flow(trigger: str = "manual", drift: dict[str, Any] | None = None) -> dict[str, Any]:
    logger = get_run_logger()
    lock = _lock_path()
    lock.write_text(
        json.dumps({"started_at": pd.Timestamp.now(tz="UTC").isoformat(), "trigger": trigger})
    )
    try:
        champion, champion_version = load_champion()
        feats, is_live = rebuild_training_set()
        challenger, info, ref = train_challenger(feats, is_live, champion)
        metrics = evaluate(champion, challenger, info["holdout"])
        entry = register_and_decide(
            champion, champion_version, challenger, ref, info["holdout"], metrics, trigger, drift
        )
    finally:
        lock.unlink(missing_ok=True)
    logger.info("retraining %s: %s", entry["decision"], entry["reason"])
    return entry


LOCK_STALE = pd.Timedelta(hours=3)


def _lock_path() -> Path:
    return path("reports_dir") / "retraining_in_progress.json"


def retrain_running() -> bool:
    p = _lock_path()
    if not p.exists():
        return False
    started = pd.Timestamp(json.loads(p.read_text())["started_at"])
    return pd.Timestamp.now(tz="UTC") - started < LOCK_STALE  # a crashed run never blocks forever


def retrain_allowed(summary: dict[str, Any]) -> tuple[bool, str]:
    rc = load_config()["retrain"]
    if retrain_running():
        return False, "a retraining run is already in progress"
    labelled = int((summary.get("cumulative") or {}).get("labelled_rows", 0))
    if labelled < rc["min_labelled_rows"]:
        return (
            False,
            f"only {labelled} labelled live rows (< {rc['min_labelled_rows']}); waiting for labels",
        )
    last = path("reports_dir") / "retraining_last.json"
    if last.exists():
        prev = json.loads(last.read_text())
        age = pd.Timestamp.now(tz="UTC") - pd.Timestamp(prev["run_at"], tz="UTC")
        if age < pd.Timedelta(minutes=rc["cooldown_minutes"]):
            return (
                False,
                f"last retraining {age.total_seconds() / 60:.0f} min ago (cooldown {rc['cooldown_minutes']} min)",
            )
    return True, "ok"


@flow(name="drift-monitor")
def monitor_flow(auto_retrain: bool = True) -> dict[str, Any]:
    logger = get_run_logger()
    summary = check_drift()
    logger.info(
        "drift_detected=%s share=%.2f pred_drift=%s live_pr_auc=%s",
        summary.get("drift_detected"),
        summary.get("share_drifted_columns", 0.0),
        summary.get("prediction_drift_score"),
        summary.get("live_pr_auc"),
    )
    if auto_retrain and summary.get("drift_detected"):
        ok, why = retrain_allowed(summary)
        if ok:
            logger.info("drift over threshold -> triggering retraining")
            summary["retraining"] = retrain_flow(trigger="drift", drift=summary)
        else:
            logger.info("drift detected but retraining skipped: %s", why)
            summary["retraining"] = {"decision": "SKIPPED", "reason": why}
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("flow", choices=["monitor", "retrain"])
    ap.add_argument("--auto-retrain", action="store_true")
    ap.add_argument("--trigger", default="manual")
    args = ap.parse_args()
    if args.flow == "monitor":
        out = monitor_flow(auto_retrain=args.auto_retrain)
        out = {k: v for k, v in out.items() if k not in ("column_drift_scores",)}
    else:
        out = retrain_flow(trigger=args.trigger)
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
