"""MLflow tracking + model registry helpers (champion / challenger aliases)."""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import mlflow
from mlflow import MlflowClient

from fraud.config import ROOT, load_config, path
from fraud.model import ModelBundle

log = logging.getLogger(__name__)

CHAMPION, CHALLENGER = "champion", "challenger"


def tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")


def model_name() -> str:
    return os.environ.get("MODEL_NAME", load_config()["train"]["registered_model_name"])


def git_commit() -> str:
    if os.environ.get("GIT_COMMIT"):
        return os.environ["GIT_COMMIT"]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True
        )
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


class FraudPyfunc(mlflow.pyfunc.PythonModel):
    """pyfunc wrapper so the registered model is loadable by generic MLflow tooling.

    Input: offline feature frame (fraud.features.history output). Output: calibrated probability.
    The API skips pyfunc and loads the bundle file directly for lower latency.
    """

    def load_context(self, context: Any) -> None:
        self.bundle = ModelBundle.load(context.artifacts["bundle"])

    def predict(self, context: Any, model_input: Any, params: Any = None) -> Any:
        return self.bundle.predict_proba(self.bundle.matrix(model_input))


def _flat(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out |= _flat(v, f"{key}.")
        elif isinstance(v, int | float | str | bool):
            out[key] = v
    return out


def log_bundle(bundle_path: Path, metadata: dict[str, Any]) -> Any:
    return mlflow.pyfunc.log_model(
        name="model",
        python_model=FraudPyfunc(),
        artifacts={"bundle": str(bundle_path)},
        pip_requirements=["lightgbm", "pandas", "numpy", "scikit-learn", "pyyaml"],
        metadata={k: v for k, v in metadata.items() if isinstance(v, str | int | float)},
    )


def register(model_uri: str, tags: dict[str, str] | None = None) -> str:
    """Register a logged model; the first version becomes champion, later ones challenger."""
    client = MlflowClient()
    name = model_name()
    mv = mlflow.register_model(model_uri, name, tags=tags)
    try:
        client.get_model_version_by_alias(name, CHAMPION)
        client.set_registered_model_alias(name, CHALLENGER, mv.version)
        log.info("registered %s v%s as %s", name, mv.version, CHALLENGER)
    except mlflow.exceptions.MlflowException:
        client.set_registered_model_alias(name, CHAMPION, mv.version)
        log.info("registered %s v%s as %s (no champion existed)", name, mv.version, CHAMPION)
    return str(mv.version)


def log_training_run(
    bundle: ModelBundle,
    bundle_path: Path,
    res: dict[str, Any],
    extra: dict[str, Any],
    names: list[str],
    cfg: dict[str, Any],
) -> str:
    mlflow.set_tracking_uri(tracking_uri())
    mlflow.set_experiment(cfg["train"]["experiment_name"])
    md = bundle.metadata
    with mlflow.start_run(run_name=f"train-{md['trained_at'][:19]}") as run:
        mlflow.set_tags(
            {
                "git_commit": git_commit(),
                "feature_version": md["feature_version"],
                "sample": str(md["sample"]),
                "kind": "initial_training",
            }
        )
        split = md["split"]
        mlflow.log_params(
            {f"split.{s}.dates": " .. ".join(split["dates"][s]) for s in split["dates"]}
            | {f"split.{s}.rows": split["rows"][s] for s in split["rows"]}
            | {f"lgbm.{k}": v for k, v in md["params"].items()}
            | {"n_features": len(names), "best_iteration": md["best_iteration"]}
            | {f"cost.{k}": v for k, v in md["cost_params"].items()}
            | {"chosen_weighting": extra["training"]["lightgbm"]["chosen_weighting"]}
        )
        for model, r in res["models"].items():
            mlflow.log_metrics(
                {
                    f"{model}.{k}": r[k]
                    for k in [
                        "pr_auc",
                        "roc_auc",
                        "recall_at_1pct_fpr",
                        "precision_at_80pct_recall",
                        "brier",
                        "test_cost",
                        "saved_vs_no_model",
                        "saved_vs_rules",
                    ]
                }
            )
        mlflow.log_metrics(
            {
                "threshold.review": bundle.thresholds.review,
                "threshold.block": bundle.thresholds.block,
            }
            | {f"latency.{k}": v for k, v in md["latency_ms"].items()}
            | {"test.no_model_cost": res["test"]["no_model_cost"]}
        )
        mlflow.log_dict(
            {"features": names, "monitor_columns": md["monitor_columns"]}, "feature_list.json"
        )
        mlflow.log_dict(split, "split.json")
        mlflow.log_dict(_flat(extra), "training_info.json")
        rd = path("reports_dir")
        for f in [
            "results.md",
            "metrics.json",
            "pr_curve.png",
            "calibration.png",
            "feature_importance.png",
            "shap_summary.png",
        ]:
            if (rd / f).exists():
                mlflow.log_artifact(str(rd / f), "reports")
        info = log_bundle(bundle_path, {"feature_version": md["feature_version"]})
        version = register(info.model_uri, {"git_commit": git_commit(), "run_id": run.info.run_id})
    return version


def load_bundle(alias: str = CHAMPION) -> tuple[ModelBundle, str]:
    """Download the bundle registered under `alias` (used by the API and the retrain flow)."""
    mlflow.set_tracking_uri(tracking_uri())
    client = MlflowClient()
    mv = client.get_model_version_by_alias(model_name(), alias)
    with tempfile.TemporaryDirectory() as tmp:
        local = mlflow.artifacts.download_artifacts(
            artifact_uri=f"models:/{model_name()}/{mv.version}", dst_path=tmp
        )
        bundle_file = next(Path(local).rglob("*.pkl"))
        bundle = ModelBundle.load(bundle_file)
    return bundle, str(mv.version)
