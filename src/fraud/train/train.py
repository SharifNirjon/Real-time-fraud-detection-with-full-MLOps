"""End-to-end training: baselines, LightGBM (+weighting comparison, Optuna),
isotonic calibration and cost-based thresholds on VALID, evaluation on TEST,
reports and the deployable model bundle.

Usage: python -m fraud.train.train [--sample] [--trials N] [--no-mlflow]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from fraud.config import load_config, path
from fraud.data.split import load_bounds
from fraud.evaluate import plots
from fraud.evaluate.cost import CostParams, Thresholds, optimize_thresholds
from fraud.evaluate.report import DISPLAY, evaluate_all, write_reports
from fraud.features.pipeline import FeaturePipeline
from fraud.model import IsotonicCalibrator, ModelBundle
from fraud.monitor.drift import write_reference
from fraud.train import lgbm
from fraud.train.baselines import RulesEngine, logistic_regression

log = logging.getLogger(__name__)

ALWAYS_MONITOR = [
    "TransactionAmt",
    "log_amt",
    "DeviceInfo_freq",
    "DeviceType_map",
    "uid_device_is_new",
    "P_emaildomain_freq",
    "hour",
]


def load_feature_splits() -> tuple[pd.DataFrame, np.ndarray]:
    feats = pd.read_parquet(path("processed_dir") / "features.parquet")
    split = load_bounds().assign(feats["TransactionDT"])
    return feats, split


def monitored_columns(names: list[str], gain: np.ndarray, top: int = 20) -> list[str]:
    ranked = [names[i] for i in np.argsort(gain)[::-1][:top]]
    return list(dict.fromkeys(ranked + [c for c in ALWAYS_MONITOR if c in names]))


def train_lightgbm(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    names: list[str],
    cfg: dict[str, Any],
    n_trials: int,
) -> tuple[dict[str, lgb.Booster], dict[str, Any], dict[str, Any]]:
    tc = cfg["train"]
    dtrain = lgb.Dataset(X_tr, y_tr, feature_name=names, free_raw_data=False)
    dvalid = lgb.Dataset(X_va, y_va, reference=dtrain, free_raw_data=False)
    base = lgbm.base_params(cfg["seed"], tc["learning_rate"])
    boosters: dict[str, lgb.Booster] = {}
    info: dict[str, Any] = {"weighting": {}}
    for name, extra in lgbm.weighting_variants(y_tr).items():
        t = time.time()
        b, score = lgbm.fit(
            base | extra, dtrain, dvalid, tc["num_boost_round"], tc["early_stopping_rounds"]
        )
        boosters[f"lgbm_{name}"] = b
        info["weighting"][name] = {
            "valid_pr_auc": score,
            "best_iteration": b.best_iteration,
            **extra,
        }
        log.info(
            "weighting=%s valid PR-AUC=%.4f iters=%d (%.0fs)",
            name,
            score,
            b.best_iteration,
            time.time() - t,
        )
    best_w = max(info["weighting"], key=lambda k: info["weighting"][k]["valid_pr_auc"])
    info["chosen_weighting"] = best_w
    tuned_base = base | lgbm.weighting_variants(y_tr)[best_w]
    t = time.time()
    # search at a higher learning rate (fewer rounds), refit the winner at the final rate
    best_params, history = lgbm.optuna_search(
        tuned_base | {"learning_rate": tc["search_learning_rate"]},
        dtrain,
        dvalid,
        n_trials,
        tc["num_boost_round"],
        tc["early_stopping_rounds"],
        cfg["seed"],
    )
    best_params["learning_rate"] = tc["learning_rate"]
    info["optuna"] = {"n_trials": n_trials, "seconds": time.time() - t, "history": history}
    final, score = lgbm.fit(
        best_params, dtrain, dvalid, tc["num_boost_round"], tc["early_stopping_rounds"]
    )
    info["final"] = {"valid_pr_auc": score, "best_iteration": final.best_iteration}
    boosters["lgbm_tuned"] = final
    # keep only the trees up to the best iteration in the deployable model
    final.free_dataset()
    return boosters, best_params, info


def run(sample: bool, n_trials: int | None, use_mlflow: bool) -> dict[str, Any]:
    cfg = load_config()
    tc = cfg["train"]
    cost = CostParams.from_config(cfg)
    n_trials = n_trials or (tc["optuna_trials_sample"] if sample else tc["optuna_trials"])
    t0 = time.time()
    feats, split = load_feature_splits()
    tr, va, te = (feats[split == s] for s in ["train", "valid", "test"])
    y_tr, y_va = tr["isFraud"].to_numpy(), va["isFraud"].to_numpy()
    log.info("train=%d valid=%d test=%d", len(tr), len(va), len(te))

    pipe = FeaturePipeline()
    X_tr_df = pipe.fit_transform(
        tr, tr["isFraud"]
    )  # encoders fitted on TRAIN only (OOF target enc)
    X_tr = pipe.to_numpy(X_tr_df)
    X_va, X_te = (pipe.to_numpy(pipe.transform(d)) for d in (va, te))
    names = pipe.feature_names
    log.info("features: %d", len(names))

    scores = pd.concat([va, te])[
        ["TransactionID", "TransactionDT", "TransactionAmt", "isFraud"]
    ].copy()
    scores["split"] = ["valid"] * len(va) + ["test"] * len(te)
    X_vt = np.vstack([X_va, X_te])

    rules = RulesEngine().fit(tr)
    scores["rules"] = np.concatenate([rules.score(va), rules.score(te)])
    t = time.time()
    lr = logistic_regression(cfg["seed"]).fit(X_tr, y_tr)
    scores["logreg"] = lr.predict_proba(X_vt)[:, 1]
    log.info("logreg fitted in %.0fs", time.time() - t)

    boosters, best_params, lgb_info = train_lightgbm(X_tr, y_tr, X_va, y_va, names, cfg, n_trials)
    for name, b in boosters.items():
        scores[name] = b.predict(X_vt, num_iteration=b.best_iteration)

    final = boosters["lgbm_tuned"]
    raw_va = scores.loc[scores["split"] == "valid", "lgbm_tuned"].to_numpy()
    calibrator = IsotonicCalibrator.fit(raw_va, y_va)  # calibration fitted on VALID
    scores["lgbm_tuned_calibrated"] = calibrator(scores["lgbm_tuned"].to_numpy())
    cal_va = scores.loc[scores["split"] == "valid", "lgbm_tuned_calibrated"].to_numpy()
    thresholds, _ = optimize_thresholds(y_va, va["TransactionAmt"].to_numpy(), cal_va, cost)

    scores.to_parquet(path("artifacts_dir") / "scores.parquet", index=False)
    models = [m for m in DISPLAY if m in scores.columns]
    res = evaluate_all(scores, models, cost)
    assert res["models"]["lgbm_tuned_calibrated"]["thresholds"] == {
        "review": thresholds.review,
        "block": thresholds.block,
    }

    # plots
    rd = path("reports_dir")
    te_s = scores[scores["split"] == "test"]
    y_te = te_s["isFraud"].to_numpy()
    plots.pr_curves(
        y_te,
        {
            DISPLAY[m]: te_s[m].to_numpy()
            for m in ["rules", "logreg", "lgbm_none", "lgbm_tuned_calibrated"]
        },
        rd / "pr_curve.png",
    )
    cal_models = {
        "LightGBM raw": "lgbm_tuned",
        "LightGBM + isotonic": "lgbm_tuned_calibrated",
        "Logistic regression": "logreg",
    }
    plots.calibration(
        {
            k: {"brier": res["models"][m]["brier"], "curve": res["models"][m]["calibration_curve"]}
            for k, m in cal_models.items()
        },
        rd / "calibration.png",
    )
    gain = final.feature_importance("gain", iteration=final.best_iteration)
    plots.feature_importance(names, gain, rd / "feature_importance.png")
    rng = np.random.default_rng(cfg["seed"])
    shap_idx = rng.choice(len(X_te), min(tc["shap_sample"], len(X_te)), replace=False)
    try:
        plots.shap_summary(final, X_te[shap_idx], names, rd / "shap_summary.png")
    except Exception as e:  # shap is optional for the pipeline to succeed
        log.warning("SHAP plot failed: %s", e)

    # deployable bundle: trees truncated at the best iteration
    final_model = lgb.Booster(model_str=final.model_to_string(num_iteration=final.best_iteration))
    split_info = json.loads((path("processed_dir") / "split.json").read_text())
    served = res["models"]["lgbm_tuned_calibrated"]
    bundle = ModelBundle(
        booster=final_model,
        pipeline=pipe,
        calibrator=calibrator,
        thresholds=Thresholds(thresholds.review, thresholds.block),
        metadata={
            "trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "feature_version": pipe.version,
            "n_features": len(names),
            "sample": sample,
            "params": best_params,
            "best_iteration": final.best_iteration,
            "split": split_info,
            "test_metrics": {
                k: served[k]
                for k in [
                    "pr_auc",
                    "roc_auc",
                    "recall_at_1pct_fpr",
                    "precision_at_80pct_recall",
                    "brier",
                    "test_cost",
                ]
            },
            "valid_pr_auc_raw": lgb_info["final"]["valid_pr_auc"],
            "monitor_columns": monitored_columns(names, gain),
            "cost_params": cost.__dict__,
        },
    )
    bundle.metadata["latency_ms"] = bundle.latency_ms(X_te, cfg["retrain"]["latency_rows"])
    bundle_path = bundle.save(path("artifacts_dir") / "model_bundle.pkl")

    # monitoring reference: out-of-sample VALID rows (features + calibrated score)
    ref_n = min(cfg["monitor"]["reference_rows"], len(va))
    ref_idx = np.sort(rng.choice(len(va), ref_n, replace=False))
    ref = pd.DataFrame(X_va[ref_idx], columns=names)
    ref["score"] = cal_va[ref_idx]
    ref["isFraud"] = y_va[ref_idx]

    extra = {
        "training": {
            "seconds": time.time() - t0,
            "n_features": len(names),
            "lightgbm": {k: v for k, v in lgb_info.items() if k != "optuna"}
            | {
                "optuna_trials": n_trials,
                "optuna_seconds": lgb_info["optuna"]["seconds"],
                "best_params": best_params,
            },
            "latency_ms": bundle.metadata["latency_ms"],
            "sample": sample,
        }
    }
    write_reports(res, extra)
    (path("artifacts_dir") / "optuna_trials.json").write_text(
        json.dumps(lgb_info["optuna"]["history"], indent=2)
    )

    if use_mlflow:
        from fraud.train.registry import log_training_run

        version = log_training_run(bundle, bundle_path, res, extra, names, cfg)
        log.info("registered model version %s", version)
    else:
        version = "local"
    write_reference(
        ref,
        {
            "model_version": version,
            "monitor_columns": bundle.metadata["monitor_columns"],
            "test_pr_auc": served["pr_auc"],
            "source": "valid split sample (out-of-sample scores)",
        },
    )
    return res


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true")
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()
    res = run(args.sample, args.trials, not args.no_mlflow)
    print((path("reports_dir") / "results.md").read_text())
    log.info("served model test PR-AUC=%.4f", res["models"]["lgbm_tuned_calibrated"]["pr_auc"])


if __name__ == "__main__":
    main()
