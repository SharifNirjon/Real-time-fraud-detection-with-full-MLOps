"""Drift + live-performance monitor.

Compares the most recent predictions in the prediction log with the training
reference (out-of-sample VALID rows of the current champion) using Evidently,
and - once delayed labels have arrived - computes live PR-AUC / recall /
precision. Writes an HTML report and a JSON summary to reports/drift/.

Usage: python -m fraud.monitor.drift
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from fraud.config import load_config, path
from fraud.evaluate.metrics import ranking_metrics
from fraud.serve.prediction_log import read_log

log = logging.getLogger(__name__)


def load_reference() -> tuple[pd.DataFrame, dict[str, Any]]:
    rd = path("reference_dir")
    ref = pd.read_parquet(rd / "reference.parquet")
    meta = json.loads((rd / "reference_meta.json").read_text())
    return ref, meta


def write_reference(ref: pd.DataFrame, meta: dict[str, Any]) -> None:
    rd = path("reference_dir")
    ref.to_parquet(rd / "reference.parquet", index=False)
    (rd / "reference_meta.json").write_text(json.dumps(meta, indent=2))


def recent_predictions(pred_dir: Path, n: int) -> pd.DataFrame:
    preds = read_log(pred_dir)
    if preds.empty:
        return preds
    preds = preds.sort_values(["logged_at"]).drop_duplicates("TransactionID", keep="last")
    return preds.sort_values(["TransactionDT", "TransactionID"]).tail(n)


def live_performance(preds: pd.DataFrame, labels: pd.DataFrame) -> dict[str, Any]:
    if preds.empty or labels.empty:
        return {"labelled_rows": 0}
    lab = labels.drop_duplicates("TransactionID", keep="last")[["TransactionID", "isFraud"]]
    j = preds.merge(lab, on="TransactionID", how="inner")
    out: dict[str, Any] = {
        "labelled_rows": len(j),
        "live_frauds": int(j["isFraud"].sum()) if len(j) else 0,
    }
    if len(j) < 200 or j["isFraud"].nunique() < 2:
        return out
    y = j["isFraud"].to_numpy()
    flagged = j["decision"].isin(["review", "block"]).to_numpy()
    m = ranking_metrics(y, j["score"].to_numpy())
    out |= {
        "live_pr_auc": m["pr_auc"],
        "live_roc_auc": m["roc_auc"],
        "live_recall": float(flagged[y == 1].mean()),
        "live_precision": float(y[flagged].mean()) if flagged.any() else 0.0,
        "live_fraud_rate": float(y.mean()),
    }
    return out


def run_drift_check(
    pred_dir: Path | None = None,
    label_dir: Path | None = None,
    out_dir: Path | None = None,
    window: int | None = None,
) -> dict[str, Any]:
    from evidently import Report
    from evidently.metrics import ValueDrift
    from evidently.presets import DataDriftPreset

    cfg = load_config()["monitor"]
    pred_dir = pred_dir or path("prediction_log_dir")
    label_dir = label_dir or path("label_log_dir")
    out_dir = out_dir or (path("reports_dir") / "drift")
    out_dir.mkdir(parents=True, exist_ok=True)
    window = window or cfg["current_window_rows"]

    ref, meta = load_reference()
    cur = recent_predictions(pred_dir, window)
    ts = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%S")
    summary: dict[str, Any] = {
        "generated_at": ts,
        "reference_model_version": meta.get("model_version"),
    }
    if len(cur) < 500:
        summary |= {
            "status": "insufficient_data",
            "current_rows": len(cur),
            "drift_detected": False,
        }
        (out_dir / "latest.json").write_text(json.dumps(summary, indent=2))
        return summary

    cols = [c for c in meta["monitor_columns"] if c in cur.columns and c in ref.columns]
    cols = [c for c in cols if ref[c].notna().sum() > 50 and cur[c].notna().sum() > 50]
    ref_df = ref[cols + ["score"]].astype("float64")
    cur_df = cur[cols + ["score"]].astype("float64")
    report = Report([DataDriftPreset(columns=cols), ValueDrift(column="score")])
    snap = report.run(current_data=cur_df, reference_data=ref_df)
    html = out_dir / f"drift_report_{ts}.html"
    snap.save_html(str(html))

    per_col: dict[str, float] = {}
    pred_drift = None
    for mres in snap.dict()["metrics"]:
        cfg_m = mres.get("config", {})
        if cfg_m.get("type", "").endswith("ValueDrift"):
            if cfg_m["column"] == "score":
                pred_drift = float(mres["value"])
            else:
                per_col[cfg_m["column"]] = float(mres["value"])
    threshold = 0.1  # Evidently default for Wasserstein (normed) on >1000 rows
    drifted = sorted([c for c, v in per_col.items() if v >= threshold], key=lambda c: -per_col[c])
    share = len(drifted) / max(len(per_col), 1)

    labels = read_log(label_dir)
    # Labels arrive late, so the drift window is (almost) never labelled. Live
    # performance uses the most recent `performance_window_rows` LABELLED predictions.
    all_preds = recent_predictions(pred_dir, 10**9)
    if labels.empty:
        perf: dict[str, Any] = {"labelled_rows": 0}
        perf_all: dict[str, Any] = {"labelled_rows": 0}
    else:
        labelled_ids = set(labels["TransactionID"])
        recent_lab = all_preds[all_preds["TransactionID"].isin(labelled_ids)].tail(
            cfg["performance_window_rows"]
        )
        perf = live_performance(recent_lab, labels)
        perf_all = live_performance(all_preds, labels)

    summary |= {
        "status": "ok",
        "current_rows": len(cur),
        "current_dt_range": [int(cur["TransactionDT"].min()), int(cur["TransactionDT"].max())],
        "current_model_versions": sorted(cur["model_version"].astype(str).unique().tolist()),
        "n_monitored_columns": len(per_col),
        "share_drifted_columns": share,
        "drifted_columns": drifted,
        "column_drift_scores": per_col,
        "prediction_drift_score": pred_drift,
        "mean_score_reference": float(ref["score"].mean()),
        "mean_score_current": float(cur["score"].mean()),
        "drift_detected": bool(
            share >= cfg["drift_share_threshold"]
            or (pred_drift is not None and pred_drift >= cfg["prediction_drift_threshold"])
        ),
        "thresholds": {
            "drift_share": cfg["drift_share_threshold"],
            "prediction_drift": cfg["prediction_drift_threshold"],
            "column_drift": threshold,
        },
        "reference_test_pr_auc": meta.get("test_pr_auc"),
        "report_html": (
            str(html.relative_to(path("reports_dir").parent))
            if html.is_relative_to(path("reports_dir").parent)
            else str(html)
        ),
        **perf,
        "cumulative": perf_all,
    }
    (out_dir / f"summary_{ts}.json").write_text(json.dumps(summary, indent=2, default=float))
    (out_dir / "latest.json").write_text(json.dumps(summary, indent=2, default=float))
    log.info(
        "drift share=%.2f pred_drift=%s detected=%s labelled=%s live_pr_auc=%s",
        share,
        pred_drift,
        summary["drift_detected"],
        perf.get("labelled_rows"),
        perf.get("live_pr_auc"),
    )
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    s = run_drift_check()
    print(
        json.dumps(
            {k: v for k, v in s.items() if k != "column_drift_scores"}, indent=2, default=float
        )
    )


if __name__ == "__main__":
    main()
