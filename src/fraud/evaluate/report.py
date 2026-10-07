"""Evaluate every model on TEST with thresholds chosen on VALID; write reports.

Usage: python -m fraud.evaluate.report   (reads artifacts/scores.parquet)
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd

from fraud.config import load_config, path
from fraud.evaluate.cost import (
    APPROVE,
    BLOCK,
    REVIEW,
    CostParams,
    decide,
    no_model_cost,
    optimize_thresholds,
    total_cost,
)
from fraud.evaluate.metrics import calibration_metrics, confusion, ranking_metrics

log = logging.getLogger(__name__)

DISPLAY = {
    "rules": "Rules engine",
    "logreg": "Logistic regression",
    "lgbm_none": "LightGBM (no reweighting)",
    "lgbm_scale_pos_weight": "LightGBM (scale_pos_weight)",
    "lgbm_sqrt_scale_pos_weight": "LightGBM (sqrt scale_pos_weight)",
    "lgbm_tuned": "LightGBM tuned (Optuna, raw)",
    "lgbm_tuned_calibrated": "LightGBM tuned + isotonic (served)",
}


def evaluate_scores(
    y_va: np.ndarray,
    a_va: np.ndarray,
    s_va: np.ndarray,
    y_te: np.ndarray,
    a_te: np.ndarray,
    s_te: np.ndarray,
    cost: CostParams,
) -> dict[str, Any]:
    t, valid_cost = optimize_thresholds(y_va, a_va, s_va, cost)
    d = decide(s_te, t)
    res: dict[str, Any] = ranking_metrics(y_te, s_te)
    cal = calibration_metrics(y_te, s_te)
    res["brier"] = cal["brier"]
    res["calibration_curve"] = cal["curve"]
    res["thresholds"] = {"review": t.review, "block": t.block}
    res["valid_cost"] = valid_cost
    res["test_cost"] = total_cost(y_te, a_te, d, cost)
    res["decisions"] = {
        "approve": int((d == APPROVE).sum()),
        "review": int((d == REVIEW).sum()),
        "block": int((d == BLOCK).sum()),
    }
    res["confusion_flagged"] = confusion(y_te, d >= REVIEW)  # review or block counts as flagged
    res["confusion_block"] = confusion(y_te, d == BLOCK)
    yb = np.asarray(y_te).astype(bool)
    res["fraud_amount_stopped"] = float(np.asarray(a_te)[yb & (d >= REVIEW)].sum())
    return res


def evaluate_all(scores: pd.DataFrame, models: list[str], cost: CostParams) -> dict[str, Any]:
    va, te = scores[scores["split"] == "valid"], scores[scores["split"] == "test"]
    y_va, a_va = va["isFraud"].to_numpy(), va["TransactionAmt"].to_numpy()
    y_te, a_te = te["isFraud"].to_numpy(), te["TransactionAmt"].to_numpy()
    out: dict[str, Any] = {
        "test": {
            "rows": len(te),
            "frauds": int(y_te.sum()),
            "fraud_rate": float(y_te.mean()),
            "no_model_cost": no_model_cost(y_te, a_te),
        },
        "cost_params": cost.__dict__,
        "models": {},
    }
    for m in models:
        out["models"][m] = evaluate_scores(
            y_va, a_va, va[m].to_numpy(), y_te, a_te, te[m].to_numpy(), cost
        )
    base, rules = out["test"]["no_model_cost"], out["models"]["rules"]["test_cost"]
    for r in out["models"].values():
        r["saved_vs_no_model"] = base - r["test_cost"]
        r["saved_vs_no_model_pct"] = 100 * (base - r["test_cost"]) / base
        r["saved_vs_rules"] = rules - r["test_cost"]
    return out


def results_markdown(res: dict[str, Any], split_info: dict[str, Any] | None = None) -> str:
    t = res["test"]
    c = res["cost_params"]
    lines = [
        "# Results (TEST split, thresholds chosen on VALID)",
        "",
        f"Test: {t['rows']:,} transactions, {t['frauds']:,} frauds ({100 * t['fraud_rate']:.2f}%). "
        f"Cost of approving everything (no model): **${t['no_model_cost']:,.0f}**.",
        f"Cost model: missed fraud = amount; review = ${c['review_cost']:.0f} per transaction; "
        f"blocking a legitimate customer = ${c['false_decline_cost']:.0f}; "
        f"review catch rate = {c['review_catch_rate']:.0%}.",
        "",
        "| Model | PR-AUC | ROC-AUC | Recall @1% FPR | Precision @80% recall | Brier "
        "| Expected cost | Saved vs no model | Saved vs rules |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in res["models"].items():
        lines.append(
            f"| {DISPLAY.get(name, name)} | {r['pr_auc']:.4f} | {r['roc_auc']:.4f} | "
            f"{r['recall_at_1pct_fpr']:.3f} | {r['precision_at_80pct_recall']:.3f} | {r['brier']:.4f} | "
            f"${r['test_cost']:,.0f} | ${r['saved_vs_no_model']:,.0f} ({r['saved_vs_no_model_pct']:.1f}%) | "
            f"${r['saved_vs_rules']:,.0f} |"
        )
    served = res["models"].get("lgbm_tuned_calibrated")
    if served:
        th, d, cf, cb = (
            served[k] for k in ["thresholds", "decisions", "confusion_flagged", "confusion_block"]
        )
        lines += [
            "",
            "## Served model at the chosen thresholds",
            "",
            f"Thresholds (calibrated probability): review >= {th['review']:.4f}, block >= {th['block']:.4f}.",
            f"Decisions on test: approve {d['approve']:,}, review {d['review']:,}, block {d['block']:,}.",
            "",
            "| Confusion (flagged = review or block) | Predicted fraud | Predicted legit |",
            "|---|---:|---:|",
            f"| Actual fraud | {cf['tp']:,} | {cf['fn']:,} |",
            f"| Actual legit | {cf['fp']:,} | {cf['tn']:,} |",
            "",
            "| Confusion (block only) | Blocked | Not blocked |",
            "|---|---:|---:|",
            f"| Actual fraud | {cb['tp']:,} | {cb['fn']:,} |",
            f"| Actual legit | {cb['fp']:,} | {cb['tn']:,} |",
        ]
    if split_info:
        lines += [
            "",
            "## Data split (by TransactionDT)",
            "",
            "| Split | Rows | Dates | Fraud rate |",
            "|---|---:|---|---:|",
        ]
        for s in ["train", "valid", "test"]:
            lines.append(
                f"| {s} | {split_info['rows'][s]:,} | {split_info['dates'][s][0]} .. "
                f"{split_info['dates'][s][1]} | {100 * split_info['fraud_rate'][s]:.2f}% |"
            )
        lines.append(
            f"| live (tail of test) | {split_info['rows']['live']:,} | "
            f"{split_info['dates']['live'][0]} .. {split_info['dates']['live'][1]} | |"
        )
    return "\n".join(lines) + "\n"


def write_reports(res: dict[str, Any], extra: dict[str, Any] | None = None) -> None:
    rd = path("reports_dir")
    split_file = path("processed_dir") / "split.json"
    split_info = json.loads(split_file.read_text()) if split_file.exists() else None
    payload = {**res, **(extra or {}), "split": split_info}
    (rd / "metrics.json").write_text(json.dumps(payload, indent=2, default=float))
    (rd / "results.md").write_text(results_markdown(res, split_info))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    scores = pd.read_parquet(path("artifacts_dir") / "scores.parquet")
    models = [c for c in DISPLAY if c in scores.columns]
    res = evaluate_all(scores, models, CostParams.from_config(load_config()))
    write_reports(res)
    print((path("reports_dir") / "results.md").read_text())


if __name__ == "__main__":
    main()
