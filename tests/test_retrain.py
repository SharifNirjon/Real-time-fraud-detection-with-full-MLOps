"""Retraining logic end to end on synthetic data (no MLflow, no API)."""

import copy
import json

import numpy as np
import pandas as pd
import pytest

from fixtures.synthetic import make_transactions
from fraud.config import load_config
from fraud.data.split import compute_bounds
from fraud.evaluate.cost import CostParams
from fraud.pipelines import retrain as R
from fraud.serve.prediction_log import ParquetLog


@pytest.fixture()
def env(tmp_path, monkeypatch, bundle):
    df = make_transactions(6000, seed=3)
    proc = tmp_path / "processed"
    proc.mkdir()
    monkeypatch.setenv("FRAUD_PROCESSED_DIR", str(proc))
    b = compute_bounds(df["TransactionDT"], 0.6, 0.2, 0.9)
    df.to_parquet(proc / "transactions.parquet", index=False)
    (proc / "split.json").write_text(json.dumps({"bounds": b.__dict__}))
    live = df[df["TransactionDT"] >= b.live_start]
    preds, labels = ParquetLog(tmp_path / "pred", "pred"), ParquetLog(tmp_path / "labels", "labels")
    raw_cols = [c for c in df.columns if c != "isFraud"]
    preds.append(
        [
            {
                "TransactionID": int(r["TransactionID"]),
                "TransactionDT": int(r["TransactionDT"]),
                "logged_at": pd.Timestamp.now(tz="UTC"),
                "score": 0.1,
                "decision": "approve",
                "raw": pd.Series({k: r[k] for k in raw_cols}).to_json(),
            }
            for r in live.to_dict("records")
        ]
    )
    labelled = live.iloc[: int(len(live) * 0.9)]  # the most recent 10% has no label yet
    labels.append(
        [
            {"TransactionID": int(t), "isFraud": int(y)}
            for t, y in zip(labelled["TransactionID"], labelled["isFraud"], strict=True)
        ]
    )
    preds.close()
    labels.close()
    champion = copy.deepcopy(bundle)
    champion.metadata |= {
        "params": {
            "objective": "binary",
            "learning_rate": 0.1,
            "num_leaves": 15,
            "min_child_samples": 10,
            "verbosity": -1,
            "deterministic": True,
            "seed": 0,
        },
        "monitor_columns": ["TransactionAmt"],
    }
    return tmp_path, df, b, live, labelled, champion


def test_build_frame_and_challenger(env):
    tmp, df, b, live, labelled, champion = env
    feats, is_live = R.build_frame(tmp / "pred", tmp / "labels")
    assert len(feats) == len(df)
    assert is_live.sum() == len(live)
    # unlabelled live rows stay as history only
    assert feats.loc[is_live, "isFraud"].isna().sum() == len(live) - len(labelled)
    # features of live rows equal those built from the original data (same history)
    from fraud.features.history import build_offline_features

    ref = build_offline_features(df).set_index("TransactionID")
    got = feats.set_index("TransactionID")
    for c in ["card1_cnt_24h", "uid_n_prev", "uid_amt_zscore"]:
        np.testing.assert_allclose(
            got.loc[live["TransactionID"], c],
            ref.loc[live["TransactionID"], c],
            rtol=1e-6,
            equal_nan=True,
        )

    cfg = copy.deepcopy(load_config())
    cfg["train"] |= {"num_boost_round": 60, "early_stopping_rounds": 10}
    cfg["monitor"]["reference_rows"] = 200
    cfg["retrain"]["latency_rows"] = 20
    challenger, info, ref_df = R.train_challenger(feats, is_live, champion, cfg)
    hold = info["holdout"]
    rt = challenger.metadata["retrain"]
    # fresh holdout: strictly after everything the challenger trained on
    pool_dt = feats.loc[
        feats["isFraud"].notna() & (feats["TransactionDT"] < rt["holdout_start_dt"]),
        "TransactionDT",
    ]
    assert pool_dt.max() < rt["holdout_start_dt"] <= hold["TransactionDT"].min()
    assert rt["pool_rows"] == int(
        (feats["isFraud"].notna() & (feats["TransactionDT"] < rt["holdout_start_dt"])).sum()
    )
    assert hold["isFraud"].notna().all()
    assert {"score", "isFraud"} <= set(ref_df.columns)
    m = R.evaluate_on(challenger, hold, CostParams())
    assert 0 <= m["pr_auc"] <= 1 and m["cost"] >= 0
