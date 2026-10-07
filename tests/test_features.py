"""Leakage and training/serving-skew tests for the feature pipeline."""

import fakeredis
import numpy as np
import pandas as pd
import pytest

from fraud.features.definitions import WINDOWS, add_base_features, derive, raw_aggregate_names
from fraud.features.history import build_offline_features, compute_raw_aggregates
from fraud.features.online import OnlineFeatureStore
from fraud.features.pipeline import FeaturePipeline


def _brute_force(df: pd.DataFrame, i: int) -> dict[str, float]:
    """Recompute aggregates for row i from rows strictly before it."""
    past = df.iloc[:i]
    row = df.iloc[i]
    out = {}
    for key in ["card1", "uid"]:
        same = past[past[key] == row[key]]
        for w, secs in WINDOWS.items():
            m = same["TransactionDT"] >= row["TransactionDT"] - secs
            out[f"{key}_cnt_{w}"] = float(m.sum())
            out[f"{key}_amt_{w}"] = float(same.loc[m, "TransactionAmt"].sum())
        out[f"{key}_n_prev"] = float(len(same))
    return out


def test_velocity_uses_past_rows_only(transactions):
    base = add_base_features(transactions)
    agg = compute_raw_aggregates(base)
    rng = np.random.default_rng(1)
    for i in rng.choice(len(base), 60, replace=False):
        expected = _brute_force(base, int(i))
        for k, v in expected.items():
            assert agg.iloc[i][k] == pytest.approx(v, abs=1e-6), (i, k)


def test_current_transaction_excluded(transactions):
    base = add_base_features(transactions)
    agg = compute_raw_aggregates(base)
    first = ~base.duplicated(subset=["card1"])
    assert (agg.loc[first, "card1_n_prev"] == 0).all()
    assert (agg.loc[first, "card1_cnt_7d"] == 0).all()
    assert agg.loc[first, "card1_last_dt"].isna().all()


def test_no_feature_uses_future_rows(transactions):
    """Features of the first k rows must not change when later rows are removed."""
    full = build_offline_features(transactions)
    k = len(transactions) // 2
    trunc = build_offline_features(transactions.iloc[:k])
    cols = [c for c in trunc.columns if c not in transactions.columns]
    pd.testing.assert_frame_equal(full.iloc[:k][cols], trunc[cols], check_exact=False, rtol=1e-6)


def test_offline_online_parity(transactions):
    """Training/serving skew check: stream rows through Redis state, compare with offline."""
    df = transactions.iloc[:1500]
    offline = build_offline_features(df)
    store = OnlineFeatureStore(fakeredis.FakeRedis())
    base = add_base_features(df)
    online_rows = []
    for rec in base.to_dict("records"):
        online_rows.append(store.read(rec))
        store.write([rec])
    online_agg = pd.DataFrame(online_rows, index=df.index)[raw_aggregate_names()]
    online = pd.concat([base, derive(base, online_agg)], axis=1)

    offline_agg = compute_raw_aggregates(base)
    pd.testing.assert_frame_equal(offline_agg, online_agg, check_exact=False, rtol=1e-7, atol=1e-6)

    pipe = FeaturePipeline()
    pipe.fit_transform(offline, df["isFraud"])
    X_off = pipe.transform(offline)
    X_on = pipe.transform(online)
    pd.testing.assert_frame_equal(X_off, X_on, check_exact=False, rtol=1e-5, atol=1e-4)


def test_target_encoding_is_out_of_fold(transactions):
    df = build_offline_features(transactions)
    pipe = FeaturePipeline()
    X = pipe.fit_transform(df, df["isFraud"])
    # a category seen once must not be encoded with its own label
    keys = df["card1"].astype(float).map(repr)
    singles = keys.map(keys.value_counts()) == 1
    if singles.any():
        assert (X.loc[singles, "card1_te"] != df.loc[singles, "isFraud"]).all()
    # train-time (OOF) encodings differ from the full-train mapping used for serving
    assert not np.allclose(X["card1_te"], pipe.transform(df)["card1_te"])
