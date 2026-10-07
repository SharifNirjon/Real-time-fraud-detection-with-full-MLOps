"""Offline, vectorised computation of the raw history aggregates.

Every value for row i uses only rows that come before i in
(TransactionDT, TransactionID) order: windows via searchsorted on prefix
sums (current row excluded), expanding stats via exclusive cumulative sums.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud.features.definitions import (
    DISTINCT_COLS,
    MAX_WINDOW,
    VELOCITY_KEYS,
    WINDOWS,
    add_base_features,
    derive,
    raw_aggregate_names,
)


def assert_time_sorted(df: pd.DataFrame) -> None:
    dt = df["TransactionDT"].to_numpy()
    tid = df["TransactionID"].to_numpy()
    ok = (np.diff(dt) > 0) | ((np.diff(dt) == 0) & (np.diff(tid) > 0))
    assert ok.all(), "rows must be sorted by (TransactionDT, TransactionID)"


def _key_aggregates(key: pd.Series, dt: np.ndarray, amt: np.ndarray) -> dict[str, np.ndarray]:
    n = len(key)
    codes = pd.factorize(key, use_na_sentinel=True)[0].astype(np.int64)
    pos_orig = np.arange(n)
    order = np.lexsort((pos_orig, codes))  # group by key, keep time order inside the group
    c_s, dt_s, amt_s = codes[order], dt[order] - dt.min(), amt[order]
    span = int(dt_s.max()) + MAX_WINDOW + 1
    composite = c_s * span + dt_s  # non-decreasing: groups are disjoint ranges
    pos = np.arange(n)
    grp_start = np.searchsorted(c_s, c_s, side="left")
    csum = np.concatenate([[0.0], np.cumsum(amt_s)])
    csq = np.concatenate([[0.0], np.cumsum(amt_s * amt_s)])
    res: dict[str, np.ndarray] = {}
    for w, secs in WINDOWS.items():
        lo = np.searchsorted(composite, composite - secs, side="left")  # first row with dt >= t-w
        res[f"cnt_{w}"] = (pos - lo).astype(float)
        res[f"amt_{w}"] = csum[pos] - csum[lo]
    n_prev = pos - grp_start
    res["n_prev"] = n_prev.astype(float)
    res["last_dt"] = np.where(n_prev > 0, dt[order][np.maximum(pos - 1, 0)], np.nan)
    res["amt_sum_prev"] = csum[pos] - csum[grp_start]
    res["amt_sumsq_prev"] = csq[pos] - csq[grp_start]
    out = {}
    for name, v in res.items():
        a = np.empty(n)
        a[order] = v
        a[codes < 0] = np.nan  # missing key: no history
        out[name] = a
    return out


def compute_raw_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """df must contain base features (uid) and be time sorted."""
    assert_time_sorted(df)
    dt = df["TransactionDT"].to_numpy(dtype=np.int64)
    amt = df["TransactionAmt"].to_numpy(dtype="float64")
    agg: dict[str, np.ndarray] = {}
    for k in VELOCITY_KEYS:
        r = _key_aggregates(df[k], dt, amt)
        for w in WINDOWS:
            agg[f"{k}_cnt_{w}"] = r[f"cnt_{w}"]
            agg[f"{k}_amt_{w}"] = r[f"amt_{w}"]
        agg[f"{k}_n_prev"] = r["n_prev"]
        agg[f"{k}_last_dt"] = r["last_dt"]
        if k == "uid":
            agg["uid_amt_sum_prev"] = r["amt_sum_prev"]
            agg["uid_amt_sumsq_prev"] = r["amt_sumsq_prev"]
    uid = df["uid"]
    for d, col in DISTINCT_COLS.items():
        v = df[col]
        present = v.notna() & uid.notna()
        seen_before = df.duplicated(subset=["uid", col], keep="first") & present
        first = (present & ~seen_before).astype(np.int64)
        n_distinct_prev = first.groupby(uid.to_numpy()).cumsum() - first
        agg[f"uid_n_{d}s_prev"] = n_distinct_prev.to_numpy(dtype=float)
        agg[f"uid_{d}_seen"] = np.where(present, seen_before.astype(float), np.nan)
    out = pd.DataFrame(agg, index=df.index)
    return out[raw_aggregate_names()]


def build_offline_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw columns + base features + history features for a time-sorted frame."""
    base = add_base_features(df)
    agg = compute_raw_aggregates(base)
    return pd.concat([base, derive(base, agg)], axis=1)
