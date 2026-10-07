"""Single source of truth for history (velocity) features.

Both the offline builder (fraud.features.history, vectorised over a whole
DataFrame) and the online store (fraud.features.online, Redis state for one
transaction) produce the same *raw aggregates* defined here, then call the
same `derive()` to turn them into model features. "Past" means every
transaction processed earlier in (TransactionDT, TransactionID) order; the
current transaction is never included.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WINDOWS: dict[str, int] = {"1h": 3600, "24h": 86400, "7d": 7 * 86400}
MAX_WINDOW = max(WINDOWS.values())
VELOCITY_KEYS: tuple[str, ...] = ("card1", "uid")
DISTINCT_COLS: dict[str, str] = {"device": "DeviceInfo", "email": "P_emaildomain"}
SECONDS_PER_DAY = 86400


def raw_aggregate_names() -> list[str]:
    names: list[str] = []
    for k in VELOCITY_KEYS:
        for w in WINDOWS:
            names += [f"{k}_cnt_{w}", f"{k}_amt_{w}"]
        names += [f"{k}_n_prev", f"{k}_last_dt"]
    names += ["uid_amt_sum_prev", "uid_amt_sumsq_prev"]
    for d in DISTINCT_COLS:
        names += [f"uid_n_{d}s_prev", f"uid_{d}_seen"]
    return names


def _key_part(s: pd.Series) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce").astype("float64")
    out = v.round().astype("Int64").astype(str)
    return out.where(v.notna(), "na")


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Stateless row-level features (work on 1 row or 1M rows alike).

    uid is an APPROXIMATE pseudo-user: card1 + addr1 + card-start day
    (day - D1, where D1 is "days since card first seen"). Different people
    can collide and one person can split across uids.
    """
    out = df.copy()
    dt = out["TransactionDT"].astype("int64")
    day = dt // SECONDS_PER_DAY
    amt = out["TransactionAmt"].astype("float64")
    start_day = pd.Series(day, index=out.index) - pd.to_numeric(out["D1"]).astype("float64")
    out["uid"] = (
        _key_part(out["card1"])
        + "_"
        + _key_part(out["addr1"])
        + "_"
        + _key_part(np.floor(start_day))
    )
    out["hour"] = ((dt // 3600) % 24).astype("float32")
    out["dow"] = (day % 7).astype("float32")
    out["log_amt"] = np.log1p(amt)
    out["amt_cents"] = ((amt - np.floor(amt)) * 100).round(2)
    p, r = out["P_emaildomain"], out["R_emaildomain"]
    out["email_match"] = np.where(p.isna() | r.isna(), np.nan, (p == r).astype(float))
    return out


def derive(df: pd.DataFrame, agg: pd.DataFrame) -> pd.DataFrame:
    """Turn raw aggregates into model features. Shared by offline and online paths."""
    amt = df["TransactionAmt"].to_numpy(dtype="float64")
    dt = df["TransactionDT"].to_numpy(dtype="float64")
    f: dict[str, np.ndarray] = {}
    for k in VELOCITY_KEYS:
        for w in WINDOWS:
            f[f"{k}_cnt_{w}"] = agg[f"{k}_cnt_{w}"].to_numpy(dtype="float64")
            f[f"{k}_amt_{w}"] = agg[f"{k}_amt_{w}"].to_numpy(dtype="float64")
        f[f"{k}_n_prev"] = agg[f"{k}_n_prev"].to_numpy(dtype="float64")
        f[f"{k}_secs_since_last"] = dt - agg[f"{k}_last_dt"].to_numpy(dtype="float64")
    n = agg["uid_n_prev"].to_numpy(dtype="float64")
    s = agg["uid_amt_sum_prev"].to_numpy(dtype="float64")
    ss = agg["uid_amt_sumsq_prev"].to_numpy(dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(n >= 1, s / n, np.nan)
        var = np.where(n >= 2, (ss - s * s / np.maximum(n, 1)) / np.maximum(n - 1, 1), np.nan)
        var = np.where(var < 1e-6, 0.0, var)  # float noise from the sum-of-squares formula
        std = np.sqrt(var)
        f["uid_amt_mean_prev"] = mean
        f["uid_amt_std_prev"] = std
        f["uid_amt_to_mean"] = amt / mean
        f["uid_amt_zscore"] = (amt - mean) / (
            std + 1.0
        )  # +$1 keeps it finite for constant spenders
    for d, col in DISTINCT_COLS.items():
        seen = agg[f"uid_{d}_seen"].to_numpy(dtype="float64")
        f[f"uid_n_{d}s_prev"] = agg[f"uid_n_{d}s_prev"].to_numpy(dtype="float64")
        present = df[col].notna().to_numpy()
        f[f"uid_{d}_is_new"] = np.where(present & (n > 0), (seen == 0).astype(float), np.nan)
    return pd.DataFrame(f, index=df.index)


def engineered_names() -> list[str]:
    base = ["hour", "dow", "log_amt", "amt_cents", "email_match"]
    dummy = pd.DataFrame(
        {
            "TransactionAmt": [1.0],
            "TransactionDT": [0],
            "DeviceInfo": [None],
            "P_emaildomain": [None],
        }
    )
    agg = pd.DataFrame({n: [np.nan] for n in raw_aggregate_names()})
    return base + list(derive(dummy, agg).columns)
