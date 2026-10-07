"""Online feature store backed by Redis.

Keeps, per card1 and per pseudo-user (uid), the state needed to produce the
same raw aggregates as fraud.features.history:
  fs:{key}:{value}:z    sorted set of "txid:amount" scored by TransactionDT (last 7d)
  fs:{key}:{value}:h    hash n, s, ss, last (expanding count / sum / sum of squares / last dt)
  fs:uid:{value}:device set of devices seen,  fs:uid:{value}:email  set of emails seen
Transactions must arrive in (TransactionDT, TransactionID) order, as in the
offline pipeline; `read` never sees the transaction being scored because
`write` happens after scoring.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from fraud.features.definitions import (
    DISTINCT_COLS,
    MAX_WINDOW,
    VELOCITY_KEYS,
    WINDOWS,
    raw_aggregate_names,
)

PREFIX = "fs"


def _k(key: str, value: Any, suffix: str) -> str:
    return f"{PREFIX}:{key}:{value}:{suffix}"


def _present(v: Any) -> bool:
    return v is not None and not (isinstance(v, float) and np.isnan(v))


class OnlineFeatureStore:
    def __init__(self, redis_client: Any) -> None:
        self.r = redis_client

    def read(self, base_row: dict[str, Any]) -> dict[str, float]:
        """Raw aggregates for one transaction (base features incl. uid already added)."""
        dt = float(base_row["TransactionDT"])
        p = self.r.pipeline(transaction=False)
        for k in VELOCITY_KEYS:
            p.zrangebyscore(_k(k, base_row[k], "z"), dt - MAX_WINDOW, "+inf", withscores=True)
            p.hmget(_k(k, base_row[k], "h"), ["n", "s", "ss", "last"])
        for d, col in DISTINCT_COLS.items():
            key = _k("uid", base_row["uid"], d)
            p.scard(key)
            p.sismember(key, str(base_row[col]) if _present(base_row[col]) else "")
        res = p.execute()
        agg: dict[str, float] = {}
        i = 0
        for k in VELOCITY_KEYS:
            members, h = res[i], res[i + 1]
            i += 2
            scores = np.array([s for _, s in members], dtype="float64")
            amounts = np.array(
                [
                    float((m.decode() if isinstance(m, bytes) else m).split(":", 1)[1])
                    for m, _ in members
                ],
                dtype="float64",
            )
            for w, secs in WINDOWS.items():
                m = scores >= dt - secs
                agg[f"{k}_cnt_{w}"] = float(m.sum())
                agg[f"{k}_amt_{w}"] = float(amounts[m].sum())
            n, s, ss, last = (float(x) if x is not None else None for x in h)
            agg[f"{k}_n_prev"] = n or 0.0
            agg[f"{k}_last_dt"] = last if last is not None else np.nan
            if k == "uid":
                agg["uid_amt_sum_prev"] = s or 0.0
                agg["uid_amt_sumsq_prev"] = ss or 0.0
        for d, col in DISTINCT_COLS.items():
            n_distinct, seen = res[i], res[i + 1]
            i += 2
            agg[f"uid_n_{d}s_prev"] = float(n_distinct)
            agg[f"uid_{d}_seen"] = float(bool(seen)) if _present(base_row[col]) else np.nan
        return agg

    def read_frame(self, base: pd.DataFrame) -> pd.DataFrame:
        rows = [self.read(r) for r in base.to_dict("records")]
        return pd.DataFrame(rows, index=base.index)[raw_aggregate_names()]

    def write(self, rows: Iterable[dict[str, Any]], pipeline_size: int = 2000) -> int:
        """Fold transactions (in time order) into the state."""
        p = self.r.pipeline(transaction=False)
        count = 0
        for row in rows:
            dt = float(row["TransactionDT"])
            amt = float(row["TransactionAmt"])
            for k in VELOCITY_KEYS:
                z = _k(k, row[k], "z")
                h = _k(k, row[k], "h")
                p.zadd(z, {f"{row['TransactionID']}:{amt!r}": dt})
                p.zremrangebyscore(z, "-inf", f"({dt - MAX_WINDOW}")
                p.hincrby(h, "n", 1)
                p.hincrbyfloat(h, "s", amt)
                p.hincrbyfloat(h, "ss", amt * amt)
                p.hset(h, "last", dt)
            for d, col in DISTINCT_COLS.items():
                if _present(row[col]):
                    p.sadd(_k("uid", row["uid"], d), str(row[col]))
            count += 1
            if count % pipeline_size == 0:
                p.execute()
        p.execute()
        return count

    def flush(self) -> None:
        cursor = 0
        while True:
            cursor, keys = self.r.scan(cursor, match=f"{PREFIX}:*", count=5000)
            if keys:
                self.r.delete(*keys)
            if cursor == 0:
                break
