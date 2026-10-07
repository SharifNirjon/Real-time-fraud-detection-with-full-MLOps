"""Warm the Redis online store with all history before the live stream starts.

Replays the same `OnlineFeatureStore.write` used at serving time, in time order,
so the state the API sees at `live_start` equals what the offline pipeline sees.

Usage: python -m fraud.serve.warm [--redis-url URL]
"""

from __future__ import annotations

import argparse
import logging
import os
import time

import pandas as pd
import redis

from fraud.config import path
from fraud.data.split import load_bounds
from fraud.features.definitions import DISTINCT_COLS, add_base_features
from fraud.features.online import OnlineFeatureStore

log = logging.getLogger(__name__)

COLS = [
    "TransactionID",
    "TransactionDT",
    "TransactionAmt",
    "card1",
    "addr1",
    "D1",
    "P_emaildomain",
    "R_emaildomain",
    *DISTINCT_COLS.values(),
]


def warm(client: redis.Redis, until_dt: int) -> int:
    df = pd.read_parquet(
        path("processed_dir") / "transactions.parquet", columns=list(dict.fromkeys(COLS))
    )
    df = df[df["TransactionDT"] < until_dt]
    base = add_base_features(df)
    store = OnlineFeatureStore(client)
    store.flush()
    keep = [
        "TransactionID",
        "TransactionDT",
        "TransactionAmt",
        "card1",
        "uid",
        *DISTINCT_COLS.values(),
    ]
    return store.write(base[keep].to_dict("records"), pipeline_size=5000)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--redis-url", default=os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
    args = ap.parse_args()
    t0 = time.time()
    client = redis.Redis.from_url(args.redis_url)
    n = warm(client, load_bounds().live_start)
    log.info(
        "warmed online store with %d transactions in %.0fs (keys=%d)",
        n,
        time.time() - t0,
        client.dbsize(),
    )


if __name__ == "__main__":
    main()
