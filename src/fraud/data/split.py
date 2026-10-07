"""Time-based train / valid / test split (never random).

Rows are ordered by (TransactionDT, TransactionID). Cut points are moved so
that rows sharing a timestamp never straddle two splits. The last
`live_frac_of_test` of the test period is the "live" stream replayed by the
simulator; everything before it is history used to warm the online store.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from fraud.config import load_config, path

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitBounds:
    """Half-open TransactionDT intervals: train=[min, valid_start), ..."""

    valid_start: int
    test_start: int
    live_start: int

    def assign(self, dt: pd.Series | np.ndarray) -> np.ndarray:
        dt = np.asarray(dt)
        return np.select(
            [dt < self.valid_start, dt < self.test_start],
            ["train", "valid"],
            default="test",
        )


def _cut(dt_sorted: np.ndarray, frac: float) -> int:
    """Timestamp at position frac, moved forward to a timestamp boundary."""
    i = int(len(dt_sorted) * frac)
    i = min(max(i, 1), len(dt_sorted) - 1)
    return int(dt_sorted[i])


def compute_bounds(
    dt: pd.Series, train_frac: float, valid_frac: float, live_frac_of_test: float
) -> SplitBounds:
    s = np.sort(np.asarray(dt))
    valid_start = _cut(s, train_frac)
    test_start = _cut(s, train_frac + valid_frac)
    test = s[s >= test_start]
    live_start = _cut(test, 1 - live_frac_of_test)
    assert valid_start < test_start < live_start, "degenerate split"
    return SplitBounds(valid_start, test_start, live_start)


def bounds_from_config(dt: pd.Series) -> SplitBounds:
    c = load_config()["split"]
    return compute_bounds(dt, c["train_frac"], c["valid_frac"], c["live_frac_of_test"])


def dt_to_date(dt: int, reference_date: str) -> str:
    return str((pd.Timestamp(reference_date) + pd.Timedelta(seconds=int(dt))).date())


def save_bounds(b: SplitBounds, df: pd.DataFrame) -> dict:
    ref = load_config()["data"]["reference_date"]
    split = b.assign(df["TransactionDT"])
    info: dict = {"bounds": asdict(b), "dates": {}, "rows": {}, "fraud_rate": {}}
    for name in ["train", "valid", "test"]:
        m = split == name
        dts = df.loc[m, "TransactionDT"]
        info["dates"][name] = [dt_to_date(dts.min(), ref), dt_to_date(dts.max(), ref)]
        info["rows"][name] = int(m.sum())
        info["fraud_rate"][name] = round(float(df.loc[m, "isFraud"].mean()), 5)
    live = df["TransactionDT"] >= b.live_start
    info["rows"]["live"] = int(live.sum())
    info["dates"]["live"] = [
        dt_to_date(b.live_start, ref),
        dt_to_date(df.loc[live, "TransactionDT"].max(), ref),
    ]
    with open(path("processed_dir") / "split.json", "w") as f:
        json.dump(info, f, indent=2)
    return info


def load_bounds() -> SplitBounds:
    with open(path("processed_dir") / "split.json") as f:
        return SplitBounds(**json.load(f)["bounds"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = pd.read_parquet(path("processed_dir") / "transactions.parquet")
    b = bounds_from_config(df["TransactionDT"])
    info = save_bounds(b, df)
    live = df[df["TransactionDT"] >= b.live_start]
    live.to_parquet(path("processed_dir") / "live.parquet", index=False)
    log.info("split: %s", json.dumps(info))


if __name__ == "__main__":
    main()
