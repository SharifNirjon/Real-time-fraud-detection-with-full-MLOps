"""Build model-independent offline features for the whole time-sorted dataset.

Usage: python -m fraud.features.build
"""

from __future__ import annotations

import logging
import time

import pandas as pd

from fraud.config import path
from fraud.features.history import build_offline_features

log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.time()
    df = pd.read_parquet(path("processed_dir") / "transactions.parquet")
    feats = build_offline_features(df)
    out = path("processed_dir") / "features.parquet"
    feats.to_parquet(out, index=False)
    log.info(
        "features rows=%d cols=%d in %.1fs -> %s", len(feats), feats.shape[1], time.time() - t0, out
    )


if __name__ == "__main__":
    main()
