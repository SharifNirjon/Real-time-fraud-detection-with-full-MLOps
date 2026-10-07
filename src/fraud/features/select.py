"""Select V-columns on the TRAIN split and write them to configs/features.yaml.

Method (documented in the YAML):
  1. drop V-columns with a train null rate above `null_threshold` (mostly null);
  2. order the rest by null rate, then column index, and greedily keep a column
     only if its absolute Pearson correlation with every already-kept column is
     below `corr_threshold` (pairwise-complete rows, on a train sample).
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
import yaml

from fraud.config import CONFIG_DIR, _load_yaml, load_config, path
from fraud.data.split import load_bounds

log = logging.getLogger(__name__)


def select_v_columns(
    train: pd.DataFrame,
    null_threshold: float,
    corr_threshold: float,
    seed: int,
    max_rows: int = 100_000,
) -> tuple[list[str], dict]:
    v_cols = [c for c in train.columns if c.startswith("V") and c[1:].isdigit()]
    null_rate = train[v_cols].isna().mean()
    candidates = [c for c in v_cols if null_rate[c] <= null_threshold]
    candidates.sort(key=lambda c: (round(null_rate[c], 3), int(c[1:])))
    sample = train[candidates]
    if len(sample) > max_rows:
        sample = sample.sample(max_rows, random_state=seed)
    corr = sample.astype("float64").corr().abs().fillna(0.0).to_numpy()
    idx = {c: i for i, c in enumerate(candidates)}
    kept: list[str] = []
    for c in candidates:
        if not kept or corr[idx[c], [idx[k] for k in kept]].max() < corr_threshold:
            kept.append(c)
    kept.sort(key=lambda c: int(c[1:]))
    info = {
        "method": "null-rate filter then greedy |pearson| pruning, fitted on train split",
        "null_threshold": null_threshold,
        "corr_threshold": corr_threshold,
        "n_total": len(v_cols),
        "n_after_null_filter": len(candidates),
        "n_kept": len(kept),
    }
    return kept, info


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--null-threshold", type=float, default=0.85)
    ap.add_argument("--corr-threshold", type=float, default=0.75)
    args = ap.parse_args()
    df = pd.read_parquet(path("processed_dir") / "transactions.parquet")
    train = df[load_bounds().assign(df["TransactionDT"]) == "train"]
    kept, info = select_v_columns(
        train, args.null_threshold, args.corr_threshold, load_config()["seed"]
    )
    fpath = CONFIG_DIR / "features.yaml"
    text = fpath.read_text()
    head = text[: text.index("v_selection:")]
    tail = yaml.safe_dump({"v_selection": info, "v_columns": kept}, sort_keys=False)
    fpath.write_text(head + tail)
    _load_yaml.cache_clear()
    log.info("V-columns: %s", info)
    np.testing.assert_equal(len(kept), info["n_kept"])


if __name__ == "__main__":
    main()
