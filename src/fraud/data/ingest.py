"""Load the raw IEEE-CIS CSVs, join identity, downcast and write parquet.

Usage: python -m fraud.data.ingest [--sample]
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from fraud.config import load_config, path

log = logging.getLogger(__name__)

# Kept at full precision: cents and amount ratios are features.
NO_DOWNCAST = {"TransactionAmt"}


def downcast(df: pd.DataFrame) -> pd.DataFrame:
    """Shrink numeric dtypes losslessly for ints, float64->float32 for floats."""
    out = df.copy()
    for col in out.columns:
        if col in NO_DOWNCAST:
            continue
        s = out[col]
        if pd.api.types.is_integer_dtype(s):
            out[col] = pd.to_numeric(s, downcast="integer")
        elif pd.api.types.is_float_dtype(s):
            out[col] = s.astype(np.float32)
    return out


def card_sample_mask(card1: pd.Series, frac: float) -> pd.Series:
    """Deterministic sample of whole card1 groups.

    Sampling by card (not by row) keeps every card's full history, so velocity
    features on the sample behave like on the full data, and time order is kept.
    """
    h = (card1.astype(np.int64).to_numpy() * 2654435761) % 2**32
    return pd.Series(h / 2**32 < frac, index=card1.index)


def load_raw(raw_dir: str | None = None) -> pd.DataFrame:
    rd = path("raw_dir") if raw_dir is None else raw_dir
    tx = pd.read_csv(f"{rd}/train_transaction.csv")
    idn = pd.read_csv(f"{rd}/train_identity.csv")
    df = tx.merge(idn, on="TransactionID", how="left", validate="one_to_one")
    assert len(df) == len(tx), "identity join must not duplicate rows"
    return df


def prepare(df: pd.DataFrame, sample_frac: float | None = None) -> pd.DataFrame:
    if sample_frac is not None:
        df = df[card_sample_mask(df["card1"], sample_frac)]
    df = df.sort_values(["TransactionDT", "TransactionID"], kind="mergesort")
    return downcast(df.reset_index(drop=True))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true", help="keep ~20%% of cards for fast iteration")
    args = ap.parse_args()
    cfg = load_config()
    raw = load_raw()
    mem_before = raw.memory_usage(deep=True).sum() / 1e6
    df = prepare(raw, cfg["data"]["sample_frac"] if args.sample else None)
    out = path("processed_dir") / "transactions.parquet"
    df.to_parquet(out, index=False)
    log.info(
        "rows=%d fraud_rate=%.4f memory %.0fMB -> %.0fMB written=%s",
        len(df),
        df["isFraud"].mean(),
        mem_before,
        df.memory_usage(deep=True).sum() / 1e6,
        out,
    )


if __name__ == "__main__":
    main()
