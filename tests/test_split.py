import numpy as np
import pandas as pd
import pytest

from fraud.data.ingest import card_sample_mask, downcast
from fraud.data.split import compute_bounds


def test_time_split_no_overlap_and_ordered(transactions):
    b = compute_bounds(transactions["TransactionDT"], 0.7, 0.15, 0.5)
    split = pd.Series(b.assign(transactions["TransactionDT"]), index=transactions.index)
    dt = transactions["TransactionDT"]
    assert dt[split == "train"].max() < dt[split == "valid"].min()
    assert dt[split == "valid"].max() < dt[split == "test"].min()
    assert b.test_start < b.live_start <= dt.max()
    # each timestamp belongs to exactly one split (ties never straddle)
    assert (transactions.assign(s=split).groupby("TransactionDT")["s"].nunique() == 1).all()
    shares = split.value_counts(normalize=True)
    assert shares["train"] == pytest.approx(0.70, abs=0.02)
    assert shares["valid"] == pytest.approx(0.15, abs=0.02)


def test_sample_keeps_whole_cards(transactions):
    m = card_sample_mask(transactions["card1"], 0.3)
    per_card = m.groupby(transactions["card1"]).nunique()
    assert (per_card == 1).all()  # a card is either fully in or fully out
    assert 0.1 < m.mean() < 0.5


def test_downcast_keeps_amount_precision():
    df = pd.DataFrame(
        {"TransactionAmt": [49.95], "C1": [1.0], "card1": np.array([5], dtype=np.int64)}
    )
    out = downcast(df)
    assert out["TransactionAmt"].dtype == np.float64 and out["TransactionAmt"][0] == 49.95
    assert out["C1"].dtype == np.float32 and out["card1"].dtype == np.int8
