"""Small synthetic IEEE-CIS-shaped data, so tests and CI never need Kaggle."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud.config import load_feature_config


def make_transactions(n: int = 3000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    fc = load_feature_config()
    n_cards = max(n // 15, 10)
    card1 = rng.integers(1000, 1000 + n_cards, n)
    # timestamps with deliberate ties, sorted
    dt = np.sort(86400 + rng.integers(0, 120 * 86400, n) // 60 * 60)
    tid = 2_987_000 + np.arange(n)
    amt = np.round(rng.lognormal(3.8, 1.0, n), 3)
    devices = np.array(["Windows", "iOS Device", "MacOS", "SM-G930V", None], dtype=object)
    emails = np.array(
        ["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", None], dtype=object
    )
    device = devices[rng.integers(0, 5, n)]
    p_email = emails[rng.integers(0, 5, n)]
    r_email = emails[rng.integers(0, 5, n)]
    logit = (
        -4 + 0.6 * (amt > 150) + 1.0 * (device == "SM-G930V") + 0.5 * (p_email == "anonymous.com")
    )
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(np.int8)
    df = pd.DataFrame(
        {
            "TransactionID": tid,
            "isFraud": y,
            "TransactionDT": dt,
            "TransactionAmt": amt,
            "ProductCD": rng.choice(["W", "C", "H", "R", "S"], n),
            "card1": card1.astype(np.int16),
            "card4": rng.choice(["visa", "mastercard", "discover"], n),
            "card6": rng.choice(["debit", "credit"], n),
            "addr1": np.where(rng.random(n) < 0.1, np.nan, (card1 % 7) + 100).astype(np.float32),
            "D1": np.where(rng.random(n) < 0.05, np.nan, 0).astype(np.float32),
            "P_emaildomain": p_email,
            "R_emaildomain": r_email,
            "DeviceInfo": device,
            "DeviceType": rng.choice(["desktop", "mobile", None], n),
        }
    )
    # D1 = days since card start; make the start day stable per card so uids are meaningful
    start = (card1 % 30).astype(np.float32)
    df["D1"] = np.where(df["D1"].isna(), np.nan, np.maximum(dt // 86400 - start, 0)).astype(
        np.float32
    )
    extra: dict[str, np.ndarray] = {}
    for c in list(dict.fromkeys(fc["numeric"] + fc["v_columns"])):
        if c not in df:
            extra[c] = np.where(rng.random(n) < 0.3, np.nan, rng.normal(size=n)).astype(np.float32)
    for c, mapping in fc["value_maps"].items():
        if c not in df:
            keys = list(mapping) + [None]
            extra[c] = np.array(keys, dtype=object)[rng.integers(0, len(keys), n)]
    for c in fc["frequency_encode"] + fc["target_encode"]:
        if c not in df and c not in extra:
            extra[c] = rng.choice(["a", "b", "c"], n)
    df = pd.concat([df, pd.DataFrame(extra)], axis=1)
    df = df.sort_values(["TransactionDT", "TransactionID"], kind="mergesort")
    return df.reset_index(drop=True)
