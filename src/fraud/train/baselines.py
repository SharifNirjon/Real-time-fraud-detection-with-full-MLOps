"""Baselines: a hand-written rules engine and logistic regression."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class RulesEngine:
    """Typical first-line fraud rules. Score = share of rules that fire (0, 1/3, 2/3, 1).

    1. high amount: above the train-set 95th percentile
    2. new device: the pseudo-user has history but has never used this device
    3. email mismatch: purchaser and recipient email domains both present and differ
    """

    amount_threshold: float = np.nan

    def fit(self, df: pd.DataFrame) -> RulesEngine:
        self.amount_threshold = float(df["TransactionAmt"].quantile(0.95))
        return self

    def flags(self, df: pd.DataFrame) -> pd.DataFrame:
        p, r = df["P_emaildomain"], df["R_emaildomain"]
        new_device = (
            df["uid_device_is_new"].fillna(0).astype(bool)
            if "uid_device_is_new" in df
            else pd.Series(False, index=df.index)
        )
        return pd.DataFrame(
            {
                "high_amount": df["TransactionAmt"] > self.amount_threshold,
                "new_device": new_device,
                "email_mismatch": p.notna() & r.notna() & (p != r),
            }
        )

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return self.flags(df).mean(axis=1).to_numpy(dtype=float)


def logistic_regression(seed: int = 42) -> Pipeline:
    """Median-impute + standardise + L2 logistic regression (all fitted on train only)."""
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(C=0.1, max_iter=2000, random_state=seed)),
        ]
    )
