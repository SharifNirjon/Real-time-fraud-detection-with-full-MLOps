"""Categorical encoders, fitted on the TRAIN split only."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

NAN_KEY = "__nan__"


def category_keys(s: pd.Series) -> pd.Series:
    """Canonical string key so 1234 (int), 1234.0 (float) and '1234' agree online/offline."""
    if pd.api.types.is_numeric_dtype(s):
        v = s.astype("float64")
        return v.map(repr).where(v.notna(), NAN_KEY)
    num = pd.to_numeric(s, errors="coerce")
    if s.notna().any() and num[s.notna()].notna().all():
        return category_keys(num)
    return s.astype(object).where(s.notna(), NAN_KEY).astype(str)


@dataclass
class FrequencyEncoder:
    """Share of train rows per category; unseen categories map to 0."""

    maps: dict[str, dict[str, float]] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame, cols: list[str]) -> FrequencyEncoder:
        for c in cols:
            self.maps[c] = category_keys(df[c]).value_counts(normalize=True).to_dict()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                f"{c}_freq": category_keys(df[c]).map(m).fillna(0.0).astype("float32")
                for c, m in self.maps.items()
            },
            index=df.index,
        )


@dataclass
class TargetEncoder:
    """Smoothed mean target per category.

    `fit_transform` returns OUT-OF-FOLD encodings for the train rows (a row's
    own label never feeds its encoding); `transform` uses the mapping fitted on
    all of train, for valid / test / serving.
    """

    smoothing: float = 100.0
    n_splits: int = 5
    seed: int = 42
    prior: float = 0.0
    maps: dict[str, dict[str, float]] = field(default_factory=dict)

    def _mapping(self, keys: pd.Series, y: pd.Series, prior: float) -> dict[str, float]:
        g = y.groupby(keys.to_numpy()).agg(["sum", "count"])
        enc = (g["sum"] + self.smoothing * prior) / (g["count"] + self.smoothing)
        return enc.to_dict()

    def fit_transform(self, df: pd.DataFrame, y: pd.Series, cols: list[str]) -> pd.DataFrame:
        y = pd.Series(np.asarray(y, dtype=float), index=df.index)
        self.prior = float(y.mean())
        from sklearn.model_selection import KFold  # lazy: not needed at serving time

        out = {}
        kf = KFold(self.n_splits, shuffle=True, random_state=self.seed)
        for c in cols:
            keys = category_keys(df[c])
            oof = np.full(len(df), self.prior)
            for tr, va in kf.split(df):
                m = self._mapping(keys.iloc[tr], y.iloc[tr], float(y.iloc[tr].mean()))
                oof[va] = keys.iloc[va].map(m).fillna(self.prior).to_numpy()
            out[f"{c}_te"] = oof.astype("float32")
            self.maps[c] = self._mapping(keys, y, self.prior)
        return pd.DataFrame(out, index=df.index)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                f"{c}_te": category_keys(df[c]).map(m).fillna(self.prior).astype("float32")
                for c, m in self.maps.items()
            },
            index=df.index,
        )
