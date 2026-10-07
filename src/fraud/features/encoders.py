"""Categorical encoders, fitted on the TRAIN split only.

Lookups use plain dicts on canonical string keys, which is fast for a single
API request and fine for the offline batch, so both paths run the same code.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

NAN_KEY = "__nan__"


def scalar_key(v: object) -> str:
    """Canonical key: numbers -> repr(float) (1234, 1234.0 -> '1234.0'), missing -> NAN_KEY."""
    if v is None or v is pd.NA:
        return NAN_KEY
    if isinstance(v, int | float | np.integer | np.floating) and not isinstance(v, bool):
        f = float(v)
        return NAN_KEY if math.isnan(f) else repr(f)
    return str(v)


def category_keys(s: pd.Series) -> list[str]:
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        return [NAN_KEY if x != x else repr(x) for x in s.to_numpy(dtype="float64").tolist()]
    return [scalar_key(v) for v in s.tolist()]


def lookup(s: pd.Series, mapping: dict[str, float], default: float) -> np.ndarray:
    return np.fromiter(
        (mapping.get(k, default) for k in category_keys(s)), dtype=np.float32, count=len(s)
    )


@dataclass
class FrequencyEncoder:
    """Share of train rows per category; unseen categories map to 0."""

    maps: dict[str, dict[str, float]] = field(default_factory=dict)

    def fit(self, df: pd.DataFrame, cols: list[str]) -> FrequencyEncoder:
        for c in cols:
            self.maps[c] = pd.Series(category_keys(df[c])).value_counts(normalize=True).to_dict()
        return self

    @property
    def names(self) -> list[str]:
        return [f"{c}_freq" for c in self.maps]

    def transform_block(self, df: pd.DataFrame) -> np.ndarray:
        return np.column_stack([lookup(df[c], m, 0.0) for c, m in self.maps.items()])


@dataclass
class TargetEncoder:
    """Smoothed mean target per category.

    `fit_transform` returns OUT-OF-FOLD encodings for the train rows (a row's
    own label never feeds its encoding); `transform_block` uses the mapping
    fitted on all of train, for valid / test / serving.
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

    @property
    def names(self) -> list[str]:
        return [f"{c}_te" for c in self.maps]

    def fit_transform_block(self, df: pd.DataFrame, y: pd.Series, cols: list[str]) -> np.ndarray:
        from sklearn.model_selection import KFold  # lazy: not needed at serving time

        y = pd.Series(np.asarray(y, dtype=float), index=df.index)
        self.prior = float(y.mean())
        kf = KFold(self.n_splits, shuffle=True, random_state=self.seed)
        out = []
        for c in cols:
            keys = pd.Series(category_keys(df[c]), index=df.index)
            oof = np.full(len(df), self.prior)
            for tr, va in kf.split(df):
                m = self._mapping(keys.iloc[tr], y.iloc[tr], float(y.iloc[tr].mean()))
                oof[va] = keys.iloc[va].map(m).fillna(self.prior).to_numpy()
            out.append(oof.astype(np.float32))
            self.maps[c] = self._mapping(keys, y, self.prior)
        return np.column_stack(out)

    def transform_block(self, df: pd.DataFrame) -> np.ndarray:
        return np.column_stack([lookup(df[c], m, self.prior) for c, m in self.maps.items()])
