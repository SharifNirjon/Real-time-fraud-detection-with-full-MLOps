"""Model-specific part of the features: value maps + fitted encoders -> model matrix.

Input is the output of history.build_offline_features (offline) or the same
columns assembled by the API (online). The fitted pipeline is pickled inside
the model bundle, so training and serving apply identical transforms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from fraud.config import load_feature_config
from fraud.features.definitions import engineered_names
from fraud.features.encoders import FrequencyEncoder, TargetEncoder


def raw_input_columns(fc: dict[str, Any]) -> tuple[list[str], list[str]]:
    """(numeric, categorical) raw columns a transaction must/can carry."""
    numeric = list(dict.fromkeys(fc["numeric"] + fc["v_columns"]))
    cat = set(fc["value_maps"]) | set(fc["frequency_encode"]) | set(fc["target_encode"])
    cat |= {"P_emaildomain", "R_emaildomain", "DeviceInfo"}
    categorical = sorted(cat - set(numeric) - {"card1", "addr1"})
    return numeric, categorical


@dataclass
class FeaturePipeline:
    config: dict[str, Any] = field(default_factory=load_feature_config)
    freq: FrequencyEncoder = field(default_factory=FrequencyEncoder)
    target: TargetEncoder = field(default_factory=TargetEncoder)
    feature_names: list[str] = field(default_factory=list)

    @property
    def version(self) -> str:
        return str(self.config["version"])

    def _plain(self, df: pd.DataFrame) -> pd.DataFrame:
        fc = self.config
        cols: dict[str, pd.Series] = {}
        for c in fc["numeric"] + fc["v_columns"]:
            cols[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
        for c, mapping in fc["value_maps"].items():
            cols[f"{c}_map"] = df[c].map(mapping).astype("float32")
        eng = [n for n in engineered_names() if n in fc["engineered"]]
        for c in eng:
            cols[c] = df[c].astype("float32")
        return pd.DataFrame(cols, index=df.index)

    def fit_transform(self, df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        fc = self.config
        self.freq.fit(df, fc["frequency_encode"])
        te = self.target.fit_transform(df, y, fc["target_encode"])
        X = pd.concat([self._plain(df), self.freq.transform(df), te], axis=1)
        self.feature_names = list(X.columns)
        return X

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = pd.concat([self._plain(df), self.freq.transform(df), self.target.transform(df)], axis=1)
        return X[self.feature_names]

    def to_numpy(self, X: pd.DataFrame) -> np.ndarray:
        return X.to_numpy(dtype=np.float32)
