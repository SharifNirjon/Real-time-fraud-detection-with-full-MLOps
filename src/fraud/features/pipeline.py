"""Model-specific part of the features: value maps + fitted encoders -> model matrix.

Input is the output of history.build_offline_features (offline) or the same
columns assembled by the API (online). The fitted pipeline is pickled inside
the model bundle, so training and serving apply identical transforms. The
matrix is assembled from NumPy blocks, so one transaction costs ~1 ms.
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

    @property
    def _numeric(self) -> list[str]:
        return list(dict.fromkeys(self.config["numeric"] + self.config["v_columns"]))

    @property
    def _engineered(self) -> list[str]:
        return [n for n in engineered_names() if n in self.config["engineered"]]

    def _plain_names(self) -> list[str]:
        return self._numeric + [f"{c}_map" for c in self.config["value_maps"]] + self._engineered

    def _plain_block(self, df: pd.DataFrame) -> np.ndarray:
        num = df[self._numeric].to_numpy(dtype=np.float32, na_value=np.nan)
        maps = [
            np.fromiter(
                (m.get(v, np.nan) if isinstance(v, str) else np.nan for v in df[c].tolist()),
                dtype=np.float32,
                count=len(df),
            )
            for c, m in self.config["value_maps"].items()
        ]
        eng = df[self._engineered].to_numpy(dtype=np.float32, na_value=np.nan)
        return np.column_stack([num, *maps, eng])

    def fit_transform(self, df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        fc = self.config
        self.freq.fit(df, fc["frequency_encode"])
        te = self.target.fit_transform_block(df, y, fc["target_encode"])
        X = np.column_stack([self._plain_block(df), self.freq.transform_block(df), te])
        self.feature_names = self._plain_names() + self.freq.names + self.target.names
        return pd.DataFrame(X, columns=self.feature_names, index=df.index)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.transform_numpy(df), columns=self.feature_names, index=df.index)

    def transform_numpy(self, df: pd.DataFrame) -> np.ndarray:
        return np.column_stack(
            [self._plain_block(df), self.freq.transform_block(df), self.target.transform_block(df)]
        ).astype(np.float32, copy=False)

    def to_numpy(self, X: pd.DataFrame) -> np.ndarray:
        return X.to_numpy(dtype=np.float32)
