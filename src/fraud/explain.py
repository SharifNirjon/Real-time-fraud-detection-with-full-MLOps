"""Fast per-prediction reason codes for LightGBM (decision-path attribution).

For every tree the row's path root -> leaf is walked; each split credits the
change in node value (leaf value - parent value, in log-odds) to the split
feature. Contributions + bias sum exactly to the raw margin. This is the
"Saabas" / treeinterpreter decomposition: an approximation of TreeSHAP that
runs in ~1 ms instead of ~50 ms for 1.5k trees, so it fits the serving
budget. Global explanations (reports/shap_summary.png) use exact TreeSHAP.
All trees are traversed at once with NumPy, one depth level per step.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np

_MISSING = {"None": 0, "Zero": 1, "NaN": 2}


@dataclass
class PathExplainer:
    feature: np.ndarray  # split feature per node (-1 for leaves)
    threshold: np.ndarray
    default_left: np.ndarray
    missing: np.ndarray
    left: np.ndarray
    right: np.ndarray
    value: np.ndarray  # internal_value for splits, leaf_value for leaves
    roots: np.ndarray
    n_features: int

    @classmethod
    def from_booster(cls, booster: lgb.Booster) -> PathExplainer:
        dump = booster.dump_model()
        cols: dict[str, list] = {
            k: []
            for k in ["feature", "threshold", "default_left", "missing", "left", "right", "value"]
        }
        roots = []

        def add(node: dict) -> int:
            idx = len(cols["feature"])
            for k in cols:
                cols[k].append(0)
            if "leaf_value" in node or "leaf_index" in node:
                cols["feature"][idx] = -1
                cols["value"][idx] = node.get("leaf_value", 0.0)
                cols["left"][idx] = cols["right"][idx] = idx
                return idx
            if node["decision_type"] != "<=":
                raise ValueError("categorical splits are not supported")
            cols["feature"][idx] = node["split_feature"]
            cols["threshold"][idx] = node["threshold"]
            cols["default_left"][idx] = node["default_left"]
            cols["missing"][idx] = _MISSING[node["missing_type"]]
            cols["value"][idx] = node["internal_value"]
            cols["left"][idx] = add(node["left_child"])
            cols["right"][idx] = add(node["right_child"])
            return idx

        for t in dump["tree_info"]:
            roots.append(add(t["tree_structure"]))
        return cls(
            feature=np.array(cols["feature"], dtype=np.int64),
            threshold=np.array(cols["threshold"], dtype=np.float64),
            default_left=np.array(cols["default_left"], dtype=bool),
            missing=np.array(cols["missing"], dtype=np.int8),
            left=np.array(cols["left"], dtype=np.int64),
            right=np.array(cols["right"], dtype=np.int64),
            value=np.array(cols["value"], dtype=np.float64),
            roots=np.array(roots, dtype=np.int64),
            n_features=booster.num_feature(),
        )

    def leaves(self, x: np.ndarray) -> np.ndarray:
        return self._walk(x)[0]

    def contributions(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        """(per-feature log-odds contributions, bias) for one row."""
        return self._walk(x)[1:]

    def _walk(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        x = np.asarray(x, dtype=np.float64).ravel()
        cur = self.roots.copy()
        contrib = np.zeros(self.n_features)
        active = self.feature[cur] >= 0
        while active.any():
            node = cur[active]
            f = self.feature[node]
            v = x[f]
            miss_type = self.missing[node]
            is_nan = np.isnan(v)
            # LightGBM semantics: missing_type None -> NaN treated as 0; Zero -> 0/NaN are missing
            v_eff = np.where(is_nan & (miss_type != 2), 0.0, v)
            is_missing = np.where(
                miss_type == 2, is_nan, np.where(miss_type == 1, is_nan | (v == 0), False)
            )
            go_left = np.where(is_missing, self.default_left[node], v_eff <= self.threshold[node])
            nxt = np.where(go_left, self.left[node], self.right[node])
            np.add.at(contrib, f, self.value[nxt] - self.value[node])
            cur[active] = nxt
            active = self.feature[cur] >= 0
        bias = float(self.value[self.roots].sum())
        return cur, contrib, bias
