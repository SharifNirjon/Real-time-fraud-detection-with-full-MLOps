"""Business cost model and two-threshold (review / block) decision policy.

Per transaction:
  approve: fraud -> lose the amount;            legit -> 0
  review:  every review costs `review_cost`; a share `review_catch_rate`
           of reviewed frauds is stopped, the rest is lost
  block:   fraud -> 0;                          legit -> `false_decline_cost`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

APPROVE, REVIEW, BLOCK = 0, 1, 2
DECISION_NAMES = {APPROVE: "approve", REVIEW: "review", BLOCK: "block"}


@dataclass(frozen=True)
class CostParams:
    review_cost: float = 5.0
    false_decline_cost: float = 25.0
    review_catch_rate: float = 1.0

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> CostParams:
        c = cfg["cost"]
        return cls(c["review_cost"], c["false_decline_cost"], c["review_catch_rate"])


@dataclass(frozen=True)
class Thresholds:
    review: float
    block: float

    def __post_init__(self) -> None:
        if self.review > self.block:
            raise ValueError("review threshold must be <= block threshold")


def decide(scores: np.ndarray, t: Thresholds) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    return np.where(s >= t.block, BLOCK, np.where(s >= t.review, REVIEW, APPROVE))


def decision_cost(y: np.ndarray, amount: np.ndarray, d: np.ndarray, c: CostParams) -> np.ndarray:
    y = np.asarray(y).astype(bool)
    amt = np.asarray(amount, dtype=float)
    cost = np.zeros(len(y))
    cost[(d == APPROVE) & y] = amt[(d == APPROVE) & y]
    rev = d == REVIEW
    cost[rev] = c.review_cost + np.where(y[rev], (1 - c.review_catch_rate) * amt[rev], 0.0)
    cost[(d == BLOCK) & ~y] = c.false_decline_cost
    return cost


def total_cost(y: np.ndarray, amount: np.ndarray, d: np.ndarray, c: CostParams) -> float:
    return float(decision_cost(y, amount, d, c).sum())


def no_model_cost(y: np.ndarray, amount: np.ndarray) -> float:
    """Approve everything: every fraud is lost."""
    return float(np.asarray(amount, dtype=float)[np.asarray(y).astype(bool)].sum())


def optimize_thresholds(
    y: np.ndarray,
    amount: np.ndarray,
    scores: np.ndarray,
    c: CostParams,
    max_candidates: int = 400,
) -> tuple[Thresholds, float]:
    """Exhaustive search over candidate (review, block) pairs, vectorised.

    Candidates are score quantiles plus +inf ("never"). For thresholds
    c_i <= c_j the cost decomposes into prefix sums over the sorted scores.
    """
    y = np.asarray(y).astype(bool)
    amt = np.asarray(amount, dtype=float)
    s = np.asarray(scores, dtype=float)
    uniq = np.unique(s)
    if len(uniq) > max_candidates:
        uniq = np.unique(np.quantile(s, np.linspace(0, 1, max_candidates)))
    cand = np.append(uniq, np.inf)

    order = np.argsort(s, kind="mergesort")
    s_sorted, y_sorted, a_sorted = s[order], y[order], amt[order]
    k = np.searchsorted(s_sorted, cand, side="left")  # rows with score < cand
    fraud_amt = np.concatenate([[0.0], np.cumsum(np.where(y_sorted, a_sorted, 0.0))])[k]
    n_below = k.astype(float)
    legit_below = np.concatenate([[0], np.cumsum(~y_sorted)])[k].astype(float)
    legit_total = float((~y).sum())

    a_i = fraud_amt[:, None]  # approve region: score < t_review
    a_j = fraud_amt[None, :]
    reviewed = n_below[None, :] - n_below[:, None]
    cost = (
        a_i
        + c.review_cost * reviewed
        + (1 - c.review_catch_rate) * (a_j - a_i)
        + c.false_decline_cost * (legit_total - legit_below[None, :])
    )
    cost = np.where(np.arange(len(cand))[:, None] <= np.arange(len(cand))[None, :], cost, np.inf)
    i, j = np.unravel_index(np.argmin(cost), cost.shape)
    return Thresholds(float(cand[i]), float(cand[j])), float(cost[i, j])
