import itertools

import numpy as np
import pytest

from fraud.evaluate.cost import (
    APPROVE,
    BLOCK,
    REVIEW,
    CostParams,
    Thresholds,
    decide,
    decision_cost,
    no_model_cost,
    optimize_thresholds,
    total_cost,
)
from fraud.evaluate.metrics import (
    calibration_metrics,
    confusion,
    precision_at_recall,
    ranking_metrics,
    recall_at_fpr,
)


def test_ranking_metrics_hand_computed():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.4, 0.35, 0.8])
    m = ranking_metrics(y, s)
    # 3 of 4 (pos, neg) pairs ordered correctly
    assert m["roc_auc"] == pytest.approx(0.75)
    # AP = 0.5 * P@0.8 (=1) + 0.5 * P@0.35 (=2/3)
    assert m["pr_auc"] == pytest.approx(0.5 * 1 + 0.5 * 2 / 3)


def test_recall_at_fpr_and_precision_at_recall():
    y = np.array([0, 0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.9, 0.8, 0.4])
    assert recall_at_fpr(y, s, 0.0) == 0.0  # top score is a negative
    assert recall_at_fpr(y, s, 0.25) == 1.0  # one FP buys both positives
    # recall>=0.5: threshold 0.8 -> P=1/2, threshold 0.4 -> P=2/3
    assert precision_at_recall(y, s, 0.5) == pytest.approx(2 / 3)


def test_confusion_and_brier():
    y = np.array([1, 0, 1, 0])
    assert confusion(y, np.array([1, 1, 0, 0])) == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
    cal = calibration_metrics(y, np.array([1.0, 0.0, 0.5, 0.5]), n_bins=2)
    assert cal["brier"] == pytest.approx((0 + 0 + 0.25 + 0.25) / 4)


def test_decide_thresholds():
    t = Thresholds(review=0.3, block=0.7)
    assert decide(np.array([0.1, 0.3, 0.69, 0.7, 0.99]), t).tolist() == [
        APPROVE,
        REVIEW,
        REVIEW,
        BLOCK,
        BLOCK,
    ]
    with pytest.raises(ValueError):
        Thresholds(review=0.8, block=0.2)


def test_cost_hand_computed():
    c = CostParams(review_cost=5, false_decline_cost=25, review_catch_rate=1.0)
    y = np.array([1, 0, 1, 0, 1])
    amt = np.array([100.0, 50.0, 10.0, 20.0, 70.0])
    d = np.array([APPROVE, APPROVE, REVIEW, BLOCK, BLOCK])
    # missed fraud 100 + review 5 + false decline 25 + caught fraud 0
    assert decision_cost(y, amt, d, c).tolist() == [100, 0, 5, 25, 0]
    assert total_cost(y, amt, d, c) == 130
    assert no_model_cost(y, amt) == 180
    half = CostParams(review_cost=5, false_decline_cost=25, review_catch_rate=0.5)
    assert total_cost(y, amt, d, half) == 135  # reviewed fraud: 5 + 0.5*10


def test_optimize_thresholds_matches_brute_force():
    rng = np.random.default_rng(0)
    n = 300
    y = rng.random(n) < 0.1
    s = np.clip(0.3 * y + rng.random(n) * 0.7, 0, 1).round(2)
    amt = rng.exponential(80, n)
    for c in [CostParams(5, 25, 1.0), CostParams(2, 10, 0.7)]:
        t, best = optimize_thresholds(y, amt, s, c)
        cands = np.append(np.unique(s), np.inf)
        brute = min(
            total_cost(y, amt, decide(s, Thresholds(a, b)), c)
            for a, b in itertools.product(cands, cands)
            if a <= b
        )
        assert best == pytest.approx(brute)
        assert total_cost(y, amt, decide(s, t), c) == pytest.approx(best)
