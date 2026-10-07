import numpy as np

from fraud.features.history import build_offline_features


def test_path_attribution_sums_to_raw_margin(bundle, transactions):
    feats = build_offline_features(transactions.iloc[:2500]).iloc[2000:2200]
    X = bundle.matrix(feats)
    raw = bundle.booster.predict(X, raw_score=True)
    assert np.isnan(X).any()  # exercise missing-value routing
    for i in range(len(X)):
        contrib, bias = bundle.explainer.contributions(X[i])
        assert abs(contrib.sum() + bias - raw[i]) < 1e-9


def test_reason_codes_top3_positive(bundle, transactions):
    feats = build_offline_features(transactions.iloc[:2100]).iloc[2000:]
    codes = bundle.reason_codes(bundle.matrix(feats))
    for c in codes:
        assert len(c) <= 3
        contribs = [r["contribution"] for r in c]
        assert contribs == sorted(contribs, reverse=True) and all(x > 0 for x in contribs)
