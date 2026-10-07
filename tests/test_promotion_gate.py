import pytest

from fraud.pipelines.retrain import promotion_gate


@pytest.mark.parametrize(
    "champ,chall,lat,promote,why",
    [
        (0.50, 0.52, 10.0, True, "gain"),  # +0.02 >= 0.005, fast
        (0.50, 0.503, 10.0, False, "below"),  # +0.003 < 0.005
        (0.50, 0.505, 10.0, True, "gain"),  # exactly the margin is enough
        (0.50, 0.45, 10.0, False, "below"),  # worse
        (0.50, 0.60, 80.0, False, "latency"),  # better but too slow
        (0.50, 0.60, 50.0, True, "gain"),  # at the budget is fine
    ],
)
def test_promotion_gate(champ, chall, lat, promote, why):
    d = promotion_gate(champ, chall, lat, min_gain=0.005, latency_budget_ms=50.0)
    assert d.promote is promote
    assert why in d.reason
