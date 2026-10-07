"""API contract tests: FastAPI TestClient + fakeredis + a fixture model."""

import math

import fakeredis
import numpy as np
import pytest
from fastapi.testclient import TestClient

from fraud.serve.app import create_app
from fraud.serve.prediction_log import read_log
from fraud.serve.schemas import Transaction
from fraud.serve.service import LoadedModel


def _payload(row: dict) -> dict:
    fields = Transaction.model_fields
    out = {}
    for k, v in row.items():
        if k not in fields:
            continue
        if hasattr(v, "item"):
            v = v.item()
        if isinstance(v, float) and math.isnan(v):
            v = None
        out[k] = v
    return out


@pytest.fixture()
def client(bundle, tmp_path, monkeypatch):
    monkeypatch.setenv("ADMIN_API_KEY", "secret")
    versions = iter(["1", "2", "3"])
    server = fakeredis.FakeServer()
    app = create_app(
        model_loader=lambda: LoadedModel(bundle, next(versions)),
        redis_factory=lambda: fakeredis.FakeRedis(server=server),
        prediction_log_dir=tmp_path / "pred",
        label_log_dir=tmp_path / "labels",
    )
    with TestClient(app) as c:
        c.tmp_path = tmp_path
        yield c


@pytest.fixture()
def live_rows(transactions):
    return [_payload(r) for r in transactions.iloc[2000:2050].to_dict("records")]


def test_score_contract(client, live_rows):
    r = client.post("/score", json=live_rows[0])
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {
        "transaction_id",
        "fraud_probability",
        "decision",
        "reason_codes",
        "model_version",
        "latency_ms",
    }
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["decision"] in {"approve", "review", "block"}
    assert body["model_version"] == "1"
    assert len(body["reason_codes"]) <= 3
    for rc in body["reason_codes"]:
        assert rc["contribution"] > 0 and isinstance(rc["feature"], str)


@pytest.mark.parametrize(
    "patch,field",
    [
        ({"TransactionAmt": -1}, "TransactionAmt"),
        ({"TransactionAmt": "abc"}, "TransactionAmt"),
        ({"ProductCD": "Z"}, "ProductCD"),
        ({"card1": None}, "card1"),
        ({"TransactionDT": -5}, "TransactionDT"),
    ],
)
def test_invalid_input_returns_clear_422(client, live_rows, patch, field):
    r = client.post("/score", json=live_rows[0] | patch)
    assert r.status_code == 422
    body = r.json()
    assert body["detail"] == "invalid transaction"
    assert field in [e["field"] for e in body["errors"]]


def test_missing_required_field(client, live_rows):
    row = dict(live_rows[0])
    del row["TransactionDT"]
    r = client.post("/score", json=row)
    assert r.status_code == 422 and r.json()["errors"][0]["field"] == "TransactionDT"


def test_state_updates_between_requests(client, live_rows):
    """Second transaction of the same card must see the first one in its velocity features."""
    a = dict(live_rows[0])
    b = a | {"TransactionID": a["TransactionID"] + 1, "TransactionDT": a["TransactionDT"] + 60}
    client.post("/score", json=a)
    client.post("/score", json=b)
    client.app.state.pred_log.flush()
    log = read_log(client.tmp_path / "pred").set_index("TransactionID")
    assert (
        log.loc[b["TransactionID"], "card1_cnt_1h"]
        == log.loc[a["TransactionID"], "card1_cnt_1h"] + 1
    )


def test_batch_scores_in_time_order(client, live_rows):
    rows = list(reversed(live_rows[:10]))
    r = client.post("/score/batch", json={"transactions": rows})
    assert r.status_code == 200
    ids = [x["transaction_id"] for x in r.json()["results"]]
    expected = [
        t["TransactionID"]
        for t in sorted(rows, key=lambda t: (t["TransactionDT"], t["TransactionID"]))
    ]
    assert ids == expected


def test_batch_validation(client):
    assert client.post("/score/batch", json={"transactions": []}).status_code == 422


def test_prediction_log_written(client, live_rows):
    for row in live_rows[:5]:
        client.post("/score", json=row)
    client.app.state.pred_log.flush()
    log = read_log(client.tmp_path / "pred")
    assert len(log) == 5
    assert {"score", "decision", "model_version", "raw", "logged_at"} <= set(log.columns)
    assert set(client.app.state.scorer.model.bundle.feature_names) <= set(log.columns)


def test_health_metadata_metrics(client, live_rows):
    assert client.get("/health").json()["status"] == "ok"
    md = client.get("/metadata").json()
    assert md["model_version"] == "1" and md["n_features"] > 50
    assert set(md["required_fields"]) >= {
        "TransactionID",
        "TransactionDT",
        "TransactionAmt",
        "card1",
    }
    client.post("/score", json=live_rows[0])
    text = client.get("/metrics/").text
    for name in [
        "fraud_requests_total",
        "fraud_request_latency_seconds_bucket",
        "fraud_score_bucket",
        "fraud_decisions_total",
        "fraud_model_info",
        "fraud_errors_total",
    ]:
        assert name in text


def test_labels_endpoint(client):
    r = client.post("/labels", json={"labels": [{"TransactionID": 1, "isFraud": 1}]})
    assert r.status_code == 200 and r.json() == {"received": 1}
    assert (
        client.post("/labels", json={"labels": [{"TransactionID": 1, "isFraud": 3}]}).status_code
        == 422
    )


def test_admin_reload_requires_key(client):
    assert client.post("/admin/reload").status_code == 401
    assert client.post("/admin/reload", headers={"X-API-Key": "wrong"}).status_code == 401
    r = client.post("/admin/reload", headers={"X-API-Key": "secret"})
    assert r.status_code == 200 and r.json() == {"previous_version": "1", "model_version": "2"}
    assert client.get("/metadata").json()["model_version"] == "2"


def test_decision_thresholds(bundle):
    t = bundle.thresholds
    p = np.array([0.0, t.review - 1e-9, t.review, t.block - 1e-9, t.block, 1.0])
    assert bundle.decide(p) == ["approve", "approve", "review", "review", "block", "block"]
