"""Endpoint tests for FraudShield's transaction scoring and evaluation APIs."""

import pytest

import app as fraudshield


@pytest.fixture
def client():
    fraudshield.app.config["TESTING"] = True
    with fraudshield.app.test_client() as test_client:
        yield test_client


def test_score_endpoint_returns_model_result(client):
    response = client.post(
        "/api/score",
        json={
            "amount": 68.50,
            "distance_km": 12,
            "attempts_10m": 1,
            "new_merchant": False,
            "country_mismatch": False,
            "channel": "Card present",
        },
    )

    assert response.status_code == 200
    result = response.get_json()
    assert 0 <= result["risk_score"] <= 100
    assert result["decision"] in {"approved", "review", "declined"}
    assert result["reasons"]
    assert result["latency_ms"] >= 0
    assert "_label_fraud" not in result


def test_score_endpoint_declines_high_risk_transaction(client):
    response = client.post(
        "/api/score",
        json={
            "amount": 1800,
            "distance_km": 2200,
            "attempts_10m": 6,
            "new_merchant": True,
            "country_mismatch": True,
            "channel": "E-commerce",
        },
    )

    assert response.status_code == 200
    assert response.get_json()["decision"] == "declined"


def test_score_endpoint_rejects_missing_required_fields(client):
    response = client.post("/api/score", json={"amount": 35, "distance_km": 4})

    assert response.status_code == 400
    assert "Required fields" in response.get_json()["error"]


@pytest.mark.parametrize(
    "transaction",
    [
        {"amount": -1, "distance_km": 4, "attempts_10m": 1},
        {"amount": 35, "distance_km": -1, "attempts_10m": 1},
        {"amount": 35, "distance_km": 4, "attempts_10m": 0},
        {"amount": "not-a-number", "distance_km": 4, "attempts_10m": 1},
    ],
)
def test_score_endpoint_rejects_invalid_values(client, transaction):
    response = client.post("/api/score", json=transaction)

    assert response.status_code == 400
    assert "nonnegative" in response.get_json()["error"]


def test_evaluation_endpoint_reports_held_out_metrics(client):
    response = client.get("/api/evaluation?threshold=0.85")

    assert response.status_code == 200
    result = response.get_json()
    assert result["test_size"] == 4000
    assert 0 <= result["precision"] <= 1
    assert 0 <= result["recall"] <= 1
    assert 0 <= result["f1"] <= 1
    assert 0 <= result["roc_auc"] <= 1
    assert sum(sum(row) for row in result["confusion_matrix"]) == result["test_size"]


def test_evaluation_endpoint_rejects_non_numeric_threshold(client):
    response = client.get("/api/evaluation?threshold=high")

    assert response.status_code == 400
    assert "threshold must be a number" in response.get_json()["error"]
