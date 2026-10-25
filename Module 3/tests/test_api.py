"""Tests HTTP de l'API à partir des exemples issus de la validation."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app


APP_DIR = Path(__file__).resolve().parents[1] / "app"
EXAMPLES = json.loads((APP_DIR / "exemples.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_model_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_predict_examples_and_threshold_consistency(client):
    for name, expected_alert in (("fraude", True), ("legitime", False)):
        response = client.post(
            "/predict",
            json={"features": EXAMPLES[name]["features"]},
        )
        assert response.status_code == 200
        result = response.json()
        assert result["alerte"] is expected_alert
        assert result["alerte"] == (result["probabilite"] > result["seuil"])


@pytest.mark.parametrize(
    "features",
    [
        {
            "type": "PAYMENT",
            "amount": 10,
            "oldbalanceOrg": 10,
            "oldbalanceDest": 0,
            "nb_tx_orig_avant": 0,
            "nb_recu_dest_avant": 0,
        },
        {
            "type": "CASH_OUT",
            "amount": -1,
            "oldbalanceOrg": 10,
            "oldbalanceDest": 0,
            "nb_tx_orig_avant": 0,
            "nb_recu_dest_avant": 0,
        },
        {
            "type": "CASH_OUT",
            "amount": 10,
            "oldbalanceOrg": 10,
            "oldbalanceDest": 0,
            "nb_tx_orig_avant": 0,
        },
        {
            "type": "CASH_OUT",
            "amount": 10,
            "oldbalanceOrg": 10,
            "oldbalanceDest": 0,
            "nb_tx_orig_avant": 0,
            "nb_recu_dest_avant": 0,
            "newbalanceOrig": 0,
        },
    ],
)
def test_invalid_transaction_returns_422(client, features):
    response = client.post("/predict", json={"features": features})
    assert response.status_code == 422


def test_batch_preserves_input_order_and_result_count(client):
    transactions = [
        EXAMPLES["fraude"]["features"],
        EXAMPLES["legitime"]["features"],
    ]
    response = client.post(
        "/predict_batch",
        json={"transactions": transactions},
    )

    assert response.status_code == 200
    results = response.json()
    assert len(results) == len(transactions)
    assert [item["alerte"] for item in results] == [True, False]
