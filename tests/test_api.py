from unittest.mock import patch
import pytest

from fastapi.testclient import TestClient

from backend.main import app
from src.features.preprocessing import FEATURE_COLUMNS


@pytest.fixture(scope="module")
def client():
    """Create an API client using a mocked ML predictor."""

    class MockFraudPredictor:
        def predict(self, transaction):
            return {
                "prediction": 0,
                "is_fraud": False,
                "risk_score": 0.15,
                "threshold": 0.990795,
                "model_name": "XGBoost",
            }

    with patch(
        "backend.main.FraudPredictor",
        return_value=MockFraudPredictor(),
    ):
        with TestClient(app) as test_client:
            yield test_client

@pytest.fixture
def valid_transaction():
    """Generate a valid synthetic transaction."""

    transaction = {
        feature: 0.0
        for feature in FEATURE_COLUMNS
    }

    transaction["Time"] = 1000.0
    transaction["Amount"] = 150.0

    return transaction


def test_root_endpoint(client):
    """Verify that the root endpoint responds."""

    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["version"] == "1.0.0"


def test_health_endpoint(client):
    """Verify API health and model availability."""

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_valid_prediction(client, valid_transaction):
    """Verify prediction response structure."""

    response = client.post(
        "/predict",
        json=valid_transaction,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["prediction"] in [0, 1]
    assert isinstance(data["is_fraud"], bool)
    assert 0 <= data["risk_score"] <= 1
    assert 0 <= data["threshold"] <= 1
    assert data["model_name"] == "XGBoost"

    assert data["is_fraud"] == bool(data["prediction"])


def test_missing_features(client):
    """Reject transactions with missing features."""

    response = client.post(
        "/predict",
        json={
            "Time": 1000,
            "Amount": 150,
        },
    )

    assert response.status_code == 422


def test_negative_amount(client, valid_transaction):
    """Reject negative transaction amounts."""

    valid_transaction["Amount"] = -100

    response = client.post(
        "/predict",
        json=valid_transaction,
    )

    assert response.status_code == 422


def test_negative_time(client, valid_transaction):
    """Reject negative transaction time."""

    valid_transaction["Time"] = -10

    response = client.post(
        "/predict",
        json=valid_transaction,
    )

    assert response.status_code == 422


def test_extra_features(client, valid_transaction):
    """Reject unexpected transaction features."""

    valid_transaction["UnexpectedFeature"] = 123

    response = client.post(
        "/predict",
        json=valid_transaction,
    )

    assert response.status_code == 422


def test_invalid_feature_type(client, valid_transaction):
    """Reject nonnumeric feature values."""

    valid_transaction["V1"] = "invalid"

    response = client.post(
        "/predict",
        json=valid_transaction,
    )

    assert response.status_code == 422


def test_multiple_predictions(client, valid_transaction):
    """Verify repeated inference requests work."""

    for _ in range(3):
        response = client.post(
            "/predict",
            json=valid_transaction,
        )

        assert response.status_code == 200