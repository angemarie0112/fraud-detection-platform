from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.features.preprocessing import FEATURE_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "fraud_pipeline.joblib"


@pytest.fixture(scope="module")
def model():
    """Load the trained fraud detection pipeline."""

    if not MODEL_PATH.exists():
        pytest.skip("Trained model artifact not found")

    return joblib.load(MODEL_PATH)


@pytest.fixture
def sample_transaction():
    """Create a synthetic transaction for inference testing."""

    transaction = {
        feature: 0.0
        for feature in FEATURE_COLUMNS
    }

    transaction["Time"] = 1000.0
    transaction["Amount"] = 150.0

    return pd.DataFrame([transaction])[FEATURE_COLUMNS]


def test_model_prediction(model, sample_transaction):
    """The model should produce a valid binary prediction."""

    prediction = model.predict(sample_transaction)

    assert prediction.shape == (1,)
    assert prediction[0] in [0, 1]


def test_model_probabilities(model, sample_transaction):
    """Prediction probabilities should be valid."""

    probabilities = model.predict_proba(sample_transaction)

    assert probabilities.shape == (1, 2)

    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)

    assert np.isclose(probabilities.sum(), 1.0)


def test_batch_predictions(model, sample_transaction):
    """The model should support batch inference."""

    batch = pd.concat(
        [sample_transaction] * 5,
        ignore_index=True,
    )

    predictions = model.predict(batch)

    assert len(predictions) == 5