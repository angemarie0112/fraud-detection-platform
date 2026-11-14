import numpy as np
import pandas as pd
import pytest

from src.features.preprocessing import (
    FEATURE_COLUMNS,
    create_preprocessor,
)


@pytest.fixture
def sample_data():
    """Create a small synthetic transaction dataset."""

    rng = np.random.default_rng(42)

    data = pd.DataFrame(
        rng.normal(size=(10, len(FEATURE_COLUMNS))),
        columns=FEATURE_COLUMNS,
    )

    data["Time"] = np.arange(10) * 100
    data["Amount"] = np.arange(10) * 50 + 10

    return data


def test_expected_feature_count():
    """Verify that the model expects 30 features."""

    assert len(FEATURE_COLUMNS) == 30


def test_expected_feature_names():
    """Verify that all required features are present."""

    assert "Time" in FEATURE_COLUMNS
    assert "Amount" in FEATURE_COLUMNS

    for i in range(1, 29):
        assert f"V{i}" in FEATURE_COLUMNS


def test_preprocessor_output_shape(sample_data):
    """Preprocessing should preserve the number of features."""

    preprocessor = create_preprocessor()

    transformed = preprocessor.fit_transform(sample_data)

    assert transformed.shape == sample_data.shape


def test_preprocessor_has_no_missing_values(sample_data):
    """Preprocessing should not introduce missing values."""

    preprocessor = create_preprocessor()

    transformed = preprocessor.fit_transform(sample_data)

    assert np.isfinite(transformed).all()


def test_scaling(sample_data):
    """Time and Amount should be standardized."""

    preprocessor = create_preprocessor()

    transformed = preprocessor.fit_transform(sample_data)

    # The first two columns are scaled Time and Amount.
    assert np.isclose(transformed[:, 0].mean(), 0, atol=1e-7)
    assert np.isclose(transformed[:, 1].mean(), 0, atol=1e-7)

    assert np.isclose(transformed[:, 0].std(), 1, atol=1e-7)
    assert np.isclose(transformed[:, 1].std(), 1, atol=1e-7)