import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from src.features.preprocessing import FEATURE_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "fraud_pipeline.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "pipeline_metadata.json"

RANDOM_STATE = 42


def load_test_data():
    """
    Reproduce the original stratified split and
    return the untouched test dataset.
    """

    from sklearn.model_selection import train_test_split

    df = pd.read_csv(DATA_PATH)

    X = df[FEATURE_COLUMNS]
    y = df["Class"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    return X_test, y_test


def evaluate():
    print("Loading trained pipeline...")

    pipeline = joblib.load(MODEL_PATH)

    with open(METADATA_PATH, "r") as file:
        metadata = json.load(file)

    threshold = metadata["threshold"]

    print("Loading test dataset...")

    X_test, y_test = load_test_data()

    print(f"Test samples: {len(X_test)}")
    print(f"Fraud cases: {y_test.sum()}")

    print("\nGenerating predictions...")

    probabilities = pipeline.predict_proba(X_test)[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    metrics = {
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_test, probabilities
        ),
        "average_precision": average_precision_score(
            y_test, probabilities
        ),
    }

    print("\n===== FINAL TEST RESULTS =====")

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    results_path = PROJECT_ROOT / "models" / "test_results.json"

    with open(results_path, "w") as file:
        json.dump(
            {key: float(value) for key, value in metrics.items()},
            file,
            indent=4,
        )

    print(f"\nTest results saved to: {results_path}")


if __name__ == "__main__":
    evaluate()