import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

from src.features.preprocessing import (
    FEATURE_COLUMNS,
    create_preprocessor,
)


# --------------------------------------------------
# Project configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
MODELS_DIR = PROJECT_ROOT / "models"

RANDOM_STATE = 42


# --------------------------------------------------
# Load and split dataset
# --------------------------------------------------

def load_data():
    print("Loading dataset...")

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

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples reserved: {len(X_test)}")

    return X_train, X_val, y_train, y_val


# --------------------------------------------------
# Create ML pipeline
# --------------------------------------------------

def create_pipeline(y_train):
    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()

    scale_pos_weight = negative_count / positive_count

    classifier = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    pipeline = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("classifier", classifier),
    ])

    return pipeline


# --------------------------------------------------
# Evaluate validation performance
# --------------------------------------------------

def evaluate(pipeline, X_val, y_val):
    probabilities = pipeline.predict_proba(X_val)[:, 1]

    precision, recall, thresholds = precision_recall_curve(
        y_val,
        probabilities,
    )

    f1_scores = (
        2 * precision[:-1] * recall[:-1]
        / (precision[:-1] + recall[:-1] + 1e-12)
    )

    best_index = np.argmax(f1_scores)
    best_threshold = float(thresholds[best_index])

    predictions = (
        probabilities >= best_threshold
    ).astype(int)

    metrics = {
        "precision": float(
            precision_score(y_val, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_val, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y_val, predictions, zero_division=0)
        ),
        "roc_auc": float(
            roc_auc_score(y_val, probabilities)
        ),
        "average_precision": float(
            average_precision_score(y_val, probabilities)
        ),
    }

    return metrics, best_threshold


# --------------------------------------------------
# Train and track experiment
# --------------------------------------------------

def train():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_val, y_train, y_val = load_data()

    pipeline = create_pipeline(y_train)

    # Configure MLflow tracking using SQLite
    MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"

    mlflow.set_tracking_uri(
        f"sqlite:///{MLFLOW_DB_PATH}"
    )

    mlflow.set_experiment("fraud-detection")

    with mlflow.start_run():

        print("\nTraining XGBoost pipeline...")

        pipeline.fit(X_train, y_train)

        print("Evaluating model...")

        metrics, threshold = evaluate(
            pipeline,
            X_val,
            y_val,
        )

        classifier = pipeline.named_steps["classifier"]

        mlflow.log_params(classifier.get_params())
        mlflow.log_metrics(metrics)
        mlflow.log_param("decision_threshold", threshold)

        model_path = MODELS_DIR / "fraud_pipeline.joblib"

        joblib.dump(pipeline, model_path)

        metadata = {
            "model_name": "XGBoost",
            "threshold": threshold,
            "feature_columns": FEATURE_COLUMNS,
            "validation_metrics": metrics,
        }

        metadata_path = MODELS_DIR / "pipeline_metadata.json"

        with open(metadata_path, "w") as file:
            json.dump(metadata, file, indent=4)

        mlflow.log_artifact(str(metadata_path))

        print("\nTraining completed successfully!")
        print(f"Model saved to: {model_path}")
        print(f"Selected threshold: {threshold:.6f}")

        print("\nValidation metrics:")

        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    train()