import json
from pathlib import Path

import joblib
import pandas as pd

from src.features.preprocessing import FEATURE_COLUMNS


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "fraud_pipeline.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "pipeline_metadata.json"


class FraudPredictor:

    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found: {MODEL_PATH}"
            )

        if not METADATA_PATH.exists():
            raise FileNotFoundError(
                f"Metadata not found: {METADATA_PATH}"
            )

        # Load only trusted model artifacts
        self.pipeline = joblib.load(MODEL_PATH)

        with open(METADATA_PATH, "r") as file:
            self.metadata = json.load(file)

        self.threshold = float(self.metadata["threshold"])
        self.model_name = self.metadata["model_name"]

    def predict(self, transaction: dict):

        # Ensure consistent feature ordering
        features = pd.DataFrame(
            [transaction],
            columns=FEATURE_COLUMNS,
        )

        # Generate model score
        score = float(
            self.pipeline.predict_proba(features)[0][1]
        )

        prediction = int(score >= self.threshold)

        return {
            "prediction": prediction,
            "is_fraud": bool(prediction),
            "risk_score": score,
            "threshold": self.threshold,
            "model_name": self.model_name,
        }