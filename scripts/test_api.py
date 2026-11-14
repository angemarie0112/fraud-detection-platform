import pandas as pd
import httpx
import os

# Load dataset
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

df = pd.read_csv(
    PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
)

# Select one legitimate transaction
legitimate = df[df["Class"] == 0].iloc[0]

# Select one fraudulent transaction
fraud = df[df["Class"] == 1].iloc[0]


def test_transaction(transaction, label):
    features = transaction.drop("Class").to_dict()

    API_URL = os.getenv(
        "API_URL",
        "http://127.0.0.1:8000"
    )

    response = httpx.post(
        f"{API_URL}/predict",
        json=features,
        timeout=10,
    )

    print(f"\n===== {label} =====")
    print("HTTP status:", response.status_code)
    print("Response:", response.json())


test_transaction(legitimate, "Legitimate Transaction")
test_transaction(fraud, "Fraudulent Transaction")