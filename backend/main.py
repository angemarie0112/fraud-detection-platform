from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from backend.predictor import FraudPredictor
from backend.schemas import (
    PredictionResponse,
    TransactionInput,
)


predictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor

    predictor = FraudPredictor()

    yield

    predictor = None


app = FastAPI(
    title="Intelligent Fraud Detection API",
    description=(
        "Machine learning API for detecting potentially "
        "fraudulent credit card transactions using XGBoost."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "message": "Fraud Detection API is running",
        "version": "1.0.0",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict_fraud(transaction: TransactionInput):

    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not available",
        )

    try:
        result = predictor.predict(
            transaction.model_dump()
        )

        return result

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed",
        )