from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    create_model,
)


# Configuration for strict input validation
class TransactionBase(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
    )


# Generate the 28 PCA feature fields
feature_fields = {
    f"V{i}": (
        float,
        Field(...)
    )
    for i in range(1, 29)
}


# Transaction input schema
TransactionInput = create_model(
    "TransactionInput",

    __base__=TransactionBase,

    Time=(
        float,
        Field(ge=0)
    ),

    Amount=(
        float,
        Field(ge=0)
    ),

    **feature_fields,
)


# Prediction response schema
class PredictionResponse(BaseModel):
    prediction: int
    is_fraud: bool
    risk_score: float
    threshold: float
    model_name: str