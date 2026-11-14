from pydantic import BaseModel, ConfigDict, Field, create_model


# Create a model with the 28 PCA features
feature_fields = {
    f"V{i}": (float, ...)
    for i in range(1, 29)
}


TransactionInput = create_model(
    "TransactionInput",
    __base__=BaseModel,
    Time=(float, Field(ge=0)),
    Amount=(float, Field(ge=0)),
    **feature_fields,
)


TransactionInput.model_config = ConfigDict(
    extra="forbid",
    allow_inf_nan=False,
)


class PredictionResponse(BaseModel):
    prediction: int
    is_fraud: bool
    risk_score: float
    threshold: float
    model_name: str