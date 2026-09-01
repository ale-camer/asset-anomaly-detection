"""Pydantic schemas for the FastAPI inference service."""


from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Input payload for anomaly scoring prediction."""

    features: list[dict[str, float]] = Field(
        ...,
        description="List of feature dictionaries required by the model. Each dict represents a single row/sample.",
        examples=[
            [
                {
                    "close": 50000.0,
                    "volume": 1200.5,
                    "close_rolling_mean_7": 49000.0,
                    "close_rolling_std_7": 500.0,
                    "volatility_std_7": 0.02,
                    "momentum_rsi_14": 55.0,
                    "price_velocity_roc_1": 0.01,
                }
            ]
        ],
    )


class PredictionResponseItem(BaseModel):
    """Single prediction response item."""

    is_anomaly: bool = Field(..., description="True if the sample is an anomaly, False otherwise.")
    anomaly_score: float = Field(..., description="Continuous anomaly score (higher means more anomalous).")


class PredictionResponse(BaseModel):
    """Output response for anomaly scoring prediction."""

    predictions: list[PredictionResponseItem] = Field(
        ..., description="List of prediction results corresponding to the input features."
    )
    model_version: str | None = Field(default=None, description="The MLflow model version used for inference.")
