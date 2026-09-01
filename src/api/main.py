"""FastAPI inference service for Anomaly Detection."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from src.api.schemas import PredictionRequest, PredictionResponse, PredictionResponseItem
from src.models.registry import load_model_from_mlflow

logger = logging.getLogger(__name__)

# Global dictionary to hold the loaded model state
app_state: dict[str, Any] = {"model": None, "model_version": None}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycle."""
    logger.info("Initializing FastAPI inference service...")
    try:
        # Load the latest production model from MLflow registry
        model = load_model_from_mlflow(model_uri="models:/asset-anomaly-detector/latest")
        app_state["model"] = model
        app_state["model_version"] = "latest"
        logger.info("Successfully loaded production anomaly detector from MLflow.")
    except Exception as e:
        logger.error(f"Failed to load model from MLflow during startup: {e}")
        # Depending on requirements, we could raise here to prevent the API from starting,
        # but for testing/resilience we might allow it to start and return 503 on /health
    yield
    logger.info("Shutting down inference service...")
    app_state["model"] = None


app = FastAPI(
    title="Asset Anomaly Detection API",
    description="Real-time inference service for cryptocurrency anomaly detection.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", tags=["Health"])
def health_check() -> JSONResponse:
    """Check API and model health status."""
    if app_state["model"] is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "detail": "Model is not loaded."},
        )
    return JSONResponse(status_code=200, content={"status": "ok", "model_loaded": True})


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(request: PredictionRequest) -> PredictionResponse:
    """Run anomaly detection inference on provided features."""
    model = app_state["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    try:
        # Convert incoming list of dictionaries to pandas DataFrame
        df = pd.DataFrame(request.features)

        # Generate binary predictions (1 = anomaly, 0 = inlier)
        # Note: If the PyFunc wrapper isn't strictly used in tests, we can fallback to standard method checks
        if hasattr(model, "predict"):
            preds = model.predict(df)
        else:
            raise ValueError("Model does not implement a predict() method.")

        # Generate continuous anomaly scores
        if hasattr(model, "score_samples"):
            scores = model.score_samples(df)
        else:
            # Fallback if no score_samples exists (e.g. some custom pyfunc wrappers)
            scores = [0.0] * len(preds)

        predictions = []
        for i in range(len(preds)):
            is_anomaly = bool(preds[i] == 1)
            anomaly_score = float(scores[i])
            predictions.append(
                PredictionResponseItem(is_anomaly=is_anomaly, anomaly_score=anomaly_score)
            )

        return PredictionResponse(
            predictions=predictions,
            model_version=app_state.get("model_version"),
        )
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=400, detail=f"Error during inference: {str(e)}") from e
