"""Model Registry packaging, artifact serialization, and ONNX export utilities."""

import pickle
from pathlib import Path
from typing import Any

import mlflow
import mlflow.models
import mlflow.pyfunc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.models.autoencoder import AutoencoderAnomalyDetector
from src.models.base import BaseAnomalyDetector


class AnomalyDetectorPyFunc(mlflow.pyfunc.PythonModel):  # type: ignore
    """MLflow PyFunc wrapper for BaseAnomalyDetector models."""

    def __init__(self, detector: BaseAnomalyDetector) -> None:
        """Initialize PyFunc wrapper.

        Args:
            detector: Fitted BaseAnomalyDetector instance.
        """
        self.detector = detector

    def predict(self, context: Any, model_input: Any) -> Any:
        """Generate binary anomaly predictions.

        Args:
            context: MLflow model context.
            model_input: Input features dataframe or numpy array.

        Returns:
            np.ndarray: Binary anomaly predictions (1 = anomaly, 0 = inlier).
        """
        return self.detector.predict(model_input)

    def score_samples(self, model_input: Any) -> Any:
        """Generate continuous anomaly scores.

        Args:
            model_input: Input features dataframe or numpy array.

        Returns:
            np.ndarray: Continuous anomaly scores.
        """
        return self.detector.score_samples(model_input)


def save_model_artifact(model: BaseAnomalyDetector, filepath: str | Path) -> Path:
    """Serialize model detector to disk using pickle.

    Args:
        model: Fitted BaseAnomalyDetector instance.
        filepath: Destination file path.

    Returns:
        Path: Saved file path.

    Raises:
        RuntimeError: If model is not fitted.
    """
    if not model.is_fitted:
        raise RuntimeError("Model must be fitted before saving artifact.")

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path


def load_model_artifact(filepath: str | Path) -> BaseAnomalyDetector:
    """Load serialized model detector from disk.

    Args:
        filepath: Source file path.

    Returns:
        BaseAnomalyDetector: Loaded detector instance.

    Raises:
        FileNotFoundError: If filepath does not exist.
        TypeError: If loaded object is not a BaseAnomalyDetector.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found at: {path}")

    with open(path, "rb") as f:
        model = pickle.load(f)

    if not isinstance(model, BaseAnomalyDetector):
        raise TypeError(f"Loaded object of type {type(model)} is not an instance of BaseAnomalyDetector.")

    return model


def log_model_to_mlflow(
    model: BaseAnomalyDetector,
    artifact_path: str = "model",
    registered_model_name: str | None = None,
    signature: mlflow.models.ModelSignature | None = None,
    input_example: pd.DataFrame | np.ndarray | None = None,
) -> Any:
    """Log anomaly detector model to MLflow and optionally register it in the Model Registry.

    Args:
        model: Fitted BaseAnomalyDetector instance.
        artifact_path: Run-relative artifact directory to save model to. Defaults to "model".
        registered_model_name: Optional name to register model in MLflow Model Registry.
        signature: Optional MLflow model signature.
        input_example: Optional example input data for signature inference.

    Returns:
        ModelInfo: MLflow ModelInfo object.

    Raises:
        RuntimeError: If model is not fitted.
    """
    if not model.is_fitted:
        raise RuntimeError("Model must be fitted before logging to MLflow.")

    pyfunc_wrapper = AnomalyDetectorPyFunc(model)
    return mlflow.pyfunc.log_model(
        artifact_path=artifact_path,
        python_model=pyfunc_wrapper,
        registered_model_name=registered_model_name,
        signature=signature,
        input_example=input_example,
    )


def load_model_from_mlflow(model_uri: str) -> Any:
    """Load model from an MLflow run or Model Registry.

    Args:
        model_uri: MLflow model URI (e.g. 'runs:/<run_id>/model' or 'models:/<name>/1').

    Returns:
        mlflow.pyfunc.PyFuncModel: Loaded MLflow PyFunc model.
    """
    return mlflow.pyfunc.load_model(model_uri)


def export_to_onnx(
    detector: AutoencoderAnomalyDetector,
    output_path: str | Path,
    input_dim: int | None = None,
    batch_size: int = 1,
) -> Path:
    """Export Autoencoder PyTorch network to ONNX format for optimized inference.

    Args:
        detector: Fitted AutoencoderAnomalyDetector instance.
        output_path: Target file path for the exported .onnx model.
        input_dim: Input feature dimension. Inferred from the encoder layer if None.
        batch_size: Batch size for dummy input tensor. Defaults to 1.

    Returns:
        Path: Path to exported ONNX model file.

    Raises:
        RuntimeError: If detector is not fitted or model network is None.
        ValueError: If input dimension cannot be determined.
    """
    if not detector.is_fitted or detector.model is None:
        raise RuntimeError("AutoencoderAnomalyDetector must be fitted before exporting to ONNX.")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if input_dim is None:
        first_layer = detector.model.encoder[0]
        if isinstance(first_layer, nn.Linear):
            input_dim = first_layer.in_features
        else:
            raise ValueError("Could not infer input_dim from model architecture.")

    detector.model.eval()
    dummy_input = torch.randn(batch_size, input_dim, dtype=torch.float32).to(detector.device)

    torch.onnx.export(
        detector.model,
        (dummy_input,),
        str(path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["reconstructed"],
        dynamic_axes={"input": {0: "batch_size"}, "reconstructed": {0: "batch_size"}},
    )

    return path
