import pickle
from pathlib import Path

import mlflow
import numpy as np
import pytest

from src.models.autoencoder import AutoencoderAnomalyDetector
from src.models.baseline import IsolationForestDetector, LOFDetector
from src.models.registry import (
    AnomalyDetectorPyFunc,
    export_to_onnx,
    load_model_artifact,
    load_model_from_mlflow,
    log_model_to_mlflow,
    save_model_artifact,
)


@pytest.fixture
def sample_features() -> np.ndarray:
    """Generate reproducible sample features."""
    np.random.seed(42)
    return np.random.normal(loc=0.0, scale=1.0, size=(60, 4))


def test_save_and_load_isolation_forest(sample_features: np.ndarray, tmp_path: Path) -> None:
    model = IsolationForestDetector(n_estimators=20, contamination=0.1, random_state=42)
    model.fit(sample_features)

    filepath = tmp_path / "iforest.pkl"
    saved_path = save_model_artifact(model, filepath)
    assert saved_path.exists()

    loaded_model = load_model_artifact(filepath)
    assert isinstance(loaded_model, IsolationForestDetector)
    assert loaded_model.is_fitted

    preds_orig = model.predict(sample_features)
    preds_loaded = loaded_model.predict(sample_features)
    np.testing.assert_array_equal(preds_orig, preds_loaded)

    scores_orig = model.score_samples(sample_features)
    scores_loaded = loaded_model.score_samples(sample_features)
    np.testing.assert_allclose(scores_orig, scores_loaded)


def test_save_and_load_lof(sample_features: np.ndarray, tmp_path: Path) -> None:
    model = LOFDetector(n_neighbors=10, contamination=0.1)
    model.fit(sample_features)

    filepath = tmp_path / "lof.pkl"
    save_model_artifact(model, filepath)

    loaded_model = load_model_artifact(filepath)
    assert isinstance(loaded_model, LOFDetector)
    assert loaded_model.is_fitted

    np.testing.assert_array_equal(model.predict(sample_features), loaded_model.predict(sample_features))
    np.testing.assert_allclose(model.score_samples(sample_features), loaded_model.score_samples(sample_features))


def test_save_and_load_autoencoder(sample_features: np.ndarray, tmp_path: Path) -> None:
    model = AutoencoderAnomalyDetector(hidden_dim=8, latent_dim=2, epochs=5, batch_size=16, random_seed=42)
    model.fit(sample_features)

    filepath = tmp_path / "autoencoder.pkl"
    save_model_artifact(model, filepath)

    loaded_model = load_model_artifact(filepath)
    assert isinstance(loaded_model, AutoencoderAnomalyDetector)
    assert loaded_model.is_fitted
    assert loaded_model.threshold == pytest.approx(model.threshold)

    np.testing.assert_array_equal(model.predict(sample_features), loaded_model.predict(sample_features))
    np.testing.assert_allclose(
        model.score_samples(sample_features),
        loaded_model.score_samples(sample_features),
        rtol=1e-5,
    )


def test_save_unfitted_model_raises(tmp_path: Path) -> None:
    model = IsolationForestDetector()
    with pytest.raises(RuntimeError, match="must be fitted before saving"):
        save_model_artifact(model, tmp_path / "unfitted.pkl")


def test_load_nonexistent_artifact_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found at"):
        load_model_artifact(tmp_path / "missing.pkl")


def test_load_invalid_type_artifact_raises(tmp_path: Path) -> None:
    invalid_file = tmp_path / "invalid.pkl"
    with open(invalid_file, "wb") as f:
        pickle.dump({"not_a_model": 123}, f)

    with pytest.raises(TypeError, match="is not an instance of BaseAnomalyDetector"):
        load_model_artifact(invalid_file)


def test_mlflow_log_and_load_model(sample_features: np.ndarray) -> None:
    model = AutoencoderAnomalyDetector(hidden_dim=4, latent_dim=2, epochs=3, batch_size=16, random_seed=42)
    model.fit(sample_features)

    with mlflow.start_run():
        model_info = log_model_to_mlflow(model, artifact_path="test_autoencoder_model")
        assert model_info.model_uri is not None

        loaded_pyfunc = load_model_from_mlflow(model_info.model_uri)
        preds_pyfunc = loaded_pyfunc.predict(sample_features)
        preds_orig = model.predict(sample_features)

        np.testing.assert_array_equal(preds_orig, preds_pyfunc)


def test_mlflow_log_unfitted_raises() -> None:
    model = AutoencoderAnomalyDetector()
    with pytest.raises(RuntimeError, match="must be fitted before logging"):
        log_model_to_mlflow(model)


def test_export_to_onnx(sample_features: np.ndarray, tmp_path: Path) -> None:
    detector = AutoencoderAnomalyDetector(hidden_dim=6, latent_dim=2, epochs=2, batch_size=16, random_seed=42)
    detector.fit(sample_features)

    onnx_path = tmp_path / "models" / "autoencoder.onnx"
    exported_path = export_to_onnx(detector, onnx_path)

    assert exported_path.exists()
    assert exported_path.stat().st_size > 0


def test_export_to_onnx_unfitted_raises(tmp_path: Path) -> None:
    detector = AutoencoderAnomalyDetector()
    with pytest.raises(RuntimeError, match="must be fitted before exporting to ONNX"):
        export_to_onnx(detector, tmp_path / "model.onnx")


def test_pyfunc_wrapper_direct_methods(sample_features: np.ndarray) -> None:
    model = IsolationForestDetector(n_estimators=10, random_state=42)
    model.fit(sample_features)

    wrapper = AnomalyDetectorPyFunc(model)
    preds = wrapper.predict(None, sample_features)
    scores = wrapper.score_samples(sample_features)

    assert len(preds) == len(sample_features)
    assert len(scores) == len(sample_features)
