from typing import Any, cast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.base import BaseAnomalyDetector


class AutoencoderNetwork(nn.Module):
    """Deep Neural Network Autoencoder for Time-Series Market Features."""

    def __init__(self, input_dim: int, hidden_dim: int = 16, latent_dim: int = 4) -> None:
        """Initialize autoencoder network.

        Args:
            input_dim: Number of input features.
            hidden_dim: Number of units in hidden layers. Defaults to 16.
            latent_dim: Number of units in bottleneck latent space. Defaults to 4.
        """
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder and decoder.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Reconstructed tensor of shape (batch_size, input_dim).
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return cast(torch.Tensor, decoded)


class AutoencoderAnomalyDetector(BaseAnomalyDetector):
    """Deep Learning anomaly detector based on PyTorch Autoencoder reconstruction error."""

    def __init__(
        self,
        hidden_dim: int = 16,
        latent_dim: int = 4,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 32,
        threshold_std_factor: float = 3.0,
        feature_cols: list[str] | None = None,
        random_seed: int = 42,
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Autoencoder Anomaly Detector.

        Args:
            hidden_dim: Number of hidden units in layers. Defaults to 16.
            latent_dim: Number of units in latent bottleneck. Defaults to 4.
            lr: Learning rate for Adam optimizer. Defaults to 1e-3.
            epochs: Number of training epochs. Defaults to 50.
            batch_size: Training batch size. Defaults to 32.
            threshold_std_factor: Multiplier for standard deviation above mean reconstruction error
                                  used to set anomaly threshold. Defaults to 3.0.
            feature_cols: Optional list of column names if input is DataFrame.
            random_seed: Random seed for reproducibility. Defaults to 42.
            device: Computing device ('cpu' or 'cuda'). Defaults to CPU if None.
            **kwargs: Extra parameters.
        """
        super().__init__(feature_cols=feature_cols, **kwargs)
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold_std_factor = threshold_std_factor
        self.random_seed = random_seed

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model: AutoencoderNetwork | None = None
        self.threshold: float = 0.0

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any = None) -> "AutoencoderAnomalyDetector":
        """Train the Autoencoder on normal market data.

        Args:
            X: Input feature matrix or DataFrame.
            y: Ignored.

        Returns:
            self
        """
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        features = self._prepare_features(X)
        input_dim = features.shape[1]

        # Log hyperparameters if MLflow tracking run is active
        self._log_mlflow_params(
            {
                "model_type": self.__class__.__name__,
                "input_dim": input_dim,
                "hidden_dim": self.hidden_dim,
                "latent_dim": self.latent_dim,
                "lr": self.lr,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "threshold_std_factor": self.threshold_std_factor,
                "random_seed": self.random_seed,
            }
        )

        self.model = AutoencoderNetwork(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
        ).to(self.device)

        tensor_x = torch.tensor(features, dtype=torch.float32)
        dataset = TensorDataset(tensor_x)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            num_batches = 0
            for batch in dataloader:
                inputs = batch[0].to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item())
                num_batches += 1

            epoch_loss = total_loss / max(1, num_batches)
            self._log_mlflow_metric("train_loss", epoch_loss, step=epoch)

        self.is_fitted = True

        # Calculate reconstruction error on training data to set anomaly threshold
        train_errors = self.score_samples(X)
        mean_err = float(np.mean(train_errors))
        std_err = float(np.std(train_errors))
        self.threshold = mean_err + self.threshold_std_factor * std_err

        self._log_mlflow_metrics(
            {
                "reconstruction_error_mean": mean_err,
                "reconstruction_error_std": std_err,
                "anomaly_threshold": self.threshold,
            }
        )

        return self

    def score_samples(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Compute per-sample reconstruction Mean Squared Error (MSE).

        Args:
            X: Input feature matrix or DataFrame.

        Returns:
            np.ndarray: 1D array of reconstruction errors (higher indicates greater anomaly).
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("AutoencoderAnomalyDetector is not fitted yet. Call 'fit' before 'score_samples'.")

        features = self._prepare_features(X)
        tensor_x = torch.tensor(features, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(tensor_x)
            # Per-sample MSE loss: mean across feature dimension
            mse_per_sample = torch.mean((tensor_x - reconstructed) ** 2, dim=1)

        scores = mse_per_sample.cpu().numpy()
        return np.asarray(scores, dtype=float)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict binary anomaly labels (1 for anomaly, 0 for normal).

        Args:
            X: Input feature matrix or DataFrame.

        Returns:
            np.ndarray: 1D binary array (1 = anomaly, 0 = normal).
        """
        if not self.is_fitted:
            raise RuntimeError("AutoencoderAnomalyDetector is not fitted yet. Call 'fit' before 'predict'.")

        scores = self.score_samples(X)
        preds = (scores > self.threshold).astype(int)
        return np.asarray(preds, dtype=int)
