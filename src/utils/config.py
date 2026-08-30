from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application and environment configuration settings."""

    # Application Settings
    environment: str = "development"
    log_level: str = "INFO"
    app_port: int = 8000
    app_host: str = "0.0.0.0"

    # Market Data API Settings
    coingecko_api_base_url: str = "https://api.coingecko.com/api/v3"
    coingecko_api_key: str | None = None

    # Storage & Lakehouse Directories
    data_raw_dir: Path = Path("./data/raw")
    data_processed_dir: Path = Path("./data/processed")
    data_features_dir: Path = Path("./data/features")

    # Database & Metadata Backend
    postgres_user: str = "mlops_user"
    postgres_password: str = "mlops_password"
    postgres_db: str = "anomaly_detection_db"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    database_url: str | None = None

    # MLflow Tracking & Registry
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "asset-anomaly-detection"
    mlflow_s3_endpoint_url: str = "http://localhost:9000"

    # MinIO / S3 Object Storage
    minio_root_user: str = "minioadmin"
    minio_root_password: str = "minioadmin"
    minio_default_buckets: str = "mlflow-artifacts,data-lake"
    aws_access_key_id: str = "minioadmin"
    aws_secret_access_key: str = "minioadmin"

    # Airflow Settings
    airflow_core_load_examples: bool = False
    airflow_core_executor: str = "LocalExecutor"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton instance of application settings."""
    return Settings()


settings: Settings = get_settings()
