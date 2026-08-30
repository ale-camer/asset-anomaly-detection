from pathlib import Path

import pytest

from src.utils.config import Settings, get_settings


def test_settings_defaults() -> None:
    """Test that default settings values are correctly loaded."""
    settings = Settings()
    assert settings.environment == "development"
    assert settings.log_level == "INFO"
    assert settings.app_port == 8000
    assert settings.app_host == "0.0.0.0"
    assert settings.coingecko_api_base_url == "https://api.coingecko.com/api/v3"
    assert isinstance(settings.data_raw_dir, Path)
    assert settings.postgres_port == 5432


def test_settings_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that environment variables override default settings."""
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("APP_PORT", "9090")
    monkeypatch.setenv("COINGECKO_API_KEY", "test-api-key")

    settings = Settings()
    assert settings.environment == "production"
    assert settings.log_level == "DEBUG"
    assert settings.app_port == 9090
    assert settings.coingecko_api_key == "test-api-key"


def test_get_settings_singleton() -> None:
    """Test that get_settings returns a valid Settings instance."""
    settings_instance = get_settings()
    assert isinstance(settings_instance, Settings)
