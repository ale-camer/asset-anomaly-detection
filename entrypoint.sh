#!/usr/bin/env bash
set -e

PORT="${PORT:-8000}"

echo "Starting Asset Anomaly Detection API on port ${PORT}..."
exec uvicorn src.api.main:app --host 0.0.0.0 --port "${PORT}"
