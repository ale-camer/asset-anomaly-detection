FROM python:3.11-slim

WORKDIR /app

# Install system runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy packaging manifests and install project dependencies
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir .

# Copy application source code and bundled production model
COPY src/ ./src/
COPY data/models/ ./data/models/
COPY entrypoint.sh ./

RUN chmod +x /app/entrypoint.sh

ENV PORT=8000
ENV MODEL_PATH=/app/data/models/production_model.pkl
EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
