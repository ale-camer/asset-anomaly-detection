"""Prometheus metrics definition for the FastAPI inference service."""

from prometheus_client import Counter, Histogram

# Total number of prediction requests received
PREDICTION_REQUESTS = Counter(
    "anomaly_prediction_requests_total",
    "Total number of prediction requests processed",
)

# Total number of anomalies detected
ANOMALIES_DETECTED = Counter(
    "anomalies_detected_total",
    "Total number of anomalies flagged by the model",
)

# Distribution of anomaly scores
ANOMALY_SCORE_HISTOGRAM = Histogram(
    "anomaly_score_distribution",
    "Distribution of anomaly scores returned by the model",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
