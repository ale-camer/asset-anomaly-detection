"""UI module for Streamlit anomaly detection dashboard."""

from src.ui.api_client import check_api_health, predict_anomalies
from src.ui.components import (
    calculate_alert_severity,
    format_timeseries_data,
    render_drift_report_html,
)

__all__ = [
    "check_api_health",
    "predict_anomalies",
    "calculate_alert_severity",
    "format_timeseries_data",
    "render_drift_report_html",
]
