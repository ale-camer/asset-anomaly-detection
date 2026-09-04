"""Streamlit Real-Time Anomaly Detection & Alerting Dashboard."""

import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.ui.api_client import check_api_health, predict_anomalies
from src.ui.components import (
    calculate_alert_severity,
    format_timeseries_data,
    generate_demo_timeseries,
    render_drift_report_html,
)
from src.utils.config import get_settings

# Configure page layout and aesthetics
st.set_page_config(
    page_title="Asset Anomaly Detection | MLOps Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_dataset() -> tuple[pd.DataFrame, str]:
    """Attempt to load real feature data from disk, falling back to synthetic demo data."""
    settings = get_settings()
    feature_dir = Path(settings.data_features_dir)

    # Check for existing parquet feature files
    if feature_dir.exists():
        parquet_files = list(feature_dir.glob("*.parquet"))
        if parquet_files:
            latest_file = max(parquet_files, key=os.path.getmtime)
            try:
                df = pd.read_parquet(latest_file)
                if not df.empty and "close" in df.columns:
                    return format_timeseries_data(df), f"Live Features ({latest_file.name})"
            except Exception:
                pass

    # Fallback to realistic demo dataset
    demo_df = generate_demo_timeseries(n_samples=150)
    return format_timeseries_data(demo_df), "Synthetic Demo Stream (BTC/USDT)"


def main() -> None:
    """Main application loop."""
    st.title("⚡ Asset Anomaly Detection & Monitoring Platform")
    st.caption("Real-Time MLOps Serving, Drift Observability & Intelligent Market Alerting")

    # ==========================================
    # Sidebar Configuration & Health Check
    # ==========================================
    st.sidebar.header("⚙️ System Configuration")
    api_url: str = st.sidebar.text_input(
        "FastAPI Service URL",
        value=os.getenv("API_BASE_URL", "http://localhost:8000"),
        help="Base URL of the running FastAPI inference backend.",
    )

    threshold: float = st.sidebar.slider(
        "Anomaly Score Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.55,
        step=0.05,
        help="Sensitivity threshold above which samples are flagged as anomalous.",
    )

    st.sidebar.divider()
    st.sidebar.subheader("🩺 Backend Status")
    health = check_api_health(base_url=api_url)

    if health["status"] == "online":
        st.sidebar.success("🟢 API Online: Ready for inference")
        if health.get("model_loaded"):
            st.sidebar.caption("📦 MLflow Model: Loaded in memory")
        else:
            st.sidebar.warning("⚠️ Model is not loaded")
    elif health["status"] == "offline":
        st.sidebar.error("🔴 API Offline (Connection Refused)")
        st.sidebar.caption("Ensure `uvicorn src.api.main:app` is running.")
    else:
        st.sidebar.warning(f"🟡 API {health['status'].upper()}: {health.get('detail', '')}")

    st.sidebar.divider()
    st.sidebar.markdown(
        """
        **Architecture Stack**:
        - **Inference**: FastAPI + MLflow
        - **Monitoring**: Evidently AI + Prometheus
        - **Orchestration**: Apache Airflow
        - **Pipeline**: Parquet Lakehouse
        """
    )

    # ==========================================
    # Load Data & Main Navigation Tabs
    # ==========================================
    df, data_source = load_dataset()

    # Recalculate anomaly flags based on sidebar threshold
    df["is_anomaly"] = df["anomaly_score"] >= threshold

    tab_overview, tab_infer, tab_drift = st.tabs([
        "📈 Time-Series & Anomalies",
        "🚀 Real-Time Inference Simulator",
        "🛡 Observability & Data Drift",
    ])

    # --------------------------------------------------------------------------
    # TAB 1: Time-Series & Anomaly Overview
    # --------------------------------------------------------------------------
    with tab_overview:
        st.subheader("Asset Price Dynamics & Anomaly Timeline")
        st.info(f"📊 Active Data Stream: **{data_source}** | Records: **{len(df)}**")

        # Top Metric Cards
        total_samples = len(df)
        total_anomalies = int(df["is_anomaly"].sum())
        anomaly_rate = (total_anomalies / total_samples) * 100 if total_samples > 0 else 0.0
        avg_score = float(df["anomaly_score"].mean())

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Data Points", f"{total_samples:,}")
        m2.metric("Detected Anomalies", f"{total_anomalies}", delta=f"{total_anomalies} flags", delta_color="inverse")
        m3.metric("Anomaly Rate", f"{anomaly_rate:.2f}%")
        m4.metric("Avg Anomaly Score", f"{avg_score:.3f}")

        # Price Time-Series Chart
        st.markdown("#### Close Price & Rolling Baseline")
        chart_columns = ["close"]
        if "close_rolling_mean_7" in df.columns:
            chart_columns.append("close_rolling_mean_7")

        price_chart_df = df.set_index("timestamp")[chart_columns]
        st.line_chart(price_chart_df)

        # Volatility & Momentum Technical Indicators
        col_vol, col_rsi = st.columns(2)
        with col_vol:
            st.markdown("#### Volatility Metric")
            vol_col = "volatility_parkinson_14" if "volatility_parkinson_14" in df.columns else None
            if vol_col:
                st.area_chart(df.set_index("timestamp")[[vol_col]])
            else:
                st.caption("No volatility feature available.")

        with col_rsi:
            st.markdown("#### RSI Indicator")
            rsi_col = "rsi_14" if "rsi_14" in df.columns else None
            if rsi_col:
                st.line_chart(df.set_index("timestamp")[[rsi_col]])
            else:
                st.caption("No RSI feature available.")

        # Anomaly Log Table
        st.markdown("#### 🚨 Flagged Anomaly Events")
        anomalies_df = df[df["is_anomaly"]].copy()

        if not anomalies_df.empty:
            display_cols = [c for c in ["timestamp", "close", "volume", "anomaly_score", "is_anomaly"] if c in df.columns]
            anomalies_df["Alert Severity"] = anomalies_df["anomaly_score"].apply(
                lambda s: calculate_alert_severity(s, threshold)["badge"]
            )
            st.dataframe(
                anomalies_df[display_cols + ["Alert Severity"]].sort_values("timestamp", ascending=False),
                use_container_width=True,
            )
        else:
            st.success("✅ No anomalies detected under the current threshold.")

    # --------------------------------------------------------------------------
    # TAB 2: Real-Time Inference Simulator
    # --------------------------------------------------------------------------
    with tab_infer:
        st.subheader("Interactive Feature Vector Inference")
        st.markdown(
            "Simulate custom or extreme market conditions and send them live to the "
            "FastAPI `/predict` endpoint to test model scoring and alerting."
        )

        preset = st.selectbox(
            "Quick Scenario Presets",
            ["Custom Input", "Normal Regime", "Flash Crash / High Volatility", "Pump & Dump Outlier"],
        )

        # Preset values defaults
        defaults: dict[str, float] = {
            "close": 64200.0,
            "volume": 1250.0,
            "close_rolling_mean_7": 63900.0,
            "close_rolling_std_7": 450.0,
            "volatility_parkinson_14": 0.025,
            "rsi_14": 52.0,
            "price_velocity_1": 0.005,
        }

        if preset == "Flash Crash / High Volatility":
            defaults.update({
                "close": 53000.0,
                "volume": 7800.0,
                "close_rolling_mean_7": 64000.0,
                "close_rolling_std_7": 2800.0,
                "volatility_parkinson_14": 0.095,
                "rsi_14": 12.0,
                "price_velocity_1": -0.14,
            })
        elif preset == "Pump & Dump Outlier":
            defaults.update({
                "close": 78000.0,
                "volume": 9500.0,
                "close_rolling_mean_7": 61000.0,
                "close_rolling_std_7": 3500.0,
                "volatility_parkinson_14": 0.11,
                "rsi_14": 92.0,
                "price_velocity_1": 0.18,
            })

        col1, col2 = st.columns(2)
        with col1:
            in_close = st.number_input("Close Price ($)", value=float(defaults["close"]), step=100.0)
            in_volume = st.number_input("Volume", value=float(defaults["volume"]), step=50.0)
            in_mean = st.number_input("Rolling Mean (7-period)", value=float(defaults["close_rolling_mean_7"]))
            in_std = st.number_input("Rolling Std (7-period)", value=float(defaults["close_rolling_std_7"]))

        with col2:
            in_vol = st.slider(
                "Parkinson Volatility (14-period)",
                min_value=0.0,
                max_value=0.20,
                value=float(defaults["volatility_parkinson_14"]),
                step=0.005,
            )
            in_rsi = st.slider("RSI (14-period)", min_value=0.0, max_value=100.0, value=float(defaults["rsi_14"]))
            in_velocity = st.slider(
                "Price Velocity (1-period return)",
                min_value=-0.25,
                max_value=0.25,
                value=float(defaults["price_velocity_1"]),
                step=0.01,
            )

        payload_features: dict[str, float] = {
            "close": in_close,
            "volume": in_volume,
            "close_rolling_mean_7": in_mean,
            "close_rolling_std_7": in_std,
            "volatility_parkinson_14": in_vol,
            "rsi_14": in_rsi,
            "price_velocity_1": in_velocity,
        }

        if st.button("🚀 Run Anomaly Inference", type="primary", use_container_width=True):
            with st.spinner("Submitting request to FastAPI backend..."):
                try:
                    res: dict[str, Any] = predict_anomalies(features=[payload_features], base_url=api_url)
                    predictions = res.get("predictions", [])

                    if predictions:
                        pred_item = predictions[0]
                        score = float(pred_item.get("anomaly_score", 0.0))
                        is_anom = bool(pred_item.get("is_anomaly", score >= threshold))
                        model_version = res.get("model_version", "production")

                        severity = calculate_alert_severity(score, threshold=threshold)

                        st.divider()
                        r1, r2, r3 = st.columns(3)
                        r1.metric("Anomaly Score", f"{score:.4f}", help="Raw score from the model.")
                        r2.metric(
                            "Classification",
                            "🚨 ANOMALY" if is_anom else "✅ INLIER",
                            delta="Flagged" if is_anom else "Normal",
                            delta_color="inverse" if is_anom else "normal",
                        )
                        r3.metric("Model Serving Version", str(model_version))

                        # Alert Banner
                        if is_anom:
                            st.error(
                                f"**{severity['badge']}**: {severity['description']} "
                                f"(Score: `{score:.4f}` >= Threshold: `{threshold:.2f}`)"
                            )
                        else:
                            st.success(
                                f"**{severity['badge']}**: {severity['description']} "
                                f"(Score: `{score:.4f}` < Threshold: `{threshold:.2f}`)"
                            )
                    else:
                        st.warning("Empty response received from inference engine.")
                except Exception as ex:
                    st.error(f"Inference error: {ex}")
                    st.info(
                        "Tip: If FastAPI is not running locally, start it via:\n"
                        "```bash\nuvicorn src.api.main:app --host 0.0.0.0 --port 8000\n```"
                    )

    # --------------------------------------------------------------------------
    # TAB 3: Observability & Data Drift
    # --------------------------------------------------------------------------
    with tab_drift:
        st.subheader("Evidently AI Data & Concept Drift Report")
        st.markdown(
            "Monitors statistical distribution drift between the training baseline "
            "and live inference traffic to flag concept degradation."
        )

        settings = get_settings()
        report_path = Path(settings.data_processed_dir) / "reports" / "data_drift_report.html"

        html_content = render_drift_report_html(report_path)

        if html_content:
            st.success(f"Loaded Drift Report from `{report_path}`")
            components.html(html_content, height=800, scrolling=True)
        else:
            st.warning(f"No generated drift report found at `{report_path}`.")
            st.markdown(
                """
                **How to generate a Drift Report:**
                1. Ensure inference or feature data is available.
                2. Execute the drift detection module:
                ```python
                from src.monitoring.drift import generate_drift_report
                generate_drift_report(reference_df, current_df)
                ```
                3. Or trigger the scheduled Airflow DAG.
                """
            )


if __name__ == "__main__":
    main()
