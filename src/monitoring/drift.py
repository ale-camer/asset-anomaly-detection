"""Data Drift and Concept Drift monitoring via Evidently AI."""

import logging
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from src.utils.config import get_settings

logger = logging.getLogger(__name__)


def generate_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    report_name: str = "data_drift_report",
    output_dir: str | Path | None = None,
) -> Path:
    """Generate a Data Drift report using Evidently AI.

    Compares a reference dataset (usually training data) with the current
    dataset (inference data) to detect feature distributions shifts.

    Args:
        reference_df: Baseline DataFrame.
        current_df: Recent production DataFrame to evaluate.
        report_name: Name of the generated report file.
        output_dir: Destination directory. Defaults to 'data/processed/reports'.

    Returns:
        Path: Path to the generated HTML report.

    Raises:
        ValueError: If either DataFrame is empty.
    """
    if reference_df.empty or current_df.empty:
        raise ValueError("Cannot generate drift report with empty DataFrames.")

    settings = get_settings()
    dest_dir = Path(output_dir or (Path(settings.data_processed_dir) / "reports"))
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest_path = dest_dir / f"{report_name}.html"

    logger.info("Generating DataDrift report for %d vs %d samples...", len(reference_df), len(current_df))

    # Initialize and run Evidently report
    drift_report = Report(metrics=[DataDriftPreset()])
    snapshot = drift_report.run(reference_data=reference_df, current_data=current_df)

    # Save to HTML
    snapshot.save_html(str(dest_path))
    logger.info("Drift report saved to %s", dest_path)

    return dest_path
