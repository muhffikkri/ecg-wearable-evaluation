"""Automated scientific interpretation of SQI results."""

from .engine import (
    SQI_INTERPRETATION,
    acquisition_text,
    artifact_text,
    build_report,
    describe_frame,
    limitations_text,
    low_sqi_factors,
    overall_finding,
    position_comparison_text,
    problematic_frames,
    sqi_interpretation_text,
    variability_text,
)

__all__ = [
    "SQI_INTERPRETATION",
    "acquisition_text",
    "artifact_text",
    "build_report",
    "describe_frame",
    "limitations_text",
    "low_sqi_factors",
    "overall_finding",
    "position_comparison_text",
    "problematic_frames",
    "sqi_interpretation_text",
    "variability_text",
]
