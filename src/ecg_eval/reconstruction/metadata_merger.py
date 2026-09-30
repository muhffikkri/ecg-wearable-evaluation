"""Merge metadata from several source folders into one canonical frame.

The rule throughout: a value the firmware actually recorded wins, and a value it
never recorded stays absent. Nothing is back-filled from another folder just to
make the output look complete, because a fabricated ``validation`` status would
be indistinguishable from a measured one.

Conflicts are the interesting case. ``calibrated/`` and ``model_ready/`` both
report ``validation``; when they disagree the canonical frame keeps the one from
the folder that produced the signal and records the divergence, so the user can
see the device's own view and the app's view side by side.
"""

from __future__ import annotations

from typing import Any

from ..models.reconstruction import FrameMapping
from ..models.source_frame import (
    FOLDER_CALIBRATED,
    FOLDER_FILTERED,
    FOLDER_MODEL_READY,
    FOLDER_PREDICTIONS,
    SourceFrame,
)

#: Metadata keys lifted into ``ECGFrame.metadata`` rather than left in extras.
PROMOTED_KEYS = (
    "channel_order",
    "lead_mapping",
    "calibration",
    "source_processing",
    "validation",
    "signal_quality",
)


def _validation_of(record: SourceFrame) -> dict[str, Any] | None:
    """The firmware's own validation verdict, whichever key carries it."""
    for key in ("validation", "signal_quality"):
        value = record.metadata.get(key)
        if isinstance(value, dict) and value.get("status"):
            return value
    return None


def merge_metadata(
    sources: dict[str, SourceFrame],
    mapping: FrameMapping,
) -> tuple[dict[str, Any], list[str]]:
    """Merge per-folder metadata into the canonical frame's ``metadata``.

    Returns ``(metadata, conflicts)``. Precedence for a single-valued key is the
    order of ``sources``, which the reconstructor builds signal-source-first.
    """
    merged: dict[str, Any] = {}
    conflicts: list[str] = []
    validations: dict[str, str] = {}

    for folder, record in sources.items():
        if folder == FOLDER_PREDICTIONS:
            continue
        for key in PROMOTED_KEYS:
            value = record.metadata.get(key)
            if value is None:
                continue
            if key == "validation":
                verdict = _validation_of(record)
                if verdict:
                    validations[folder] = str(verdict.get("status"))
                    merged.setdefault("validation", verdict)
                continue
            if key not in merged:
                merged[key] = value
            elif merged[key] != value:
                conflicts.append(f"{folder} disagrees on {key}")

    # A single disagreement about device-reported validity is worth naming: the
    # two folders ran different checks on the same underlying recording.
    if len(set(validations.values())) > 1:
        detail = ", ".join(f"{folder}={status}" for folder, status in sorted(validations.items()))
        conflicts.append(f"validation status differs across folders ({detail})")

    merged["source_metadata"] = {
        folder: {
            "created_at": record.created_at,
            "sampling_rate_hz": record.sampling_rate,
            "duration_s": record.duration_s,
            "unit": record.unit,
            "dtype": record.dtype,
        }
        for folder, record in sources.items()
        if folder != FOLDER_PREDICTIONS
    }

    device_processing = {
        folder: record.metadata.get("source_processing")
        for folder, record in sources.items()
        if record.metadata.get("source_processing")
    }
    if device_processing:
        merged["device_processing"] = device_processing

    calibration = sources.get(FOLDER_CALIBRATED) or sources.get(FOLDER_FILTERED)
    if calibration is not None and calibration.metadata.get("calibration"):
        merged["calibration"] = calibration.metadata["calibration"]

    model_ready = sources.get(FOLDER_MODEL_READY)
    if model_ready is not None:
        lead_statistics = model_ready.metadata.get("lead_statistics")
        if lead_statistics:
            merged["lead_statistics"] = lead_statistics
        steps = model_ready.metadata.get("preprocessing_steps_applied")
        if steps:
            merged["device_preprocessing_steps"] = steps

    if mapping.warnings:
        merged["merge_warnings"] = list(mapping.warnings)

    return merged, conflicts


def merge_prediction(record: SourceFrame | None) -> dict[str, Any]:
    """Extract the prediction payload for the canonical frame.

    Returns an empty dict when no prediction was linked, which is a normal
    outcome: IDEA-REVISED.md section 5 forbids requiring one.
    """
    if record is None:
        return {}
    prediction: dict[str, Any] = {
        "status": record.metadata.get("status"),
        "label": record.metadata.get("label"),
        "confidence_percent": record.metadata.get("confidence_percent"),
        "probabilities": record.metadata.get("probabilities") or {},
        "threshold": record.metadata.get("threshold"),
        "latency_ms": record.metadata.get("latency_ms"),
        "runtime": record.metadata.get("runtime"),
    }
    if record.metadata.get("error"):
        prediction["error"] = record.metadata["error"]
    if record.metadata.get("input_validation_status"):
        prediction["input_validation_status"] = record.metadata["input_validation_status"]
    if record.metadata.get("input_warnings"):
        prediction["input_warnings"] = list(record.metadata["input_warnings"])
    if record.metadata.get("model_sha256"):
        prediction["model_sha256"] = record.metadata["model_sha256"]
    return prediction


__all__ = ["PROMOTED_KEYS", "merge_metadata", "merge_prediction"]