"""Reader for ``predictions/``: the on-device inference result per frame.

Files are ``frame_000001_prediction.json`` and carry a full result, including
the hash of the input array they were computed from::

    "frame_id": "000001"
    "source_file": ".../model_ready/frame_000001_input.npy"
    "source_sha256": "5d5c68ed..."
    "prediction": "Normal", "confidence_percent": 99.73, ...

``source_sha256`` is optional corroboration for the mapper: when present it can
confirm that the prediction really belongs to the model_ready frame on disk.
``latest_prediction.json`` and ``mqtt_publish_state.json`` are session-level
state, not per-frame, and are skipped.

A prediction is never required to reconstruct an ECG frame
(IDEA-REVISED.md section 5); the reader therefore reports what it finds and
leaves absence to the reconstruction report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..models.source_frame import FOLDER_PREDICTIONS, PREDICTION_SUFFIX, SourceFrame
from .source_common import parse_frame_id, parse_frame_number, read_json

# Session-level files that live in predictions/ but belong to no single frame.
NON_FRAME_FILES = ("latest_prediction.json", "mqtt_publish_state.json")


def read_predictions(root: Path | str) -> list[SourceFrame]:
    """Read ``<root>/predictions/`` into one :class:`SourceFrame` per frame."""
    directory = Path(root) / FOLDER_PREDICTIONS
    frames: list[SourceFrame] = []
    if not directory.is_dir():
        return frames

    for path in sorted(directory.glob(f"frame_*_{PREDICTION_SUFFIX}.json")):
        record = SourceFrame(folder=FOLDER_PREDICTIONS)
        record.files = {"json": str(path)}
        try:
            record.frame_number = parse_frame_number(path.stem)
        except ValueError as exc:
            record.errors.append(str(exc))
            frames.append(record)
            continue

        try:
            payload = read_json(path)
        except Exception as exc:  # noqa: BLE001
            record.errors.append(f"unreadable json: {exc}")
            frames.append(record)
            continue

        declared = payload.get("frame_id")
        if declared is not None:
            try:
                # predictions record the bare "000001", not "frame_000001".
                declared_number = parse_frame_id(str(declared))
            except ValueError:
                record.errors.append(f"unparseable json frame_id {declared!r}")
            else:
                if declared_number != record.frame_number:
                    record.errors.append(
                        f"json frame_id {declared!r} disagrees with filename "
                        f"frame {record.frame_number:06d}"
                    )

        record.created_at = payload.get("created_at")
        record.sampling_rate = (
            None if payload.get("sampling_rate_hz") is None else float(payload["sampling_rate_hz"])
        )
        record.unit = payload.get("unit")
        record.dtype = payload.get("dtype")

        record.metadata = {
            "schema_version": payload.get("schema_version"),
            "status": payload.get("status"),
            "label": payload.get("prediction"),
            "confidence_percent": payload.get("confidence_percent"),
            "probabilities": payload.get("probabilities"),
            "threshold": payload.get("threshold"),
            "latency_ms": payload.get("latency_ms"),
            "runtime": payload.get("runtime"),
            "model_path": payload.get("model_path"),
            "model_sha256": payload.get("model_sha256"),
            "warning": payload.get("warning"),
            "error": payload.get("error"),
            "input_validation_status": payload.get("input_validation_status"),
            "input_warnings": payload.get("input_warnings"),
            "source_file": payload.get("source_file"),
            "source_sha256": payload.get("source_sha256"),
            # Normalised to the same shape model_ready uses, so the mapper's
            # pointer index can read both folders through one lookup.
            "recorded_source_files": (
                {"source_file": payload["source_file"]} if payload.get("source_file") else {}
            ),
        }
        frames.append(record)

    return frames


__all__ = ["NON_FRAME_FILES", "read_predictions"]
