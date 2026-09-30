"""Build JSONL records from canonical frames.

Field-by-field origin, as IDEA-REVISED.md section 7 requires it to be recorded:

=====================  =========================================================
JSONL field            source
=====================  =========================================================
``message_id``         derived: ``{device_id}-{session_id}-frame_{frame_id}``
``device_id``          recorded in ``source_metadata.device_id``
``session_id``         ``session.json`` / ``source_metadata.session_id``
``patient_id``         **not present in the raw tree** -> reported unavailable
``frame_id``           the canonical frame number
``created_at``         the signal source folder's ``created_at_utc``
``sampling_rate_hz``   the signal source folder's ``sample_rate_hz``
``duration_s``         the signal source folder's ``duration_seconds``
``validation``         ``model_ready`` ``validation`` / ``signal_quality``
``ecg``                the reconstruction's chosen signal source
``prediction``         ``predictions/`` when linked, else omitted
``system``             **not present in the raw tree** -> reported unavailable
``stress_test``        partially derivable: ``enabled`` and ``frame_counter``
``network``            **not present in the raw tree** -> reported unavailable
=====================  =========================================================

Anything the raw tree does not carry is reported as unavailable rather than
filled with a plausible default. A generated ``system.cpu_usage_percent`` would
read as a measurement the device never made.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..models.frame import ECGFrame
from .schema_validator import ECG_FORMAT, RecordValidation, validate_record

#: Fields no Raspberry Pi folder in the observed tree can supply.
UNAVAILABLE_IN_RAW = ("patient_id", "system", "network")


def _device_id(frame: ECGFrame) -> str:
    """Device id from recorded metadata, never from a filename."""
    device = frame.provenance.get("device_id")
    if device:
        return str(device)
    for entry in (frame.metadata.get("source_metadata") or {}).values():
        if isinstance(entry, dict) and entry.get("device_id"):
            return str(entry["device_id"])
    return "unknown-device"


def _validation_block(frame: ECGFrame) -> dict[str, Any]:
    """The device's own verdict, carried through unchanged."""
    verdict = frame.metadata.get("validation") or frame.metadata.get("signal_quality")
    if not isinstance(verdict, dict):
        return {"status": "WARNING", "warnings": []}
    block: dict[str, Any] = {
        "status": verdict.get("status") or "WARNING",
        "warnings": list(verdict.get("warnings") or verdict.get("reasons") or []),
    }
    failures = verdict.get("failures")
    if failures:
        block["failures"] = list(failures)
    return block


def _prediction_block(frame: ECGFrame) -> dict[str, Any] | None:
    prediction = frame.metadata.get("prediction")
    if not isinstance(prediction, dict) or not prediction:
        return None
    block: dict[str, Any] = {"status": prediction.get("status") or "PASS"}
    if prediction.get("label"):
        block["label"] = prediction["label"]
    if prediction.get("confidence_percent") is not None:
        block["confidence_percent"] = prediction["confidence_percent"]
    if prediction.get("probabilities"):
        block["probabilities"] = dict(prediction["probabilities"])
    if prediction.get("threshold") is not None:
        block["threshold"] = prediction["threshold"]
    if prediction.get("latency_ms") is not None:
        block["latency_ms"] = prediction["latency_ms"]
    if prediction.get("runtime"):
        block["runtime"] = prediction["runtime"]
    return block


def _stress_test(frame: ECGFrame) -> dict[str, Any]:
    """Only ``enabled`` is derivable, and only as false: the raw tree records no
    stress-test run. ``frame_counter`` comes from the frame number itself."""
    block: dict[str, Any] = {"enabled": False}
    try:
        block["frame_counter"] = int(frame.frame_id)
    except (TypeError, ValueError):
        pass
    return block


def _samples(frame: ECGFrame) -> list[list[float]]:
    """``samples_by_time``: one row per time step, one value per lead.

    Values go out as full float64 repr of the stored float32, matching what the
    device writes, so a generated file is not distinguishable from a recorded
    one by precision alone.
    """
    signal = np.asarray(frame.signal, dtype=np.float32)
    return [[float(value) for value in row] for row in signal]


@dataclass
class BuiltRecord:
    """A generated record plus its validation outcome and field origins."""

    frame_id: str
    record: dict[str, Any]
    validation: RecordValidation
    #: ``{field: source folder}`` for fields the builder filled in.
    origins: dict[str, str] = field(default_factory=dict)
    unavailable_fields: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.validation.ok

    def line(self) -> str:
        return json.dumps(self.record, separators=(",", ":"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "origins": dict(self.origins),
            "unavailable_fields": list(self.unavailable_fields),
            "validation": self.validation.to_dict(),
        }


def build_record(frame: ECGFrame, *, validate: bool = True) -> BuiltRecord:
    """Build one JSONL record from a canonical frame."""
    device_id = _device_id(frame)
    origins: dict[str, str] = {}
    unavailable: list[str] = []

    record: dict[str, Any] = {
        "message_id": f"{device_id}-{frame.session_id}-frame_{frame.frame_id}",
        "device_id": device_id,
        "session_id": frame.session_id,
        "frame_id": frame.frame_id,
        "created_at": frame.timestamp or "",
        "sampling_rate_hz": float(frame.sampling_rate),
        "duration_s": float(frame.duration_s),
        "validation": _validation_block(frame),
        "ecg": {"format": ECG_FORMAT, "samples": _samples(frame)},
    }
    origins.update(
        {
            "message_id": "derived",
            "device_id": "source_metadata.device_id",
            "session_id": "session metadata",
            "frame_id": "canonical frame number",
            "created_at": f"{frame.provenance.get('signal_source') or 'source'}.created_at_utc",
            "sampling_rate_hz": f"{frame.provenance.get('signal_source') or 'source'}.sample_rate_hz",
            "duration_s": f"{frame.provenance.get('signal_source') or 'source'}.duration_seconds",
            "validation": "model_ready validation/signal_quality",
            "ecg": str(frame.provenance.get("signal_source") or "source"),
        }
    )

    prediction = _prediction_block(frame)
    if prediction is not None:
        record["prediction"] = prediction
        origins["prediction"] = "predictions/"
    else:
        unavailable.append("prediction")

    record["stress_test"] = _stress_test(frame)
    origins["stress_test"] = "derived from the frame number"

    for name in UNAVAILABLE_IN_RAW:
        unavailable.append(name)

    if not record["created_at"]:
        unavailable.append("created_at")

    expected = int(round(float(frame.duration_s) * float(frame.sampling_rate)))
    validation = (
        validate_record(record, frame_id=frame.frame_id, expected_samples=expected)
        if validate
        else RecordValidation(frame_id=frame.frame_id).finalise()
    )
    # validate_record already reports template fields absent from the record.
    # Merge rather than extend, or a field missing for both reasons is listed
    # twice and the manifest's count stops matching the record.
    known = set(validation.unavailable_fields)
    for name in unavailable:
        if name not in known:
            validation.unavailable_fields.append(name)
            known.add(name)

    return BuiltRecord(
        frame_id=frame.frame_id,
        record=record,
        validation=validation,
        origins=origins,
        unavailable_fields=unavailable,
    )


__all__ = ["UNAVAILABLE_IN_RAW", "BuiltRecord", "build_record"]