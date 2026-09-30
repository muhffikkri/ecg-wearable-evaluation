"""The JSONL schema, derived from ``templates/json-web.jsonl``.

IDEA-REVISED.md section 7 makes the template the source of truth: read it,
understand every field, map the canonical frame onto it, and validate before
writing. So the schema below is a transcription of the template rather than an
independent design, and :func:`load_template` lets a test prove the two still
agree.

A real line from ``data/29-09-2026/ses000000000005.jsonl``::

    message_id        "device01-ses000000000005-frame_000001"
    device_id         "device01"
    session_id        "ses000000000005"
    patient_id        "PAT-001"        -- in the template, absent in that file
    frame_id          "000001"         -- zero-padded string, not an int
    created_at        "2026-09-28T20:05:52+07:00"
    sampling_rate_hz  250.0
    duration_s        10.0
    validation        {"status": "PASS", "warnings": []}
    ecg               {"format": "samples_by_time", "samples": [[i, ii, iii], ...]}
    prediction        {"status": "PASS", "label": "Normal", "confidence_percent": 99.67,
                      "probabilities": {...}, "threshold": 0.5, "latency_ms": 251.73,
                      "runtime": "ai-edge-litert"}
    system            {"cpu_usage_percent": 14.2, ...}
    stress_test       {"enabled": false, "frame_counter": 1}
    network           {"mqtt_publish_latency_ms": 74219.7, "wifi_rssi_dbm": -60,
                      "mqtt_connected": true}

``ecg.samples`` stores full float64 repr of the float32 samples, which is what the
device writes; rounding here would make generated files differ from recorded
ones for no benefit.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
TEMPLATE_PATH = REPO_ROOT / "templates" / "json-web.jsonl"

ECG_FORMAT = "samples_by_time"

#: Top-level keys the template defines, in template order.
TEMPLATE_FIELDS = (
    "message_id",
    "device_id",
    "session_id",
    "patient_id",
    "frame_id",
    "created_at",
    "sampling_rate_hz",
    "duration_s",
    "validation",
    "ecg",
    "prediction",
    "system",
    "stress_test",
    "network",
)

#: Without these a record cannot be read back as a frame.
REQUIRED_FIELDS = (
    "message_id",
    "device_id",
    "session_id",
    "frame_id",
    "created_at",
    "sampling_rate_hz",
    "duration_s",
    "validation",
    "ecg",
)

#: Present in the template but legitimately absent in real recordings.
OPTIONAL_FIELDS = ("patient_id", "prediction", "system", "stress_test", "network")

VALIDATION_STATUSES = ("PASS", "WARNING", "ERROR")


@dataclass
class ValidationIssue:
    """One problem with a generated record, at WARNING or ERROR severity."""

    field: str
    severity: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {"field": self.field, "severity": self.severity, "message": self.message}


@dataclass
class RecordValidation:
    """Outcome of checking one generated record against the schema."""

    frame_id: str = ""
    status: str = "VALID"
    issues: list[ValidationIssue] = field(default_factory=list)
    #: Fields the schema defines that no source provided. Reported, never faked.
    unavailable_fields: list[str] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "ERROR"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [issue for issue in self.issues if issue.severity == "WARNING"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, field_name: str, message: str) -> None:
        self.issues.append(ValidationIssue(field_name, "ERROR", message))

    def warn(self, field_name: str, message: str) -> None:
        self.issues.append(ValidationIssue(field_name, "WARNING", message))

    def finalise(self) -> "RecordValidation":
        self.status = "ERROR" if self.errors else ("WARNING" if self.warnings else "VALID")
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "status": self.status,
            "issues": [issue.to_dict() for issue in self.issues],
            "unavailable_fields": list(self.unavailable_fields),
        }


#: ``templates/json-web.jsonl`` abbreviates its sample list with ``[ ... ]``, so
#: the file is a schema illustration, not parseable JSON. The row width is
#: inferred from the one complete row that precedes the elision.
ELISION_RE = re.compile(r"\[\s*\.\.\.\s*\]")
_FIRST_ROW_RE = re.compile(r"\[\s*-?[\d.]+(?:[eE][+-]?\d+)?(?:\s*,\s*-?[\d.]+(?:[eE][+-]?\d+)?)+\s*\]")


def expand_elisions(text: str) -> str:
    """Replace ``[ ... ]`` placeholders with a zero row of the right width."""
    if not ELISION_RE.search(text):
        return text
    match = _FIRST_ROW_RE.search(text)
    width = match.group(0).count(",") + 1 if match else 3
    filler = "[" + ",".join("0.0" for _ in range(width)) + "]"
    return ELISION_RE.sub(filler, text)


def load_template(path: Path | str | None = None) -> dict[str, Any]:
    """Parse ``templates/json-web.jsonl`` and return its first record.

    The template's sample list is elided, so elisions are expanded first. Only
    the field structure is meaningful here; the numeric values are illustrative.
    """
    target = Path(path) if path else TEMPLATE_PATH
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                return json.loads(expand_elisions(stripped))
    raise ValueError(f"{target} contains no JSON record")


def template_fields(path: Path | str | None = None) -> tuple[str, ...]:
    """Top-level keys the template actually declares."""
    return tuple(load_template(path).keys())


def validate_record(
    record: dict[str, Any],
    *,
    frame_id: str = "",
    expected_samples: int | None = None,
) -> RecordValidation:
    """Check one generated record against the template schema.

    Checks structure and types. It cannot judge whether the *values* are
    physiologically sensible -- that is what the SQI pipeline is for.
    """
    result = RecordValidation(frame_id=frame_id or str(record.get("frame_id") or ""))

    if not isinstance(record, dict):
        result.error("(record)", "record is not a JSON object")
        return result.finalise()

    for name in REQUIRED_FIELDS:
        if name not in record:
            result.error(name, f"required field {name!r} is missing")

    for name in TEMPLATE_FIELDS:
        if name not in record and name not in REQUIRED_FIELDS:
            result.unavailable_fields.append(name)

    unknown = [name for name in record if name not in TEMPLATE_FIELDS]
    if unknown:
        result.warn("(record)", f"fields outside the template schema: {sorted(unknown)}")

    for name in ("message_id", "device_id", "session_id", "frame_id", "created_at"):
        if name in record and not isinstance(record[name], str):
            result.error(name, f"{name} must be a string, got {type(record[name]).__name__}")

    frame_value = record.get("frame_id")
    if isinstance(frame_value, str) and frame_value and not frame_value.isdigit():
        result.error("frame_id", f"frame_id {frame_value!r} is not a zero-padded numeric string")

    for name in ("sampling_rate_hz", "duration_s"):
        value = record.get(name)
        if value is not None and not isinstance(value, (int, float)):
            result.error(name, f"{name} must be numeric, got {type(value).__name__}")
        elif isinstance(value, (int, float)) and value <= 0:
            result.error(name, f"{name} must be positive, got {value}")

    validation = record.get("validation")
    if validation is not None:
        if not isinstance(validation, dict):
            result.error("validation", "validation must be an object")
        else:
            status = validation.get("status")
            if status is not None and status not in VALIDATION_STATUSES:
                result.warn(
                    "validation.status",
                    f"status {status!r} is outside {VALIDATION_STATUSES}",
                )
            warnings = validation.get("warnings")
            if warnings is not None and not isinstance(warnings, list):
                result.error("validation.warnings", "warnings must be a list")

    ecg = record.get("ecg")
    if not isinstance(ecg, dict):
        if "ecg" in record:
            result.error("ecg", "ecg must be an object")
    else:
        ecg_format = ecg.get("format", ECG_FORMAT)
        if ecg_format != ECG_FORMAT:
            result.error("ecg.format", f"unsupported ecg.format {ecg_format!r}")
        samples = ecg.get("samples")
        if not isinstance(samples, list) or not samples:
            result.error("ecg.samples", "ecg.samples must be a non-empty list")
        else:
            widths = {len(row) for row in samples if isinstance(row, list)}
            if len(widths) > 1:
                result.error("ecg.samples", f"rows have inconsistent channel counts {sorted(widths)}")
            elif widths and expected_samples is not None and len(samples) != expected_samples:
                result.error(
                    "ecg.samples",
                    f"{len(samples)} samples present, {expected_samples} implied by "
                    "duration_s x sampling_rate_hz",
                )

    prediction = record.get("prediction")
    if prediction is not None and not isinstance(prediction, dict):
        result.error("prediction", "prediction must be an object")

    return result.finalise()


__all__ = [
    "ECG_FORMAT",
    "ELISION_RE",
    "OPTIONAL_FIELDS",
    "REQUIRED_FIELDS",
    "TEMPLATE_FIELDS",
    "TEMPLATE_PATH",
    "VALIDATION_STATUSES",
    "RecordValidation",
    "ValidationIssue",
    "expand_elisions",
    "load_template",
    "template_fields",
    "validate_record",
]