"""JSONL ingestion.

The canonical JSONL entry schema is defined by ``templates/json-web.jsonl``.
This reader treats that template as the source of truth and does not invent
fields that contradict it (IDEA.md section 5).

Real, verified entry structure (one JSON object per line = one frame)::

    message_id        str    "device01-ses000000000005-frame_000001"
    device_id         str    "device01"
    session_id        str    "ses000000000005"
    patient_id        str    optional, present in the template, absent in some
                               real recordings
    frame_id          str    "000001"  (zero-padded string, not an int)
    created_at        str    ISO-8601 with offset
    sampling_rate_hz  float  250.0
    duration_s        float  10.0
    validation        obj    {status, warnings[]}
    ecg               obj    {format: "samples_by_time",
                             samples: [[lead_i, lead_ii, lead_iii], ...]}
    prediction        obj    device-side model output -- retained as metadata,
                               never used by the evaluation
    system            obj    device telemetry
    stress_test       obj
    network           obj

``ecg.samples`` is stored as ``samples_by_time``: one list per time step, one
value per lead, in the device channel order Lead I / Lead II / Lead III.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from ..models.frame import ECGFrame, SOURCE_JSONL

#: Fields the reader understands. Anything else is preserved under ``extra``.
KNOWN_FIELDS = (
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

SUPPORTED_ECG_FORMATS = ("samples_by_time",)
DEFAULT_SAMPLING_RATE = 250.0


@dataclass
class MalformedRecord:
    """A line that could not be ingested. Never silently discarded."""

    source_file: str
    line_number: int
    reason: str
    excerpt: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_file": self.source_file,
            "line_number": self.line_number,
            "reason": self.reason,
            "excerpt": self.excerpt,
        }


@dataclass
class JSONLReadResult:
    """Frames plus a validation report for one file."""

    frames: list[ECGFrame] = field(default_factory=list)
    malformed: list[MalformedRecord] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_lines: int = 0

    @property
    def status(self) -> str:
        if self.malformed:
            return "MALFORMED"
        if self.warnings:
            return "WARNING"
        return "OK"


def _samples_to_array(samples: Any) -> np.ndarray:
    if not isinstance(samples, list) or not samples:
        raise ValueError("ecg.samples is empty or not a list")
    if isinstance(samples[0], list):
        array = np.asarray(samples, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError(f"ecg.samples has unexpected rank {array.ndim}")
        return array
    # Flat list of scalars -> single lead.
    return np.asarray(samples, dtype=np.float32).reshape(-1, 1)


def subject_id_for(session_id: str, manifest: dict[str, str] | None = None) -> str:
    """Resolve a subject id for a session.

    IDEA.md section 4 warns that filenames are not sufficient to identify
    subjects, so a configurable manifest is consulted first. The fallback is
    the session id itself, which keeps the mapping honest and traceable rather
    than guessing.
    """
    if manifest:
        if session_id in manifest:
            return manifest[session_id]
        stem = Path(session_id).stem
        if stem in manifest:
            return manifest[stem]
    return session_id


def read_jsonl_file(
    path: Path | str,
    *,
    date_folder: str | None = None,
    subject_manifest: dict[str, str] | None = None,
) -> JSONLReadResult:
    """Parse one JSONL recording line by line into canonical frames.

    Accepts a path or a string, so a path handed back by the JSONL writer can be
    read straight back without the caller wrapping it first.
    """
    path = Path(path)
    result = JSONLReadResult()
    seen_frame_ids: set[str] = set()

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            result.n_lines += 1
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                result.malformed.append(
                    MalformedRecord(str(path), line_number, f"invalid JSON: {exc.msg}", stripped[:160])
                )
                continue

            try:
                frame = _record_to_frame(
                    record, path=path, date_folder=date_folder,
                    subject_manifest=subject_manifest, line_number=line_number,
                )
            except ValueError as exc:
                result.malformed.append(
                    MalformedRecord(str(path), line_number, str(exc), stripped[:160])
                )
                continue

            if frame.frame_id in seen_frame_ids:
                result.warnings.append(
                    f"duplicate frame_id {frame.frame_id!r} on line {line_number}"
                )
            seen_frame_ids.add(frame.frame_id)
            result.frames.append(frame)

    result.frames.sort(key=lambda f: (f.timestamp or "", f.frame_id))
    for index, frame in enumerate(result.frames):
        frame.record_index = index

    if not result.frames and not result.malformed:
        result.warnings.append("file contains no JSON records")
    return result


def _record_to_frame(
    record: dict[str, Any],
    *,
    path: Path,
    date_folder: str | None,
    subject_manifest: dict[str, str] | None,
    line_number: int,
) -> ECGFrame:
    if not isinstance(record, dict):
        raise ValueError("line is not a JSON object")

    ecg = record.get("ecg")
    if not isinstance(ecg, dict):
        raise ValueError("missing 'ecg' object")
    ecg_format = ecg.get("format", "samples_by_time")
    if ecg_format not in SUPPORTED_ECG_FORMATS:
        raise ValueError(f"unsupported ecg.format {ecg_format!r}")

    signal = _samples_to_array(ecg.get("samples"))

    session_id = str(record.get("session_id") or path.stem)
    frame_id = str(record.get("frame_id") or f"line{line_number:06d}")
    sampling_rate = float(record.get("sampling_rate_hz") or DEFAULT_SAMPLING_RATE)
    duration_s = float(record.get("duration_s") or (signal.shape[0] / sampling_rate))

    expected_samples = int(round(duration_s * sampling_rate))
    metadata: dict[str, Any] = {
        "device_id": record.get("device_id"),
        "patient_id": record.get("patient_id"),
        "message_id": record.get("message_id"),
        "declared_duration_s": duration_s,
        "declared_sampling_rate_hz": sampling_rate,
        "expected_samples": expected_samples,
        "ecg_format": ecg_format,
        "channel_order": ["Lead I", "Lead II", "Lead III"][: signal.shape[1]],
        "validation": record.get("validation"),
        "prediction": record.get("prediction"),
        "system": record.get("system"),
        "stress_test": record.get("stress_test"),
        "network": record.get("network"),
    }
    extra = {k: v for k, v in record.items() if k not in KNOWN_FIELDS}
    if extra:
        metadata["extra_fields"] = extra

    provenance = {
        "reader": "jsonl_reader",
        "source_path": str(path),
        "line_number": line_number,
        "schema_fields": sorted(record.keys()),
    }

    frame = ECGFrame(
        subject_id=subject_id_for(session_id, subject_manifest),
        session_id=session_id,
        frame_id=frame_id,
        source_file=str(path),
        source_format=SOURCE_JSONL,
        timestamp=record.get("created_at"),
        sampling_rate=sampling_rate,
        duration_s=duration_s,
        signal=signal,
        metadata=metadata,
        provenance=provenance,
        date_folder=date_folder,
    )

    if expected_samples and signal.shape[0] != expected_samples:
        frame.metadata["sample_count_mismatch"] = (
            f"{signal.shape[0]} samples present, {expected_samples} implied by "
            f"duration_s x sampling_rate_hz"
        )
    return frame


def iter_jsonl_files(data_dir: Path) -> Iterator[Path]:
    """Yield every ``*.jsonl`` under ``data/``, excluding converted outputs."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return
    for path in sorted(data_dir.rglob("*.jsonl")):
        parts = {p.lower() for p in path.parts}
        if parts & {"processed", "results", "annotations"}:
            continue
        yield path


__all__ = [
    "DEFAULT_SAMPLING_RATE",
    "JSONLReadResult",
    "KNOWN_FIELDS",
    "MalformedRecord",
    "SUPPORTED_ECG_FORMATS",
    "iter_jsonl_files",
    "read_jsonl_file",
    "subject_id_for",
]
