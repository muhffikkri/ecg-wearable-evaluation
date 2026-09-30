"""Per-source frame records read straight from the Raspberry Pi folders.

A :class:`SourceFrame` is deliberately *not* the canonical frame. It is the
unmerged, single-folder view of one recording, and it exists so that
:mod:`ecg_eval.reconstruction` can decide how the folders relate to each other
instead of assuming a conversion (IDEA-REVISED.md sections 4 and 5).

Observed layout of ``data/<date>/raw/<subject>/`` for the 29-09-2026 session
(20 frames, 250 Hz, 10 s, 3 leads, float32 mV):

===============  =========================  ==========================
folder           files                      content
===============  =========================  ==========================
``raw_adc/``     ``frame_*.{bin,csv,json,np}`` uncalibrated ADC counts
``calibrated/``  ``frame_*_mv.{csv,json,np}``  mV after gain/offset
``filtered/``    ``frame_*_mv.{csv,json,np}``  mV after the 50 Hz notch
``model_ready/`` ``frame_*_input.{json,npy}`` filtered mV as AI input
``predictions/`` ``frame_*_prediction.json``  on-device inference result
``logs/``        (empty in this session)     device logs
===============  =========================  ==========================

``model_ready/`` is byte-for-byte equal to ``filtered/`` in this session, and
its JSON records ``source_file`` pointing at ``filtered/frame_*_mv.npy``; that
recorded pointer is the evidence the mapper uses.  It is never assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._paths import strip_root

# Folder names under data/<date>/raw/<subject>/.
FOLDER_RAW_ADC = "raw_adc"
FOLDER_CALIBRATED = "calibrated"
FOLDER_FILTERED = "filtered"
FOLDER_MODEL_READY = "model_ready"
FOLDER_PREDICTIONS = "predictions"
FOLDER_LOGS = "logs"

RAW_KNOWN_SUBDIRS: tuple[str, ...] = (
    FOLDER_CALIBRATED,
    FOLDER_MODEL_READY,
    FOLDER_FILTERED,
    FOLDER_PREDICTIONS,
    FOLDER_LOGS,
    FOLDER_RAW_ADC,
)

# Folders that carry an ECG array, in the order the pipeline prefers them when
# the analysis lead has to be chosen.
SIGNAL_FOLDERS: tuple[str, ...] = (FOLDER_CALIBRATED, FOLDER_FILTERED, FOLDER_MODEL_READY)

# Folders a dedicated reader is registered for. ``raw_adc/`` is deliberately
# absent: it holds uncalibrated ADC counts, which this study never analyses, so
# it is reported as present-but-not-read rather than as zero frames.
READ_FOLDERS: tuple[str, ...] = (
    FOLDER_CALIBRATED,
    FOLDER_FILTERED,
    FOLDER_MODEL_READY,
    FOLDER_PREDICTIONS,
    FOLDER_LOGS,
)

# Suffix emitted by the firmware for each signal folder. ``calibrated`` and
# ``filtered`` share it, which is why frame identity cannot come from the
# filename alone once both are read.
CALIBRATED_SUFFIX = "mv"
MODEL_READY_SUFFIX = "input"
PREDICTION_SUFFIX = "prediction"


@dataclass
class SourceFrame:
    """One recording as published by a single Raspberry Pi output folder.

    Attributes mirror what the firmware actually writes, so a field that is
    absent on disk stays ``None`` instead of being invented. Nothing here is
    cross-checked against another folder; that is the mapper's job.
    """

    folder: str
    frame_number: int | None = None
    signal: np.ndarray | None = field(default=None, repr=False)  # type: ignore[assignment]
    sampling_rate: float | None = None
    duration_s: float | None = None
    unit: str | None = None
    dtype: str | None = None
    channel_order: list[str] | None = None
    created_at: str | None = None
    files: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def frame_id(self) -> str:
        """Zero-padded frame number, or an empty string when unparseable."""
        return "" if self.frame_number is None else f"{self.frame_number:06d}"

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def shape(self) -> tuple[int, ...]:
        return () if self.signal is None else tuple(self.signal.shape)

    def relative_files(self, root: str) -> dict[str, str]:
        """File map with ``root`` stripped, for compact provenance output."""
        return {key: strip_root(value, root) for key, value in self.files.items()}

    def summary(self) -> dict[str, Any]:
        return {
            "folder": self.folder,
            "frame_id": self.frame_id,
            "shape": list(self.shape),
            "sampling_rate_hz": self.sampling_rate,
            "duration_s": self.duration_s,
            "unit": self.unit,
            "channel_order": list(self.channel_order or ()),
            "created_at": self.created_at,
            "files": dict(self.files),
            "errors": list(self.errors),
        }


@dataclass
class SourceDataset:
    """Everything found under one ``data/<date>/raw/<subject>/`` folder."""

    root: str
    subject_id: str
    session_id: str = ""
    device_id: str | None = None
    session_metadata: dict[str, Any] = field(default_factory=dict)
    frames: dict[str, list[SourceFrame]] = field(default_factory=dict)
    predictions: list[SourceFrame] = field(default_factory=list)
    logs: list[dict[str, Any]] = field(default_factory=list)
    present_dirs: dict[str, bool] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)

    def by_folder(self, folder: str) -> list[SourceFrame]:
        return self.frames.get(folder, [])

    def frame_map(self, folder: str) -> dict[int, SourceFrame]:
        """``{frame_number: SourceFrame}`` for one folder, bad reads dropped."""
        return {
            frm.frame_number: frm
            for frm in self.by_folder(folder)
            if frm.frame_number is not None
        }

    def prediction_map(self) -> dict[int, SourceFrame]:
        return {
            frm.frame_number: frm
            for frm in self.predictions
            if frm.frame_number is not None
        }

    def count(self, folder: str) -> int:
        if folder == FOLDER_PREDICTIONS:
            return len(self.predictions)
        if folder == FOLDER_LOGS:
            return len(self.logs)
        return len(self.by_folder(folder))

    def frame_numbers(self, folder: str) -> set[int]:
        return set(self.frame_map(folder))

    def not_read(self) -> list[str]:
        """Present folders with no reader, so absence is never read as "empty"."""
        return [
            name
            for name in RAW_KNOWN_SUBDIRS
            if name not in READ_FOLDERS and self.present_dirs.get(name)
        ]

    def status(self) -> str:
        if self.errors:
            return "ERROR"
        if self.warnings:
            return "WARNING"
        if not any(self.present_dirs.get(f) for f in SIGNAL_FOLDERS):
            return "ERROR"
        return "VALID"

    def summary(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "subject_id": self.subject_id,
            "session_id": self.session_id,
            "device_id": self.device_id,
            "status": self.status(),
            "present_dirs": dict(self.present_dirs),
            "counts": {f: self.count(f) for f in READ_FOLDERS},
            "present_not_read": self.not_read(),
            "session_metadata": dict(self.session_metadata),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }


__all__ = [
    "CALIBRATED_SUFFIX",
    "FOLDER_CALIBRATED",
    "FOLDER_FILTERED",
    "FOLDER_LOGS",
    "FOLDER_MODEL_READY",
    "FOLDER_PREDICTIONS",
    "FOLDER_RAW_ADC",
    "MODEL_READY_SUFFIX",
    "PREDICTION_SUFFIX",
    "RAW_KNOWN_SUBDIRS",
    "READ_FOLDERS",
    "SIGNAL_FOLDERS",
    "SourceDataset",
    "SourceFrame",
]
