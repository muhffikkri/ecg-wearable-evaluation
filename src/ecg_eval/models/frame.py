"""Canonical internal frame representation.

Every downstream layer -- visualization, annotation, segmentation,
preprocessing, R-peak detection, SQI, fuzzy evaluation and export -- consumes
:class:`ECGFrame` and never the original JSONL or raw-folder structures
(IDEA.md section 9).

Verified against the real dataset:

* ``data/29-09-2026/ses*.jsonl`` lines carry ``ecg.samples`` as a list of
  ``[lead_i, lead_ii, lead_iii]`` triples, 250 Hz, 10 s, float, 2500 samples.
* ``data/29-09-2026/Bryan/calibrated/frame_*_mv.npy`` is ``(2500, 3)``
  float32 in mV with the same channel order, described by the sibling
  ``.json`` (unit mV, sample_rate_hz, lead_mapping, calibration).

Both sources therefore normalise to ``(n_samples, 3)`` in mV.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._paths import strip_root

SOURCE_JSONL = "jsonl"
SOURCE_RAW_CALIBRATED = "raw_calibrated"
SOURCE_RAW_MODEL_READY = "raw_model_ready"
#: Built by merging several Raspberry Pi folders, not by reading one file.
SOURCE_RAW_RECONSTRUCTED = "raw_reconstructed"

CHANNEL_ORDER = ("Lead I", "Lead II", "Lead III")


@dataclass
class ECGFrame:
    """One analysis unit: a fixed-length multi-lead ECG recording.

    The default analysis unit for this study is a 10-second window
    (IDEA.md section 18).

    The same class serves both data paths of IDEA-REVISED.md section 10, so the
    analysis pipeline cannot tell whether a frame came from an existing JSONL or
    from a reconstruction. For a reconstructed frame ``signal`` is the
    reconstruction's chosen ECG source and ``provenance`` records where every
    part came from.
    """

    subject_id: str
    session_id: str
    frame_id: str
    source_file: str
    source_format: str
    timestamp: str | None = None
    sampling_rate: float = 250.0
    duration_s: float = 10.0
    signal: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    internal_id: str = ""
    date_folder: str | None = None
    record_index: int = -1
    #: True when a model_ready/filtered array was merged into this frame.
    model_input_available: bool = False
    #: True when an on-device prediction was merged in. Never required.
    prediction_available: bool = False

    def __post_init__(self) -> None:
        self.signal = np.asarray(self.signal, dtype=np.float32)
        if self.signal.ndim == 1:
            self.signal = self.signal[:, None]
        if not self.internal_id:
            self.internal_id = f"{self.subject_id}|{self.session_id}|{self.frame_id}"

    # -- shape ----------------------------------------------------------
    @property
    def signal_shape(self) -> tuple[int, int]:
        return tuple(self.signal.shape)  # type: ignore[return-value]

    @property
    def n_samples(self) -> int:
        return int(self.signal.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.signal.shape[1]) if self.signal.ndim > 1 else 1

    @property
    def duration_actual_s(self) -> float:
        return self.n_samples / self.sampling_rate if self.sampling_rate else float("nan")

    def lead(self, name: str = "Lead II") -> np.ndarray:
        """Return one lead by name, falling back to channel 0."""
        try:
            index = CHANNEL_ORDER.index(name)
        except ValueError:
            index = 0
        if index >= self.n_channels:
            index = 0
        return self.signal[:, index]

    def channel_order(self) -> tuple[str, ...]:
        declared = self.metadata.get("channel_order") or self.provenance.get("channel_order")
        if declared:
            return tuple(str(c) for c in declared)
        return CHANNEL_ORDER[: self.n_channels] or ("ch0",)

    # -- integrity ------------------------------------------------------
    def content_hash(self) -> str:
        """Hash of the actual samples.

        Part of the analysis cache key, so an edited source file invalidates
        cached results (IDEA.md section 50).
        """
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(self.signal).tobytes())
        digest.update(self.internal_id.encode("utf-8"))
        return digest.hexdigest()[:16]

    def signal_issues(self) -> list[str]:
        """Cheap per-frame signal sanity checks (IDEA.md section 43)."""
        issues: list[str] = []
        if self.n_samples == 0:
            issues.append("empty frame")
            return issues
        if not np.all(np.isfinite(self.signal)):
            n_nan = int(np.isnan(self.signal).sum())
            n_inf = int(np.isinf(self.signal).sum())
            if n_nan:
                issues.append(f"{n_nan} NaN sample(s)")
            if n_inf:
                issues.append(f"{n_inf} infinite sample(s)")
        # Flat-line detection: zero dynamic range on the analysis lead.
        peak_to_peak = float(np.nanmax(self.lead()) - np.nanmin(self.lead()))
        if peak_to_peak == 0.0:
            issues.append("flat line (zero dynamic range on analysis lead)")
        return issues

    def summary(self) -> dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "session_id": self.session_id,
            "frame_id": self.frame_id,
            "internal_id": self.internal_id,
            "date_folder": self.date_folder,
            "source_format": self.source_format,
            "source_file": self.source_file,
            "timestamp": self.timestamp,
            "sampling_rate": self.sampling_rate,
            "duration_s": self.duration_actual_s,
            "n_samples": self.n_samples,
            "n_channels": self.n_channels,
            "model_input_available": self.model_input_available,
            "prediction_available": self.prediction_available,
            "mapping_status": self.provenance.get("mapping_status"),
            "mapping_method": self.provenance.get("mapping_method"),
            "device_validation": (self.metadata.get("validation") or {}).get("status"),
        }

    def provenance_view(self) -> dict[str, Any]:
        """Provenance with the recording root stripped, for compact display."""
        root = self.provenance.get("raw_root", "")
        files = self.provenance.get("source_files") or {}
        return {
            "source_type": self.provenance.get("source_type"),
            "mapping_status": self.provenance.get("mapping_status"),
            "mapping_method": self.provenance.get("mapping_method"),
            "signal_source": self.provenance.get("signal_source"),
            "source_files": {
                key: strip_root(value, root) for key, value in files.items()
            },
        }

    def label(self) -> str:
        return f"{self.subject_id} · {self.session_id} · frame {self.frame_id}"


__all__ = [
    "CHANNEL_ORDER",
    "ECGFrame",
    "SOURCE_JSONL",
    "SOURCE_RAW_CALIBRATED",
    "SOURCE_RAW_MODEL_READY",
    "SOURCE_RAW_RECONSTRUCTED",
]
