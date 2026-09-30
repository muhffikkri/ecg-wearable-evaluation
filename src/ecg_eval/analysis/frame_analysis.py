"""Segmentation into the 10-second evaluation unit.

If a source frame already represents the configured analysis duration it is
used directly -- no resampling, no reconstruction (IDEA.md section 18). Only
continuous recordings that are not already analysis frames get windowed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from ..models.frame import ECGFrame


@dataclass
class Segment10s:
    """A 10-second analysis window carved from one source frame."""

    segment_id: str
    subject_id: str
    session_id: str
    position: str
    start_time: str | None
    end_time: str | None
    sample_count: int
    sampling_rate: float
    start_sample: int
    signal: np.ndarray
    source_frame: ECGFrame
    metadata: dict[str, Any]

    def record(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "subject_id": self.subject_id,
            "session_id": self.session_id,
            "position": self.position,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "sample_count": self.sample_count,
            "sampling_rate": self.sampling_rate,
            "start_sample": self.start_sample,
        }


def segment_frame(
    frame: ECGFrame,
    *,
    position: str,
    analysis_duration_s: float = 10.0,
    max_windows: int | None = None,
) -> list[Segment10s]:
    """Split one canonical frame into analysis windows.

    A frame whose duration already equals ``analysis_duration_s`` (within one
    sample) yields exactly one segment covering the whole frame, so the
    intended analysis frame passes through untouched.

    A frame longer than the window is cut into whole windows only; any trailing
    partial window is dropped rather than zero-padded, because padding would
    fabricate samples that were never recorded. The dropped length is recorded on
    the last segment so it is visible instead of implicit.
    """
    fs = frame.sampling_rate
    window = int(round(analysis_duration_s * fs))
    if window <= 0:
        return []

    n_samples = frame.n_samples

    if abs(n_samples - window) <= 1:
        return [
            Segment10s(
                segment_id=f"{frame.internal_id}#w0",
                subject_id=frame.subject_id,
                session_id=frame.session_id,
                position=position,
                start_time=frame.timestamp,
                end_time=frame.timestamp,
                sample_count=n_samples,
                sampling_rate=fs,
                start_sample=0,
                signal=frame.signal,
                source_frame=frame,
                metadata={"segmentation": "source frame used directly"},
            )
        ]

    segments: list[Segment10s] = []
    n_windows = n_samples // window
    dropped_samples = n_samples - n_windows * window
    if max_windows is not None:
        n_windows = min(n_windows, max_windows)
    for index in range(n_windows):
        start = index * window
        stop = start + window
        metadata: dict[str, Any] = {
            "segmentation": "windowed",
            "window_index": index,
            "total_samples": n_samples,
        }
        if dropped_samples:
            metadata["dropped_trailing_samples"] = int(dropped_samples)
        if max_windows is not None and index == n_windows - 1 and n_samples // window > n_windows:
            metadata["truncated_by_max_windows"] = True
        segments.append(
            Segment10s(
                segment_id=f"{frame.internal_id}#w{index}",
                subject_id=frame.subject_id,
                session_id=frame.session_id,
                position=position,
                start_time=frame.timestamp,
                end_time=frame.timestamp,
                sample_count=window,
                sampling_rate=fs,
                start_sample=start,
                signal=frame.signal[start:stop, :],
                source_frame=frame,
                metadata=metadata,
            )
        )
    return segments


def segment_frames(
    frames: Iterable[ECGFrame],
    labels: dict[str, str],
    *,
    analysis_duration_s: float = 10.0,
) -> list[Segment10s]:
    """Segment every frame, skipping frames with no static-position label."""
    out: list[Segment10s] = []
    for frame in frames:
        label = labels.get(frame.internal_id, "")
        if not label:
            continue
        out.extend(segment_frame(frame, position=label, analysis_duration_s=analysis_duration_s))
    return out


__all__ = ["Segment10s", "segment_frame", "segment_frames"]
