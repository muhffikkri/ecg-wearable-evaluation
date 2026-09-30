"""Annotation manager: label resolution and per-frame lookup.

Separates "what label does this frame have" from "how it is stored", so the UI
can preview changes before saving (IDEA.md section 48).
"""

from __future__ import annotations

from typing import Iterable

from ..models.annotation import (
    ASSIGNABLE_LABELS,
    STATIC_POSITIONS,
    TRANSITION,
    UNLABELED,
    Segment,
    SubjectAnnotation,
    validate_annotations,
)
from . import storage

#: Order used for the timeline legend in the UI.
LABEL_ORDER = (*ASSIGNABLE_LABELS, UNLABELED)


def get_or_create(
    annotations: list[SubjectAnnotation], subject_id: str, session_id: str
) -> SubjectAnnotation:
    """Fetch the subject's annotation record, creating an empty one if absent."""
    return _ensure(annotations, subject_id, session_id)


def _ensure(annotations: list[SubjectAnnotation], subject_id: str, session_id: str) -> SubjectAnnotation:
    index = storage.annotation_index(annotations)
    key = (subject_id, session_id)
    if key not in index:
        record = SubjectAnnotation(subject_id=subject_id, session_id=session_id)
        annotations.append(record)
        return record
    return index[key]


def assign(
    annotations: list[SubjectAnnotation],
    subject_id: str,
    session_id: str,
    label: str,
    start_frame: int,
    end_frame: int,
    *,
    start_time: str | None = None,
    end_time: str | None = None,
    note: str = "",
) -> SubjectAnnotation:
    """Assign a label to an inclusive frame range, replacing overlaps."""
    if label not in ASSIGNABLE_LABELS:
        raise ValueError(f"label must be one of {ASSIGNABLE_LABELS}, got {label!r}")
    lo, hi = min(start_frame, end_frame), max(start_frame, end_frame)
    segment = Segment(
        label=label, start_frame=lo, end_frame=hi,
        start_time=start_time, end_time=end_time, note=note,
    )
    return storage.add_segment(annotations, subject_id, session_id, segment, replace_overlaps=True)


def clear_range(
    annotations: list[SubjectAnnotation], subject_id: str, session_id: str, start_frame: int, end_frame: int
) -> int:
    record = _ensure(annotations, subject_id, session_id)
    return record.remove_range(start_frame, end_frame)


def labels_for_frames(
    annotations: Iterable[SubjectAnnotation], subject_id: str, session_id: str, n_frames: int
) -> list[str]:
    """One label per frame index, ``UNLABELED`` where nothing is assigned."""
    index = storage.annotation_index(list(annotations))
    record = index.get((subject_id, session_id))
    mapping = record.labels_by_frame(n_frames) if record else {}
    return [mapping.get(i, UNLABELED) for i in range(n_frames)]


def frames_for_label(
    annotations: Iterable[SubjectAnnotation], subject_id: str, session_id: str, label: str
) -> list[int]:
    """Frame indices carrying ``label``, taken from the segments directly.

    Deliberately not built on top of ``labels_for_frames``: resolving a whole
    session per query would allocate one label per frame even when the caller
    only wants a handful of matches.
    """
    index = storage.annotation_index(list(annotations))
    record = index.get((subject_id, session_id))
    if record is None:
        return []
    found: set[int] = set()
    for seg in record.segments:
        if seg.label == label:
            found.update(seg.frame_indices())
    return sorted(found)


def analyzable_frames(
    annotations: Iterable[SubjectAnnotation], subject_id: str, session_id: str, n_frames: int
) -> list[int]:
    """Frame indices eligible for the main position comparison.

    Only SUPINE, SITTING and STANDING qualify. TRANSITION frames are excluded
    so movement artifacts during posture changes are never counted as
    stationary-position quality (IDEA.md section 16).
    """
    labels = labels_for_frames(annotations, subject_id, session_id, n_frames)
    return [i for i, label in enumerate(labels) if label in STATIC_POSITIONS]


def per_position_counts(
    annotations: Iterable[SubjectAnnotation], subject_id: str, session_id: str, n_frames: int
) -> dict[str, int]:
    labels = labels_for_frames(annotations, subject_id, session_id, n_frames)
    counts = {position: 0 for position in STATIC_POSITIONS}
    counts[TRANSITION] = labels.count(TRANSITION)
    counts[UNLABELED] = labels.count(UNLABELED)
    for label in labels:
        if label in STATIC_POSITIONS:
            counts[label] += 1
    return counts


def validate(
    annotations: Iterable[SubjectAnnotation], frames_by_session: dict[tuple[str, str], list]
) -> list[str]:
    frame_counts = {key: len(frames) for key, frames in frames_by_session.items()}
    return validate_annotations(annotations, frame_counts)


def auto_fill_unlabeled(
    annotations: list[SubjectAnnotation],
    subject_id: str,
    session_id: str,
    n_frames: int,
) -> int:
    """Do NOT guess positions.

    Provided as an explicit no-op helper so the UI can state clearly that
    unannotated frames are never auto-assigned. Returns 0.
    """
    return 0


__all__ = [
    "LABEL_ORDER",
    "analyzable_frames",
    "assign",
    "auto_fill_unlabeled",
    "clear_range",
    "frames_for_label",
    "get_or_create",
    "labels_for_frames",
    "per_position_counts",
    "validate",
]
