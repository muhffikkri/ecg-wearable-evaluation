"""Annotation storage.

Annotations live in their own directory and never touch the raw data
(IDEA.md sections 17, 44)::

    annotations/
      annotations.json          structured, per subject session
      annotation_manifest.csv   flat one-row-per-segment view

Writes are atomic so an interrupted save cannot corrupt the annotation file.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ..models.annotation import Segment, SubjectAnnotation

logger = logging.getLogger(__name__)

ANNOTATIONS_FILE = "annotations.json"
MANIFEST_FILE = "annotation_manifest.csv"
FORMAT_VERSION = 1


class AnnotationFileError(RuntimeError):
    """The annotation file exists but cannot be read as an annotation document.

    Raised rather than degrading to an empty annotation set: a silently empty
    set looks identical to "never annotated", and the next save would overwrite
    hours of labelling with an empty file.
    """


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=str(path.parent), delete=False, suffix=".tmp"
    )
    try:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    finally:
        handle.close()
    os.replace(handle.name, path)


def load_annotations(directory: Path | str) -> list[SubjectAnnotation]:
    """Load annotations.

    A missing file is an empty set, which is a legitimate state. A file that
    exists but is unparseable raises :class:`AnnotationFileError` so the UI can
    refuse to overwrite it. Individual malformed records are skipped with a
    warning, because losing one segment is recoverable but losing the file is not.
    """
    directory = Path(directory)
    path = directory / ANNOTATIONS_FILE
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise AnnotationFileError(
            f"{path} is not valid JSON ({exc}); move it aside to start a new file"
        ) from exc

    if isinstance(payload, dict):
        records = payload.get("annotations", [])
    elif isinstance(payload, list):
        records = payload
    else:
        raise AnnotationFileError(
            f"{path} holds a {type(payload).__name__}, expected an object or a list"
        )
    if not isinstance(records, list):
        raise AnnotationFileError(f"{path}: 'annotations' must be a list, got {type(records).__name__}")

    out: list[SubjectAnnotation] = []
    skipped = 0
    for record in records:
        try:
            out.append(SubjectAnnotation.from_dict(record))
        except Exception as exc:  # noqa: BLE001
            skipped += 1
            logger.warning("skipping malformed annotation record in %s: %s", path.name, exc)
    if skipped:
        logger.warning("loaded %d of %d annotation records from %s", len(out), len(records), path.name)
    return out


def save_annotations(
    directory: Path | str,
    annotations: Iterable[SubjectAnnotation],
    *,
    notes: str = "",
) -> Path:
    """Write annotations atomically and regenerate the CSV manifest."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    records = [ann.to_dict() for ann in annotations]
    payload = {
        "format_version": FORMAT_VERSION,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "notes": notes,
        "labels": {
            "assignable": ["SUPINE", "SITTING", "STANDING", "TRANSITION"],
            "static_positions": ["SUPINE", "SITTING", "STANDING"],
            "system": "UNLABELED",
        },
        "count": len(records),
        "annotations": records,
    }
    path = directory / ANNOTATIONS_FILE
    _atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False))
    write_manifest(directory, annotations)
    logger.info("saved %d annotation record(s) to %s", len(records), path)
    return path


def write_manifest(directory: Path | str, annotations: Iterable[SubjectAnnotation]) -> Path:
    """Flat CSV view: one row per labelled segment."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / MANIFEST_FILE
    rows = []
    for ann in annotations:
        for seg in ann.segments:
            rows.append(
                {
                    "subject_id": ann.subject_id,
                    "session_id": ann.session_id,
                    "label": seg.label,
                    "start_frame": seg.start_frame,
                    "end_frame": seg.end_frame,
                    "n_frames": seg.n_frames,
                    "start_time": seg.start_time or "",
                    "end_time": seg.end_time or "",
                    "note": seg.note,
                }
            )
    fieldnames = [
        "subject_id", "session_id", "label", "start_frame", "end_frame",
        "n_frames", "start_time", "end_time", "note",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def annotation_index(annotations: Iterable[SubjectAnnotation]) -> dict[tuple[str, str], SubjectAnnotation]:
    return {(ann.subject_id, ann.session_id): ann for ann in annotations}


def annotation_hash(annotations: Iterable[SubjectAnnotation]) -> str:
    """Stable hash of annotation content, used in the analysis cache key."""
    import hashlib

    payload = json.dumps(
        [ann.to_dict() for ann in sorted(annotations, key=lambda a: (a.subject_id, a.session_id))],
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def add_segment(
    annotations: list[SubjectAnnotation],
    subject_id: str,
    session_id: str,
    segment: Segment,
    *,
    replace_overlaps: bool = True,
) -> SubjectAnnotation:
    """Insert a segment, creating the subject record if needed.

    With ``replace_overlaps`` the new segment overwrites any existing label in
    the same range, which is what the annotation UI's "assign range" action
    needs. Overlaps are otherwise reported by the validator.
    """
    existing = annotation_index(annotations).get((subject_id, session_id))
    if existing is None:
        existing = SubjectAnnotation(subject_id=subject_id, session_id=session_id)
        annotations.append(existing)
    if replace_overlaps:
        existing.remove_range(segment.start_frame, segment.end_frame)
    existing.add(segment)
    return existing


def clear_subject(annotations: list[SubjectAnnotation], subject_id: str, session_id: str) -> bool:
    target = annotation_index(annotations).get((subject_id, session_id))
    if target is None:
        return False
    target.segments.clear()
    return True


def summary(annotations: Iterable[SubjectAnnotation]) -> dict[str, Any]:
    annotations = list(annotations)
    counts: dict[str, int] = {}
    for ann in annotations:
        for label, n in ann.count_by_label().items():
            counts[label] = counts.get(label, 0) + n
    return {
        "records": len(annotations),
        "segments": sum(len(a.segments) for a in annotations),
        "frames_by_label": counts,
        "static_frames": sum(
            ann.count_by_label().get(label, 0)
            for ann in annotations
            for label in ("SUPINE", "SITTING", "STANDING")
        ),
    }


__all__ = [
    "ANNOTATIONS_FILE",
    "MANIFEST_FILE",
    "AnnotationFileError",
    "add_segment",
    "annotation_hash",
    "annotation_index",
    "clear_subject",
    "load_annotations",
    "save_annotations",
    "summary",
    "write_manifest",
]
