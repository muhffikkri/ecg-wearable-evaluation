"""Body-position annotation model.

Annotations are stored separately from the raw ECG data
(IDEA.md section 17) and support both frame-index and timestamp addressing
(section 14).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

SUPINE = "SUPINE"
SITTING = "SITTING"
STANDING = "STANDING"
TRANSITION = "TRANSITION"
UNLABELED = "UNLABELED"

#: Only these three enter the primary position comparison (IDEA.md section 16).
STATIC_POSITIONS = (SUPINE, SITTING, STANDING)
#: Labels the user may assign.
ASSIGNABLE_LABELS = (SUPINE, SITTING, STANDING, TRANSITION)
#: Everything the application can report for a frame.
ALL_LABELS = ASSIGNABLE_LABELS + (UNLABELED,)


@dataclass
class Segment:
    """A contiguous labelled span of one session.

    ``start_frame``/``end_frame`` are inclusive indices into the session's
    frame list. Timestamps are optional and are used only when the source
    provides them.
    """

    label: str
    start_frame: int
    end_frame: int
    start_time: str | None = None
    end_time: str | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.label not in ALL_LABELS:
            raise ValueError(f"unknown label {self.label!r}; allowed: {ALL_LABELS}")
        if self.end_frame < self.start_frame:
            raise ValueError(
                f"end_frame ({self.end_frame}) precedes start_frame ({self.start_frame})"
            )

    @property
    def n_frames(self) -> int:
        return self.end_frame - self.start_frame + 1

    def frame_indices(self) -> list[int]:
        return list(range(self.start_frame, self.end_frame + 1))

    def overlaps(self, other: "Segment") -> bool:
        return not (self.end_frame < other.start_frame or other.end_frame < self.start_frame)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "label": self.label,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
        }
        if self.start_time:
            payload["start_time"] = self.start_time
        if self.end_time:
            payload["end_time"] = self.end_time
        if self.note:
            payload["note"] = self.note
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Segment":
        return cls(
            label=str(payload["label"]),
            start_frame=int(payload["start_frame"]),
            end_frame=int(payload["end_frame"]),
            start_time=payload.get("start_time"),
            end_time=payload.get("end_time"),
            note=str(payload.get("note", "")),
        )


@dataclass
class SubjectAnnotation:
    """All annotated segments for one subject session."""

    subject_id: str
    session_id: str
    segments: list[Segment] = field(default_factory=list)
    source_file: str | None = None
    notes: str = ""

    def add(self, segment: Segment) -> None:
        self.segments.append(segment)
        self.segments.sort(key=lambda s: (s.start_frame, s.end_frame))

    def remove_range(self, start_frame: int, end_frame: int) -> int:
        """Clear one label over an inclusive range, splitting partial overlaps.

        A segment that only partially overlaps the cleared range is trimmed
        rather than dropped, so clearing frames 4-6 out of SUPINE 0-9 leaves
        SUPINE 0-3 and SUPINE 7-9 intact.
        """
        lo, hi = min(start_frame, end_frame), max(start_frame, end_frame)
        kept: list[Segment] = []
        removed = 0
        for seg in self.segments:
            if seg.end_frame < lo or seg.start_frame > hi:
                kept.append(seg)
                continue
            removed += 1
            if seg.start_frame < lo:
                kept.append(
                    Segment(seg.label, seg.start_frame, lo - 1, seg.start_time, seg.end_time, seg.note)
                )
            if seg.end_frame > hi:
                kept.append(Segment(seg.label, hi + 1, seg.end_frame, seg.start_time, seg.end_time, seg.note))
        self.segments = sorted(kept, key=lambda s: (s.start_frame, s.end_frame))
        return removed

    def labels_by_frame(self, n_frames: int) -> dict[int, str]:
        """Map frame index -> label. Later segments win on overlap."""
        mapping: dict[int, str] = {}
        for seg in self.segments:
            for idx in seg.frame_indices():
                if 0 <= idx < n_frames:
                    mapping[idx] = seg.label
        return mapping

    def overlapping_pairs(self) -> list[tuple[Segment, Segment]]:
        pairs: list[tuple[Segment, Segment]] = []
        for i, a in enumerate(self.segments):
            for b in self.segments[i + 1 :]:
                if a.overlaps(b):
                    pairs.append((a, b))
        return pairs

    def count_by_label(self) -> dict[str, int]:
        counts = {label: 0 for label in ALL_LABELS}
        for seg in self.segments:
            counts[seg.label] += seg.n_frames
        return counts

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "subject_id": self.subject_id,
            "session_id": self.session_id,
            "segments": [s.to_dict() for s in self.segments],
        }
        if self.source_file:
            payload["source_file"] = self.source_file
        if self.notes:
            payload["notes"] = self.notes
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SubjectAnnotation":
        return cls(
            subject_id=str(payload["subject_id"]),
            session_id=str(payload["session_id"]),
            segments=[Segment.from_dict(s) for s in payload.get("segments", [])],
            source_file=payload.get("source_file"),
            notes=str(payload.get("notes", "")),
        )


def validate_annotations(annotations: Iterable[SubjectAnnotation], frame_counts: dict[str, int]) -> list[str]:
    """Dataset-level annotation checks (IDEA.md section 43)."""
    problems: list[str] = []
    for ann in annotations:
        key = (ann.subject_id, ann.session_id)
        display = f"{ann.subject_id}/{ann.session_id}"
        n_frames = frame_counts.get(key)
        for a, b in ann.overlapping_pairs():
            problems.append(f"{display}: overlapping segments {a.to_dict()} and {b.to_dict()}")
        for seg in ann.segments:
            if seg.start_frame < 0:
                problems.append(f"{display}: negative start_frame in {seg.to_dict()}")
            if n_frames is not None and seg.end_frame >= n_frames:
                problems.append(
                    f"{display}: segment end_frame {seg.end_frame} outside recording "
                    f"(0..{n_frames - 1})"
                )
    return problems


__all__ = [
    "ALL_LABELS",
    "ASSIGNABLE_LABELS",
    "STATIC_POSITIONS",
    "SUPINE",
    "SITTING",
    "STANDING",
    "TRANSITION",
    "UNLABELED",
    "Segment",
    "SubjectAnnotation",
    "validate_annotations",
]
