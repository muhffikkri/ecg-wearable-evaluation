"""Dataset-level validation and inventory.

Produces the inventory the UI shows before any analysis
(IDEA.md sections 10, 12, 43), and never hides a problem: malformed records
and signal issues are surfaced, not dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from ..models.annotation import STATIC_POSITIONS, SubjectAnnotation
from ..models.frame import ECGFrame
from .jsonl_reader import MalformedRecord


@dataclass
class SessionInventory:
    subject_id: str
    session_id: str
    date_folder: str
    source_format: str
    source_file: str
    n_frames: int
    sampling_rate: float | None
    frame_duration_s: float | None
    n_channels: int
    status: str
    issues: list[str] = field(default_factory=list)
    n_malformed: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        return {
            "Subject": self.subject_id,
            "Session": self.session_id,
            "Date": self.date_folder,
            "Source": self.source_format,
            "Frames": self.n_frames,
            "Sampling rate (Hz)": self.sampling_rate,
            "Frame duration (s)": self.frame_duration_s,
            "Channels": self.n_channels,
            "Malformed": self.n_malformed,
            "Status": self.status,
            "Issues": "; ".join(self.issues),
            "File": self.source_file,
        }


@dataclass
class DatasetInventory:
    sessions: list[SessionInventory] = field(default_factory=list)
    malformed: list[MalformedRecord] = field(default_factory=list)
    n_frames: int = 0
    expected_subjects: int = 12
    expected_positions: int = 3
    expected_frames_per_position: int = 6
    expected_total_frames: int = 216

    @property
    def subjects(self) -> list[str]:
        seen: list[str] = []
        for session in self.sessions:
            if session.subject_id not in seen:
                seen.append(session.subject_id)
        return seen

    @property
    def n_subjects(self) -> int:
        return len(self.subjects)

    @property
    def malformed_count(self) -> int:
        return len(self.malformed)

    def by_subject(self) -> dict[str, list[SessionInventory]]:
        grouped: dict[str, list[SessionInventory]] = {}
        for session in self.sessions:
            grouped.setdefault(session.subject_id, []).append(session)
        return grouped

    def rows(self) -> list[dict[str, Any]]:
        return [s.to_row() for s in self.sessions]

    @property
    def expected_design(self) -> str:
        """Where the expected frame count comes from, in words."""
        return (
            f"{self.expected_subjects} subjects x {self.expected_positions} positions x "
            f"{self.expected_frames_per_position} frames"
        )

    def coverage(self) -> dict[str, Any]:
        """Expected vs actual, reported transparently (IDEA.md section 31)."""
        return {
            "expected_subjects": self.expected_subjects,
            "actual_subjects": self.n_subjects,
            "missing_subjects": max(0, self.expected_subjects - self.n_subjects),
            "expected_frames": self.expected_total_frames,
            "expected_design": self.expected_design,
            "actual_frames": self.n_frames,
            "missing_frames": max(0, self.expected_total_frames - self.n_frames),
            "malformed_records": self.malformed_count,
        }


def validate_frames(frames: Iterable[ECGFrame], max_abs_limit_mv: float = 1000.0) -> dict[str, list[str]]:
    """Signal-level checks (IDEA.md section 43)."""
    issues: dict[str, list[str]] = {}
    for frame in frames:
        problems = list(frame.signal_issues())
        if frame.n_samples and np.nanmax(np.abs(frame.signal)) > max_abs_limit_mv:
            problems.append(
                f"amplitude {float(np.nanmax(np.abs(frame.signal))):.1f} mV exceeds "
                f"limit {max_abs_limit_mv} mV"
            )
        if problems:
            issues[frame.internal_id] = problems
    return issues


def validate_sampling_rates(frames: Iterable[ECGFrame], tolerance: float = 1e-6) -> list[str]:
    """Flag sessions whose frames disagree about sampling rate or sample count."""
    per_session: dict[tuple[str, str], set[tuple[float, int]]] = {}
    for frame in frames:
        key = (frame.subject_id, frame.session_id)
        per_session.setdefault(key, set()).add((round(frame.sampling_rate, 6), frame.n_samples))
    problems: list[str] = []
    for (subject, session), variants in per_session.items():
        if len(variants) > 1:
            rendered = ", ".join(f"{fs:g} Hz/{n} samples" for fs, n in sorted(variants))
            problems.append(f"{subject}/{session}: inconsistent sampling across frames ({rendered})")
    return problems


def build_inventory(
    frames_by_session: dict[tuple[str, str], list[ECGFrame]],
    malformed: list[MalformedRecord],
    *,
    expected_subjects: int = 12,
    expected_positions: int = 3,
    expected_frames_per_position: int = 6,
    expected_total_frames: int = 216,
    max_abs_limit_mv: float = 1000.0,
) -> DatasetInventory:
    """Assemble the inventory table shown on the Dataset page."""
    inventory = DatasetInventory(
        malformed=malformed,
        expected_subjects=expected_subjects,
        expected_positions=expected_positions,
        expected_frames_per_position=expected_frames_per_position,
        expected_total_frames=expected_total_frames,
    )

    for (subject, session), frames in sorted(frames_by_session.items()):
        issues: list[str] = []
        session_malformed = [m for m in malformed if subject in m.source_file]
        if not frames:
            issues.append("no readable frames")
        rates = {round(f.sampling_rate, 6) for f in frames}
        durations = {round(f.duration_actual_s, 3) for f in frames}
        if len(rates) > 1:
            issues.append(f"inconsistent sampling rate: {sorted(rates)}")
        if len(durations) > 1:
            issues.append(f"inconsistent frame duration: {sorted(durations)}")

        signal_issues = validate_frames(frames, max_abs_limit_mv=max_abs_limit_mv)
        if signal_issues:
            issues.append(f"{len(signal_issues)} frame(s) with signal issues")
        if session_malformed:
            issues.append(f"{len(session_malformed)} malformed line(s)")

        device_statuses = {
            str((f.metadata.get("validation") or {}).get("status"))
            for f in frames
            if (f.metadata.get("validation") or {}).get("status")
        }
        status = "OK"
        if not frames or session_malformed:
            status = "MALFORMED"
        elif issues or (device_statuses - {"PASS", "OK"}):
            status = "WARNING"

        first = frames[0] if frames else None
        inventory.sessions.append(
            SessionInventory(
                subject_id=subject,
                session_id=session,
                date_folder=(first.date_folder if first else "") or "",
                source_format=(first.source_format if first else "unknown"),
                source_file=(first.source_file if first else ""),
                n_frames=len(frames),
                sampling_rate=(first.sampling_rate if first else None),
                frame_duration_s=(round(durations.pop(), 3) if len(durations) == 1 else None),
                n_channels=(first.n_channels if first else 0),
                status=status,
                issues=issues,
                n_malformed=len(session_malformed),
                extra={"device_validation": sorted(device_statuses)} if device_statuses else {},
            )
        )
        inventory.n_frames += len(frames)

    return inventory


def annotation_coverage(
    frames_by_session: dict[tuple[str, str], list[ECGFrame]],
    annotations: list[SubjectAnnotation],
    *,
    expected_total_frames: int = 216,
    expected_subjects: int = 12,
    expected_positions: int = 3,
    expected_frames_per_position: int = 6,
) -> dict[str, Any]:
    """Frame accounting for the Analysis page (IDEA.md sections 31, 49).

    The design is passed in rather than assumed, and echoed back as
    ``expected_design`` so the UI and the report can state where the expected
    frame count came from instead of repeating a hard-coded 216
    (IDEA.md section 62).
    """
    annotated = 0
    static = 0
    transition = 0
    unlabeled = 0
    per_subject: dict[str, dict[str, int]] = {}

    ann_index = {(a.subject_id, a.session_id): a for a in annotations}
    for (subject, session), frames in frames_by_session.items():
        counts = {label: 0 for label in list(STATIC_POSITIONS) + ["TRANSITION", "UNLABELED"]}
        ann = ann_index.get((subject, session))
        mapping = ann.labels_by_frame(len(frames)) if ann else {}
        for index in range(len(frames)):
            label = mapping.get(index, "UNLABELED")
            counts[label] += 1
            annotated += label != "UNLABELED"
            static += label in STATIC_POSITIONS
            transition += label == "TRANSITION"
            unlabeled += label == "UNLABELED"
        per_subject[f"{subject}/{session}"] = counts

    analyzed = static
    return {
        "expected_frames": expected_total_frames,
        "expected_design": (
            f"{expected_subjects} subjects x {expected_positions} positions x "
            f"{expected_frames_per_position} frames"
        ),
        "total_frames": sum(len(f) for f in frames_by_session.values()),
        "annotated_frames": annotated,
        "analyzable_frames": analyzed,
        "excluded_transitions": transition,
        "unlabeled_frames": unlabeled,
        "missing_frames": max(0, expected_total_frames - analyzed),
        "per_session": per_subject,
    }


__all__ = [
    "DatasetInventory",
    "SessionInventory",
    "annotation_coverage",
    "build_inventory",
    "validate_frames",
    "validate_sampling_rates",
]
