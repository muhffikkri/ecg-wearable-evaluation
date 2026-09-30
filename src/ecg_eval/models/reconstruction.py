"""Models describing how a raw recording was mapped and merged.

These types carry the reasoning, not just the outcome: for every canonical
frame there is a :class:`FrameMapping` recording which folder each source came
from, which method established the link, and whether that link is proven
(IDEA-REVISED.md sections 9C and 12).

The distinction that matters is :data:`MAPPING_VERIFIED` versus
:data:`MAPPING_UNRESOLVED`. A frame that cannot be mapped safely is reported as
unresolved and kept visible to the user; it is never silently paired with
whatever frame happened to share its number.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# -- mapping outcomes ---------------------------------------------------

MAPPING_VERIFIED = "verified"
MAPPING_UNRESOLVED = "unresolved"

# -- how the link was established ---------------------------------------
#: The model_ready JSON names the file it was built from, so the link is
#: recorded by the firmware rather than inferred.
MAPPING_METHOD_RECORDED_POINTER = "recorded_pointer"
#: Both sides carry a timestamp that matched within a tolerance.
MAPPING_METHOD_TIMESTAMP = "timestamp"
#: No link could be established; the frame is kept but flagged.
MAPPING_METHOD_NONE = "none"

MAPPING_METHODS = (
    MAPPING_METHOD_RECORDED_POINTER,
    MAPPING_METHOD_TIMESTAMP,
    MAPPING_METHOD_NONE,
)

# -- overall run states (IDEA-REVISED.md section 13) --------------------

STATUS_VALID = "VALID"
STATUS_WARNING = "WARNING"
STATUS_ERROR = "ERROR"
STATUS_UNRESOLVED = "UNRESOLVED"


@dataclass
class SourceRef:
    """A pointer to one source frame that took part in a reconstruction."""

    folder: str
    frame_number: int | None = None
    files: dict[str, str] = field(default_factory=dict)
    #: Files the firmware recorded as the origin of this frame, when it did.
    recorded_source_files: dict[str, str] = field(default_factory=dict)

    @property
    def frame_id(self) -> str:
        return "" if self.frame_number is None else f"{self.frame_number:06d}"

    @property
    def primary_file(self) -> str:
        for key in ("npy", "json"):
            if key in self.files:
                return self.files[key]
        return next(iter(self.files.values()), "")

    def to_dict(self) -> dict[str, Any]:
        return {
            "folder": self.folder,
            "frame_id": self.frame_id,
            "files": dict(self.files),
            "recorded_source_files": dict(self.recorded_source_files),
        }


@dataclass
class FrameMapping:
    """The record of how one canonical frame was assembled.

    ``status`` is ``verified`` only when at least one source is linked by
    evidence. ``reasons`` explains every gap so the UI can show why a frame is
    unresolved instead of just that it is.
    """

    frame_id: str
    status: str = MAPPING_VERIFIED
    method: str = MAPPING_METHOD_RECORDED_POINTER
    signal_source: str = ""
    sources: dict[str, SourceRef] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def resolved(self) -> bool:
        return self.status == MAPPING_VERIFIED

    def has(self, folder: str) -> bool:
        return folder in self.sources

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "status": self.status,
            "method": self.method,
            "signal_source": self.signal_source,
            "sources": {name: ref.to_dict() for name, ref in self.sources.items()},
            "evidence": dict(self.evidence),
            "reasons": list(self.reasons),
            "warnings": list(self.warnings),
        }


@dataclass
class ReconstructionResult:
    """Canonical frames plus the full mapping report that produced them."""

    subject_id: str
    session_id: str
    source_root: str = ""
    frames: list[Any] = field(default_factory=list)  # list[ECGFrame]
    mappings: list[FrameMapping] = field(default_factory=list)
    #: Frames present in a source folder but absent from ``frames``.
    dropped: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)

    # -- aggregate ------------------------------------------------------
    def counts(self) -> dict[str, int]:
        return {
            "canonical_frames": len(self.frames),
            "verified": sum(1 for m in self.mappings if m.resolved),
            "unresolved": sum(1 for m in self.mappings if not m.resolved),
            "with_prediction": sum(1 for m in self.mappings if m.has("predictions")),
            "with_model_ready": sum(1 for m in self.mappings if m.has("model_ready")),
            "with_calibrated": sum(1 for m in self.mappings if m.has("calibrated")),
            "dropped": len(self.dropped),
        }

    def unresolved(self) -> list[FrameMapping]:
        return [m for m in self.mappings if not m.resolved]

    def mapping_for(self, frame_id: str) -> FrameMapping | None:
        for mapping in self.mappings:
            if mapping.frame_id == frame_id:
                return mapping
        return None

    def status(self) -> str:
        """Worst state across the run; errors outrank unresolved outrank warnings."""
        if self.errors:
            return STATUS_ERROR
        if self.unresolved():
            return STATUS_UNRESOLVED
        if self.warnings or self.dropped:
            return STATUS_WARNING
        return STATUS_VALID

    def mapping_methods(self) -> dict[str, int]:
        tally: dict[str, int] = {}
        for mapping in self.mappings:
            tally[mapping.method] = tally.get(mapping.method, 0) + 1
        return tally

    def to_dict(self, *, include_mappings: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "subject_id": self.subject_id,
            "session_id": self.session_id,
            "source_root": self.source_root,
            "status": self.status(),
            "counts": self.counts(),
            "mapping_methods": self.mapping_methods(),
            "dropped": list(self.dropped),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }
        if include_mappings:
            payload["mappings"] = [m.to_dict() for m in self.mappings]
        return payload


__all__ = [
    "MAPPING_METHODS",
    "MAPPING_METHOD_NONE",
    "MAPPING_METHOD_RECORDED_POINTER",
    "MAPPING_METHOD_TIMESTAMP",
    "MAPPING_UNRESOLVED",
    "MAPPING_VERIFIED",
    "STATUS_ERROR",
    "STATUS_UNRESOLVED",
    "STATUS_VALID",
    "STATUS_WARNING",
    "FrameMapping",
    "ReconstructionResult",
    "SourceRef",
]