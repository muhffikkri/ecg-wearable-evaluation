"""Provenance for a reconstructed canonical frame.

IDEA-REVISED.md section 12 asks one question: *which Raspberry Pi file did this
JSONL frame actually come from?* The answer has to survive the round trip into
JSONL, so it is built here in one place and attached to both the in-memory
frame and the generated record.
"""

from __future__ import annotations

from typing import Any

from ..models.frame import ECGFrame
from ..models.reconstruction import FrameMapping, SourceRef

SOURCE_TYPE_RAW = "raspberry_pi_raw"
SOURCE_TYPE_JSONL = "jsonl"


def source_files(mapping: FrameMapping) -> dict[str, str]:
    """Flatten the mapping's sources into ``{folder: primary file}``."""
    return {folder: ref.primary_file for folder, ref in mapping.sources.items()}


def build_provenance(
    mapping: FrameMapping,
    *,
    raw_root: str,
    device_id: str | None = None,
    session_metadata_file: str | None = None,
) -> dict[str, Any]:
    """Build the provenance block recorded for one canonical frame."""
    provenance: dict[str, Any] = {
        "source_type": SOURCE_TYPE_RAW,
        "raw_root": raw_root,
        "source_files": source_files(mapping),
        "mapping_method": mapping.method,
        "mapping_status": mapping.status,
        "mapping_evidence": dict(mapping.evidence),
        "signal_source": mapping.signal_source,
        "device_id": device_id,
    }
    if session_metadata_file:
        provenance["session_metadata_file"] = session_metadata_file
    if mapping.reasons:
        provenance["unresolved_reasons"] = list(mapping.reasons)
    if mapping.warnings:
        provenance["warnings"] = list(mapping.warnings)
    return provenance


def content_sha256(record: SourceFrame | SourceRef) -> str:
    """Return a recorded checksum if the source published one, else empty."""
    metadata = getattr(record, "metadata", {}) or {}
    source_metadata = metadata.get("source_metadata") or {}
    checksum = source_metadata.get("sha256_checksum")
    return str(checksum) if checksum else ""


def frame_provenance_view(frame: ECGFrame) -> dict[str, Any]:
    """Compact provenance for the dataset viewer."""
    return frame.provenance_view()


__all__ = [
    "SOURCE_TYPE_JSONL",
    "SOURCE_TYPE_RAW",
    "build_provenance",
    "content_sha256",
    "frame_provenance_view",
    "source_files",
]