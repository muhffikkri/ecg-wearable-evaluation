"""Write generated JSONL, and record where every line came from.

The writer is deliberately conservative about paths. IDEA-REVISED.md section 8
requires the raw tree to stay untouched, so:

* output never goes inside the recording folder that was read;
* writing is all-or-nothing per run -- a run that produced invalid records
  writes nothing unless ``allow_invalid`` is set;
* a sidecar manifest is written next to the output recording the per-frame
  field origins and provenance, so a generated dataset is as traceable as a
  recorded one.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ..models.frame import ECGFrame
from ..reconstruction.provenance import SOURCE_TYPE_RAW
from .jsonl_builder import BuiltRecord, build_record

MANIFEST_SUFFIX = ".manifest.json"


@dataclass
class WriteResult:
    """Outcome of writing one generated JSONL dataset."""

    output_path: str = ""
    manifest_path: str = ""
    subject_id: str = ""
    session_id: str = ""
    written: int = 0
    skipped: list[dict[str, Any]] = field(default_factory=list)
    unavailable_fields: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    dry_run: bool = False

    @property
    def status(self) -> str:
        if self.skipped:
            return "ERROR" if not self.written else "WARNING"
        if self.warnings:
            return "WARNING"
        return "VALID"

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_path": self.output_path,
            "manifest_path": self.manifest_path,
            "subject_id": self.subject_id,
            "session_id": self.session_id,
            "status": self.status,
            "written": self.written,
            "skipped": list(self.skipped),
            "unavailable_fields": list(self.unavailable_fields),
            "warnings": list(self.warnings),
            "dry_run": self.dry_run,
        }


def _is_inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def assert_output_outside_source(
    output_dir: Path, source_root: Path | None
) -> None:
    """Refuse to write into the recording folder that was read.

    Raises:
        ValueError: when ``output_dir`` lies inside ``source_root``.
    """
    if source_root is None:
        return
    if _is_inside(Path(output_dir), Path(source_root)):
        raise ValueError(
            f"refusing to write generated JSONL into the raw recording folder "
            f"{source_root}: the source tree must stay read-only "
            "(IDEA-REVISED.md section 8)"
        )


def build_manifest(
    result: WriteResult,
    built: list[BuiltRecord],
    *,
    source_root: str,
    source_type: str = SOURCE_TYPE_RAW,
) -> dict[str, Any]:
    """The sidecar describing a generated dataset."""
    origins: dict[str, list[str]] = {}
    for entry in built:
        for name, origin in entry.origins.items():
            origins.setdefault(name, [])
            if origin not in origins[name]:
                origins[name].append(origin)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_type": source_type,
        "source_root": source_root,
        "subject_id": result.subject_id,
        "session_id": result.session_id,
        "schema": "templates/json-web.jsonl",
        "frames_written": result.written,
        "field_origins": origins,
        "unavailable_fields": result.unavailable_fields,
        "frames": [entry.to_dict() for entry in built],
    }


def write_jsonl(
    frames: Iterable[ECGFrame],
    output_dir: Path | str,
    *,
    filename: str | None = None,
    source_root: Path | str | None = None,
    dry_run: bool = False,
    allow_invalid: bool = False,
    write_manifest: bool = True,
) -> WriteResult:
    """Generate a JSONL dataset from canonical frames.

    Args:
        frames: canonical frames, from either data path.
        output_dir: destination folder, created if absent. Must not sit inside
            ``source_root``.
        filename: defaults to ``<subject>_<session>.jsonl``.
        source_root: the recording folder that was read, if any; used only to
            refuse an in-place write.
        dry_run: build and validate everything, write nothing.
        allow_invalid: write records that failed validation, reporting each one.
            Off by default, so a malformed dataset cannot be produced silently.
        write_manifest: also write the ``*.manifest.json`` sidecar.

    Returns:
        A :class:`WriteResult`. ``written`` counts lines actually emitted.
    """
    frames = list(frames)
    output_dir = Path(output_dir)
    source = Path(source_root) if source_root else None
    if source is not None:
        assert_output_outside_source(output_dir, source)

    subject_id = frames[0].subject_id if frames else ""
    session_id = frames[0].session_id if frames else ""
    target = output_dir / (filename or f"{subject_id}_{session_id}.jsonl")

    result = WriteResult(
        output_path=str(target),
        subject_id=subject_id,
        session_id=session_id,
        dry_run=dry_run,
    )

    built: list[BuiltRecord] = []
    unavailable: list[str] = []

    for frame in frames:
        entry = build_record(frame)
        built.append(entry)
        for name in entry.unavailable_fields:
            if name not in unavailable:
                unavailable.append(name)
        if entry.ok or allow_invalid:
            if not entry.ok:
                result.warnings.append(
                    f"frame {entry.frame_id}: wrote an invalid record "
                    f"({len(entry.validation.errors)} error(s))"
                )
        else:
            result.skipped.append(
                {
                    "frame_id": entry.frame_id,
                    "errors": [issue.to_dict() for issue in entry.validation.errors],
                }
            )

    result.unavailable_fields = unavailable
    result.written = len(frames) - len(result.skipped)

    if dry_run:
        return result

    output_dir.mkdir(parents=True, exist_ok=True)
    # Write to a sibling temp file and rename, so an interrupted run cannot leave
    # a half-written dataset that looks complete.
    temporary = target.with_name(target.name + ".partial")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for entry in built:
            if entry.ok or allow_invalid:
                handle.write(entry.line() + "\n")
    os.replace(temporary, target)

    if write_manifest:
        manifest_path = target.with_name(target.stem + MANIFEST_SUFFIX)
        manifest = build_manifest(
            result, built, source_root=str(source) if source else ""
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        result.manifest_path = str(manifest_path)

    return result


__all__ = [
    "MANIFEST_SUFFIX",
    "WriteResult",
    "assert_output_outside_source",
    "build_manifest",
    "write_jsonl",
]