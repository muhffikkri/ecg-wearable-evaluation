"""Ingestion pipeline: discovery -> format detection -> parsing -> inventory.

This is the single entry point the UI and the tests both use. It never
modifies anything under ``data/`` (IDEA.md section 44).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import REPO_ROOT, Config
from ..models.frame import ECGFrame
from .jsonl_reader import MalformedRecord, iter_jsonl_files, read_jsonl_file
from .raw_reader import discover_raw_roots, read_raw_directory
from .validator import DatasetInventory, annotation_coverage, build_inventory

logger = logging.getLogger(__name__)


@dataclass
class Dataset:
    """An ingested, validated collection of canonical frames."""

    frames_by_session: dict[tuple[str, str], list[ECGFrame]] = field(default_factory=dict)
    inventory: DatasetInventory | None = None
    subject_manifest: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    # -- access ---------------------------------------------------------
    def all_frames(self) -> list[ECGFrame]:
        out: list[ECGFrame] = []
        for frames in self.frames_by_session.values():
            out.extend(frames)
        return out

    def sessions(self) -> list[tuple[str, str]]:
        return sorted(self.frames_by_session.keys())

    def subjects(self) -> list[str]:
        seen: list[str] = []
        for subject, _ in self.sessions():
            if subject not in seen:
                seen.append(subject)
        return seen

    def session_frames(self, subject_id: str, session_id: str) -> list[ECGFrame]:
        return self.frames_by_session.get((subject_id, session_id), [])

    def frame_by_id(self, internal_id: str) -> ECGFrame | None:
        for frames in self.frames_by_session.values():
            for frame in frames:
                if frame.internal_id == internal_id:
                    return frame
        return None

    @property
    def n_frames(self) -> int:
        return sum(len(f) for f in self.frames_by_session.values())

    def source_formats(self) -> list[str]:
        return sorted({f.source_format for f in self.all_frames()})

    def sampling_rates(self) -> list[float]:
        return sorted({round(f.sampling_rate, 3) for f in self.all_frames()})


def load_subject_manifest(path: Path | None) -> dict[str, str]:
    """Load ``configs/subject_manifest.json`` if present.

    IDEA.md section 4: filenames alone must not be assumed to identify
    subjects, so the mapping is external and configurable.
    """
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("subject manifest unreadable: %s", exc)
        return {}
    if isinstance(payload, dict):
        return {str(k): str(v) for k, v in payload.items()}
    return {}


def ingest(
    data_dir: Path | str | None = None,
    config: Config | None = None,
    *,
    subject_manifest_path: Path | str | None = None,
    include_raw: bool = True,
) -> Dataset:
    """Discover, parse and validate everything under ``data_dir``."""
    config = config or Config(name="default", data={}, path=Path("."))
    data_dir = Path(data_dir) if data_dir else REPO_ROOT / str(config.get("paths.data_dir", "data"))
    if not data_dir.exists():
        logger.warning("data directory not found: %s", data_dir)
        return Dataset()

    manifest_path = (
        Path(subject_manifest_path)
        if subject_manifest_path
        else REPO_ROOT / str(config.get("paths.configs_dir", "configs")) / "subject_manifest.json"
    )
    manifest = load_subject_manifest(manifest_path)
    if manifest:
        logger.info("subject manifest loaded with %d entries", len(manifest))

    dataset = Dataset(subject_manifest=manifest)
    malformed: list[MalformedRecord] = []

    for path in iter_jsonl_files(data_dir):
        date_folder = path.parent.name if path.parent != data_dir else ""
        result = read_jsonl_file(path, date_folder=date_folder, subject_manifest=manifest)
        malformed.extend(result.malformed)
        for issue in result.warnings:
            dataset.warnings.append(f"{path.name}: {issue}")
        if not result.frames:
            dataset.warnings.append(f"{path.relative_to(data_dir)}: no frames ingested")
            continue
        key = (result.frames[0].subject_id, result.frames[0].session_id)
        dataset.frames_by_session.setdefault(key, []).extend(result.frames)
        logger.info(
            "ingested %s: %d frames (%d malformed, status %s)",
            path.name, len(result.frames), len(result.malformed), result.status,
        )

    if include_raw:
        for root in discover_raw_roots(data_dir):
            date_folder = root.parent.name
            result = read_raw_directory(root, date_folder=date_folder, subject_manifest=manifest)
            for warning in result.warnings:
                dataset.warnings.append(f"{root.name}: {warning}")
            for bad in result.malformed:
                malformed.append(
                    MalformedRecord(
                        source_file=str(bad.get("source_file", root)),
                        line_number=int(bad.get("frame_number", 0)),
                        reason=str(bad.get("reason", "unknown")),
                    )
                )
            if not result.frames:
                continue
            key = (result.frames[0].subject_id, result.frames[0].session_id)
            dataset.frames_by_session.setdefault(key, []).extend(result.frames)
            logger.info("ingested raw %s: %d frames", root.name, len(result.frames))

    dataset.inventory = build_inventory(
        dataset.frames_by_session,
        malformed,
        expected_subjects=int(config.get("dataset.expected_subjects", 12)),
        expected_positions=int(config.get("dataset.expected_positions", 3)),
        expected_frames_per_position=int(config.get("dataset.expected_frames_per_position", 6)),
        expected_total_frames=int(config.get("dataset.expected_total_frames", 216)),
        max_abs_limit_mv=float(config.get("validation.max_abs_limit_mv", 1000.0)),
    )

    coverage = dataset.inventory.coverage()
    logger.info(
        "inventory: %d subject(s), %d frame(s), %d malformed",
        coverage["actual_subjects"], coverage["actual_frames"], coverage["malformed_records"],
    )
    return dataset


def dataset_summary(dataset: Dataset) -> dict[str, Any]:
    """Compact summary for the sidebar (IDEA.md section 46)."""
    inv = dataset.inventory
    return {
        "subjects": dataset.subjects(),
        "sessions": len(dataset.sessions()),
        "frames": dataset.n_frames,
        "sampling_rates": dataset.sampling_rates(),
        "source_formats": dataset.source_formats(),
        "statuses": sorted({s.status for s in inv.sessions}) if inv else [],
        "malformed": inv.malformed_count if inv else 0,
        "coverage": inv.coverage() if inv else {},
    }


__all__ = [
    "Dataset",
    "annotation_coverage",
    "dataset_summary",
    "ingest",
    "load_subject_manifest",
]
