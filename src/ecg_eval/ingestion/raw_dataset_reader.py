"""Discovery and inspection of one Raspberry Pi recording folder.

This is the entry point IDEA-REVISED.md section 9A describes as "Scan Raw
Dataset". It walks ``data/<date>/raw/<subject>/``, delegates to the per-folder
readers and returns a :class:`~ecg_eval.models.source_frame.SourceDataset` that
holds every folder *separately*.

Nothing is merged here and nothing is written. Merging is
:mod:`ecg_eval.reconstruction`'s job, so that the evidence for how folders
relate stays visible instead of being hidden inside a converter.

Observed for ``data/29-09-2026/Bryan``::

    calibrated/   20 frames   frame_000001_mv.{csv,json,npy}
    filtered/     20 frames   frame_000001_mv.{csv,json,npy}
    model_ready/  20 frames   frame_000001_input.{json,npy}
    predictions/  20 frames   + latest_prediction.json, mqtt_publish_state.json
    logs/         empty
    raw_adc/      20 frames + ecg_metadata.sqlite, raw_ecg.csv
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..models.source_frame import (
    FOLDER_CALIBRATED,
    FOLDER_FILTERED,
    FOLDER_MODEL_READY,
    FOLDER_PREDICTIONS,
    RAW_KNOWN_SUBDIRS,
    SIGNAL_FOLDERS,
    SourceDataset,
)
from .calibrated_reader import read_calibrated
from .log_reader import read_logs
from .model_ready_reader import read_model_ready
from .prediction_reader import read_predictions

# ``filtered`` and ``calibrated`` are structurally identical (frame_*_mv.*), so
# the calibrated reader is reused with an explicit folder name.
FILTERED_READER_NOTE = "filtered/ shares the frame_*_mv layout and is read like calibrated/"


def discover_raw_roots(data_dir: Path | str) -> list[Path]:
    """Find every Raspberry Pi recording folder under ``data_dir``.

    A recording folder is any directory that contains at least one of the known
    output subfolders. Both ``data/<date>/<subject>/`` and
    ``data/<date>/raw/<subject>/`` are accepted so an older flat layout still
    opens.
    """
    data_dir = Path(data_dir)
    roots: list[Path] = []
    if not data_dir.is_dir():
        return roots

    for path in sorted(data_dir.rglob("*")):
        if not path.is_dir():
            continue
        if any((path / name).is_dir() for name in RAW_KNOWN_SUBDIRS):
            roots.append(path)
    return roots


def read_session_metadata(root: Path) -> dict[str, Any]:
    """Read the optional ``session.json`` written next to the folders."""
    path = Path(root) / "session.json"
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return {"_error": f"session.json unreadable: {exc}"}
    return payload if isinstance(payload, dict) else {"_error": "session.json is not an object"}


def _resolve_session_id(root: Path, session_metadata: dict[str, Any]) -> str:
    """Prefer the firmware session id; fall back to the folder name."""
    declared = session_metadata.get("session_id")
    return str(declared) if declared else f"raw_{root.name}"


def _resolve_device_id(session_metadata: dict[str, Any], frames: dict[str, list]) -> str | None:
    """Device id only ever comes from recorded metadata, never from the path."""
    for folder in (FOLDER_CALIBRATED, FOLDER_MODEL_READY):
        for record in frames.get(folder, []):
            source_metadata = record.metadata.get("source_metadata") or {}
            if source_metadata.get("device_id"):
                return str(source_metadata["device_id"])
    return None


def read_raw_dataset(
    root: Path | str,
    *,
    subject_id: str | None = None,
    load_signals: bool = True,
) -> SourceDataset:
    """Scan one recording folder without merging anything.

    Args:
        root: the subject folder containing ``calibrated/``, ``model_ready/`` etc.
        subject_id: overrides the subject id, which otherwise defaults to the
            folder name (the only identifier the folder itself provides).
        load_signals: passed through to the array readers.
    """
    root = Path(root)
    session_metadata = read_session_metadata(root)
    dataset = SourceDataset(
        root=str(root),
        subject_id=subject_id or root.name,
        session_id=_resolve_session_id(root, session_metadata),
        session_metadata=session_metadata,
    )
    dataset.present_dirs = {name: (root / name).is_dir() for name in RAW_KNOWN_SUBDIRS}

    if session_metadata.get("_error"):
        dataset.warnings.append(str(session_metadata["_error"]))

    if not any(dataset.present_dirs[name] for name in SIGNAL_FOLDERS):
        dataset.errors.append(
            {
                "folder": "",
                "frame_id": "",
                "reason": f"no signal folder present (looked for {', '.join(SIGNAL_FOLDERS)})",
            }
        )
        return dataset

    dataset.frames[FOLDER_CALIBRATED] = read_calibrated(
        root, folder=FOLDER_CALIBRATED, load_signals=load_signals
    )
    dataset.frames[FOLDER_FILTERED] = read_calibrated(
        root, folder=FOLDER_FILTERED, load_signals=load_signals
    )
    dataset.frames[FOLDER_MODEL_READY] = read_model_ready(root, load_signals=load_signals)
    dataset.predictions = read_predictions(root)

    dataset.logs, log_warnings = read_logs(root)
    dataset.warnings.extend(log_warnings)

    dataset.device_id = _resolve_device_id(session_metadata, dataset.frames)

    _record_frame_errors(dataset)
    return dataset


def _record_frame_errors(dataset: SourceDataset) -> None:
    """Fold per-frame reader errors into the dataset-level report.

    Every entry is a ``{"folder", "frame_id", "reason"}`` dict so a caller can
    group by folder without type-checking each item.
    """
    for folder, records in dataset.frames.items():
        for record in records:
            for message in record.errors:
                dataset.errors.append(
                    {"folder": folder, "frame_id": record.frame_id, "reason": message}
                )
    for record in dataset.predictions:
        for message in record.errors:
            dataset.errors.append(
                {"folder": FOLDER_PREDICTIONS, "frame_id": record.frame_id, "reason": message}
            )


__all__ = [
    "FILTERED_READER_NOTE",
    "discover_raw_roots",
    "read_raw_dataset",
    "read_session_metadata",
]