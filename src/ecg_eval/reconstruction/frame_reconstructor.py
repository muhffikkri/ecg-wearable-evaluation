"""Turn a scanned raw recording into canonical frames.

This is the "Data Reconstruction" stage of IDEA-REVISED.md sections 5, 6 and 14::

    SourceDataset  ->  map_frames  ->  merge_metadata  ->  ECGFrame

The output is the same :class:`~ecg_eval.models.frame.ECGFrame` the JSONL reader
produces, so annotation and analysis downstream cannot tell a reconstructed frame
from a pre-existing one.

Two guarantees worth stating:

* **Non-destructive.** Nothing is written, moved or renamed. The caller decides
  whether to generate JSONL; this module only builds in-memory frames.
* **Nothing disappears.** A source frame that cannot become a canonical frame is
  recorded in ``dropped`` with a reason, and a frame whose mapping is not proven
  is emitted but flagged ``unresolved``. Neither is silently discarded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..models.frame import ECGFrame, SOURCE_RAW_RECONSTRUCTED
from ..models.reconstruction import FrameMapping, ReconstructionResult
from ..models.source_frame import (
    FOLDER_CALIBRATED,
    FOLDER_FILTERED,
    FOLDER_MODEL_READY,
    FOLDER_PREDICTIONS,
    SIGNAL_FOLDERS,
    SourceDataset,
    SourceFrame,
)
from .frame_mapper import map_frames
from .metadata_merger import merge_metadata, merge_prediction
from .provenance import build_provenance

#: Folder order used when resolving metadata precedence: the folder that supplies
#: the signal is consulted first.
SOURCE_PRECEDENCE = (FOLDER_CALIBRATED, FOLDER_FILTERED, FOLDER_MODEL_READY, FOLDER_PREDICTIONS)


def _resolve_sources(
    dataset: SourceDataset, mapping: FrameMapping
) -> dict[str, SourceFrame]:
    """Fetch the actual :class:`SourceFrame` objects a mapping refers to."""
    resolved: dict[str, SourceFrame] = {}
    for folder in SOURCE_PRECEDENCE:
        ref = mapping.sources.get(folder)
        if ref is None or ref.frame_number is None:
            continue
        if folder == FOLDER_PREDICTIONS:
            record = dataset.prediction_map().get(ref.frame_number)
        else:
            record = dataset.frame_map(folder).get(ref.frame_number)
        if record is not None:
            resolved[folder] = record
    return resolved


def _ordered(resolved: dict[str, SourceFrame], signal_source: str) -> dict[str, SourceFrame]:
    """Order sources so the signal folder comes first for metadata precedence."""
    ordered: dict[str, SourceFrame] = {}
    if signal_source in resolved:
        ordered[signal_source] = resolved[signal_source]
    for folder in SOURCE_PRECEDENCE:
        if folder in resolved and folder not in ordered:
            ordered[folder] = resolved[folder]
    return ordered


def _build_frame(
    dataset: SourceDataset,
    mapping: FrameMapping,
    resolved: dict[str, SourceFrame],
    *,
    subject_id: str,
    session_id: str,
    record_index: int,
) -> tuple[ECGFrame | None, list[str]]:
    """Build one canonical frame. Returns ``(frame, problems)``."""
    problems: list[str] = []

    ordered = _ordered(resolved, mapping.signal_source)
    signal_record = ordered.get(mapping.signal_source) if mapping.signal_source else None
    if signal_record is None or signal_record.signal is None:
        return None, [f"source folder {mapping.signal_source or '(none)'} has no readable array"]

    sampling_rate = signal_record.sampling_rate
    if not sampling_rate:
        sampling_rate = float(dataset.session_metadata.get("sampling_rate_hz") or 250.0)
        problems.append(
            f"{mapping.signal_source} recorded no sample rate; using session value {sampling_rate:g} Hz"
        )

    duration_s = signal_record.duration_s
    if not duration_s:
        duration_s = signal_record.signal.shape[0] / sampling_rate if sampling_rate else 0.0

    metadata, conflicts = merge_metadata(ordered, mapping)
    metadata["folders_merged"] = list(ordered)

    prediction_record = ordered.get(FOLDER_PREDICTIONS)
    prediction = merge_prediction(prediction_record)
    if prediction:
        metadata["prediction"] = prediction

    signal = np.asarray(signal_record.signal, dtype=np.float32)
    other = [name for name, record in ordered.items()
             if name not in (mapping.signal_source, FOLDER_PREDICTIONS) and record.signal is not None]
    for name in other:
        other_shape = ordered[name].signal.shape
        if other_shape != signal.shape:
            problems.append(
                f"{name} array shape {list(other_shape)} != {mapping.signal_source} "
                f"shape {list(signal.shape)}"
            )

    frame = ECGFrame(
        subject_id=subject_id,
        session_id=session_id,
        frame_id=mapping.frame_id,
        source_file=sorted(ordered[folder].files.get("npy", "") for folder in ordered
                           if folder != FOLDER_PREDICTIONS)[0],
        source_format=SOURCE_RAW_RECONSTRUCTED,
        timestamp=signal_record.created_at,
        sampling_rate=sampling_rate,
        duration_s=duration_s,
        signal=signal,
        metadata=metadata,
        provenance=build_provenance(
            mapping,
            raw_root=dataset.root,
            device_id=dataset.device_id,
            session_metadata_file=str(Path(dataset.root) / "session.json")
            if (Path(dataset.root) / "session.json").is_file()
            else None,
        ),
        record_index=record_index,
        model_input_available=FOLDER_MODEL_READY in ordered,
        prediction_available=prediction_record is not None,
    )
    if conflicts:
        frame.metadata["merge_conflicts"] = conflicts
        problems.extend(conflicts)

    return frame, problems


def reconstruct(
    dataset: SourceDataset | Path | str,
    *,
    subject_id: str | None = None,
    include_unresolved: bool = True,
) -> ReconstructionResult:
    """Reconstruct canonical frames from one scanned recording folder.

    Args:
        dataset: a :class:`SourceDataset`, or a path to scan on the fly -- which
            is the common case for the reconstruction tab.
        subject_id: overrides the dataset's subject id.
        include_unresolved: when false, frames whose mapping is unproven are left
            out of ``frames`` and recorded in ``dropped`` instead of being emitted
            with a flag. The default keeps them visible.
    """
    if not isinstance(dataset, SourceDataset):
        from ..ingestion.raw_dataset_reader import read_raw_dataset

        dataset = read_raw_dataset(dataset)

    subject = subject_id or dataset.subject_id
    result = ReconstructionResult(
        subject_id=subject,
        session_id=dataset.session_id,
        source_root=dataset.root,
        warnings=list(dataset.warnings),
        errors=list(dataset.errors),
    )

    mappings = map_frames(dataset)
    result.mappings = mappings

    # Anything with a signal folder but no mapping at all is dropped, not lost.
    mapped_numbers = {m.frame_id for m in mappings}
    for folder in SIGNAL_FOLDERS:
        for record in dataset.by_folder(folder):
            if record.frame_id and record.frame_id not in mapped_numbers:
                result.dropped.append(
                    {
                        "folder": folder,
                        "frame_id": record.frame_id,
                        "reason": "no canonical frame was mapped for this source frame",
                    }
                )

    index = 0
    for mapping in mappings:
        resolved = _resolve_sources(dataset, mapping)
        frame, problems = _build_frame(
            dataset,
            mapping,
            resolved,
            subject_id=subject,
            session_id=dataset.session_id,
            record_index=index,
        )
        if frame is None:
            result.dropped.append(
                {
                    "frame_id": mapping.frame_id,
                    "reason": "; ".join(problems) or "signal source unreadable",
                    "mapping_status": mapping.status,
                }
            )
            continue
        if not mapping.resolved and not include_unresolved:
            result.dropped.append(
                {
                    "frame_id": mapping.frame_id,
                    "reason": "; ".join(mapping.reasons) or "mapping unresolved",
                    "mapping_status": mapping.status,
                }
            )
            continue
        if problems:
            result.warnings.extend(problems)
        result.frames.append(frame)
        index += 1

    return result


def reconstruct_directory(
    root: Path | str,
    *,
    subject_id: str | None = None,
    include_unresolved: bool = True,
) -> ReconstructionResult:
    """Scan and reconstruct in one step. Thin alias for ``reconstruct(path)``."""
    return reconstruct(root, subject_id=subject_id, include_unresolved=include_unresolved)


__all__ = ["SOURCE_PRECEDENCE", "reconstruct", "reconstruct_directory"]