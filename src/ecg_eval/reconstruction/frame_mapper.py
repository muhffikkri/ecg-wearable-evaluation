"""Frame mapping: decide which Raspberry Pi files describe the same recording.

This is the piece IDEA-REVISED.md section 9C is about. It must not assume that
``calibrated frame 20 = model_ready frame 1``; it must establish the link from
evidence the firmware actually recorded, and report ``UNRESOLVED`` when it
cannot.

Evidence available in the real 29-09-2026 session, strongest first:

1. ``source_metadata.measurement_id`` -- a UUID identifying the physical
   measurement. Unique per frame in ``calibrated/`` and present in every
   ``filtered/`` frame, always naming the same frame number. This joins
   ``calibrated`` to ``filtered``.
2. ``source_file`` / ``source_metadata_file`` pointers. ``model_ready`` names
   the ``filtered`` file it was built from; ``predictions`` names the
   ``model_ready`` array it was computed from.
3. Timestamps, used only as a last resort and only when they match within
   ``timestamp_tolerance_s``.

The firmware writes no pointer from ``calibrated`` to ``raw_adc`` other than the
shared ``measurement_id``, and ``model_ready`` carries no ``measurement_id`` at
all -- so the real chain is a three-hop proof, not one join::

    calibrated --measurement_id--> filtered --source_file--> model_ready
                                                         --source_file--> prediction

Nothing here reads samples; it only decides membership. Ambiguity (two
candidates for one slot) is recorded as unresolved rather than resolved by
picking the first.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from ..models.reconstruction import (
    MAPPING_METHOD_MEASUREMENT_ID,
    MAPPING_METHOD_NONE,
    MAPPING_METHOD_RECORDED_POINTER,
    MAPPING_METHOD_TIMESTAMP,
    MAPPING_UNRESOLVED,
    MAPPING_VERIFIED,
    FrameMapping,
    SourceRef,
)
from ..models.source_frame import (
    FOLDER_CALIBRATED,
    FOLDER_FILTERED,
    FOLDER_MODEL_READY,
    FOLDER_PREDICTIONS,
    SIGNAL_FOLDERS,
    SourceDataset,
    SourceFrame,
)

#: Tolerance for the timestamp fallback. Frames are 10 s long and the firmware
#: stamps them at the end of the capture, so a few seconds is already generous.
DEFAULT_TIMESTAMP_TOLERANCE_S = 2.0


def measurement_id(record: SourceFrame) -> str | None:
    """The physical-measurement UUID a calibrated/filtered frame was built from."""
    source_metadata = record.metadata.get("source_metadata") or {}
    value = source_metadata.get("measurement_id")
    return str(value) if value else None


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _index_by_measurement_id(records: list[SourceFrame]) -> dict[str, list[SourceFrame]]:
    index: dict[str, list[SourceFrame]] = {}
    for record in records:
        key = measurement_id(record)
        if key:
            index.setdefault(key, []).append(record)
    return index


def _index_by_pointer(records: list[SourceFrame]) -> dict[str, list[SourceFrame]]:
    """Index records by the basename of the ``source_file`` they name."""
    index: dict[str, list[SourceFrame]] = {}
    for record in records:
        pointer = (record.metadata.get("recorded_source_files") or {}).get("source_file")
        if not pointer:
            continue
        index.setdefault(Path(str(pointer)).name, []).append(record)
    return index


def _pick(
    candidates: list[SourceFrame],
    reasons: list[str],
    label: str,
) -> SourceFrame | None:
    """Return the single candidate, or ``None`` when there is not exactly one."""
    if not candidates:
        return None
    if len(candidates) > 1:
        numbers = ", ".join(sorted(c.frame_id for c in candidates))
        reasons.append(f"ambiguous: {len(candidates)} candidates for {label} ({numbers})")
        return None
    return candidates[0]


def _source_ref(record: SourceFrame) -> SourceRef:
    return SourceRef(
        folder=record.folder,
        frame_number=record.frame_number,
        files=dict(record.files),
        recorded_source_files=dict(record.metadata.get("recorded_source_files") or {}),
    )


def _match_by_timestamp(
    anchor: SourceFrame,
    candidates: list[SourceFrame],
    tolerance_s: float,
) -> SourceFrame | None:
    """Last-resort link. Only used when neither a UUID nor a pointer matched."""
    anchor_time = _parse_timestamp(anchor.created_at)
    if anchor_time is None:
        return None
    within: list[SourceFrame] = []
    for candidate in candidates:
        candidate_time = _parse_timestamp(candidate.created_at)
        if candidate_time is None:
            continue
        if abs((candidate_time - anchor_time).total_seconds()) <= tolerance_s:
            within.append(candidate)
    return within[0] if len(within) == 1 else None


def map_frames(
    dataset: SourceDataset,
    *,
    timestamp_tolerance_s: float = DEFAULT_TIMESTAMP_TOLERANCE_S,
) -> list[FrameMapping]:
    """Build one :class:`FrameMapping` per canonical frame in a raw recording.

    The canonical frame is anchored on ``calibrated/`` when present, because
    IDEA-REVISED.md section 3 names it the source of the calibrated ECG signal.
    A session with only ``filtered/`` or ``model_ready/`` is still mapped, using
    whichever signal folder exists as the anchor.

    Returns mappings ordered by frame number. A frame whose own signal source is
    missing is not returned here; the reconstructor reports it as dropped.
    """
    calibrated = dataset.frame_map(FOLDER_CALIBRATED)
    filtered = dataset.frame_map(FOLDER_FILTERED)
    model_ready = dataset.frame_map(FOLDER_MODEL_READY)
    predictions = dataset.prediction_map()

    # filtered -> calibrated and calibrated -> filtered, via measurement_id.
    calibrated_by_mid = _index_by_measurement_id(list(calibrated.values()))
    filtered_by_mid = _index_by_measurement_id(list(filtered.values()))

    # model_ready -> filtered and prediction -> model_ready, via recorded pointer.
    model_ready_by_pointer = _index_by_pointer(list(model_ready.values()))
    predictions_by_pointer = _index_by_pointer(predictions.values())

    # Anchor numbering: prefer calibrated, then filtered, then model_ready.
    if calibrated:
        anchors = sorted(calibrated)
    elif filtered:
        anchors = sorted(filtered)
    else:
        anchors = sorted(model_ready)

    mappings: list[FrameMapping] = []

    for number in anchors:
        reasons: list[str] = []
        warnings: list[str] = []
        evidence: dict[str, Any] = {}
        methods: list[str] = []

        anchor_record = calibrated.get(number) or filtered.get(number) or model_ready[number]
        sources: dict[str, SourceRef] = {}

        # The anchor needs no link: it defines the frame. A session with a single
        # signal folder is therefore mapped, not stranded as unresolved.
        sources[anchor_record.folder] = _source_ref(anchor_record)

        # -- filtered, joined on the shared measurement UUID -------------
        filtered_record: SourceFrame | None = filtered.get(number)
        if number in filtered:
            filtered_record = filtered[number]
            sources.setdefault(FOLDER_FILTERED, _source_ref(filtered_record))

        mid = measurement_id(anchor_record)
        if mid and filtered_record is not None and number not in calibrated:
            # Anchor came from filtered, so find the calibrated twin.
            twin = _pick(calibrated_by_mid.get(mid, []), reasons, f"calibrated for measurement {mid}")
            if twin is not None:
                sources[FOLDER_CALIBRATED] = _source_ref(twin)
                methods.append(MAPPING_METHOD_MEASUREMENT_ID)
                evidence["measurement_id"] = mid
        elif mid and number in calibrated:
            # Confirm the filtered frame really is the same measurement.
            partners = filtered_by_mid.get(mid, [])
            partner = _pick(partners, reasons, f"filtered for measurement {mid}")
            if partner is not None:
                if partner.frame_number != number:
                    warnings.append(
                        f"measurement {mid} maps to filtered frame {partner.frame_id} "
                        f"but the anchor is frame {number:06d}"
                    )
                if number not in sources:
                    sources[FOLDER_FILTERED] = _source_ref(partner)
                methods.append(MAPPING_METHOD_MEASUREMENT_ID)
                evidence["measurement_id"] = mid
            elif partners == []:
                warnings.append(
                    f"measurement {mid} (frame {number:06d}) has no filtered/ counterpart"
                )

        # -- model_ready, joined on the pointer it records ---------------
        model_ready_record: SourceFrame | None = None
        if filtered_record is not None:
            pointer_name = f"frame_{filtered_record.frame_number:06d}_mv.npy"
            candidates = model_ready_by_pointer.get(pointer_name, [])
            model_ready_record = _pick(
                candidates, reasons, f"model_ready built from {pointer_name}"
            )
            if model_ready_record is not None:
                sources[FOLDER_MODEL_READY] = _source_ref(model_ready_record)
                methods.append(MAPPING_METHOD_RECORDED_POINTER)
                evidence["model_ready_source_file"] = pointer_name
            elif candidates or model_ready:
                reasons.append(
                    f"filtered frame {filtered_record.frame_id} has no model_ready frame "
                    f"recording source_file {pointer_name}"
                )
        elif model_ready and anchor_record.folder != FOLDER_MODEL_READY:
            # No filtered folder to point into, and the anchor is not itself a
            # model_ready frame, so the only remaining link is a timestamp.
            candidates = [
                record
                for record in model_ready.values()
                if measurement_id(record) in (None, mid)
            ]
            record = _match_by_timestamp(anchor_record, candidates, timestamp_tolerance_s)
            if record is not None:
                model_ready_record = record
                sources[FOLDER_MODEL_READY] = _source_ref(record)
                methods.append(MAPPING_METHOD_TIMESTAMP)
                evidence["timestamp_tolerance_s"] = timestamp_tolerance_s
            else:
                reasons.append(
                    "no filtered/ folder to anchor model_ready, and no timestamp match "
                    f"within {timestamp_tolerance_s}s"
                )

        # -- prediction, joined on the pointer it records ----------------
        if model_ready_record is not None:
            pointer_name = f"frame_{model_ready_record.frame_number:06d}_input.npy"
            candidates = predictions_by_pointer.get(pointer_name, [])
            record = _pick(candidates, reasons, f"prediction for {pointer_name}")
            if record is not None:
                sources[FOLDER_PREDICTIONS] = _source_ref(record)
                methods.append(MAPPING_METHOD_RECORDED_POINTER)
                evidence["prediction_source_file"] = pointer_name

        # A prediction sharing this frame's number but not linked by pointer must
        # never disappear without a trace, so the mapper says so explicitly.
        if FOLDER_PREDICTIONS not in sources and predictions:
            same_number = predictions.get(number)
            if same_number is not None:
                pointer = same_number.metadata.get("source_file")
                reasons.append(
                    f"prediction frame {number:06d} is not linked: it names {pointer!r}, "
                    "which is not this frame's model_ready array"
                )

        method = _dominant_method(methods)
        signal_source = _signal_source(sources)
        # Verified means: we hold a signal source, and nothing we tried to link
        # failed. A frame that is itself an anchor needs no cross-folder proof.
        resolved = bool(signal_source) and not reasons

        mappings.append(
            FrameMapping(
                frame_id=f"{number:06d}",
                status=MAPPING_VERIFIED if resolved else MAPPING_UNRESOLVED,
                method=method,
                signal_source=signal_source,
                sources=sources,
                evidence=evidence,
                reasons=reasons,
                warnings=warnings,
            )
        )

    return mappings


def _dominant_method(methods: list[str]) -> str:
    """Prefer the strongest evidence recorded for this frame."""
    for candidate in (MAPPING_METHOD_MEASUREMENT_ID, MAPPING_METHOD_RECORDED_POINTER, MAPPING_METHOD_TIMESTAMP):
        if candidate in methods:
            return candidate
    return MAPPING_METHOD_NONE


def _signal_source(sources: dict[str, SourceRef]) -> str:
    """Which folder supplies ``signal``; calibrated wins, per section 3."""
    for folder in SIGNAL_FOLDERS:
        if folder in sources:
            return folder
    return ""


__all__ = [
    "DEFAULT_TIMESTAMP_TOLERANCE_S",
    "map_frames",
    "measurement_id",
]