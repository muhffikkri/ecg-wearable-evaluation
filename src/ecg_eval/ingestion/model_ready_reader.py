"""Reader for ``model_ready/``: the array the on-device model actually consumed.

One frame is two files::

    frame_000001_input.npy    (2500, 3) float32 mV
    frame_000001_input.json   shape/dtype, lead statistics, input validation

The JSON also carries the pointers that make frame mapping provable rather than
assumed (IDEA-REVISED.md section 9C)::

    "source_file": ".../filtered/frame_000001_mv.npy"
    "source_metadata_file": ".../filtered/frame_000001_mv.json"

Those two names are the only trustworthy link to the other folders, so they are
surfaced as :attr:`SourceFrame.metadata['recorded_source_files']` and parsed
into a frame number by :func:`recorded_source_number`. ``filtered/`` is not
model input by definition: it is the notched signal the firmware copied.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..models.source_frame import FOLDER_MODEL_READY, SourceFrame
from .source_common import collect_frame_files, load_npy, parse_frame_id, parse_frame_number, read_json

# Keys in the model_ready JSON that point at the file this input was built from.
SOURCE_POINTER_KEYS = ("source_file", "source_metadata_file")


def recorded_source_number(sidecar: dict[str, Any]) -> int | None:
    """Frame number of the ``filtered/`` frame a model_ready input came from.

    Returns ``None`` when the firmware recorded no pointer, which forces the
    mapper to report ``UNRESOLVED`` instead of falling back to the frame's own
    number. Never falls back to the model_ready frame number: that is the exact
    assumption IDEA-REVISED.md forbids.
    """
    pointer = sidecar.get("source_file")
    if not pointer:
        return None
    try:
        return parse_frame_number(Path(pointer).stem)
    except ValueError:
        return None


def read_model_ready(root: Path | str, *, load_signals: bool = True) -> list[SourceFrame]:
    """Read ``<root>/model_ready/`` into one :class:`SourceFrame` per frame."""
    directory = Path(root) / FOLDER_MODEL_READY
    frames: list[SourceFrame] = []

    for number, files in sorted(collect_frame_files(directory).items()):
        record = SourceFrame(folder=FOLDER_MODEL_READY, frame_number=number)
        record.files = {key: str(path) for key, path in files.items()}

        sidecar: dict[str, Any] = {}
        if "json" in files:
            try:
                sidecar = read_json(files["json"])
            except Exception as exc:  # noqa: BLE001
                record.errors.append(f"unreadable json: {exc}")
        else:
            record.errors.append("missing frame_*_input.json")

        declared_id = sidecar.get("frame_id")
        if declared_id:
            try:
                if parse_frame_id(str(declared_id)) != number:
                    record.errors.append(
                        f"json frame_id {declared_id!r} disagrees with filename frame {number:06d}"
                    )
            except ValueError:
                record.errors.append(f"unparseable json frame_id {declared_id!r}")

        record.sampling_rate = (
            None if sidecar.get("sample_rate_hz") is None else float(sidecar["sample_rate_hz"])
        )
        record.duration_s = (
            None if sidecar.get("duration_seconds") is None else float(sidecar["duration_seconds"])
        )
        record.unit = sidecar.get("unit")
        record.dtype = sidecar.get("dtype")
        record.channel_order = sidecar.get("channel_order")
        record.created_at = sidecar.get("created_at") or sidecar.get("created_at_utc")

        if "npy" not in files:
            record.errors.append("missing frame_*_input.npy")
        elif load_signals:
            try:
                signal = load_npy(files["npy"])
            except Exception as exc:  # noqa: BLE001
                record.errors.append(f"unreadable npy: {exc}")
            else:
                declared = sidecar.get("shape")
                if declared and list(declared) != list(signal.shape):
                    record.errors.append(
                        f"declared shape {list(declared)} != actual {list(signal.shape)}"
                    )
                record.signal = signal

        record.metadata = {
            "canonical_format": sidecar.get("canonical_format"),
            "memory_layout": sidecar.get("memory_layout"),
            "channel_count": sidecar.get("channel_count"),
            "calibration_method": sidecar.get("calibration_method"),
            "calibration_scale_mV_per_count": sidecar.get("calibration_scale_mV_per_count"),
            "lead_statistics": sidecar.get("lead_statistics"),
            "validation": sidecar.get("validation"),
            "signal_quality": sidecar.get("signal_quality"),
            "source_processing": sidecar.get("source_processing"),
            "preprocessing_config": sidecar.get("preprocessing_config"),
            "preprocessing_steps_applied": sidecar.get("preprocessing_steps_applied"),
            "recorded_source_files": {k: sidecar[k] for k in SOURCE_POINTER_KEYS if sidecar.get(k)},
            "recorded_source_frame": recorded_source_number(sidecar),
        }
        frames.append(record)

    return frames


__all__ = ["SOURCE_POINTER_KEYS", "read_model_ready", "recorded_source_number"]
