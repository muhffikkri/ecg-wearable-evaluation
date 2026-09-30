"""Reader for ``calibrated/``: the gain/offset-corrected mV signal.

One frame is three files that must agree with each other::

    frame_000001_mv.npy    (2500, 3) float32 mV
    frame_000001_mv.json   calibration, channel order, lead mapping, source metadata
    frame_000001_mv.csv    the same samples as text

The JSON is authoritative for rate, unit and channel order; the NPY is
authoritative for the samples. When the declared shape disagrees with the array
this is reported as a warning rather than corrected, because a silent reshape
would fabricate data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..models.source_frame import FOLDER_CALIBRATED, SourceFrame
from .source_common import collect_frame_files, load_npy, read_json


def _sampling_rate(sidecar: dict[str, Any]) -> float | None:
    source = sidecar.get("source_metadata") or {}
    value = sidecar.get("sample_rate_hz", source.get("sample_rate_hz"))
    return None if value is None else float(value)


def read_calibrated(
    root: Path | str,
    *,
    folder: str = FOLDER_CALIBRATED,
    load_signals: bool = True,
) -> list[SourceFrame]:
    """Read ``<root>/<folder>/`` into one :class:`SourceFrame` per frame.

    Args:
        root: the subject folder, i.e. the parent of ``folder``.
        folder: ``calibrated/`` by default. ``filtered/`` is read through the
            same function because the firmware writes the identical
            ``frame_*_mv.{csv,json,npy}`` layout there; only the recorded folder
            name differs.
        load_signals: when false only the sidecars are read. Inspection views use
            this so a 20-frame subject costs 20 small JSON reads instead of 20
            array reads.
    """
    directory = Path(root) / folder
    frames: list[SourceFrame] = []

    for number, files in sorted(collect_frame_files(directory).items()):
        record = SourceFrame(folder=folder, frame_number=number)
        record.files = {key: str(path) for key, path in files.items()}

        sidecar: dict[str, Any] = {}
        if "json" in files:
            try:
                sidecar = read_json(files["json"])
            except Exception as exc:  # noqa: BLE001 - surfaced in the report
                record.errors.append(f"unreadable json: {exc}")

        record.sampling_rate = _sampling_rate(sidecar)
        record.duration_s = (
            float(sidecar["duration_seconds"]) if sidecar.get("duration_seconds") is not None else None
        )
        record.unit = sidecar.get("unit")
        record.dtype = sidecar.get("dtype")
        record.channel_order = sidecar.get("channel_order") or (
            (sidecar.get("source_metadata") or {}).get("channel_order")
        )
        record.created_at = sidecar.get("created_at_utc") or sidecar.get("created_at")

        if "npy" not in files:
            record.errors.append("missing frame_*_mv.npy")
        elif not load_signals:
            pass
        else:
            try:
                signal = load_npy(files["npy"])
            except Exception as exc:  # noqa: BLE001
                record.errors.append(f"unreadable npy: {exc}")
            else:
                declared = sidecar.get("shape") or (sidecar.get("source_metadata") or {}).get("shape")
                if declared and list(declared) != list(signal.shape):
                    record.errors.append(
                        f"declared shape {list(declared)} != actual {list(signal.shape)}"
                    )
                record.signal = signal
                if record.duration_s is None and record.sampling_rate:
                    record.duration_s = signal.shape[0] / record.sampling_rate

        if record.unit and record.unit.lower() != "mv":
            record.errors.append(f"unexpected unit {record.unit!r} in {folder}/")

        record.metadata = {
            "calibration": sidecar.get("calibration"),
            "lead_mapping": sidecar.get("lead_mapping")
            or (sidecar.get("source_metadata") or {}).get("lead_mapping"),
            "ch3_derived": sidecar.get("ch3_derived"),
            "ch3_max_error_mV": sidecar.get("ch3_max_error_mV"),
            "source_frame": sidecar.get("source_frame"),
            "source_metadata": sidecar.get("source_metadata"),
            "has_csv": "csv" in files,
        }
        frames.append(record)

    return frames


__all__ = ["read_calibrated"]
