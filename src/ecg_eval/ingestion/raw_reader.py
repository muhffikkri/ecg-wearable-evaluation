"""Raw-folder ingestion for the MQTT-mistake recording.

The raw tree under ``data/<date>/Bryan/`` contains::

    calibrated/   frame_000001_mv.{csv,json,npy}   <- (2500, 3) float32 mV
    model_ready/  frame_000001_input.{json,npy}    <- (2500, 3) float32 mV
    filtered/     intermediate firmware output
    raw_adc/      uncalibrated ADC counts
    logs/  predictions/

``calibrated/`` is the primary source for wearable evaluation
(IDEA.md section 6). ``model_ready/`` is readable too and is used as a
cross-check, because its JSON carries ``lead_statistics`` and ``validation``
that make drift between the two representations directly visible.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..models.frame import ECGFrame, SOURCE_RAW_CALIBRATED, SOURCE_RAW_MODEL_READY

FRAME_RE = re.compile(r"frame_(?P<number>\d+)(?P<suffix>[a-z_]*)")

RAW_KNOWN_SUBDIRS = ("calibrated", "model_ready", "filtered", "raw_adc", "predictions", "logs")


@dataclass
class RawReadResult:
    frames: list[ECGFrame] = field(default_factory=list)
    malformed: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    layout: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        if self.malformed:
            return "MALFORMED"
        if self.warnings:
            return "WARNING"
        return "OK"


def discover_raw_roots(data_dir: Path) -> list[Path]:
    """Find raw recording roots: any dir directly under a date folder."""
    data_dir = Path(data_dir)
    roots: list[Path] = []
    if not data_dir.exists():
        return roots
    for date_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        if date_dir.name.lower() in {"processed", "results", "annotations", "configs"}:
            continue
        for child in sorted(p for p in date_dir.iterdir() if p.is_dir()):
            if (child / "calibrated").is_dir() or (child / "model_ready").is_dir():
                roots.append(child)
    return roots


def _parse_frame_number(name: str) -> tuple[int, str]:
    match = FRAME_RE.search(name)
    if not match:
        raise ValueError(f"cannot parse frame number from {name!r}")
    return int(match.group("number")), match.group("suffix")


def _load_npy(path: Path) -> np.ndarray:
    array = np.load(path, allow_pickle=False)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"unexpected npy rank {array.ndim} (shape {array.shape})")
    return array.astype(np.float32, copy=False)


def read_raw_directory(
    root: Path,
    *,
    date_folder: str | None = None,
    subject_manifest: dict[str, str] | None = None,
    prefer: str = "calibrated",
) -> RawReadResult:
    """Read one raw recording root into canonical frames.

    ``prefer`` selects the primary representation. ``model_ready`` is used as
    a fallback when ``calibrated`` is missing, and its presence is reported in
    the layout so the UI can show that a subject came from the raw folder.
    """
    from .jsonl_reader import subject_id_for

    result = RawReadResult()
    root = Path(root)
    subject_id = subject_manifest.get(root.name, root.name) if subject_manifest else root.name
    session_id = f"raw_{root.name}"

    layout: dict[str, Any] = {
        "root": str(root),
        "subject_id": subject_id,
        "present_dirs": {name: (root / name).is_dir() for name in RAW_KNOWN_SUBDIRS},
    }

    source_dir = root / prefer
    if not source_dir.is_dir():
        fallback = "model_ready" if prefer == "calibrated" else "calibrated"
        if (root / fallback).is_dir():
            result.warnings.append(
                f"{prefer}/ missing, fell back to {fallback}/"
            )
            source_dir = root / fallback
            prefer = fallback
        else:
            result.warnings.append("neither calibrated/ nor model_ready/ present")
            layout["prefer"] = prefer
            result.layout = layout
            return result

    layout["prefer"] = prefer
    source_format = (
        SOURCE_RAW_CALIBRATED if prefer == "calibrated" else SOURCE_RAW_MODEL_READY
    )

    entries: dict[int, dict[str, Path]] = {}
    for path in sorted(source_dir.glob("frame_*.npy")):
        try:
            number, _ = _parse_frame_number(path.stem)
        except ValueError as exc:
            result.malformed.append({"source_file": str(path), "reason": str(exc)})
            continue
        entries.setdefault(number, {})["npy"] = path
        stem = path.stem
        json_path = path.with_name(f"{stem}.json")
        if json_path.exists():
            entries[number]["json"] = json_path
        csv_path = path.with_name(f"{stem}.csv")
        if csv_path.exists():
            entries[number]["csv"] = csv_path

    for number in sorted(entries):
        files = entries[number]
        try:
            signal = _load_npy(files["npy"])
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            result.malformed.append(
                {"source_file": str(files["npy"]), "frame_number": number, "reason": f"unreadable npy: {exc}"}
            )
            continue

        sidecar: dict[str, Any] = {}
        if "json" in files:
            try:
                sidecar = json.loads(files["json"].read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                result.malformed.append(
                    {"source_file": str(files["json"]), "frame_number": number, "reason": f"unreadable json: {exc}"}
                )

        sampling_rate = float(
            sidecar.get("sample_rate_hz")
            or sidecar.get("source_metadata", {}).get("sample_rate_hz")
            or 250.0
        )
        duration_s = float(sidecar.get("duration_seconds") or sidecar.get("duration_seconds") or 0.0)
        if not duration_s:
            duration_s = signal.shape[0] / sampling_rate if sampling_rate else 0.0

        declared_shape = sidecar.get("shape") or sidecar.get("source_metadata", {}).get("shape")
        if declared_shape and list(declared_shape) != list(signal.shape):
            result.warnings.append(
                f"frame {number:06d}: declared shape {declared_shape} != actual {list(signal.shape)}"
            )

        metadata: dict[str, Any] = {
            "unit": sidecar.get("unit", "mV"),
            "dtype": sidecar.get("dtype"),
            "channel_order": sidecar.get("channel_order")
            or sidecar.get("source_metadata", {}).get("channel_order"),
            "lead_mapping": sidecar.get("lead_mapping")
            or sidecar.get("source_metadata", {}).get("lead_mapping"),
            "calibration": sidecar.get("calibration"),
            "validation": sidecar.get("validation") or sidecar.get("signal_quality"),
            "source_processing": sidecar.get("source_processing"),
            "source_metadata": sidecar.get("source_metadata"),
            "raw_representation": prefer,
            "has_csv": "csv" in files,
            "has_sidecar_json": "json" in files,
        }
        extra = {k: v for k, v in sidecar.items() if k not in {
            "unit", "dtype", "channel_order", "lead_mapping", "calibration", "validation",
            "source_processing", "source_metadata", "shape", "sample_rate_hz",
            "duration_seconds", "signal_quality",
        }}
        if extra:
            metadata["extra_fields"] = extra

        frame = ECGFrame(
            subject_id=subject_id,
            session_id=session_id,
            frame_id=f"{number:06d}",
            source_file=str(files["npy"]),
            source_format=source_format,
            timestamp=sidecar.get("created_at_utc") or sidecar.get("created_at"),
            sampling_rate=sampling_rate,
            duration_s=duration_s,
            signal=signal,
            metadata=metadata,
            provenance={
                "reader": "raw_reader",
                "representation": prefer,
                "npy": str(files["npy"]),
                "json": str(files["json"]) if "json" in files else None,
                "csv": str(files["csv"]) if "csv" in files else None,
                "original_session_id": sidecar.get("source_metadata", {}).get("session_id"),
            },
            date_folder=date_folder,
        )
        result.frames.append(frame)

    result.frames.sort(key=lambda f: f.frame_id)
    for index, frame in enumerate(result.frames):
        frame.record_index = index

    # Report the model_ready counterpart count so the reconstruction tab can show
    # the asymmetry between the two representations.
    for other in ("calibrated", "model_ready"):
        other_dir = root / other
        if other_dir.is_dir():
            layout[f"{other}_frame_files"] = len(list(other_dir.glob("frame_*.npy")))

    result.layout = layout
    return result


__all__ = ["RawReadResult", "RAW_KNOWN_SUBDIRS", "discover_raw_roots", "read_raw_directory"]
