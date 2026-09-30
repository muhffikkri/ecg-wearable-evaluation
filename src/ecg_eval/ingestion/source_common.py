"""Helpers shared by the per-folder Raspberry Pi readers.

Kept separate so ``calibrated_reader``, ``model_ready_reader``,
``prediction_reader`` and ``log_reader`` describe only what is specific to
their folder. Every failure is turned into a recorded error string rather than
an exception, because IDEA-REVISED.md section 13 forbids silently dropping
data: a corrupt or missing file has to reach the validation report.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

# frame_000001_mv / frame_000001_input / frame_000001_prediction
FRAME_RE = re.compile(r"^frame_(?P<number>\d+)(?:_(?P<suffix>[a-z]+))?$")

# The ``frame_id`` *field* inside sidecar JSON is not written the same way in
# every folder: calibrated and model_ready record ``"frame_000001"`` while
# predictions record the bare ``"000001"``. Both are accepted here.
FRAME_ID_RE = re.compile(r"^(?:frame_)?(?P<number>\d+)(?:_(?P<suffix>[a-z]+))?$")


def parse_frame_number(stem: str) -> int:
    """Return the frame number encoded in a firmware *filename* stem.

    Raises:
        ValueError: if the stem is not a ``frame_<n>[_<suffix>]`` name, so the
            caller can record it as a malformed entry instead of guessing.
    """
    match = FRAME_RE.match(stem)
    if not match:
        raise ValueError(f"cannot parse frame number from {stem!r}")
    return int(match.group("number"))


def parse_frame_id(value: str) -> int:
    """Return the frame number from a ``frame_id`` field in any observed encoding.

    Accepts ``"frame_000001"``, ``"000001"`` and ``"frame_000001_input"``.
    """
    match = FRAME_ID_RE.match(value.strip())
    if not match:
        raise ValueError(f"cannot parse frame id from {value!r}")
    return int(match.group("number"))


def load_npy(path: Path) -> np.ndarray:
    """Load a signal array and normalise it to 2-D float32 ``(samples, channels)``."""
    array = np.load(path, allow_pickle=False)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"unexpected npy rank {array.ndim} (shape {array.shape})")
    return array.astype(np.float32, copy=False)


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON sidecar, raising on malformed content for the caller to report."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object, got {type(payload).__name__}")
    return payload


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Stream a SHA-256 of a file, used to verify recorded ``source_sha256`` values."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def collect_frame_files(directory: Path, stem_prefix: str = "frame_") -> dict[int, dict[str, Path]]:
    """Group ``frame_*`` files in one folder by frame number.

    Returns ``{frame_number: {"npy": Path, "json": Path, "csv": Path, "bin": Path}}``.
    A frame number present only in a non-array file is still reported, so a
    prediction or sidecar without its signal is visible rather than invisible.
    """
    grouped: dict[int, dict[str, Path]] = {}
    if not directory.is_dir():
        return grouped
    for path in sorted(directory.iterdir()):
        if not path.is_file() or not path.name.startswith(stem_prefix):
            continue
        stem = path.stem
        try:
            number = parse_frame_number(stem)
        except ValueError:
            continue
        extension = path.suffix.lower().lstrip(".")
        if extension in {"npy", "json", "csv", "bin"}:
            grouped.setdefault(number, {})[extension] = path
    return grouped


__all__ = [
    "FRAME_ID_RE",
    "FRAME_RE",
    "collect_frame_files",
    "load_npy",
    "parse_frame_id",
    "parse_frame_number",
    "read_json",
    "sha256_file",
]
