"""Reader for ``logs/``: free-form device output, kept for traceability.

The 29-09-2026 session ships an empty ``logs/``, so this reader is written to
degrade quietly: it returns whatever it can parse and records the rest as
warnings. Logs never contribute ECG samples and never gate reconstruction --
they exist so an analyst can see what the device said around a frame
(IDEA-REVISED.md section 9A reads ``logs/`` "sesuai kebutuhan").
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..models.source_frame import FOLDER_LOGS

LOG_SUFFIXES = (".log", ".txt", ".jsonl")
MAX_LOG_BYTES = 5_000_000


def _read_text_lines(path: Path) -> list[str]:
    if path.stat().st_size > MAX_LOG_BYTES:
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def _entry_from_jsonl(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for line in _read_text_lines(path):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            entries.append({"raw": line})
            continue
        entries.append(payload if isinstance(payload, dict) else {"raw": payload})
    return entries


def read_logs(root: Path | str) -> tuple[list[dict[str, Any]], list[str]]:
    """Read ``<root>/logs/``.

    Returns ``(entries, warnings)``. Each entry carries the ``file`` it came
    from so the UI can show provenance; a file too large or unreadable yields a
    warning instead of an exception.
    """
    directory = Path(root) / FOLDER_LOGS
    entries: list[dict[str, Any]] = []
    warnings: list[str] = []
    if not directory.is_dir():
        return entries, warnings

    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in LOG_SUFFIXES:
            continue
        name = path.name
        try:
            if path.suffix.lower() == ".jsonl":
                parsed = _entry_from_jsonl(path)
            else:
                parsed = [{"raw": line} for line in _read_text_lines(path) if line.strip()]
        except OSError as exc:
            warnings.append(f"logs/{name} unreadable: {exc}")
            continue
        if not parsed:
            warnings.append(f"logs/{name} produced no entries (empty or over {MAX_LOG_BYTES} bytes)")
            continue
        for entry in parsed:
            entries.append({"file": name, **entry})

    return entries, warnings


__all__ = ["LOG_SUFFIXES", "read_logs"]
