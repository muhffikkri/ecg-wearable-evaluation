"""Result export.

Writes the tree required by IDEA.md section 40::

    results/
    ├── frame_results.csv
    ├── subject_results.csv
    ├── position_results.csv
    ├── overall_results.json
    ├── interpretation.md
    ├── run_provenance.json
    ├── plots/
    └── reports/

Raw data under ``data/`` is never modified.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .statistics import overall_summary, position_summary, subject_summary

logger = logging.getLogger(__name__)

FRAME_CSV = "frame_results.csv"
SUBJECT_CSV = "subject_results.csv"
POSITION_CSV = "position_results.csv"
OVERALL_JSON = "overall_results.json"
PROVENANCE_JSON = "run_provenance.json"
INTERPRETATION_MD = "interpretation.md"
PLOTS_DIR = "plots"
REPORTS_DIR = "reports"


def _default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, set):
        return sorted(value)
    return str(value)


def ensure_dirs(results_dir: Path | str) -> dict[str, Path]:
    root = Path(results_dir)
    paths = {
        "root": root,
        "plots": root / PLOTS_DIR,
        "reports": root / REPORTS_DIR,
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def export_results(
    results_dir: Path | str,
    results: Sequence[Any],
    *,
    provenance: dict[str, Any] | None = None,
    dataset_summary: dict[str, Any] | None = None,
    annotation_summary: dict[str, Any] | None = None,
    coverage: dict[str, Any] | None = None,
    comparison: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write every result artefact. Returns the paths written."""
    paths = ensure_dirs(results_dir)
    written: dict[str, Path] = {}

    frame_df = pd.DataFrame([r.to_row() for r in results])
    if frame_df.empty:
        frame_df = pd.DataFrame(columns=["subject_id", "session_id", "position", "frame_id"])
    frame_path = paths["root"] / FRAME_CSV
    frame_df.to_csv(frame_path, index=False)
    written["frame_results"] = frame_path

    valid = frame_df[frame_df["valid"]] if not frame_df.empty and "valid" in frame_df else frame_df
    subject_df = subject_summary(valid) if not valid.empty else pd.DataFrame()
    subject_path = paths["root"] / SUBJECT_CSV
    subject_df.to_csv(subject_path, index=False)
    written["subject_results"] = subject_path

    position_df = position_summary(valid) if not valid.empty else pd.DataFrame()
    position_path = paths["root"] / POSITION_CSV
    position_df.to_csv(position_path, index=False)
    written["position_results"] = position_path

    overall = {
        "overall": overall_summary(valid) if not valid.empty else {},
        "dataset": dataset_summary or {},
        "annotations": annotation_summary or {},
        "coverage": coverage or {},
        "statistics": comparison or {},
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    overall_path = paths["root"] / OVERALL_JSON
    overall_path.write_text(
        json.dumps(overall, indent=2, default=_default, allow_nan=False), encoding="utf-8"
    )
    written["overall_results"] = overall_path

    if provenance:
        provenance_path = paths["root"] / PROVENANCE_JSON
        provenance_path.write_text(
            json.dumps(provenance, indent=2, default=_default, allow_nan=False), encoding="utf-8"
        )
        written["run_provenance"] = provenance_path

    logger.info("exported %d frame result(s) to %s", len(results), paths["root"])
    return written


def write_markdown(path: Path | str, markdown: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
    return path


def frames_to_records(results: Iterable[Any]) -> list[dict[str, Any]]:
    return [r.to_row() for r in results]


__all__ = [
    "FRAME_CSV",
    "INTERPRETATION_MD",
    "OVERALL_JSON",
    "PLOTS_DIR",
    "POSITION_CSV",
    "PROVENANCE_JSON",
    "REPORTS_DIR",
    "SUBJECT_CSV",
    "ensure_dirs",
    "export_results",
    "frames_to_records",
    "write_markdown",
]
