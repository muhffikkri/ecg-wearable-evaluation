"""Descriptive statistics helpers.

Deliberately explicit about NaN handling: frames with an invalid result stay
in the frame-level table but are excluded from aggregates, and the exclusion
count is reported.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import pandas as pd

SQI_COLUMNS = ("qSQI", "pSQI", "kSQI", "basSQI")
CLASS_COLUMNS = ("fuzzy_excellent", "fuzzy_barely_acceptable", "fuzzy_unacceptable")


def describe(values: Sequence[float]) -> dict[str, float]:
    """mean, median, SD, min, max and interquartile range.

    SD is the sample standard deviation (ddof=1) and is undefined for n < 2.
    """
    array = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if array.size == 0:
        return {
            "n": 0, "mean": math.nan, "median": math.nan, "sd": math.nan,
            "min": math.nan, "max": math.nan, "iqr": math.nan,
            "q1": math.nan, "q3": math.nan,
        }
    q1, q3 = (float(v) for v in np.percentile(array, [25, 75]))
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "sd": float(np.std(array, ddof=1)) if array.size > 1 else math.nan,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
    }


def class_distribution(classes: Sequence[str]) -> dict[str, float]:
    """Percentage of frames in each quality class."""
    total = len(classes)
    counts: dict[str, int] = {}
    for value in classes:
        counts[value] = counts.get(value, 0) + 1
    return {name: (100.0 * count / total if total else math.nan) for name, count in counts.items()}


def results_to_frame(results: Sequence[Any]) -> pd.DataFrame:
    """Frame results as a DataFrame, one row per analysed frame."""
    if not results:
        return pd.DataFrame(
            columns=[
                "subject_id", "session_id", "position", "frame_id", "internal_id",
                *SQI_COLUMNS, *CLASS_COLUMNS, "quality_class", "valid", "invalid_reason",
            ]
        )
    return pd.DataFrame([r.to_row() for r in results])


def aggregate_by(df: pd.DataFrame, keys: list[str], *, valid_only: bool = True) -> pd.DataFrame:
    """Group and compute descriptive stats for every SQI plus class shares."""
    if df.empty:
        return pd.DataFrame()
    frame = df[df["valid"]] if valid_only else df
    if frame.empty:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for key_values, group in frame.groupby(keys, dropna=False):
        key_values = key_values if isinstance(key_values, tuple) else (key_values,)
        record: dict[str, Any] = dict(zip(keys, key_values))
        record["n_frames"] = int(len(group))
        for column in SQI_COLUMNS:
            stats = describe(group[column].tolist())
            for stat_name, stat_value in stats.items():
                if stat_name == "n":
                    continue
                record[f"{column}_{stat_name}"] = stat_value
        distribution = class_distribution(group["quality_class"].tolist())
        for class_name, pct in distribution.items():
            record[f"pct_{class_name}"] = pct
        record["pct_unassigned"] = 100.0 - sum(pct for pct in distribution.values())
        records.append(record)
    return pd.DataFrame(records)


def position_summary(df: pd.DataFrame) -> pd.DataFrame:
    """SUPINE vs SITTING vs STANDING, the primary comparison."""
    if df.empty:
        return pd.DataFrame()
    table = aggregate_by(df, ["position"])
    if table.empty:
        return table
    order = {"SUPINE": 0, "SITTING": 1, "STANDING": 2}
    table = table.assign(_order=table["position"].map(order).fillna(99)).sort_values("_order")
    return table.drop(columns="_order").reset_index(drop=True)


def subject_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per subject per position, for inter-subject variability."""
    return aggregate_by(df, ["subject_id", "position"])


def subject_position_matrix(df: pd.DataFrame, value: str = "quality_score") -> pd.DataFrame:
    """Subjects x positions matrix used by the heatmap."""
    if df.empty:
        return pd.DataFrame()
    frame = df[df["valid"]]
    if frame.empty:
        return pd.DataFrame()
    pivot = frame.pivot_table(
        index="subject_id", columns="position", values=value, aggfunc="mean"
    )
    order = [p for p in ("SUPINE", "SITTING", "STANDING") if p in pivot.columns]
    return pivot[order] if order else pivot


def overall_summary(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {}
    valid = df[df["valid"]]
    summary: dict[str, Any] = {
        "total_frames": int(len(df)),
        "valid_frames": int(len(valid)),
        "invalid_frames": int(len(df) - len(valid)),
        "subjects": int(df["subject_id"].nunique()),
        "positions": sorted(df["position"].dropna().unique().tolist()),
    }
    if not valid.empty:
        summary["class_percentages"] = class_distribution(valid["quality_class"].tolist())
        for column in SQI_COLUMNS:
            summary[column] = describe(valid[column].tolist())
    return summary


__all__ = [
    "CLASS_COLUMNS",
    "SQI_COLUMNS",
    "aggregate_by",
    "class_distribution",
    "describe",
    "overall_summary",
    "position_summary",
    "results_to_frame",
    "subject_position_matrix",
    "subject_summary",
]
