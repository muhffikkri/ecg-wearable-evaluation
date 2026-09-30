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


CLASS_LEVELS = ("Excellent", "Barely Acceptable", "Unacceptable")

#: The classes a frame must reach to count as usable. "Acceptable" is a
#: *recording* judgement, not a statement about the wearer, which is why the two
#: levels below are kept as a named group rather than inlined at each call site.
ACCEPTED_CLASSES = ("Excellent", "Barely Acceptable")


def acceptable_share(
    df: pd.DataFrame, levels: Sequence[str] = ACCEPTED_CLASSES
) -> pd.DataFrame:
    """Per subject and position, the share of frames that reached an accepted class.

    One row per subject per position, so the value describes a single person's
    recording under a single activity. That is the unit the per-activity box
    plot needs: one observation per participant, so a participant who recorded
    one activity cannot drag another activity's spread.
    """
    if df.empty:
        return pd.DataFrame()
    frame = df[df["valid"]] if "valid" in df else df
    if frame.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for (subject, position), group in frame.groupby(["subject_id", "position"], dropna=False):
        total = len(group)
        accepted = int(group["quality_class"].isin(levels).sum())
        rows.append(
            {
                "subject_id": subject,
                "position": position,
                "n_frames": int(total),
                "n_accepted": accepted,
                "pct_accepted": 100.0 * accepted / total if total else math.nan,
            }
        )
    order = {"SUPINE": 0, "SITTING": 1, "STANDING": 2}
    table = pd.DataFrame(rows)
    table["_order"] = table["position"].map(order).fillna(99)
    return table.sort_values(["_order", "subject_id"]).drop(columns="_order").reset_index(drop=True)


def class_table(df: pd.DataFrame, levels: Sequence[str] = CLASS_LEVELS) -> pd.DataFrame:
    """Per position: how many frames landed in each class, and the share.

    Every requested level gets a column even when it has no members, so the
    table never implies a class was skipped rather than empty.
    """
    if df.empty:
        return pd.DataFrame()
    frame = df[df["valid"]] if "valid" in df else df
    if frame.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for position, group in frame.groupby("position", dropna=False):
        total = len(group)
        counts = group["quality_class"].value_counts()
        row: dict[str, Any] = {"position": position, "n_frames": int(total)}
        for level in levels:
            row[f"count_{level}"] = int(counts.get(level, 0))
            row[f"pct_{level}"] = 100.0 * int(counts.get(level, 0)) / total if total else math.nan
        row["pct_accepted"] = sum(
            row[f"pct_{level}"] for level in levels if level in ACCEPTED_CLASSES
        )
        rows.append(row)

    order = {"SUPINE": 0, "SITTING": 1, "STANDING": 2}
    table = pd.DataFrame(rows)
    table["_order"] = table["position"].map(order).fillna(99)
    return table.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def box_outliers(
    values: Sequence[float], *, fences: float = 1.5
) -> tuple[dict[str, float], list[float]]:
    """Tukey five-number summary plus the points beyond the fences.

    Returns the summary and the outlier values separately, so a caller can draw
    the box and the outlier dots as two independent traces. Plotly can do this
    itself with ``boxpoints='outliers'``, but then the dots cannot be given
    their own legend entry or colour without also restyling the box.
    """
    array = np.asarray(
        [v for v in values if v is not None and np.isfinite(v)], dtype=np.float64
    )
    if array.size == 0:
        return {}, []
    q1, q3 = (float(v) for v in np.percentile(array, [25, 75]))
    iqr = q3 - q1
    low = q1 - fences * iqr
    high = q3 + fences * iqr
    inside = array[(array >= low) & (array <= high)]
    summary = {
        "n": int(array.size),
        "min": float(np.min(inside)),
        "q1": q1,
        "median": float(np.median(array)),
        "q3": q3,
        "max": float(np.max(inside)),
        "lower_fence": low,
        "upper_fence": high,
    }
    return summary, [float(v) for v in array if v < low or v > high]


def class_distribution(classes: Sequence[str]) -> dict[str, float]:
    """Percentage of frames in each quality class.

    All three classes are always present. A class with no frames is reported as
    0.0% rather than omitted, because an absent key reads as "not measured"
    where the honest answer is "none occurred" -- a reader would otherwise see
    "n/a Barely Acceptable" for a class that simply had no members.
    """
    total = len(classes)
    counts: dict[str, int] = {level: 0 for level in CLASS_LEVELS}
    for value in classes:
        counts[value] = counts.get(value, 0) + 1
    return {
        name: (100.0 * count / total if total else math.nan)
        for name, count in counts.items()
    }


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


_POSITION_ORDER = {"SUPINE": 0, "SITTING": 1, "STANDING": 2}


def _in_position_order(table: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Sort a position-keyed table into Berbaring, Duduk, Berdiri order."""
    table = table.copy()
    table["_order"] = table["position"].map(_POSITION_ORDER).fillna(99)
    keys = ["_order", *by] if by else ["_order"]
    return table.sort_values(keys).drop(columns="_order").reset_index(drop=True)


def activity_recap(
    df: pd.DataFrame,
    levels: Sequence[str] = CLASS_LEVELS,
    columns: Sequence[str] = SQI_COLUMNS,
) -> pd.DataFrame:
    """Per-activity recap: class counts, acceptance share and SQI summaries.

    This is the aggregate the report is built around. Unlike
    :func:`acceptable_share`, which gives one value per person per activity,
    every row here pools all frames of one activity, so the row states what the
    whole activity produced. The SQI columns keep the underlying spread
    (``mean``/``median``/``sd``) so the recap table and the box plot drawn from
    the same frames can be checked against each other.
    """
    if df.empty:
        return pd.DataFrame()
    frame = df[df["valid"]] if "valid" in df else df
    if frame.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for position, group in frame.groupby("position", dropna=False):
        total = len(group)
        counts = group["quality_class"].value_counts()
        row: dict[str, Any] = {"position": position, "n_frames": int(total)}
        for level in levels:
            row[f"count_{level}"] = int(counts.get(level, 0))
            row[f"pct_{level}"] = (
                100.0 * int(counts.get(level, 0)) / total if total else math.nan
            )
        row["pct_accepted"] = sum(
            row[f"pct_{level}"] for level in levels if level in ACCEPTED_CLASSES
        )
        for column in columns:
            if column not in group:
                continue
            stats = describe(group[column].dropna().tolist())
            row[f"{column}_mean"] = stats["mean"]
            row[f"{column}_median"] = stats["median"]
            row[f"{column}_sd"] = stats["sd"]
        rows.append(row)

    return _in_position_order(pd.DataFrame(rows), [])


__all__ = [
    "ACCEPTED_CLASSES",
    "CLASS_COLUMNS",
    "CLASS_LEVELS",
    "SQI_COLUMNS",
    "acceptable_share",
    "activity_recap",
    "aggregate_by",
    "box_outliers",
    "class_distribution",
    "class_table",
    "describe",
    "overall_summary",
    "position_summary",
    "results_to_frame",
    "subject_position_matrix",
    "subject_summary",
]
