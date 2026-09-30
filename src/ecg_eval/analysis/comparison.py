"""Optional repeated-measures statistics.

The three positions are measured on the same subjects, so a non-parametric
repeated-measures test (Friedman) is the appropriate default. The layer
reports WHETHER a statistically detectable difference exists and provides the
underlying values; it never declares a position clinically better or worse
(IDEA.md section 34).
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats

POSITIONS = ("SUPINE", "SITTING", "STANDING")
SQI_COLUMNS = ("qSQI", "pSQI", "kSQI", "basSQI")


def subject_position_samples(df: pd.DataFrame, column: str) -> tuple[list[str], list[list[float]]]:
    """Build per-subject sample vectors, keeping only complete subjects.

    Returns ``(subjects, samples)`` where ``samples[i]`` is the list of values
    subject ``i`` contributed, ordered SUPINE, SITTING, STANDING. A subject
    contributes only if it has at least one finite value in all three
    positions, which is what a repeated-measures test requires.
    """
    if df.empty or column not in df:
        return [], []
    frame = df[df["valid"]] if "valid" in df else df
    subjects = sorted(frame["subject_id"].dropna().unique().tolist())
    samples: list[list[float]] = []
    complete: list[str] = []
    for subject in subjects:
        row: list[float] = []
        for position in POSITIONS:
            values = frame[(frame["subject_id"] == subject) & (frame["position"] == position)][column]
            row.append(float(np.nanmean(values.to_numpy(dtype=float))) if len(values) else float("nan"))
        if all(np.isfinite(v) for v in row):
            samples.append(row)
            complete.append(subject)
    return complete, samples


def position_groups(samples: list[list[float]]) -> list[np.ndarray]:
    """Transpose subject-major samples into the position-major groups
    ``friedmanchisquare`` expects: one array per position, one value per subject.

    Frames are averaged within a subject and position first, so each subject
    contributes exactly one value per condition and the test is not skewed by
    subjects who happened to record more frames.
    """
    if not samples:
        return []
    array = np.asarray(samples, dtype=float)
    return [array[:, index] for index in range(array.shape[1])]


def friedman_test(df: pd.DataFrame, column: str) -> dict[str, Any]:
    """Friedman test across the three positions for one SQI."""
    subjects, samples = subject_position_samples(df, column)
    if len(subjects) < 3:
        return {
            "test": "Friedman",
            "column": column,
            "status": "insufficient data",
            "n_subjects": len(subjects),
            "note": "A repeated-measures test needs at least 3 subjects with data in all three positions.",
        }
    # friedmanchisquare takes one array per CONDITION, not one per subject, so
    # the subject-major samples must be transposed first.
    groups = position_groups(samples)
    if all(np.allclose(groups[0], group) for group in groups[1:]):
        # Every subject supplied identical values, so the ranking term is zero
        # and scipy returns nan with a divide warning. Reporting that as "no
        # difference" would be a false negative (IDEA.md section 34).
        return {
            "test": "Friedman",
            "column": column,
            "status": "not computable",
            "n_subjects": len(subjects),
            "note": (
                "All subjects have identical values across the three positions, so the "
                "test has no ranking information to work with."
            ),
        }
    try:
        statistic, p_value = stats.friedmanchisquare(*groups)
    except ValueError as exc:
        return {
            "test": "Friedman",
            "column": column,
            "status": "not computable",
            "n_subjects": len(subjects),
            "note": str(exc),
        }

    k = len(POSITIONS)
    n = len(subjects)
    statistic = float(np.asarray(statistic).reshape(-1)[0])
    p_value = float(np.asarray(p_value).reshape(-1)[0])
    # Kendall's coefficient of concordance, W in [0, 1].
    w = statistic / (n * (k - 1)) if n and k > 1 else math.nan

    if not (math.isfinite(statistic) and math.isfinite(p_value)):
        return {
            "test": "Friedman",
            "column": column,
            "status": "not computable",
            "n_subjects": n,
            "note": "The test statistic was not finite; no conclusion can be drawn.",
        }

    return {
        "test": "Friedman",
        "column": column,
        "status": "computed",
        "n_subjects": n,
        "positions": list(POSITIONS),
        "statistic": statistic,
        "p_value": p_value,
        "kendall_w": w,
        "subjects": subjects,
    }


def position_comparison(
    df: pd.DataFrame,
    *,
    alpha: float = 0.05,
    run_friedman: bool = True,
) -> dict[str, Any]:
    """Run the repeated-measures layer for all four SQIs.

    Returns the test outcomes plus the per-position descriptive values, so a
    reader can inspect the numbers behind any statement.
    """
    output: dict[str, Any] = {
        "alpha": alpha,
        "tests": {},
        "per_position": {},
        "detectable_difference": {},
        "disclaimer": (
            "A detectable difference is a statement about the sample, not about "
            "clinical performance. This project evaluates ECG signal quality only."
        ),
    }

    for column in SQI_COLUMNS:
        if column not in df:
            continue
        frame = df[df["valid"]] if "valid" in df else df
        output["per_position"][column] = {
            position: {
                "n": int(len(frame[frame["position"] == position])),
                "mean": _safe_mean(frame[frame["position"] == position][column]),
                "median": _safe_median(frame[frame["position"] == position][column]),
                "sd": _safe_sd(frame[frame["position"] == position][column]),
            }
            for position in POSITIONS
        }
        if run_friedman:
            outcome = friedman_test(df, column)
            output["tests"][column] = outcome
            p_value = outcome.get("p_value")
            output["detectable_difference"][column] = (
                outcome.get("status") == "computed"
                and p_value is not None
                and math.isfinite(float(p_value))
                and float(p_value) < alpha
            )
    return output


def _safe_mean(series: pd.Series) -> float:
    values = [float(v) for v in series.tolist() if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else math.nan


def _safe_median(series: pd.Series) -> float:
    values = [float(v) for v in series.tolist() if v is not None and np.isfinite(v)]
    return float(np.median(values)) if values else math.nan


def _safe_sd(series: pd.Series) -> float:
    values = [float(v) for v in series.tolist() if v is not None and np.isfinite(v)]
    return float(np.std(values, ddof=1)) if len(values) > 1 else math.nan


__all__ = [
    "POSITIONS",
    "SQI_COLUMNS",
    "friedman_test",
    "position_comparison",
    "position_groups",
    "subject_position_samples",
]
