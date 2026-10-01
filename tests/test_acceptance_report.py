"""Acceptance-category reporting: summary, decision table and figure."""

from __future__ import annotations

import pandas as pd
import pytest

from ecg_eval.analysis.statistics import ACCEPTANCE_COLUMNS, acceptance_summary
from ecg_eval.models.result import (
    EXCELLENT,
    OPTIMAL,
    SUSPICIOUS,
    UNDEFINED,
    UNQUALIFIED,
    SQI_KEYS,
)
from ecg_eval.visualization import (
    ACCEPTANCE_LABELS_ID,
    acceptance_coverage_table,
    acceptance_decision_table,
    acceptance_figure,
    sqi_label_id,
)

LEVEL_OF = {key: OPTIMAL for key in SQI_KEYS}


def frame_table(spec: dict[str, dict[str, list[str]]]) -> pd.DataFrame:
    """Build a frame-level table from ``position -> index -> [levels]``."""
    rows: list[dict[str, object]] = []
    for position, per_index in spec.items():
        length = len(next(iter(per_index.values())))
        for offset in range(length):
            row: dict[str, object] = {
                "position": position,
                "valid": True,
                "quality_class": EXCELLENT,
                "fusion_class": EXCELLENT,
            }
            for key in SQI_KEYS:
                levels = per_index.get(key)
                row[f"{key}_acceptance"] = levels[offset] if levels else OPTIMAL
            rows.append(row)
    return pd.DataFrame(rows)


def uniform(level: str, count: int) -> list[str]:
    return [level] * count


# ---------------------------------------------------------------------------
# acceptance_summary
# ---------------------------------------------------------------------------
def test_acceptance_summary_reports_the_majority_verdict():
    frame = frame_table(
        {
            "STANDING": {
                "qSQI": [OPTIMAL, OPTIMAL, UNQUALIFIED],
                "pSQI": uniform(OPTIMAL, 3),
                "kSQI": uniform(OPTIMAL, 3),
                "basSQI": uniform(OPTIMAL, 3),
            },
            "SUPINE": {
                "qSQI": [UNQUALIFIED, UNQUALIFIED, OPTIMAL],
                "pSQI": uniform(SUSPICIOUS, 3),
                "kSQI": uniform(OPTIMAL, 3),
                "basSQI": uniform(UNQUALIFIED, 3),
            },
        }
    )
    summary = acceptance_summary(frame)

    # Rows come back in Berbaring, Duduk, Berdiri order whatever the input order.
    assert list(summary["position"]) == ["SUPINE", "STANDING"]

    supine = summary.iloc[0]
    assert supine["qSQI_level"] == UNQUALIFIED
    assert supine["qSQI_n"] == 2
    assert supine["qSQI_pct"] == pytest.approx(200.0 / 3.0)
    assert supine["pSQI_level"] == SUSPICIOUS
    assert supine["basSQI_level"] == UNQUALIFIED
    assert not supine["qSQI_tied"]

    assert summary.iloc[1]["qSQI_level"] == OPTIMAL


def test_acceptance_summary_tie_goes_to_the_more_cautious_defined_level():
    frame = frame_table(
        {
            "SUPINE": {
                "qSQI": [OPTIMAL, UNQUALIFIED],
                "pSQI": [OPTIMAL, UNDEFINED],
                "kSQI": uniform(SUSPICIOUS, 2),
                "basSQI": [OPTIMAL, SUSPICIOUS],
            }
        }
    )
    row = acceptance_summary(frame).iloc[0]

    assert row["qSQI_level"] == UNQUALIFIED
    assert row["qSQI_tied"]
    # An inapplicable criterion must never displace a real verdict.
    assert row["pSQI_level"] == OPTIMAL
    assert row["pSQI_tied"]
    # A level with no members is never reported just because it is in the list.
    assert not row["kSQI_tied"]
    # A genuine 1-1 tie between two defined levels is still flagged.
    assert row["basSQI_tied"]
    assert row["basSQI_level"] == SUSPICIOUS


def test_acceptance_summary_excludes_invalid_frames():
    frame = frame_table({"SUPINE": {"qSQI": [OPTIMAL, OPTIMAL]}})
    frame.loc[1, "valid"] = False
    summary = acceptance_summary(frame)
    assert summary.iloc[0]["n_frames"] == 1
    assert summary.iloc[0]["qSQI_pct"] == pytest.approx(100.0)


def test_acceptance_summary_is_empty_safe():
    assert acceptance_summary(pd.DataFrame()).empty
    empty_but_valid = pd.DataFrame({"position": [], "valid": []})
    assert acceptance_summary(empty_but_valid).empty


# ---------------------------------------------------------------------------
# Decision and coverage tables
# ---------------------------------------------------------------------------
def test_decision_table_has_one_column_per_index_and_one_row_per_activity():
    frame = frame_table(
        {
            "SUPINE": {
                "qSQI": uniform(UNQUALIFIED, 2),
                "pSQI": uniform(OPTIMAL, 2),
                "kSQI": uniform(OPTIMAL, 2),
                "basSQI": uniform(SUSPICIOUS, 2),
            },
            "SITTING": {
                "qSQI": uniform(UNQUALIFIED, 2),
                "pSQI": uniform(OPTIMAL, 2),
                "kSQI": uniform(OPTIMAL, 2),
                "basSQI": uniform(OPTIMAL, 2),
            },
            "STANDING": {
                "qSQI": uniform(UNQUALIFIED, 2),
                "pSQI": uniform(UNDEFINED, 2),
                "kSQI": uniform(OPTIMAL, 2),
                "basSQI": uniform(OPTIMAL, 2),
            },
        }
    )
    table = acceptance_decision_table(acceptance_summary(frame))

    assert list(table.columns) == ["Aktivitas", *[sqi_label_id(k) for k in SQI_KEYS]]
    assert list(table["Aktivitas"]) == ["Berbaring", "Duduk", "Berdiri"]
    assert len(table) == 3

    supine = table.iloc[0]
    assert supine[sqi_label_id("qSQI")] == "Unqualified"
    assert supine[sqi_label_id("pSQI")] == "Optimal"
    assert supine[sqi_label_id("basSQI")] == "Suspicious"
    # The heart-rate-dependent index can be inapplicable on one activity only.
    assert table.iloc[2][sqi_label_id("pSQI")] == ACCEPTANCE_LABELS_ID[UNDEFINED]


def test_decision_table_marks_a_tie():
    frame = frame_table({"SUPINE": {"qSQI": [OPTIMAL, UNQUALIFIED]}})
    table = acceptance_decision_table(acceptance_summary(frame))
    assert "seri" in table.iloc[0][sqi_label_id("qSQI")]


def test_coverage_table_reports_the_share_behind_each_verdict():
    frame = frame_table({"SUPINE": {"qSQI": [OPTIMAL, OPTIMAL, UNQUALIFIED]}})
    coverage = acceptance_coverage_table(acceptance_summary(frame))
    assert list(coverage.columns) == ["Aktivitas", *[sqi_label_id(k) for k in SQI_KEYS]]
    assert coverage.iloc[0][sqi_label_id("qSQI")] == "66.7%"


def test_decision_and_coverage_tables_are_empty_safe():
    assert acceptance_decision_table(pd.DataFrame()).empty
    assert acceptance_coverage_table(pd.DataFrame()).empty


def test_decision_table_ignores_an_index_without_a_verdict_column():
    frame = frame_table({"SUPINE": {"qSQI": uniform(OPTIMAL, 2)}})
    frame = frame.drop(columns=[f"{k}_acceptance" for k in SQI_KEYS if k != "qSQI"])
    summary = acceptance_summary(frame)
    table = acceptance_decision_table(summary)
    assert list(table.columns) == ["Aktivitas", sqi_label_id("qSQI")]


# ---------------------------------------------------------------------------
# acceptance_figure
# ---------------------------------------------------------------------------
def test_acceptance_figure_stacks_one_trace_per_level_and_index():
    frame = frame_table(
        {
            "SUPINE": {
                "qSQI": uniform(UNQUALIFIED, 3),
                "pSQI": [OPTIMAL, OPTIMAL, SUSPICIOUS],
                "kSQI": uniform(OPTIMAL, 3),
                "basSQI": [OPTIMAL, UNDEFINED, OPTIMAL],
            },
            "SITTING": {key: uniform(OPTIMAL, 3) for key in SQI_KEYS},
            "STANDING": {key: uniform(SUSPICIOUS, 3) for key in SQI_KEYS},
        }
    )
    figure = acceptance_figure(frame)
    assert len(figure.data) == len(SQI_KEYS) * len(ACCEPTANCE_COLUMNS)


def test_acceptance_figure_shares_sum_to_one_hundred_per_activity():
    frame = frame_table(
        {
            "SUPINE": {"qSQI": [OPTIMAL, UNQUALIFIED], "pSQI": [OPTIMAL, UNDEFINED]},
            "SITTING": {"qSQI": [SUSPICIOUS, SUSPICIOUS], "pSQI": [OPTIMAL, OPTIMAL]},
        }
    )
    figure = acceptance_figure(frame, columns=["qSQI", "pSQI"])

    for index in range(0, len(figure.data), len(ACCEPTANCE_COLUMNS)):
        totals: dict[str, float] = {}
        for trace in figure.data[index : index + len(ACCEPTANCE_COLUMNS)]:
            for position, value in zip(trace.y, trace.x):
                totals[position] = totals.get(position, 0.0) + float(value)
        assert totals, "a panel produced no traces"
        for total in totals.values():
            assert total == pytest.approx(100.0)


def test_acceptance_figure_is_empty_safe_and_skips_missing_columns():
    assert not acceptance_figure(pd.DataFrame()).data
    frame = frame_table({"SUPINE": {"qSQI": uniform(OPTIMAL, 2)}})
    frame = frame.drop(columns=[f"{k}_acceptance" for k in SQI_KEYS if k != "qSQI"])
    figure = acceptance_figure(frame)
    assert len(figure.data) == len(ACCEPTANCE_COLUMNS)
