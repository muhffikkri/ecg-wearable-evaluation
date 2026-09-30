"""Regression tests for the activity table and the Indonesian figure captions.

The acceptance table and the participant/activity box plot are report artefacts:
their numbers must be derived from the frame results, and their legends and
interpretations must be Indonesian and must never assert more than the numbers
support.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ecg_eval.analysis.statistics import (
    ACCEPTED_CLASSES,
    acceptable_share,
    activity_recap,
    box_outliers,
    class_table,
)
from ecg_eval.interpretation.engine import problematic_frames
from ecg_eval.visualization import (
    CLASS_LABELS_ID,
    POSITION_LABELS_ID,
    POSITION_LABELS_ID,
    POSITION_SUMMARY_LABELS,
    RECAP_STATS,
    SQI_LABELS,
    acceptance_table,
    acceptance_table_interpretation,
    acceptance_table_legend,
    activity_recap_distribution_interpretation,
    activity_recap_distribution_legend,
    activity_recap_figure,
    activity_recap_interpretation,
    activity_recap_legend,
    activity_recap_table,
    ecg_legend,
    fuzzy_matrix_interpretation,
    fuzzy_matrix_legend,
    heatmap_interpretation,
    heatmap_legend,
    membership_interpretation,
    membership_legend,
    multilead_legend,
    position_label,
    psd_interpretation,
    psd_legend,
    quality_by_activity_figure,
    quality_by_activity_interpretation,
    quality_by_activity_legend,
    quality_distribution_interpretation,
    quality_distribution_legend,
    sqi_box_interpretation,
    sqi_box_legend,
    sqi_label_id,
    sqi_stat_columns,
    sqi_stat_display_names,
    sqi_trend_interpretation,
    sqi_trend_legend,
)

POSITIONS = ("SUPINE", "SITTING", "STANDING")
GRADES = ("Excellent", "Barely Acceptable", "Unacceptable")


def make_frames(per_subject: dict[str, list[str]] | None = None) -> pd.DataFrame:
    """Build a frame-results frame from ``subject -> list of quality classes``."""
    rows = []
    for subject, classes in (per_subject or {}).items():
        for position in POSITIONS:
            for index, quality in enumerate(classes):
                rows.append(
                    {
                        "subject_id": subject,
                        "session_id": f"session-{subject}",
                        "position": position,
                        "frame_id": f"{subject}-{position}-{index:03d}",
                        "quality_class": quality,
                        "valid": True,
                        "qSQI": 0.9,
                        "pSQI": 0.6,
                        "kSQI": 0.5,
                        "basSQI": 0.4,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture()
def frames() -> pd.DataFrame:
    """Eight subjects whose acceptance differs by activity, with one outlier.

    Baseline acceptance per subject is 100/90/80 percent, which keeps the
    interquartile range wide enough that ordinary variation is not flagged.
    ``S04`` collapses to zero while sitting, which is the only value that falls
    outside the 1.5 x IQR fences of its activity.

    The four indices drift slightly per subject and per frame so the box plots
    have a real spread, and a two-frame stretch of the first subject while
    supine sits far below the rest so it shows up as an outlying frame.
    """
    baseline = {
        "S00": 100, "S01": 100, "S02": 90, "S03": 80,
        "S04": 100, "S05": 90, "S06": 80, "S07": 100,
    }
    rows = []
    for subject_index, (subject, percent) in enumerate(baseline.items()):
        # Ten frames per activity so the percentages land on whole numbers.
        accepted = int(percent / 10)
        classes = ["Excellent"] * accepted + ["Unacceptable"] * (10 - accepted)
        for position in POSITIONS:
            for index, quality in enumerate(classes):
                drift = 0.005 * subject_index + 0.001 * index
                noisy = subject == "S00" and position == "SUPINE" and index < 2
                rows.append(
                    {
                        "subject_id": subject,
                        "session_id": f"session-{subject}",
                        "position": position,
                        "frame_id": f"{subject}-{position}-{index:03d}",
                        "quality_class": (
                            "Unacceptable" if (subject == "S04" and position == "SITTING")
                            else quality
                        ),
                        "valid": True,
                        "qSQI": (0.20 if noisy else 0.90 + drift),
                        "pSQI": 0.60 + 0.4 * drift,
                        "kSQI": 3.50 + 0.8 * drift,
                        "basSQI": 0.40 + 0.5 * drift,
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- statistics


def test_accepted_classes_are_the_two_usable_levels():
    assert set(ACCEPTED_CLASSES) == {"Excellent", "Barely Acceptable"}
    assert "Unacceptable" not in ACCEPTED_CLASSES


def test_class_table_counts_sum_to_n_frames(frames):
    table = class_table(frames)
    assert list(table["position"]) == list(POSITIONS)
    for _, row in table.iterrows():
        graded = sum(int(row[f"count_{level}"]) for level in GRADES)
        assert graded == int(row["n_frames"])


def test_class_table_percentages_recompute_from_counts(frames):
    table = class_table(frames)
    for _, row in table.iterrows():
        total = int(row["n_frames"])
        for level in GRADES:
            assert row[f"pct_{level}"] == pytest.approx(
                100.0 * int(row[f"count_{level}"]) / total
            )


def test_class_table_accepted_percentage_matches_accepted_classes(frames):
    table = class_table(frames)
    for _, row in table.iterrows():
        expected = 100.0 * sum(int(row[f"count_{l}"]) for l in ACCEPTED_CLASSES) / int(row["n_frames"])
        assert row["pct_accepted"] == pytest.approx(expected)


def test_class_table_excludes_invalid_frames(frames):
    dirty = frames.copy()
    dirty.loc[dirty.index[:5], "valid"] = False
    clean = class_table(frames)
    dirty_table = class_table(dirty)
    assert int(dirty_table["n_frames"].sum()) == int(clean["n_frames"].sum()) - 5


def test_acceptable_share_one_value_per_subject_and_position(frames):
    share = acceptable_share(frames)
    assert len(share) == frames["subject_id"].nunique() * len(POSITIONS)
    assert not share.duplicated(["subject_id", "position"]).any()
    assert list(share["position"].unique()) == [p for p in POSITIONS if p in set(share["position"])]


def test_acceptable_share_percentage_matches_its_own_counts(frames):
    share = acceptable_share(frames)
    for _, row in share.iterrows():
        assert row["pct_accepted"] == pytest.approx(
            100.0 * int(row["n_accepted"]) / int(row["n_frames"])
        )


def test_acceptable_share_counts_barely_acceptable_as_usable():
    df = make_frames({"S00": ["Barely Acceptable"] * 4})
    row = acceptable_share(df).iloc[0]
    assert int(row["n_accepted"]) == 4
    assert row["pct_accepted"] == pytest.approx(100.0)


def test_box_outliers_uses_the_one_point_five_iqr_rule():
    values = [10.0, 11.0, 12.0, 13.0, 14.0, 60.0]
    summary, flagged = box_outliers(values)
    lower = summary["q1"] - 1.5 * (summary["q3"] - summary["q1"])
    upper = summary["q3"] + 1.5 * (summary["q3"] - summary["q1"])
    assert summary["lower_fence"] == pytest.approx(lower)
    assert summary["upper_fence"] == pytest.approx(upper)
    assert flagged == [60.0]
    assert 10.0 <= summary["min"] and summary["max"] <= 14.0


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_box_outliers_tolerates_tiny_groups(n):
    summary, flagged = box_outliers([float(v) for v in range(n)])
    assert summary["n"] == n
    assert isinstance(flagged, list)


def test_box_outliers_empty_is_empty():
    summary, flagged = box_outliers([])
    assert summary == {}
    assert flagged == []


# ------------------------------------------------------------ activity recap


def test_activity_recap_has_one_row_per_activity(frames):
    recap = activity_recap(frames)
    assert list(recap["position"]) == list(POSITIONS)
    assert int(recap["n_frames"].sum()) == len(frames)


def test_activity_recap_agrees_with_class_table_on_the_class_columns(frames):
    recap = activity_recap(frames)
    classes = class_table(frames)
    for position in POSITIONS:
        row = recap[recap["position"] == position].iloc[0]
        other = classes[classes["position"] == position].iloc[0]
        assert int(row["n_frames"]) == int(other["n_frames"])
        for level in GRADES:
            assert int(row[f"count_{level}"]) == int(other[f"count_{level}"])
            assert row[f"pct_{level}"] == pytest.approx(other[f"pct_{level}"])


def test_activity_recap_accepted_share_matches_class_table(frames):
    recap = activity_recap(frames)
    classes = class_table(frames)
    for position in POSITIONS:
        row = recap[recap["position"] == position].iloc[0]
        other = classes[classes["position"] == position].iloc[0]
        assert row["pct_accepted"] == pytest.approx(other["pct_accepted"])


def test_activity_recap_summarises_every_index(frames):
    recap = activity_recap(frames)
    for key in ("qSQI", "pSQI", "kSQI", "basSQI"):
        for stat in ("mean", "median", "sd"):
            assert f"{key}_{stat}" in recap.columns


def test_activity_recap_statistics_match_the_frames_behind_them(frames):
    recap = activity_recap(frames)
    valid = frames[frames["valid"]]
    for position in POSITIONS:
        values = valid[valid["position"] == position]["qSQI"].dropna()
        row = recap[recap["position"] == position].iloc[0]
        assert row["qSQI_mean"] == pytest.approx(float(values.mean()))
        assert row["qSQI_median"] == pytest.approx(float(values.median()))
        assert row["qSQI_sd"] == pytest.approx(float(values.std(ddof=1)))


def test_activity_recap_excludes_invalid_frames(frames):
    dirty = frames.copy()
    dirty.loc[dirty.index[:7], "valid"] = False
    assert int(activity_recap(dirty)["n_frames"].sum()) == len(frames) - 7


def test_activity_recap_sd_is_undefined_for_a_single_frame():
    df = make_frames({"S00": ["Excellent"]})
    row = activity_recap(df).iloc[0]
    assert int(row["n_frames"]) == 1
    assert not np.isfinite(row["qSQI_sd"])


def test_activity_recap_empty_stays_empty():
    assert activity_recap(pd.DataFrame()).empty
    assert activity_recap_table(activity_recap(pd.DataFrame())).empty


# -------------------------------------------------------------------- table


def test_acceptance_table_uses_indonesian_headers(frames):
    table = acceptance_table(class_table(frames))
    assert list(table["Aktivitas"]) == [POSITION_LABELS_ID[p] for p in POSITIONS]
    assert "Sangat Baik (n)" in table.columns
    assert "Cukup Diterima (%)" in table.columns
    assert "Tidak Diterima (%)" in table.columns
    assert "Diterima (%)" in table.columns


def test_acceptance_table_headers_carry_no_machine_keys(frames):
    columns = list(acceptance_table(class_table(frames)).columns)
    for column in columns:
        for key in (*POSITION_LABELS_ID, *CLASS_LABELS_ID, "SUPINE", "qSQI"):
            assert key not in column


def test_acceptance_table_totals_match_the_frame_counts(frames):
    table = acceptance_table(class_table(frames))
    assert int(table["Total Frame"].sum()) == len(frames)
    per_activity = {row["Aktivitas"]: int(row["Total Frame"]) for _, row in table.iterrows()}
    for position in POSITIONS:
        expected = int((frames["position"] == position).sum())
        assert per_activity[POSITION_LABELS_ID[position]] == expected


def test_acceptance_table_percentages_are_rendered_as_percent_strings(frames):
    table = acceptance_table(class_table(frames))
    for column in ("Sangat Baik (%)", "Tidak Diterima (%)", "Diterima (%)"):
        for value in table[column]:
            assert isinstance(value, str) and value.endswith("%")


def test_acceptance_table_empty_stays_empty():
    assert acceptance_table(pd.DataFrame()).empty


# ------------------------------------------------- recap table and its text


def test_recap_table_headers_are_indonesian(frames):
    table = activity_recap_table(activity_recap(frames))
    assert list(table["Aktivitas"]) == [POSITION_LABELS_ID[p] for p in POSITIONS]
    assert "Diterima (%)" in table.columns
    for key in ("qSQI", "pSQI", "kSQI", "basSQI"):
        assert not any(key in column for column in table.columns)


def test_position_summary_labels_drop_the_machine_keys():
    assert POSITION_SUMMARY_LABELS["position"] == "Aktivitas"
    assert POSITION_SUMMARY_LABELS["n_frames"] == "Total Frame"
    # A reader must not have to decode SUPINE/SITTING/STANDING in a table.
    assert not any(
        bad in name for name in POSITION_SUMMARY_LABELS.values()
        for bad in ("SUPINE", "SITTING", "STANDING", "n_frames")
    )


def test_position_summary_table_headers_are_all_indonesian():
    from app_ui.page_interpretation import _DISPLAY_NAMES

    frame = pd.DataFrame({
        "position": ["SUPINE", "SITTING", "STANDING"],
        "n_frames": [4, 4, 4],
        **{column: [0.5, 0.5, 0.5] for column in sqi_stat_columns()},
    })
    renamed = frame[["position", "n_frames", *sqi_stat_columns()]].rename(
        columns=_DISPLAY_NAMES
    )
    assert list(renamed.columns) == ["Aktivitas", "Total Frame", *sqi_stat_display_names().values()]
    assert not renamed.columns.duplicated().any()
    for column in renamed.columns:
        for bad in ("SUPINE", "SITTING", "STANDING", "n_frames", "mean", "median", "sd"):
            assert bad not in column


def test_problematic_frame_table_headers_are_indonesian():
    # The "Problematic frames" table is a reading surface, not a data dump, so
    # it must not push machine keys at the reader.
    raw = pd.DataFrame([{
        "subject_id": "S00", "position": "SUPINE", "frame_id": "F-1",
        "qSQI": 0.9, "pSQI": 0.6, "kSQI": 3.5, "basSQI": 0.4,
        "fuzzy_excellent": 0.9, "fuzzy_barely_acceptable": 0.05,
        "fuzzy_unacceptable": 0.05, "quality_class": "Excellent",
    }])
    shown = (
        problematic_frames(raw)[[
            "subject_id", "position", "frame_id",
            *SQI_LABELS,
            "fuzzy_excellent", "fuzzy_unacceptable", "quality_class",
        ]].rename(columns={
            **POSITION_SUMMARY_LABELS,
            "subject_id": "Peserta",
            "frame_id": "Frame",
            "fuzzy_excellent": "Keanggotaan Sangat Baik",
            "fuzzy_unacceptable": "Keanggotaan Tidak Diterima (U)",
            "quality_class": "Kelas Kualitas",
            **{key: sqi_label_id(key) for key in SQI_LABELS},
        })
    )
    assert list(shown.columns) == [
        "Peserta", "Aktivitas", "Frame",
        *[sqi_label_id(key) for key in SQI_LABELS],
        "Keanggotaan Sangat Baik", "Keanggotaan Tidak Diterima (U)",
        "Kelas Kualitas",
    ]
    for column in shown.columns:
        for bad in ("SUPINE", "subject_id", "frame_id", "quality_class", "fuzzy_"):
            assert bad not in column
    # The value columns stay numeric so the reader can rank them.
    assert shown["Keanggotaan Tidak Diterima (U)"].dtype.kind == "f"


def test_recap_table_uses_full_indonesian_stat_names(frames):
    table = activity_recap_table(activity_recap(frames))
    stats = {suffix: name for suffix, name in RECAP_STATS}
    assert stats["mean"] == "rata-rata"
    assert stats["median"] == "tengah"
    assert stats["sd"] == "simpangan baku"
    for _suffix, name in RECAP_STATS:
        assert name == name.lower()
        assert any(column.endswith(name) for column in table.columns)


def test_summary_stat_columns_and_display_names_line_up():
    columns = sqi_stat_columns()
    assert columns == [
        f"{key}_{suffix}" for key in SQI_LABELS for suffix, _name in RECAP_STATS
    ]

    display = sqi_stat_display_names()
    assert set(display) == set(columns)
    # Every reader-facing header names the index in Indonesian and spells the
    # statistic out, so no visible column mixes languages.
    for key in SQI_LABELS:
        for suffix, name in RECAP_STATS:
            header = display[f"{key}_{suffix}"]
            assert header.startswith(sqi_label_id(key))
            assert name in header
            assert suffix not in header
    assert not any(
        bad in header
        for header in display.values()
        for bad in ("SUPINE", "SITTING", "STANDING")
    )


def test_summary_stat_display_names_rename_a_real_frame():
    # The mapping has to line up with the columns the analysis actually emits,
    # otherwise the "By position" table would render blank headers.
    frame = pd.DataFrame({column: [0.5] for column in sqi_stat_columns()})
    renamed = frame.rename(columns=sqi_stat_display_names())
    assert list(renamed.columns) == list(sqi_stat_display_names().values())
    assert not renamed.columns.duplicated().any()


def test_recap_table_headers_carry_no_english_stat_words(frames):
    columns = " ".join(activity_recap_table(activity_recap(frames)).columns)
    for word in ("mean", "median", "sd ", "std"):
        assert word not in columns


def test_recap_table_percentages_are_strings_and_indices_are_decimals(frames):
    table = activity_recap_table(activity_recap(frames))
    for value in table["Diterima (%)"]:
        assert value.endswith("%")
    sample = next(c for c in table.columns if c.endswith("· tengah"))
    for value in table[sample]:
        assert isinstance(value, str)
        # Three decimals, so small index differences stay visible.
        assert len(value.split(".")[1]) == 3


def test_recap_table_index_values_match_the_recap(frames):
    recap = activity_recap(frames)
    table = activity_recap_table(recap)
    column = next(c for c in table.columns if c.startswith("Deteksi R-peak") and c.endswith("tengah"))
    for index, row in table.iterrows():
        expected = recap.iloc[index]["qSQI_median"]
        assert float(row[column]) == pytest.approx(expected, abs=5e-4)


def test_recap_table_totals_match_the_frames(frames):
    table = activity_recap_table(activity_recap(frames))
    assert int(table["Total Frame"].sum()) == len(frames)


def test_recap_table_renders_a_single_missing_sd_as_a_dash():
    df = make_frames({"S00": ["Excellent"]})
    table = activity_recap_table(activity_recap(df))
    column = next(
        c for c in table.columns
        if c.startswith("Deteksi R-peak") and c.endswith("simpangan baku")
    )
    # One frame per activity, so every row has an undefined sample SD.
    assert set(table[column]) == {"-"}
    mean_column = next(
        c for c in table.columns
        if c.startswith("Deteksi R-peak") and c.endswith("rata-rata")
    )
    assert set(table[mean_column]) == {"0.900"}


# ------------------------------------------------- recap box plot and text


def test_recap_figure_gives_each_index_its_own_panel(frames):
    figure = activity_recap_figure(frames)
    titles = [a.text for a in figure.layout.annotations]
    assert titles == [
        "Deteksi R-peak", "Distribusi Daya Spektral QRS",
        "Kurtosis Sinyal", "Daya Relatif Baseline",
    ]


def test_recap_figure_draws_every_activity_in_every_panel(frames):
    figure = activity_recap_figure(frames)
    boxes = [t for t in figure.data if t.type == "box"]
    # One box per panel per activity, and each box holds a single activity row.
    assert len(boxes) == 4 * len(POSITIONS)
    for box in boxes:
        assert box.orientation == "h"
        assert len(set(box.y)) == 1
    # Every panel covers all three activities, four boxes at a time.
    activity_sets = [set(b.y[0] for b in boxes[i:i + len(POSITIONS)])
                     for i in range(0, len(boxes), len(POSITIONS))]
    assert activity_sets == [{position_label(p) for p in POSITIONS}] * 4


def test_recap_figure_uses_every_frame_as_an_observation(frames):
    valid = frames[frames["valid"]]
    figure = activity_recap_figure(frames)
    for key in ("qSQI", "pSQI", "kSQI", "basSQI"):
        # Each frame of an activity must appear exactly once in that index's
        # boxes, so the box plot cannot be quietly dropping observations. The
        # panel is identified by its Indonesian name, not the machine key.
        drawn = sum(
            len(t.x) for t in figure.data
            if t.type == "box" and sqi_label_id(key) in str(t.hovertemplate)
        )
        assert drawn == len(valid), key
    for position in POSITIONS:
        drawn = sum(
            len(t.x) for t in figure.data
            if t.type == "box" and set(t.y) == {position_label(position)}
        )
        assert drawn == 4 * int(valid["position"].eq(position).sum())


def test_recap_figure_panels_are_ordered_supine_sitting_standing(frames):
    figure = activity_recap_figure(frames)
    for suffix in ("", "2", "3", "4"):
        axis = figure.layout[f"yaxis{suffix}"]
        assert list(axis.categoryarray) == [position_label(p) for p in reversed(POSITIONS)]


def test_recap_figure_marks_outliers_blue(frames):
    figure = activity_recap_figure(frames)
    dots = [t for t in figure.data if t.name.startswith("Penyimpang")]
    assert dots, "fixture should produce at least one outlying frame"
    for dot in dots:
        assert dot.marker.color == "#1d4ed8"
    # One legend entry only, even though several panels have dots.
    assert sum(1 for t in figure.data if t.name.startswith("Penyimpang") and t.showlegend) == 1


def test_recap_figure_outlier_count_matches_the_fences(frames):
    valid = frames[frames["valid"]]
    figure = activity_recap_figure(frames)
    expected = 0
    for key in ("qSQI", "pSQI", "kSQI", "basSQI"):
        for position in POSITIONS:
            values = valid[valid["position"] == position][key].dropna().tolist()
            expected += len(box_outliers(values)[1])
    drawn = sum(len(t.x) for t in figure.data if t.name.startswith("Penyimpang"))
    assert drawn == expected


def test_recap_figure_names_are_indonesian(frames):
    figure = activity_recap_figure(frames)
    names = " ".join(str(t.name) for t in figure.data)
    assert "Sebaran seluruh frame" in names
    for position in POSITIONS:
        assert position not in names


def test_recap_figure_handles_missing_and_empty_input():
    assert not activity_recap_figure(pd.DataFrame()).data
    one_column = make_frames({"S00": ["Excellent"] * 6})
    one_column = one_column[["subject_id", "position", "quality_class", "valid", "qSQI"]]
    figure = activity_recap_figure(one_column)
    assert [a.text for a in figure.layout.annotations] == ["Deteksi R-peak"]
    assert sum(1 for t in figure.data if t.type == "box") == len(POSITIONS)


def test_recap_figure_uses_light_background(frames):
    assert activity_recap_figure(frames).layout.paper_bgcolor == "#ffffff"


def test_recap_distribution_interpretation_covers_every_index(frames):
    text = activity_recap_distribution_interpretation(frames)
    for key in ("Deteksi R-peak", "Distribusi Daya Spektral QRS",
                "Kurtosis Sinyal", "Daya Relatif Baseline"):
        assert key in text
    assert "IQR" in text


def test_recap_distribution_interpretation_reports_the_widest_spread():
    # One index is spread wide, the rest are flat.
    rows = []
    for subject in range(9):
        for position in POSITIONS:
            for frame in range(4):
                rows.append(
                    {
                        "subject_id": f"S{subject}", "session_id": f"s{subject}",
                        "position": position, "frame_id": f"{subject}-{position}-{frame}",
                        "quality_class": "Excellent", "valid": True,
                        "qSQI": 0.9, "pSQI": 0.5, "kSQI": 4.0, "basSQI": 0.4,
                    }
                )
    df = pd.DataFrame(rows)
    wide = df.copy()
    mask = (wide["position"] == "STANDING") & (wide["frame_id"].str.endswith("0"))
    wide.loc[mask, "qSQI"] = 0.1
    text = activity_recap_distribution_interpretation(wide)
    assert "Deteksi R-peak" in text
    assert "Lebar sebaran antar frame terbesar pada **Deteksi R-peak**" in text


def test_recap_distribution_interpretation_handles_a_tie_across_activities():
    rows = []
    for position in POSITIONS:
        for frame in range(4):
            rows.append(
                {
                    "subject_id": "S00", "session_id": "s",
                    "position": position, "frame_id": f"{position}-{frame}",
                    "quality_class": "Excellent", "valid": True,
                    "qSQI": 0.5, "pSQI": 0.5, "kSQI": 2.0, "basSQI": 0.5,
                }
            )
    text = activity_recap_distribution_interpretation(pd.DataFrame(rows))
    assert "nilainya sama pada ketiga aktivitas" in text


def test_recap_distribution_interpretation_does_not_call_no_outliers_uniform():
    text = activity_recap_distribution_interpretation(make_frames({"S00": ["Excellent"] * 8}))
    assert "tidak otomatis berarti sebarannya sempit" in text


def test_recap_legend_warns_against_comparing_panels():
    text = activity_recap_distribution_legend()
    assert "skalanya sendiri" in text
    assert "tidak boleh dibandingkan langsung" in text
    for position in POSITIONS:
        assert position not in text


def test_recap_legend_states_the_unit_of_observation():
    assert "setiap frame menyumbang satu nilai" in activity_recap_distribution_legend()


def test_recap_table_legend_names_the_acceptance_columns():
    text = activity_recap_legend()
    assert "Total Frame" in text
    assert "Diterima (%)" in text
    assert "ddof=1" in text


def test_recap_interpretation_does_not_pick_a_busiest_activity_on_a_tie(frames):
    text = activity_recap_interpretation(activity_recap(frames))
    assert "Jumlah frame tiap aktivitas sama" in text
    assert "menyumbang frame terbanyak" not in text


def test_recap_interpretation_names_the_busiest_activity_when_unequal():
    rows = []
    for position, count in (("SUPINE", 6), ("SITTING", 2), ("STANDING", 2)):
        for frame in range(count):
            rows.append(
                {
                    "subject_id": "S00", "session_id": "s",
                    "position": position, "frame_id": f"{position}-{frame}",
                    "quality_class": "Excellent", "valid": True,
                    "qSQI": 0.9, "pSQI": 0.5, "kSQI": 4.0, "basSQI": 0.4,
                }
            )
    text = activity_recap_interpretation(activity_recap(pd.DataFrame(rows)))
    assert "Berbaring menyumbang frame terbanyak" in text
    assert "Jumlah frame tiap aktivitas tidak sama" in text


# ------------------------------------------------------------------- figure


def test_quality_figure_is_horizontal_with_positions_on_y(frames):
    figure = quality_by_activity_figure(acceptable_share(frames))
    assert figure.layout.yaxis.title.text == "Aktivitas"
    assert "Persentase" in figure.layout.xaxis.title.text
    for trace in figure.data:
        if trace.type == "box":
            assert trace.orientation == "h"


def test_quality_figure_orders_activities_supine_first(frames):
    figure = quality_by_activity_figure(acceptable_share(frames))
    # Plotly draws the first category at the bottom, so the stored order is the
    # reverse of the reading order Berbaring -> Duduk -> Berdiri.
    assert list(figure.layout.yaxis.categoryarray) == [
        position_label(p) for p in reversed(POSITIONS)
    ]


def test_quality_figure_x_axis_is_a_percentage_scale(frames):
    figure = quality_by_activity_figure(acceptable_share(frames))
    low, high = figure.layout.xaxis.range
    assert low <= 0 <= 100 <= high


def test_quality_figure_draws_outliers_as_separate_blue_dots(frames):
    share = acceptable_share(frames)
    figure = quality_by_activity_figure(share)
    dots = [t for t in figure.data if t.name == "Penyimpang (di luar 1,5 x IQR)"]
    assert len(dots) == 1
    assert dots[0].marker.color == "#1d4ed8"
    expected = []
    for position in POSITIONS:
        values = share[share["position"] == position]["pct_accepted"].tolist()
        expected.extend(box_outliers(values)[1])
    assert sorted(float(v) for v in dots[0].x) == sorted(expected)
    # The fixture is built so exactly one value falls outside its fences.
    assert len(expected) == 1


def test_quality_figure_omits_the_outlier_legend_entry_when_there_are_none(frames):
    """An empty legend entry would imply a category that is not on the chart."""
    share = acceptable_share(frames)
    share.loc[share["position"] == "SITTING", "pct_accepted"] = 80.0
    figure = quality_by_activity_figure(share)
    assert [t.name for t in figure.data if t.name.startswith("Penyimpang")] == []


def test_quality_figure_never_reveals_a_machine_position_name(frames):
    figure = quality_by_activity_figure(acceptable_share(frames))
    rendered = " ".join(
        [str(figure.layout.title.text), str(figure.layout.yaxis.title.text)]
        + [str(t.name) for t in figure.data]
    )
    for position in POSITIONS:
        assert position not in rendered


def test_quality_figure_handles_empty_input():
    figure = quality_by_activity_figure(pd.DataFrame())
    assert not figure.data


def test_quality_figure_handles_a_single_activity():
    share = acceptable_share(make_frames({"S00": ["Excellent"] * 4}))[
        lambda d: d["position"] == "SUPINE"
    ]
    figure = quality_by_activity_figure(share)
    assert [t.name for t in figure.data if t.type == "box"] == ["Berbaring"]


# ------------------------------------------------------ legends and reading


LEGEND_BUILDERS = {
    "acceptance_table": acceptance_table_legend,
    "quality_by_activity": quality_by_activity_legend,
    "quality_distribution": quality_distribution_legend,
    "sqi_box": sqi_box_legend,
    "sqi_trend": sqi_trend_legend,
    "membership": membership_legend,
    "recap": activity_recap_legend,
    "recap_distribution": activity_recap_distribution_legend,
    "heatmap": heatmap_legend,
    "psd": psd_legend,
    "ecg": ecg_legend,
    "multilead": multilead_legend,
    "fuzzy_matrix": fuzzy_matrix_legend,
}


@pytest.mark.parametrize("name", sorted(LEGEND_BUILDERS))
def test_every_legend_starts_in_indonesian(name):
    text = LEGEND_BUILDERS[name]()
    assert text.startswith("**Legenda")
    assert len(text) > 120


@pytest.mark.parametrize("name", sorted(LEGEND_BUILDERS))
def test_every_legend_states_the_scope_limit(name):
    assert "bukan kondisi kesehatan" in LEGEND_BUILDERS[name]()


def test_legends_name_activities_in_indonesian():
    text = quality_by_activity_legend()
    for label in POSITION_LABELS_ID.values():
        assert label in text
    for position in POSITIONS:
        assert position not in text


def test_legend_explains_the_percent_denominator():
    assert "10 detik" in acceptance_table_legend()
    assert "1,5 x IQR" in quality_by_activity_legend()
    assert "biru" in quality_by_activity_legend()


def test_acceptance_interpretation_does_not_name_a_tie_as_high_and_low():
    """Nominally identical activities must not be reported as top and bottom."""
    flat = make_frames({"S00": ["Excellent"] * 8, "S01": ["Excellent"] * 8})
    text = acceptance_table_interpretation(class_table(flat))
    assert "tertinggi" not in text
    assert "terendah" not in text
    assert "terbanyak atau tersedikit" in text
    assert "sama rata" in text


def test_quality_distribution_interpretation_does_not_name_a_tie_as_high_and_low():
    flat = make_frames({"S00": ["Excellent"] * 8, "S01": ["Excellent"] * 8})
    text = quality_distribution_interpretation(flat)
    assert "tertinggi" not in text
    assert "terendah" not in text
    assert "sama" in text
    # A single activity may only be named once per clause.
    assert text.count("**Berbaring**") <= 1


def test_quality_distribution_interpretation_names_the_real_extremes():
    df = make_frames({"S00": ["Excellent"] * 6, "S01": ["Unacceptable"] * 6})
    # Make one activity stand out on the accepted side.
    df.loc[
        (df["subject_id"] == "S01") & (df["position"] == "SUPINE"), "quality_class"
    ] = "Excellent"
    text = quality_distribution_interpretation(df)
    assert text.count("tertinggi") == 2
    # The comparison is stated as two separate maxima, never as a top and bottom.
    assert "terendah" not in text
    assert "tidak ada aktivitas yang menonjol" not in text


def test_acceptance_interpretation_reports_top_and_bottom_when_they_differ(frames):
    table = class_table(frames)
    best = table.loc[table["pct_accepted"].idxmax()]
    worst = table.loc[table["pct_accepted"].idxmin()]
    assert float(best["pct_accepted"]) != float(worst["pct_accepted"])
    text = acceptance_table_interpretation(table)
    assert f"**{position_label(best['position'])}** memiliki persentase frame diterima tertinggi" in text
    assert f"**{position_label(worst['position'])}** terendah" in text


def test_acceptance_interpretation_still_reports_wide_spread_when_tied_on_bad():
    """A tie on acceptance must not suppress the Unacceptable comparison."""
    rows = []
    for subject, standing in (("S00", "Excellent"), ("S01", "Unacceptable")):
        for position in POSITIONS:
            for index in range(4):
                rows.append(
                    {
                        "subject_id": subject, "session_id": f"s-{subject}",
                        "position": position, "frame_id": f"{subject}-{position}-{index}",
                        "quality_class": standing if position == "STANDING" else "Excellent",
                        "valid": True,
                    }
                )
    text = acceptance_table_interpretation(class_table(pd.DataFrame(rows)))
    assert "**Berdiri** menyumbang proporsi *Tidak Diterima* tertinggi" in text


def test_acceptance_interpretation_quotes_its_own_numbers(frames):
    table = class_table(frames)
    text = acceptance_table_interpretation(table)
    best = table.loc[table["pct_accepted"].idxmax()]
    worst = table.loc[table["pct_accepted"].idxmin()]
    assert position_label(best["position"]) in text
    assert position_label(worst["position"]) in text
    assert f"{best['pct_accepted']:.1f}%" in text


def test_acceptance_interpretation_does_not_claim_quality_when_all_are_bad():
    all_bad = make_frames(
        {f"S{i:02d}": ["Unacceptable"] * 6 for i in range(3)}
    )
    table = class_table(all_bad)
    text = acceptance_table_interpretation(table)
    assert text.count("0.0%") >= 2
    # A zero spread means the numbers match, never that the recordings are good.
    assert "memenuhi syarat" not in text


def test_acceptance_interpretation_mentions_wide_and_narrow_spreads():
    # Wide: one subject records nothing acceptable while standing.
    wide_rows = []
    for subject, standing in (("S00", "Excellent"), ("S01", "Unacceptable")):
        for position in POSITIONS:
            for index in range(4):
                wide_rows.append(
                    {
                        "subject_id": subject, "session_id": f"s-{subject}",
                        "position": position, "frame_id": f"{subject}-{position}-{index}",
                        "quality_class": standing if position == "STANDING" else "Excellent",
                        "valid": True,
                    }
                )
    wide_text = acceptance_table_interpretation(class_table(pd.DataFrame(wide_rows)))
    assert "berbeda cukup jelas" in wide_text

    # Narrow: everybody acceptable everywhere, so the percentages merely match.
    narrow = make_frames({"S00": ["Excellent"] * 6, "S01": ["Excellent"] * 6})
    narrow_text = acceptance_table_interpretation(class_table(narrow))
    assert "relatif sama" in narrow_text
    assert "berbeda cukup jelas" not in narrow_text


def test_quality_interpretation_lists_every_activity(frames):
    text = quality_by_activity_interpretation(acceptable_share(frames))
    for label in POSITION_LABELS_ID.values():
        assert f"**{label}**" in text
    assert "penyimpang" in text.lower()


def test_quality_interpretation_names_the_outlier_count(frames):
    share = acceptable_share(frames)
    total = sum(
        len(box_outliers(share[share["position"] == p]["pct_accepted"].tolist())[1])
        for p in POSITIONS
    )
    assert total == 1
    text = quality_by_activity_interpretation(share)
    assert f"**{total}**" in text
    assert "tidak otomatis berarti sebarannya sempit" not in text


def test_quality_interpretation_reports_no_count_when_nothing_is_flagged(frames):
    share = acceptable_share(frames)
    share.loc[share["position"] == "SITTING", "pct_accepted"] = 80.0
    text = quality_by_activity_interpretation(share)
    assert "Tidak ada peserta yang berada di luar pagar" in text


def test_quality_interpretation_does_not_call_a_spread_uniform():
    """No outliers does not mean the values are tightly clustered."""
    wide_without_outliers = pd.DataFrame(
        {
            "subject_id": [f"S{i:02d}" for i in range(10)],
            "position": ["SUPINE"] * 10,
            "n_frames": [10] * 10,
            "n_accepted": [10, 9, 8, 10, 7, 9, 8, 10, 6, 9],
            "pct_accepted": [100.0, 90.0, 80.0, 100.0, 70.0, 90.0, 80.0, 100.0, 60.0, 90.0],
        }
    )
    text = quality_by_activity_interpretation(wide_without_outliers)
    assert "tidak otomatis berarti sebarannya sempit" in text


def test_quality_interpretation_describes_the_widest_spread():
    share = pd.DataFrame(
        {
            "subject_id": ["S00", "S01", "S02", "S00", "S01", "S02"],
            "position": ["SUPINE"] * 3 + ["SITTING"] * 3,
            "n_frames": [10] * 6,
            "n_accepted": [10, 9, 8, 10, 10, 10],
            "pct_accepted": [100.0, 90.0, 80.0, 100.0, 100.0, 100.0],
        }
    )
    text = quality_by_activity_interpretation(share)
    assert "**Berbaring** menunjukkan sebaran antar peserta paling lebar" in text
    assert "**Duduk** paling rapat" in text


def test_psd_interpretation_reports_the_band_share():
    freqs = np.linspace(0.5, 40.0, 400)
    psd = np.exp(-((freqs - 10.0) ** 2) / 2.0) + 1e-6
    text = psd_interpretation(freqs, psd, (5.0, 15.0), (0.5, 40.0))
    assert "5–15 Hz" in text
    assert "10.0 Hz" in text
    assert "%" in text


def test_membership_interpretation_names_the_top_class():
    text = membership_interpretation(
        {"Excellent": 0.61, "Barely Acceptable": 0.3, "Unacceptable": 0.09}
    )
    assert "Sangat Baik" in text
    assert "0.610" in text


def test_fuzzy_matrix_interpretation_covers_every_factor():
    text = fuzzy_matrix_interpretation(
        {"qSQI": {"Excellent": 0.8, "Unacceptable": 0.2},
         "pSQI": {"Excellent": 0.2, "Unacceptable": 0.8}}
    )
    assert "Deteksi R-peak" in text
    assert "Distribusi Daya Spektral QRS" in text
    assert "Tidak Diterima" in text


def test_heatmap_interpretation_finds_extremes():
    # S00/Berbaring = 0.20 is the minimum, S00/Duduk = 0.90 is the maximum.
    matrix = pd.DataFrame(
        [[0.2, 0.9], [0.5, 0.6]],
        index=["S00", "S01"],
        columns=["SUPINE", "SITTING"],
    )
    text = heatmap_interpretation(matrix)
    assert "tertinggi adalah **S00 pada Duduk** (0.900)" in text
    assert "terendah adalah **S00 pada Berbaring** (0.200)" in text


def test_heatmap_interpretation_skips_missing_combinations():
    matrix = pd.DataFrame(
        [[np.nan], [0.4]],
        index=["S00", "S01"],
        columns=["SUPINE"],
    )
    text = heatmap_interpretation(matrix)
    assert "S00" not in text
    assert "S01 pada Berbaring" in text


def test_sqi_box_and_trend_interpretations_use_display_names(frames):
    box_text = sqi_box_interpretation(frames)
    trend_text = sqi_trend_interpretation(frames)
    for text in (box_text, trend_text):
        assert "Deteksi R-peak" in text
        assert "qSQI" not in text
    assert "median tertinggi" in box_text
    assert "trennya" in trend_text


def test_quality_distribution_interpretation_uses_the_frame_counts(frames):
    text = quality_distribution_interpretation(frames)
    assert position_label(class_table(frames).loc[0, "position"]) in text


EMPTY_INTERPRETERS = {
    "acceptance": lambda: acceptance_table_interpretation(pd.DataFrame()),
    "recap": lambda: activity_recap_interpretation(pd.DataFrame()),
    "recap_distribution": lambda: activity_recap_distribution_interpretation(pd.DataFrame()),
    "quality_by_activity": lambda: quality_by_activity_interpretation(pd.DataFrame()),
    "quality_distribution": lambda: quality_distribution_interpretation(pd.DataFrame()),
    "sqi_box": lambda: sqi_box_interpretation(pd.DataFrame()),
    "sqi_trend": lambda: sqi_trend_interpretation(pd.DataFrame()),
    "membership": lambda: membership_interpretation({}),
    "fuzzy_matrix": lambda: fuzzy_matrix_interpretation({}),
    "psd": lambda: psd_interpretation([], []),
    "heatmap": lambda: heatmap_interpretation(pd.DataFrame()),
}


@pytest.mark.parametrize("name", sorted(EMPTY_INTERPRETERS))
def test_every_interpreter_survives_empty_input(name):
    text = EMPTY_INTERPRETERS[name]()
    assert "Belum ada data" in text


@pytest.mark.parametrize("name", sorted(EMPTY_INTERPRETERS))
def test_no_interpreter_makes_a_claim_on_empty_input(name):
    text = EMPTY_INTERPRETERS[name]().lower()
    for forbidden in ("tertinggi", "terendah", "lebih baik", "lebih buruk"):
        assert forbidden not in text


def test_interpretations_never_claim_a_health_outcome():
    builders = [
        acceptance_table_interpretation(class_table(make_frames(
            {"S00": ["Excellent"] * 6, "S01": ["Unacceptable"] * 6}))),
        activity_recap_interpretation(activity_recap(make_frames(
            {"S00": ["Excellent"] * 6, "S01": ["Unacceptable"] * 6}))),
        activity_recap_distribution_interpretation(make_frames(
            {"S00": ["Excellent"] * 6, "S01": ["Unacceptable"] * 6})),
        quality_by_activity_interpretation(acceptable_share(make_frames(
            {"S00": ["Excellent"] * 6, "S01": ["Unacceptable"] * 6}))),
        sqi_box_interpretation(make_frames(
            {"S00": ["Excellent"] * 6, "S01": ["Unacceptable"] * 6})),
        sqi_trend_interpretation(make_frames(
            {"S00": ["Excellent"] * 6, "S01": ["Unacceptable"] * 6})),
    ]
    forbidden = (
        "diagnos", "penyakit", "kardiovaskular", "aritmia", "arrhythmia",
        "pasien sehat", "tidak sehat", "sehat", "sakit",
    )
    for text in builders:
        lowered = text.lower()
        for word in forbidden:
            assert word not in lowered, (word, text)