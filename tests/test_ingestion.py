"""Ingestion tests: discovery, parsing, validation and inventory."""

from __future__ import annotations

import json

import numpy as np
import pytest

from ecg_eval.ingestion import (
    annotation_coverage,
    build_inventory,
    ingest,
    read_jsonl_file,
    validate_frames,
    validate_sampling_rates,
)
from ecg_eval.models.annotation import SUPINE, Segment, SubjectAnnotation


def test_reads_valid_frames(jsonl_file):
    result = read_jsonl_file(jsonl_file, date_folder="29-09-2026")
    assert len(result.frames) == 2
    assert result.status == "MALFORMED"  # one bad line was found, not hidden
    assert result.frames[0].n_samples == 2500
    assert result.frames[0].n_channels == 3
    assert result.frames[0].sampling_rate == 250.0
    assert result.frames[0].frame_id == "000001"


def test_malformed_line_is_reported_not_dropped(jsonl_file):
    result = read_jsonl_file(jsonl_file)
    assert len(result.malformed) == 1
    assert result.malformed[0].line_number == 2
    assert "invalid JSON" in result.malformed[0].reason
    # The valid frames around it survive.
    assert len(result.frames) == 2


def test_frames_are_sorted_and_indexed(jsonl_file):
    result = read_jsonl_file(jsonl_file)
    assert [f.frame_id for f in result.frames] == ["000001", "000002"]
    assert [f.record_index for f in result.frames] == [0, 1]


def test_canonical_frame_shape_and_lead(make_frame):
    frame = make_frame()
    assert frame.signal_shape == (2500, 3)
    assert frame.lead("Lead II").shape == (2500,)
    assert frame.channel_order() == ("Lead I", "Lead II", "Lead III")
    assert frame.duration_actual_s == pytest.approx(10.0)


def test_content_hash_changes_with_signal(make_frame, clean_signal):
    a = make_frame(clean_signal)
    b = make_frame(clean_signal + 0.5)
    assert a.content_hash() != b.content_hash()
    assert a.content_hash() == make_frame(clean_signal).content_hash()


def test_nan_detected_as_signal_issue(make_frame, clean_signal):
    corrupt = clean_signal.copy()
    corrupt[100] = np.nan
    frame = make_frame(corrupt)
    assert any("NaN" in issue for issue in frame.signal_issues())


def test_flat_line_detected(make_frame):
    frame = make_frame(np.zeros(2500, dtype=np.float32))
    assert any("flat line" in issue for issue in frame.signal_issues())


def test_missing_ecg_object_is_malformed(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"message_id": "x", "session_id": "s", "frame_id": "1"}) + "\n")
    result = read_jsonl_file(path)
    assert result.frames == []
    assert "missing 'ecg' object" in result.malformed[0].reason


def test_unsupported_ecg_format_is_malformed(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps(
            {"session_id": "s", "frame_id": "1",
             "ecg": {"format": "volt", "samples": [[1.0]]}}
        ) + "\n"
    )
    result = read_jsonl_file(path)
    assert "unsupported ecg.format" in result.malformed[0].reason


def test_sample_count_mismatch_recorded(jsonl_file):
    result = read_jsonl_file(jsonl_file)
    # 2500 samples at 250 Hz for 10 s is consistent, so no mismatch is flagged.
    assert "sample_count_mismatch" not in result.frames[0].metadata


def test_duplicate_frame_id_warns(jsonl_file):
    lines = jsonl_file.read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    jsonl_file.write_text("\n".join([json.dumps(record), json.dumps(record)]) + "\n", encoding="utf-8")
    result = read_jsonl_file(jsonl_file)
    assert any("duplicate frame_id" in w for w in result.warnings)


def test_inventory_reports_coverage(jsonl_file):
    result = read_jsonl_file(jsonl_file)
    inventory = build_inventory(
        {("S01", "ses1"): result.frames}, result.malformed,
        expected_subjects=12, expected_total_frames=216,
    )
    coverage = inventory.coverage()
    assert coverage["expected_frames"] == 216
    assert coverage["actual_frames"] == 2
    assert coverage["missing_frames"] == 214
    assert coverage["actual_subjects"] == 1
    assert coverage["missing_subjects"] == 11
    assert coverage["malformed_records"] == 1


def test_validate_sampling_rates_flags_inconsistency(make_frame):
    a = make_frame(sampling_rate=250.0, frame_id="1")
    b = make_frame(sampling_rate=500.0, frame_id="2")
    problems = validate_sampling_rates([a, b])
    assert problems and "inconsistent sampling" in problems[0]


def test_validate_frames_reports_amplitude_limit(make_frame):
    loud = make_frame(np.full(2500, 5000.0, dtype=np.float32) + np.arange(2500))
    issues = validate_frames([loud], max_abs_limit_mv=1000.0)
    assert any("exceeds limit" in problem for problems in issues.values() for problem in problems)


def test_annotation_coverage_counts(make_frame):
    frames = [make_frame(frame_id=f"{i:06d}") for i in range(6)]
    annotation = SubjectAnnotation(
        subject_id="S01", session_id="ses_test",
        segments=[
            Segment(SUPINE, 0, 1),
            Segment("TRANSITION", 2, 2),
        ],
    )
    coverage = annotation_coverage({("S01", "ses_test"): frames}, [annotation])
    assert coverage["annotated_frames"] == 3
    assert coverage["analyzable_frames"] == 2      # transitions excluded
    assert coverage["excluded_transitions"] == 1
    assert coverage["unlabeled_frames"] == 3


def test_ingest_real_dataset():
    """Ingest the repository's actual data directory."""
    dataset = ingest()
    assert dataset.n_frames > 0
    formats = dataset.source_formats()
    assert "jsonl" in formats
    # The raw MQTT-mistake recording is picked up as well.
    assert "raw_calibrated" in formats
    assert dataset.sampling_rates() == [250.0]
