"""Tests for JSONL Generation: schema, builder and writer.

Two things are pinned here because they are easy to get subtly wrong:

* The schema must stay a transcription of ``templates/json-web.jsonl``, not an
  independent design. ``test_schema_matches_the_template`` fails the moment they
  diverge.
* A generated dataset must read back through the existing JSONL reader to a
  bit-identical signal, so a reconstruction can replace a recorded file without
  the analysis pipeline noticing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ecg_eval.ingestion import read_jsonl_file
from ecg_eval.jsonl import (
    ECG_FORMAT,
    OPTIONAL_FIELDS,
    REQUIRED_FIELDS,
    TEMPLATE_FIELDS,
    UNAVAILABLE_IN_RAW,
    RecordValidation,
    assert_output_outside_source,
    build_record,
    expand_elisions,
    load_template,
    template_fields,
    validate_record,
    write_jsonl,
)
from ecg_eval.models import SOURCE_RAW_RECONSTRUCTED, ECGFrame

N_SAMPLES = 250


def _frame(**overrides) -> ECGFrame:
    time = np.arange(N_SAMPLES, dtype=np.float32) / 250.0
    signal = np.column_stack([np.sin(time), np.cos(time), np.sin(2 * time)]).astype(np.float32)
    payload = {
        "subject_id": "S01",
        "session_id": "ses000000000005",
        "frame_id": "000001",
        "source_file": "raw/calibrated/frame_000001_mv.npy",
        "source_format": SOURCE_RAW_RECONSTRUCTED,
        "timestamp": "2026-09-28T20:05:52+07:00",
        "sampling_rate": 250.0,
        "duration_s": 1.0,
        "signal": signal,
        "metadata": {"validation": {"status": "PASS", "warnings": []}},
        "provenance": {
            "source_type": "raspberry_pi_raw",
            "device_id": "device01",
            "signal_source": "calibrated",
            "mapping_status": "verified",
            "raw_root": "raw",
        },
    }
    payload.update(overrides)
    return ECGFrame(**payload)


def _frames(n: int = 2) -> list[ECGFrame]:
    return [_frame(frame_id=f"{i:06d}") for i in range(1, n + 1)]


# -- template and schema ------------------------------------------------


def test_template_is_not_plain_json() -> None:
    """The template elides its sample list, so it needs expanding to parse."""
    text = (Path(TEMPLATE_FIELDS and "templates/json-web.jsonl")).read_text(encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        json.loads(text.strip())
    assert json.loads(expand_elisions(text.strip()))


def test_expand_elisions_infers_the_row_width() -> None:
    expanded = expand_elisions('{"ecg":{"samples":[[1.0,2.0,3.0],[ ... ]]}}')

    assert json.loads(expanded)["ecg"]["samples"] == [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]


def test_expand_elisions_leaves_clean_json_alone() -> None:
    assert expand_elisions('{"a":[1,2]}') == '{"a":[1,2]}'


def test_schema_matches_the_template() -> None:
    """IDEA-REVISED.md section 7: the template is the source of truth."""
    assert template_fields() == TEMPLATE_FIELDS


def test_template_declares_every_known_field() -> None:
    template = load_template()

    for name in REQUIRED_FIELDS + OPTIONAL_FIELDS:
        assert name in template


def test_template_ecg_format_is_the_one_we_emit() -> None:
    assert load_template()["ecg"]["format"] == ECG_FORMAT


# -- record validation --------------------------------------------------


def test_a_good_record_validates() -> None:
    assert validate_record(build_record(_frame()).record).status == "VALID"


def test_missing_required_fields_are_errors() -> None:
    record = build_record(_frame()).record
    del record["ecg"]
    del record["frame_id"]

    result = validate_record(record)

    assert result.status == "ERROR"
    assert {issue.field for issue in result.errors} == {"ecg", "frame_id"}


def test_unavailable_optional_fields_are_reported_not_errors() -> None:
    record = build_record(_frame()).record

    result = validate_record(record)

    assert result.ok
    assert set(result.unavailable_fields) >= set(UNAVAILABLE_IN_RAW)


def test_a_non_numeric_frame_id_is_an_error() -> None:
    record = build_record(_frame()).record
    record["frame_id"] = "frame-1"

    result = validate_record(record)

    assert any("zero-padded numeric string" in issue.message for issue in result.errors)


def test_a_non_positive_sampling_rate_is_an_error() -> None:
    record = build_record(_frame()).record
    record["sampling_rate_hz"] = 0

    assert validate_record(record).status == "ERROR"


def test_ragged_sample_rows_are_an_error() -> None:
    record = build_record(_frame()).record
    record["ecg"]["samples"] = [[1.0, 2.0, 3.0], [1.0, 2.0]]

    result = validate_record(record)

    assert any("inconsistent channel counts" in issue.message for issue in result.errors)


def test_a_sample_count_mismatch_is_an_error() -> None:
    record = build_record(_frame()).record

    result = validate_record(record, expected_samples=N_SAMPLES + 1)

    assert any("implied by" in issue.message for issue in result.errors)


def test_fields_outside_the_schema_are_a_warning_not_an_error() -> None:
    record = build_record(_frame()).record
    record["surprise"] = 1

    result = validate_record(record)

    assert result.status == "WARNING"
    assert result.ok


def test_an_unknown_validation_status_is_a_warning() -> None:
    record = build_record(_frame()).record
    record["validation"]["status"] = "MAYBE"

    result = validate_record(record)

    assert result.status == "WARNING"
    assert any(issue.field == "validation.status" for issue in result.warnings)


def test_validation_finalise_prioritises_errors() -> None:
    result = RecordValidation(frame_id="000001")
    result.warn("a", "w")
    assert result.finalise().status == "WARNING"
    result.error("b", "e")
    assert result.finalise().status == "ERROR"


# -- builder ------------------------------------------------------------


def test_record_matches_the_template_shape() -> None:
    record = build_record(_frame()).record

    assert set(record) <= set(TEMPLATE_FIELDS)
    assert set(record) >= set(REQUIRED_FIELDS)


def test_message_id_follows_the_device_convention() -> None:
    record = build_record(_frame()).record

    assert record["message_id"] == "device01-ses000000000005-frame_000001"


def test_device_id_falls_back_when_unrecorded() -> None:
    frame = _frame(provenance={"source_type": "raspberry_pi_raw", "signal_source": "calibrated"})

    assert build_record(frame).record["device_id"] == "unknown-device"


def test_samples_are_samples_by_time() -> None:
    record = build_record(_frame()).record

    assert record["ecg"]["format"] == ECG_FORMAT
    assert len(record["ecg"]["samples"]) == N_SAMPLES
    assert all(len(row) == 3 for row in record["ecg"]["samples"])


def test_sample_precision_is_not_rounded() -> None:
    """Matching the device's full repr keeps generated and recorded files alike."""
    stored = np.float32(1.3148092538833618)
    frame = _frame(signal=np.full((4, 1), stored, dtype=np.float32))

    value = build_record(frame).record["ecg"]["samples"][0][0]

    assert value == float(stored)
    assert len(str(value)) > 10


def test_validation_block_carries_the_device_verdict() -> None:
    frame = _frame(
        metadata={
            "validation": {
                "status": "WARNING",
                "warnings": ["Large DC baseline on Lead Ii: +9.07 mV."],
                "failures": ["baseline drift above limit"],
            }
        }
    )

    block = build_record(frame).record["validation"]

    assert block["status"] == "WARNING"
    assert block["warnings"] == ["Large DC baseline on Lead Ii: +9.07 mV."]
    assert block["failures"] == ["baseline drift above limit"]


def test_empty_failures_are_omitted_rather_than_emitted_empty() -> None:
    frame = _frame(
        metadata={"validation": {"status": "PASS", "warnings": [], "failures": []}}
    )

    block = build_record(frame).record["validation"]

    assert block == {"status": "PASS", "warnings": []}


def test_signal_quality_is_accepted_as_the_validation_source() -> None:
    """Some firmware writes signal_quality instead of validation."""
    frame = _frame(
        metadata={
            "validation": None,
            "signal_quality": {"status": "WARNING", "reasons": ["Large DC baseline"]},
        }
    )

    block = build_record(frame).record["validation"]

    assert block["status"] == "WARNING"
    assert block["warnings"] == ["Large DC baseline"]


def test_a_frame_without_any_verdict_defaults_to_warning() -> None:
    frame = _frame(metadata={})

    assert build_record(frame).record["validation"]["status"] == "WARNING"


def test_prediction_is_included_when_linked() -> None:
    frame = _frame(
        metadata={
            "validation": {"status": "PASS", "warnings": []},
            "prediction": {
                "status": "PASS",
                "label": "Normal",
                "confidence_percent": 99.67,
                "probabilities": {"Normal": 99.67, "AF": 0.4},
                "threshold": 0.5,
                "latency_ms": 251.73,
                "runtime": "ai-edge-litert",
            },
        }
    )

    block = build_record(frame).record["prediction"]

    assert block["label"] == "Normal"
    assert block["confidence_percent"] == 99.67
    assert block["runtime"] == "ai-edge-litert"


def test_prediction_is_omitted_when_absent() -> None:
    built = build_record(_frame())

    assert "prediction" not in built.record
    assert "prediction" in built.unavailable_fields


def test_unavailable_fields_cover_what_the_raw_tree_lacks() -> None:
    built = build_record(_frame())

    for name in UNAVAILABLE_IN_RAW:
        assert name in built.unavailable_fields
        assert name not in built.record


def test_origins_name_the_source_of_each_field() -> None:
    origins = build_record(_frame()).origins

    assert origins["ecg"] == "calibrated"
    assert origins["validation"] == "model_ready validation/signal_quality"
    assert origins["message_id"] == "derived"


def test_a_missing_timestamp_is_reported() -> None:
    built = build_record(_frame(timestamp=None))

    assert "created_at" in built.unavailable_fields


# -- writer -------------------------------------------------------------


def test_write_jsonl_emits_one_line_per_frame(tmp_path: Path) -> None:
    result = write_jsonl(_frames(3), tmp_path)

    assert result.status == "VALID"
    assert result.written == 3
    lines = Path(result.output_path).read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3


def test_write_jsonl_round_trips_bit_exactly(tmp_path: Path) -> None:
    frames = _frames(1)

    result = write_jsonl(frames, tmp_path)
    back = read_jsonl_file(result.output_path)

    assert back.status == "OK"
    assert len(back.frames) == 1
    assert np.array_equal(frames[0].signal, back.frames[0].signal)
    assert back.frames[0].sampling_rate == 250.0
    assert back.frames[0].frame_id == "000001"


def test_write_jsonl_writes_a_manifest(tmp_path: Path) -> None:
    result = write_jsonl(_frames(2), tmp_path)

    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))

    assert manifest["schema"] == "templates/json-web.jsonl"
    assert manifest["frames_written"] == 2
    assert len(manifest["frames"]) == 2
    assert "ecg" in manifest["field_origins"]
    assert set(UNAVAILABLE_IN_RAW) <= set(manifest["unavailable_fields"])


def test_write_jsonl_can_skip_the_manifest(tmp_path: Path) -> None:
    result = write_jsonl(_frames(1), tmp_path, write_manifest=False)

    assert result.manifest_path == ""
    assert not list(tmp_path.glob("*.manifest.json"))


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    result = write_jsonl(_frames(2), tmp_path, dry_run=True)

    assert result.written == 2
    assert result.dry_run is True
    assert not list(tmp_path.glob("*.jsonl"))


def test_dry_run_still_exposes_the_payload_it_would_write(tmp_path: Path) -> None:
    """A dry run has no file on disk, so it must still be readable.

    The UI reads the result to offer a download; reading a path that a dry
    run never created crashed the Reconstruction page.
    """
    result = write_jsonl(_frames(2), tmp_path, dry_run=True)

    assert not Path(result.output_path).exists()
    payload = result.preview_text()
    assert payload.count(chr(10)) == 2
    assert json.loads(payload.splitlines()[0])["frame_id"]


def test_dry_run_payload_matches_the_written_file(tmp_path: Path) -> None:
    """Preview and real output must be byte-identical, not two code paths."""
    dry = write_jsonl(_frames(2), tmp_path / "a", dry_run=True)
    real = write_jsonl(_frames(2), tmp_path / "b")

    assert dry.payload == Path(real.output_path).read_text(encoding="utf-8")


def test_filename_defaults_to_subject_and_session(tmp_path: Path) -> None:
    result = write_jsonl(_frames(1), tmp_path)

    assert Path(result.output_path).name == "S01_ses000000000005.jsonl"


def test_filename_can_be_overridden(tmp_path: Path) -> None:
    result = write_jsonl(_frames(1), tmp_path, filename="custom.jsonl")

    assert Path(result.output_path).name == "custom.jsonl"


def test_writing_into_the_source_recording_folder_is_refused(tmp_path: Path) -> None:
    source = tmp_path / "Bryan"
    source.mkdir()

    with pytest.raises(ValueError, match="source tree must stay read-only"):
        write_jsonl(_frames(1), source / "gen", source_root=source)


def test_the_refusal_check_accepts_a_sibling_directory(tmp_path: Path) -> None:
    source = tmp_path / "Bryan"
    source.mkdir()
    (tmp_path / "gen").mkdir()

    assert_output_outside_source(tmp_path / "gen", source)


def test_no_partial_file_is_left_behind(tmp_path: Path) -> None:
    result = write_jsonl(_frames(2), tmp_path)

    assert not list(tmp_path.glob("*.partial"))


def test_an_invalid_record_is_skipped_not_written(tmp_path: Path) -> None:
    frame = _frame(frame_id="not-numeric")

    result = write_jsonl([frame], tmp_path)

    assert result.written == 0
    assert result.status == "ERROR"
    assert result.skipped[0]["frame_id"] == "not-numeric"


def test_an_invalid_record_can_be_forced_through(tmp_path: Path) -> None:
    frame = _frame(frame_id="not-numeric")

    result = write_jsonl([frame], tmp_path, allow_invalid=True)

    assert result.status == "WARNING"
    assert result.written == 1


def test_an_empty_frame_list_writes_an_empty_file(tmp_path: Path) -> None:
    result = write_jsonl([], tmp_path)

    assert result.written == 0
    assert Path(result.output_path).read_text(encoding="utf-8") == ""