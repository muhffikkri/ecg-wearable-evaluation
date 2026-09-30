"""Tests for the per-folder Raspberry Pi readers.

The fixtures mirror the real 29-09-2026 session, including the two details that
make the naive mapping wrong:

* ``model_ready`` records ``source_file`` pointing at ``filtered/``, so the link
  between the two folders is *recorded*, not assumed.
* ``predictions`` writes a bare ``"000001"`` frame id while the other folders
  write ``"frame_000001"``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ecg_eval.ingestion import (
    collect_frame_files,
    discover_raw_roots,
    parse_frame_id,
    parse_frame_number,
    read_calibrated,
    read_logs,
    read_model_ready,
    read_predictions,
    read_raw_dataset,
    read_session_metadata,
    recorded_source_number,
)
from ecg_eval.models.source_frame import (
    FOLDER_CALIBRATED,
    FOLDER_FILTERED,
    FOLDER_MODEL_READY,
    FOLDER_PREDICTIONS,
)

N_SAMPLES = 2500
N_FRAMES = 3


def _calibrated_sidecar(number: int, *, session_id: str = "session_28092026_202645") -> dict:
    return {
        "channel_order": ["Lead I", "Lead II", "Lead III"],
        "created_at_utc": f"2026-09-28T13:27:0{number}.844340Z",
        "dtype": "float32",
        "duration_seconds": 10.0,
        "sample_count": N_SAMPLES,
        "sample_rate_hz": 250,
        "shape": [N_SAMPLES, 3],
        "source_frame": f"frame_{number:06d}",
        "unit": "mV",
        "calibration": {"method": "multipoint_rpeak", "scale_mV_per_count": 0.000116913},
        "lead_mapping": {"lead_i_mV": "Lead I = LA - RA"},
        "source_metadata": {
            "device_id": "device01",
            "session_id": session_id,
            "frame_index": number,
            "sha256_checksum": "e" * 64,
            "state": "testing",
        },
    }


def _model_ready_sidecar(number: int, *, points_at: int | None = None) -> dict:
    """Mirror the real JSON, whose ``source_file`` points into ``filtered/``."""
    target = number if points_at is None else points_at
    return {
        "calibration_method": "multipoint_rpeak",
        "channel_count": 3,
        "channel_order": ["Lead I", "Lead II", "Lead III"],
        "created_at": f"2026-09-28T13:27:0{target}.053234Z",
        "dtype": "float32",
        "duration_seconds": 10.0,
        "frame_id": f"frame_{target:06d}",
        "memory_layout": "samples x channels",
        "sample_rate_hz": 250,
        "shape": [N_SAMPLES, 3],
        "unit": "mV",
        "preprocessing_config": None,
        "preprocessing_steps_applied": [],
        "source_file": (
            f"/home/pi/sessions/session_28092026_202645/filtered/frame_{target:06d}_mv.npy"
        ),
        "source_metadata_file": (
            f"/home/pi/sessions/session_28092026_202645/filtered/frame_{target:06d}_mv.json"
        ),
        "source_processing": {
            "filters": [
                {
                    "frequency_hz": 50.0,
                    "name": "notch_50hz",
                    "phase": "zero_phase_forward_backward",
                    "quality_factor": 30.0,
                }
            ]
        },
        "validation": {"status": "WARNING", "warnings": ["Large DC baseline on Lead Ii"]},
    }


def _prediction_sidecar(number: int) -> dict:
    return {
        "schema_version": 1,
        "status": "PASS",
        "frame_id": f"{number:06d}",
        "source_file": (
            f"/home/pi/sessions/session_28092026_202645/model_ready/frame_{number:06d}_input.npy"
        ),
        "source_sha256": "a" * 64,
        "created_at": "2026-09-28T13:27:04.898854Z",
        "sampling_rate_hz": 250.0,
        "shape": [N_SAMPLES, 3],
        "dtype": "float32",
        "unit": "mV",
        "prediction": "Normal",
        "confidence_percent": 99.73,
        "probabilities": {"Normal": 99.73, "AF": 0.15},
        "threshold": 0.5,
        "latency_ms": 251.26,
        "runtime": "ai-edge-litert",
    }


def _signal(offset: float = 0.0) -> np.ndarray:
    time = np.arange(N_SAMPLES, dtype=np.float32) / 250.0
    return np.column_stack([np.sin(2 * np.pi * 1.2 * time) + offset] * 3).astype(np.float32)


@pytest.fixture
def raw_tree(tmp_path: Path) -> Path:
    """A recording folder laid out exactly like the real 29-09-2026 session."""
    root = tmp_path / "29-09-2026" / "Bryan"
    for folder in (FOLDER_CALIBRATED, FOLDER_FILTERED, FOLDER_MODEL_READY, FOLDER_PREDICTIONS, "logs"):
        (root / folder).mkdir(parents=True)

    for number in range(1, N_FRAMES + 1):
        # calibrated: gain applied, so it differs from filtered.
        base = _signal(offset=9.0)
        np.save(root / FOLDER_CALIBRATED / f"frame_{number:06d}_mv.npy", base)
        (root / FOLDER_CALIBRATED / f"frame_{number:06d}_mv.json").write_text(
            json.dumps(_calibrated_sidecar(number)), encoding="utf-8"
        )
        (root / FOLDER_CALIBRATED / f"frame_{number:06d}_mv.csv").write_text(
            "lead_i,lead_ii,lead_iii\n", encoding="utf-8"
        )

        # filtered: what model_ready copied, and what the mapper must link to.
        filtered = _signal()
        np.save(root / FOLDER_FILTERED / f"frame_{number:06d}_mv.npy", filtered)
        (root / FOLDER_FILTERED / f"frame_{number:06d}_mv.json").write_text(
            json.dumps(_calibrated_sidecar(number)), encoding="utf-8"
        )

        np.save(root / FOLDER_MODEL_READY / f"frame_{number:06d}_input.npy", filtered)
        (root / FOLDER_MODEL_READY / f"frame_{number:06d}_input.json").write_text(
            json.dumps(_model_ready_sidecar(number)), encoding="utf-8"
        )

        (root / FOLDER_PREDICTIONS / f"frame_{number:06d}_prediction.json").write_text(
            json.dumps(_prediction_sidecar(number)), encoding="utf-8"
        )

    # Session-level files that must not be mistaken for frames.
    (root / FOLDER_PREDICTIONS / "latest_prediction.json").write_text("{}", encoding="utf-8")
    (root / FOLDER_PREDICTIONS / "mqtt_publish_state.json").write_text("{}", encoding="utf-8")
    (root / "session.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "session_id": "session_28092026_202645",
                "frame_count": N_FRAMES,
                "sampling_rate_hz": 250.0,
                "duration_per_frame_s": 10.0,
                "channel_order": ["Lead I", "Lead II", "Lead III"],
                "status": "STOPPED",
            }
        ),
        encoding="utf-8",
    )
    return root


# -- frame id parsing ---------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("frame_000001", 1),
        ("000001", 1),
        ("frame_000001_input", 1),
        ("frame_000020_prediction", 20),
        (" 000042 ", 42),
    ],
)
def test_parse_frame_id_accepts_every_observed_encoding(value: str, expected: int) -> None:
    assert parse_frame_id(value) == expected


@pytest.mark.parametrize("value", ["", "frame_", "abc", "frame_abc"])
def test_parse_frame_id_rejects_garbage(value: str) -> None:
    with pytest.raises(ValueError):
        parse_frame_id(value)


def test_parse_frame_number_is_strict_about_filenames() -> None:
    """A bare number is a valid ``frame_id`` but never a valid filename stem."""
    assert parse_frame_number("frame_000007_mv") == 7
    with pytest.raises(ValueError):
        parse_frame_number("000007")


# -- per-folder readers -------------------------------------------------


def test_calibrated_reader_returns_signal_and_metadata(raw_tree: Path) -> None:
    frames = read_calibrated(raw_tree)

    assert [f.frame_id for f in frames] == ["000001", "000002", "000003"]
    first = frames[0]
    assert first.signal is not None and first.signal.shape == (N_SAMPLES, 3)
    assert first.sampling_rate == 250.0
    assert first.duration_s == 10.0
    assert first.unit == "mV"
    assert first.channel_order == ["Lead I", "Lead II", "Lead III"]
    assert first.metadata["calibration"]["method"] == "multipoint_rpeak"
    assert first.metadata["has_csv"] is True
    assert first.ok


def test_filtered_is_read_through_the_same_reader_with_a_folder_override(raw_tree: Path) -> None:
    frames = read_calibrated(raw_tree, folder=FOLDER_FILTERED)

    assert [f.folder for f in frames] == [FOLDER_FILTERED] * N_FRAMES
    assert all(f.signal is not None and f.signal.shape == (N_SAMPLES, 3) for f in frames)


def test_model_ready_reader_records_the_filtered_pointer(raw_tree: Path) -> None:
    frames = read_model_ready(raw_tree)

    assert [f.frame_id for f in frames] == ["000001", "000002", "000003"]
    pointers = frames[0].metadata["recorded_source_files"]
    assert pointers["source_file"].endswith("filtered/frame_000001_mv.npy")
    assert frames[0].metadata["recorded_source_frame"] == 1
    assert frames[0].metadata["source_processing"]["filters"][0]["name"] == "notch_50hz"
    assert frames[0].metadata["validation"]["status"] == "WARNING"


def test_recorded_source_number_returns_none_without_a_pointer() -> None:
    """No pointer means UNRESOLVED later, never a guess based on the frame number."""
    assert recorded_source_number({}) is None
    assert recorded_source_number({"source_file": "/x/not-a-frame.npy"}) is None
    assert recorded_source_number({"source_file": "/x/frame_000012_mv.npy"}) == 12


def test_prediction_reader_accepts_bare_numeric_frame_ids(raw_tree: Path) -> None:
    frames = read_predictions(raw_tree)

    assert [f.frame_id for f in frames] == ["000001", "000002", "000003"]
    assert all(f.ok for f in frames), [f.errors for f in frames]
    assert frames[0].metadata["label"] == "Normal"
    assert frames[0].metadata["confidence_percent"] == 99.73


def test_prediction_reader_skips_session_level_files(raw_tree: Path) -> None:
    """``latest_prediction.json`` belongs to no frame and must not become one."""
    frames = read_predictions(raw_tree)

    assert all("latest" not in f.files.get("json", "") for f in frames)
    assert all("mqtt" not in f.files.get("json", "") for f in frames)


def test_prediction_reader_reports_a_frame_id_disagreement(tmp_path: Path) -> None:
    root = tmp_path / "subject"
    (root / FOLDER_PREDICTIONS).mkdir(parents=True)
    payload = _prediction_sidecar(9)
    (root / FOLDER_PREDICTIONS / "frame_000001_prediction.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    frames = read_predictions(root)

    assert len(frames) == 1
    assert not frames[0].ok
    assert "disagrees" in frames[0].errors[0]


def test_log_reader_is_empty_for_an_empty_folder(raw_tree: Path) -> None:
    entries, warnings = read_logs(raw_tree)

    assert entries == []
    assert warnings == []


def test_log_reader_returns_entries_with_their_file_name(raw_tree: Path) -> None:
    (raw_tree / "logs" / "device.log").write_text(
        "boot ok\nframe 1 captured\n", encoding="utf-8"
    )

    entries, warnings = read_logs(raw_tree)

    assert warnings == []
    assert [e["raw"] for e in entries] == ["boot ok", "frame 1 captured"]
    assert {e["file"] for e in entries} == {"device.log"}


def test_log_reader_survives_unreadable_files(raw_tree: Path) -> None:
    (raw_tree / "logs" / "device.log").write_text("boot ok\n", encoding="utf-8")
    (raw_tree / "logs" / "broken.jsonl").write_text("{not json}\n", encoding="utf-8")

    entries, warnings = read_logs(raw_tree)

    assert [e["raw"] for e in entries if e["file"] == "broken.jsonl"] == ["{not json}"]
    assert warnings == []


# -- orchestration ------------------------------------------------------


def test_discover_raw_roots_finds_the_subject_folder(raw_tree: Path) -> None:
    roots = discover_raw_roots(raw_tree.parents[1])

    assert roots == [raw_tree]


def test_discover_raw_roots_is_empty_for_a_missing_directory(tmp_path: Path) -> None:
    assert discover_raw_roots(tmp_path / "nope") == []


def test_read_session_metadata(raw_tree: Path) -> None:
    metadata = read_session_metadata(raw_tree)

    assert metadata["session_id"] == "session_28092026_202645"
    assert metadata["frame_count"] == N_FRAMES


def test_read_raw_dataset_keeps_every_folder_separate(raw_tree: Path) -> None:
    dataset = read_raw_dataset(raw_tree)

    assert dataset.subject_id == "Bryan"
    assert dataset.session_id == "session_28092026_202645"
    assert dataset.device_id == "device01"
    assert dataset.count(FOLDER_CALIBRATED) == N_FRAMES
    assert dataset.count(FOLDER_FILTERED) == N_FRAMES
    assert dataset.count(FOLDER_MODEL_READY) == N_FRAMES
    assert dataset.count(FOLDER_PREDICTIONS) == N_FRAMES
    assert dataset.status() == "VALID"
    assert dataset.errors == []


def test_read_raw_dataset_reports_filtered_and_model_ready_as_identical(raw_tree: Path) -> None:
    """The relationship the mapper relies on, proven rather than assumed."""
    dataset = read_raw_dataset(raw_tree)
    filtered = dataset.frame_map(FOLDER_FILTERED)
    model_ready = dataset.frame_map(FOLDER_MODEL_READY)
    calibrated = dataset.frame_map(FOLDER_CALIBRATED)

    for number in filtered:
        assert np.array_equal(filtered[number].signal, model_ready[number].signal)
        assert not np.array_equal(calibrated[number].signal, filtered[number].signal)


def test_read_raw_dataset_flags_a_folder_without_any_signal(tmp_path: Path) -> None:
    root = tmp_path / "empty_subject"
    (root / FOLDER_PREDICTIONS).mkdir(parents=True)

    dataset = read_raw_dataset(root)

    assert dataset.status() == "ERROR"
    assert any("no signal folder" in error["reason"] for error in dataset.errors)
    assert all(isinstance(error, dict) for error in dataset.errors)


def test_read_raw_dataset_collects_frame_errors(tmp_path: Path) -> None:
    root = tmp_path / "subject"
    (root / FOLDER_CALIBRATED).mkdir(parents=True)
    (root / FOLDER_CALIBRATED / "frame_000001_mv.json").write_text(
        json.dumps(_calibrated_sidecar(1)), encoding="utf-8"
    )

    dataset = read_raw_dataset(root)

    assert dataset.status() == "ERROR"
    assert any("missing frame_*_mv.npy" in error["reason"] for error in dataset.errors)


def test_read_raw_dataset_can_skip_arrays_for_inspection(raw_tree: Path) -> None:
    """Sidecar-only inspection must still report rate, unit and mapping evidence."""
    dataset = read_raw_dataset(raw_tree, load_signals=False)

    record = dataset.by_folder(FOLDER_MODEL_READY)[0]
    assert record.signal is None
    assert record.sampling_rate == 250.0
    assert record.metadata["recorded_source_frame"] == 1
    assert dataset.count(FOLDER_MODEL_READY) == N_FRAMES


def test_summary_marks_raw_adc_as_present_but_not_read(raw_tree: Path) -> None:
    (raw_tree / "raw_adc").mkdir()

    summary = read_raw_dataset(raw_tree).summary()

    assert summary["present_not_read"] == ["raw_adc"]
    assert "raw_adc" not in summary["counts"]


# -- file grouping ------------------------------------------------------


def test_collect_frame_files_groups_by_number(tmp_path: Path) -> None:
    for name in ("frame_000001_mv.npy", "frame_000001_mv.json", "frame_000002_mv.npy"):
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")

    grouped = collect_frame_files(tmp_path)

    assert sorted(grouped) == [1, 2]
    assert set(grouped[1]) == {"npy", "json"}


def test_collect_frame_files_is_empty_for_a_missing_directory(tmp_path: Path) -> None:
    assert collect_frame_files(tmp_path / "nope") == {}