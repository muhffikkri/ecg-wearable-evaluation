"""Tests for the ``build_jsonl_dataset.py`` command-line entry point.

The CLI is the only way to build a dataset without opening the app, so its
argument handling and its refusal to write into a recording are covered here
rather than assumed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "build_jsonl_dataset.py"
DATA = REPO_ROOT / "data"
RECORDING = DATA / "29-09-2026" / "Bryan"

EXIT_OK = 0
EXIT_INVALID = 1
EXIT_USAGE = 2


def _recording(tmp_path: Path, *, frames: int = 2) -> Path:
    """A minimal recording laid out like a real Raspberry Pi session."""
    import uuid

    root = tmp_path / "data" / "29-09-2026" / "S01"
    root.mkdir(parents=True)
    session_id = "session_test_000000"
    (root / "session.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "session_id": session_id,
                "started_at": "2026-09-28T20:26:45+07:00",
                "ended_at": "2026-09-28T20:26:55+07:00",
                "sampling_rate_hz": 250,
                "duration_per_frame_s": 1.0,
                "frame_count": frames,
                "channel_order": ["Lead I", "Lead II", "Lead III"],
                "unit": "mV",
                "status": "STOPPED",
            }
        ),
        encoding="utf-8",
    )

    rng = np.random.default_rng(7)
    for folder, suffix in (("calibrated", "_mv"), ("filtered", "_mv")):
        (root / folder).mkdir(parents=True, exist_ok=True)
    (root / "model_ready").mkdir(parents=True, exist_ok=True)
    (root / "predictions").mkdir(parents=True, exist_ok=True)

    for index in range(1, frames + 1):
        name = f"frame_{index:06d}"
        measurement_id = str(uuid.uuid4())
        signal = rng.normal(0, 0.1, size=(250, 3)).astype(np.float32)

        shared = {
            "source_frame": name,
            "dtype": "float32",
            "shape": [250, 3],
            "sample_count": 250,
            "sample_rate_hz": 250,
            "duration_seconds": 1.0,
            "unit": "mV",
            "channel_order": ["Lead I", "Lead II", "Lead III"],
            "created_at_utc": f"2026-09-28T13:27:0{index}.000000Z",
            "source_metadata": {"measurement_id": measurement_id, "device_id": "device01"},
        }
        np.save(root / "calibrated" / f"{name}{suffix}.npy", signal)
        (root / "calibrated" / f"{name}{suffix}.json").write_text(
            json.dumps(shared), encoding="utf-8"
        )
        np.save(root / "filtered" / f"{name}{suffix}.npy", signal)
        (root / "filtered" / f"{name}{suffix}.json").write_text(
            json.dumps({**shared, "source_metadata": {**shared["source_metadata"], "measurement_id": measurement_id}}),
            encoding="utf-8",
        )

        np.save(root / "model_ready" / f"{name}_input.npy", signal)
        (root / "model_ready" / f"{name}_input.json").write_text(
            json.dumps(
                {
                    "frame_id": name,
                    "channel_order": ["Lead I", "Lead II", "Lead III"],
                    "sample_rate_hz": 250,
                    "samples_per_channel": 250,
                    "shape": [250, 3],
                    "dtype": "float32",
                    "unit": "mV",
                    "source_file": f"/device/{session_id}/filtered/{name}_mv.npy",
                    "validation": {"status": "PASS", "warnings": []},
                    "signal_quality": {"status": "PASS", "reasons": []},
                }
            ),
            encoding="utf-8",
        )
        (root / "predictions" / f"{name}_prediction.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "PASS",
                    "frame_id": f"{index:06d}",
                    "source_file": f"/device/{session_id}/model_ready/{name}_input.npy",
                    "prediction": "Normal",
                    "confidence_percent": 99.7,
                    "probabilities": {"Normal": 99.7, "AF": 0.1},
                    "threshold": 0.5,
                    "latency_ms": 250.0,
                    "runtime": "ai-edge-litert",
                }
            ),
            encoding="utf-8",
        )
    return root


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        encoding="utf-8",
    )


# -- argument handling ---------------------------------------------------


def test_a_missing_path_is_a_usage_error(tmp_path: Path) -> None:
    result = run(str(tmp_path / "nope"))

    assert result.returncode == EXIT_USAGE
    assert "not found" in result.stderr


def test_a_file_instead_of_a_directory_is_a_usage_error(tmp_path: Path) -> None:
    target = tmp_path / "record.jsonl"
    target.write_text("{}\n", encoding="utf-8")

    result = run(str(target))

    assert result.returncode == EXIT_USAGE
    assert "got a file" in result.stderr


def test_a_directory_with_no_recordings_is_a_usage_error(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()

    result = run(str(tmp_path / "empty"))

    assert result.returncode == EXIT_USAGE


def test_filename_cannot_be_shared_between_recordings(tmp_path: Path) -> None:
    first = _recording(tmp_path / "a")
    second = _recording(tmp_path / "b", frames=1)

    result = run(str(first), str(second), "--output", str(tmp_path / "out"), "--filename", "x.jsonl")

    assert result.returncode == EXIT_USAGE
    assert "--filename cannot be used" in result.stderr


# -- safety --------------------------------------------------------------


def test_writing_into_the_recording_is_refused(tmp_path: Path) -> None:
    root = _recording(tmp_path)

    result = run(str(root), "--output", str(root / "converted.jsonl"))

    assert result.returncode == EXIT_USAGE
    assert "read-only" in result.stderr
    assert not (root / "converted.jsonl").exists()


def test_the_recording_is_never_modified(tmp_path: Path) -> None:
    root = _recording(tmp_path)
    before = {
        path: path.stat().st_mtime_ns
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }

    result = run(str(root), "--output", str(tmp_path / "out"))

    assert result.returncode == EXIT_OK, result.stderr
    after = {
        path: path.stat().st_mtime_ns
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    assert before == after


# -- conversion ----------------------------------------------------------


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    root = _recording(tmp_path)
    out = tmp_path / "out"

    result = run(str(root), "--output", str(out), "--dry-run")

    assert result.returncode == EXIT_OK, result.stderr
    assert "would write" in result.stdout
    assert not out.exists()


def test_conversion_writes_records_and_a_manifest(tmp_path: Path) -> None:
    root = _recording(tmp_path, frames=3)
    out = tmp_path / "out"

    result = run(str(root), "--output", str(out), "--json")

    assert result.returncode == EXIT_OK, result.stderr
    summary = json.loads(result.stdout)[0]
    assert summary["frames_reconstructed"] == 3
    assert summary["records_written"] == 3
    assert summary["unavailable_fields"] == ["patient_id", "system", "network"]

    written = out / "S01_session_test_000000.jsonl"
    assert written.is_file()
    lines = written.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    for line in lines:
        record = json.loads(line)
        assert record["device_id"] == "device01"
        assert record["sampling_rate_hz"] == 250.0
        assert len(record["ecg"]["samples"]) == 250

    manifest = json.loads((out / "S01_session_test_000000.manifest.json").read_text(encoding="utf-8"))
    assert manifest["frames_written"] == 3
    assert manifest["unavailable_fields"] == ["patient_id", "system", "network"]


def test_the_signal_survives_the_round_trip(tmp_path: Path) -> None:
    root = _recording(tmp_path, frames=2)
    out = tmp_path / "out"

    result = run(str(root), "--output", str(out))
    assert result.returncode == EXIT_OK, result.stderr

    lines = (out / "S01_session_test_000000.jsonl").read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines, start=1):
        record = json.loads(line)
        source = np.load(root / "calibrated" / f"frame_{index:06d}_mv.npy")
        assert np.array_equal(
            np.asarray(record["ecg"]["samples"], dtype=np.float32), source
        )


def test_the_subject_id_can_be_overridden(tmp_path: Path) -> None:
    root = _recording(tmp_path, frames=1)
    out = tmp_path / "out"

    result = run(str(root), "--output", str(out), "--subject-id", "PAT-042")

    assert result.returncode == EXIT_OK, result.stderr
    assert (out / "PAT-042_session_test_000000.jsonl").is_file()
    record = json.loads(
        (out / "PAT-042_session_test_000000.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert record["message_id"].startswith("device01-session_test_000000-frame_")


def test_a_date_folder_discovers_its_recordings(tmp_path: Path) -> None:
    root = _recording(tmp_path)
    out = tmp_path / "out"

    result = run(str(root.parent), "--output", str(out), "--json")

    assert result.returncode == EXIT_OK, result.stderr
    summaries = json.loads(result.stdout)
    assert len(summaries) == 1
    assert summaries[0]["subject_id"] == "S01"


def test_the_output_can_be_read_back_by_the_ingestion_reader(tmp_path: Path) -> None:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from ecg_eval.ingestion.jsonl_reader import read_jsonl_file

    root = _recording(tmp_path, frames=2)
    out = tmp_path / "out"
    result = run(str(root), "--output", str(out))
    assert result.returncode == EXIT_OK, result.stderr

    ingested = read_jsonl_file(out / "S01_session_test_000000.jsonl")

    assert ingested.status == "OK"
    assert len(ingested.frames) == 2
    assert not ingested.malformed


# -- the real recording --------------------------------------------------


@pytest.mark.skipif(not RECORDING.is_dir(), reason="recorded session is not present")
def test_the_real_recording_converts_and_validates(tmp_path: Path) -> None:
    out = tmp_path / "out"

    result = run(str(RECORDING), "--output", str(out), "--json")

    assert result.returncode == EXIT_OK, result.stderr
    summary = json.loads(result.stdout)[0]
    assert summary["frames_reconstructed"] == 20
    assert summary["frames_unresolved"] == 0
    assert summary["records_written"] == 20

    written = next(out.glob("*.jsonl"))
    lines = written.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 20
    for line in lines:
        record = json.loads(line)
        assert len(record["ecg"]["samples"]) == 2500
        assert len(record["ecg"]["samples"][0]) == 3
        assert record["prediction"]["label"] in {"Normal", "AF", "Takikardia", "Bradikardia"}