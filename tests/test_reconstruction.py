"""Tests for Data Reconstruction: mapping, merging and canonical frame assembly.

The evidence chain these tests pin down was established by inspecting the real
29-09-2026 recording before any of it was written:

* ``calibrated`` and ``filtered`` share ``source_metadata.measurement_id``, unique
  per frame, and it always names the same frame number.
* ``model_ready`` records ``source_file`` naming the ``filtered`` array.
* ``predictions`` records ``source_file`` naming the ``model_ready`` array.
* ``model_ready`` carries no ``measurement_id`` at all.

So the mapper must never rely on frame numbers agreeing across folders, even
though in this dataset they do.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ecg_eval.ingestion import read_raw_dataset
from ecg_eval.models import (
    MAPPING_METHOD_MEASUREMENT_ID,
    MAPPING_METHOD_NONE,
    MAPPING_METHOD_RECORDED_POINTER,
    MAPPING_UNRESOLVED,
    MAPPING_VERIFIED,
    STATUS_UNRESOLVED,
    STATUS_VALID,
    SOURCE_RAW_RECONSTRUCTED,
)
from ecg_eval.reconstruction import (
    map_frames,
    measurement_id,
    reconstruct,
    reconstruct_directory,
)

N_SAMPLES = 2500
N_FRAMES = 3


def _signal(offset: float = 0.0) -> np.ndarray:
    time = np.arange(N_SAMPLES, dtype=np.float32) / 250.0
    return np.column_stack([np.sin(2 * np.pi * 1.2 * time) + offset] * 3).astype(np.float32)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def _mv_sidecar(number: int, *, measurement_id_value: str | None) -> dict:
    payload = {
        "channel_order": ["Lead I", "Lead II", "Lead III"],
        "created_at_utc": f"2026-09-28T13:27:0{number}.844340Z",
        "dtype": "float32",
        "duration_seconds": 10.0,
        "shape": [N_SAMPLES, 3],
        "sample_rate_hz": 250,
        "source_frame": f"frame_{number:06d}",
        "unit": "mV",
        "source_metadata": {
            "device_id": "device01",
            "session_id": "session_28092026_202645",
            "frame_index": number,
            "sha256_checksum": f"{number:064d}",
        },
    }
    if measurement_id_value is not None:
        payload["source_metadata"]["measurement_id"] = measurement_id_value
    return payload


def _input_sidecar(number: int, *, points_at: int | None = None, with_pointer: bool = True) -> dict:
    target = number if points_at is None else points_at
    payload = {
        "channel_order": ["Lead I", "Lead II", "Lead III"],
        "created_at": f"2026-09-28T13:27:0{target}.053234Z",
        "dtype": "float32",
        "duration_seconds": 10.0,
        "frame_id": f"frame_{target:06d}",
        "sample_rate_hz": 250,
        "shape": [N_SAMPLES, 3],
        "unit": "mV",
        "validation": {"status": "WARNING", "warnings": ["Large DC baseline on Lead Ii"]},
    }
    if with_pointer:
        payload["source_file"] = f"/home/pi/s/filtered/frame_{target:06d}_mv.npy"
        payload["source_metadata_file"] = f"/home/pi/s/filtered/frame_{target:06d}_mv.json"
    return payload


def _prediction_sidecar(number: int, *, points_at: int | None = None, with_pointer: bool = True) -> dict:
    target = number if points_at is None else points_at
    payload = {
        "frame_id": f"{number:06d}",
        "created_at": "2026-09-28T13:27:04.898854Z",
        "sampling_rate_hz": 250.0,
        "status": "PASS",
        "prediction": "Normal",
        "confidence_percent": 99.73,
        "probabilities": {"Normal": 99.73, "AF": 0.15},
        "threshold": 0.5,
        "latency_ms": 251.26,
        "runtime": "ai-edge-litert",
    }
    if with_pointer:
        payload["source_file"] = f"/home/pi/s/model_ready/frame_{target:06d}_input.npy"
    return payload


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A recording where every folder is present and links up cleanly."""
    root = tmp_path / "Bryan"
    for number in range(1, N_FRAMES + 1):
        mid = f"mid-{number:06d}"
        _write_npy(root / "calibrated" / f"frame_{number:06d}_mv.npy", _signal(offset=9.0))
        _write_json(root / "calibrated" / f"frame_{number:06d}_mv.json",
                    _mv_sidecar(number, measurement_id_value=mid))
        _write_npy(root / "filtered" / f"frame_{number:06d}_mv.npy", _signal())
        _write_json(root / "filtered" / f"frame_{number:06d}_mv.json",
                    _mv_sidecar(number, measurement_id_value=mid))
        _write_npy(root / "model_ready" / f"frame_{number:06d}_input.npy", _signal())
        _write_json(root / "model_ready" / f"frame_{number:06d}_input.json",
                    _input_sidecar(number))
        _write_json(root / "predictions" / f"frame_{number:06d}_prediction.json",
                    _prediction_sidecar(number))
    _write_json(root / "session.json", {"session_id": "session_28092026_202645",
                                        "sampling_rate_hz": 250.0, "frame_count": N_FRAMES})
    return root


# -- evidence helpers ---------------------------------------------------


def test_measurement_id_is_read_from_source_metadata(tree: Path) -> None:
    records = read_raw_dataset(tree).by_folder("calibrated")

    assert measurement_id(records[0]) == "mid-000001"


def test_measurement_id_is_none_when_absent(tree: Path) -> None:
    root = tree
    _write_json(root / "calibrated" / "frame_000004_mv.json", _mv_sidecar(4, measurement_id_value=None))

    records = read_raw_dataset(root).by_folder("calibrated")

    assert measurement_id(records[-1]) is None


# -- mapping ------------------------------------------------------------


def test_map_frames_verifies_every_folder_by_evidence(tree: Path) -> None:
    mappings = map_frames(read_raw_dataset(tree))

    assert [m.frame_id for m in mappings] == ["000001", "000002", "000003"]
    assert all(m.resolved for m in mappings)
    assert all(m.method == MAPPING_METHOD_MEASUREMENT_ID for m in mappings)
    for mapping in mappings:
        assert set(mapping.sources) == {"calibrated", "filtered", "model_ready", "predictions"}
        assert mapping.signal_source == "calibrated"
        assert mapping.reasons == []


def test_mapping_records_the_uuid_and_the_pointers(tree: Path) -> None:
    mapping = map_frames(read_raw_dataset(tree))[0]

    assert mapping.evidence["measurement_id"] == "mid-000001"
    assert mapping.evidence["model_ready_source_file"] == "frame_000001_mv.npy"
    assert mapping.evidence["prediction_source_file"] == "frame_000001_input.npy"


def test_mapping_does_not_rely_on_frame_numbers_agreeing(tmp_path: Path) -> None:
    """model_ready 1..3 points at filtered 7..9, which sit at calibrated 7..9."""
    root = tmp_path / "offset"
    for number in (7, 8, 9):
        mid = f"mid-{number:06d}"
        _write_npy(root / "calibrated" / f"frame_{number:06d}_mv.npy", _signal())
        _write_json(root / "calibrated" / f"frame_{number:06d}_mv.json",
                    _mv_sidecar(number, measurement_id_value=mid))
        _write_npy(root / "filtered" / f"frame_{number:06d}_mv.npy", _signal())
        _write_json(root / "filtered" / f"frame_{number:06d}_mv.json",
                    _mv_sidecar(number, measurement_id_value=mid))
    for index in (1, 2, 3):
        _write_npy(root / "model_ready" / f"frame_{index:06d}_input.npy", _signal())
        _write_json(root / "model_ready" / f"frame_{index:06d}_input.json",
                    _input_sidecar(index, points_at=index + 6))

    mappings = map_frames(read_raw_dataset(root))

    assert [m.frame_id for m in mappings] == ["000007", "000008", "000009"]
    assert all(m.has("model_ready") for m in mappings)
    assert [m.sources["model_ready"].frame_id for m in mappings] == ["000001", "000002", "000003"]
    assert all(m.resolved for m in mappings)


def test_mapping_is_unresolved_without_a_recorded_pointer(tree: Path) -> None:
    _write_json(tree / "model_ready" / "frame_000001_input.json",
                _input_sidecar(1, with_pointer=False))

    mappings = map_frames(read_raw_dataset(tree))

    assert mappings[0].status == MAPPING_UNRESOLVED
    # The calibrated/filtered link still holds; it is the model_ready hop that
    # has no proof, which is exactly why the frame cannot be verified.
    assert mappings[0].method == MAPPING_METHOD_MEASUREMENT_ID
    assert mappings[0].signal_source == "calibrated"
    assert any("no model_ready frame recording source_file" in r for r in mappings[0].reasons)
    assert not mappings[0].has("model_ready")


def test_mapping_reports_an_ambiguous_candidate(tmp_path: Path) -> None:
    """Two model_ready frames claiming the same source file is not resolvable."""
    root = tmp_path / "ambiguous"
    mid = "mid-000001"
    _write_npy(root / "calibrated" / "frame_000001_mv.npy", _signal())
    _write_json(root / "calibrated" / "frame_000001_mv.json", _mv_sidecar(1, measurement_id_value=mid))
    _write_npy(root / "filtered" / "frame_000001_mv.npy", _signal())
    _write_json(root / "filtered" / "frame_000001_mv.json", _mv_sidecar(1, measurement_id_value=mid))
    for index in (1, 2):
        _write_npy(root / "model_ready" / f"frame_{index:06d}_input.npy", _signal())
        _write_json(root / "model_ready" / f"frame_{index:06d}_input.json",
                    _input_sidecar(index, points_at=1))

    mapping = map_frames(read_raw_dataset(root))[0]

    assert mapping.status == MAPPING_UNRESOLVED
    assert any("ambiguous" in reason for reason in mapping.reasons)
    assert not mapping.has("model_ready")


def test_mapping_anchors_on_calibrated_when_filtered_is_absent(tmp_path: Path) -> None:
    root = tmp_path / "cal_only"
    for number in range(1, N_FRAMES + 1):
        _write_npy(root / "calibrated" / f"frame_{number:06d}_mv.npy", _signal())
        _write_json(root / "calibrated" / f"frame_{number:06d}_mv.json",
                    _mv_sidecar(number, measurement_id_value=f"mid-{number:06d}"))

    mappings = map_frames(read_raw_dataset(root))

    assert len(mappings) == N_FRAMES
    assert all(m.resolved for m in mappings)
    assert all(m.signal_source == "calibrated" for m in mappings)
    assert any("no filtered/ counterpart" in w for w in mappings[0].warnings)


def test_mapping_anchors_on_model_ready_when_only_it_exists(tmp_path: Path) -> None:
    """With one signal folder the frame is its own anchor, so it is verified."""
    root = tmp_path / "mr_only"
    for number in range(1, N_FRAMES + 1):
        _write_npy(root / "model_ready" / f"frame_{number:06d}_input.npy", _signal())
        _write_json(root / "model_ready" / f"frame_{number:06d}_input.json",
                    _input_sidecar(number))

    mappings = map_frames(read_raw_dataset(root))

    assert len(mappings) == N_FRAMES
    assert all(m.signal_source == "model_ready" for m in mappings)
    assert all(m.resolved for m in mappings)
    assert all(m.method == MAPPING_METHOD_NONE for m in mappings)
    assert all(m.reasons == [] for m in mappings)


def test_unlinked_prediction_is_reported_not_dropped(tmp_path: Path) -> None:
    """A prediction whose pointer names another frame must not vanish."""
    root = tmp_path / "bad_pred"
    mid = "mid-000001"
    _write_npy(root / "calibrated" / "frame_000001_mv.npy", _signal())
    _write_json(root / "calibrated" / "frame_000001_mv.json", _mv_sidecar(1, measurement_id_value=mid))
    _write_npy(root / "filtered" / "frame_000001_mv.npy", _signal())
    _write_json(root / "filtered" / "frame_000001_mv.json", _mv_sidecar(1, measurement_id_value=mid))
    _write_json(root / "predictions" / "frame_000001_prediction.json",
                _prediction_sidecar(1, points_at=99))

    mapping = map_frames(read_raw_dataset(root))[0]

    assert not mapping.has("predictions")
    assert any("is not linked" in reason for reason in mapping.reasons)


def test_a_prediction_is_never_required(tree: Path) -> None:
    for path in (tree / "predictions").glob("frame_*.json"):
        path.unlink()

    mappings = map_frames(read_raw_dataset(tree))

    assert all(m.resolved for m in mappings)
    assert all(not m.has("predictions") for m in mappings)


# -- reconstruction -----------------------------------------------------


def test_reconstruct_builds_canonical_frames(tree: Path) -> None:
    result = reconstruct(tree)

    assert result.status() == STATUS_VALID
    assert result.counts()["canonical_frames"] == N_FRAMES
    assert result.counts()["unresolved"] == 0
    frame = result.frames[0]
    assert frame.source_format == SOURCE_RAW_RECONSTRUCTED
    assert frame.signal.shape == (N_SAMPLES, 3)
    assert frame.sampling_rate == 250.0
    assert frame.duration_s == 10.0
    assert frame.record_index == 0
    assert frame.model_input_available is True
    assert frame.prediction_available is True


def test_reconstructed_signal_comes_from_the_named_source(tree: Path) -> None:
    frame = reconstruct(tree).frames[0]

    assert np.array_equal(frame.signal, _signal(offset=9.0))
    assert not np.array_equal(frame.signal, _signal())


def test_reconstruct_merges_metadata_from_several_folders(tree: Path) -> None:
    frame = reconstruct(tree).frames[0]

    assert frame.metadata["validation"]["status"] == "WARNING"
    assert frame.metadata["folders_merged"][0] == "calibrated"
    assert "calibrated" in frame.metadata["source_metadata"]
    assert "model_ready" in frame.metadata["source_metadata"]
    assert frame.metadata["prediction"]["label"] == "Normal"
    assert frame.metadata["prediction"]["confidence_percent"] == 99.73


def test_reconstruct_records_provenance_per_folder(tree: Path) -> None:
    frame = reconstruct(tree).frames[0]

    provenance = frame.provenance_view()
    assert provenance["source_type"] == "raspberry_pi_raw"
    assert provenance["mapping_status"] == MAPPING_VERIFIED
    assert provenance["mapping_method"] == MAPPING_METHOD_MEASUREMENT_ID
    assert provenance["signal_source"] == "calibrated"
    assert provenance["source_files"]["calibrated"] == "calibrated/frame_000001_mv.npy"
    assert provenance["source_files"]["model_ready"] == "model_ready/frame_000001_input.npy"
    assert provenance["source_files"]["predictions"] == "predictions/frame_000001_prediction.json"


def test_provenance_keeps_a_foreign_device_path_verbatim(tree: Path) -> None:
    """The firmware path points at the Raspberry Pi, not the local root."""
    frame = reconstruct(tree).frames[0]

    pointer = frame.provenance["mapping_evidence"]["model_ready_source_file"]
    assert pointer == "frame_000001_mv.npy"


def test_reconstruct_emits_unresolved_frames_by_default(tree: Path) -> None:
    _write_json(tree / "model_ready" / "frame_000002_input.json",
                _input_sidecar(2, with_pointer=False))

    result = reconstruct(tree)

    assert result.status() == STATUS_UNRESOLVED
    assert len(result.frames) == N_FRAMES
    assert [m.frame_id for m in result.unresolved()] == ["000002"]
    assert result.frames[1].provenance["unresolved_reasons"]


def test_reconstruct_can_drop_unresolved_frames_instead(tree: Path) -> None:
    _write_json(tree / "model_ready" / "frame_000002_input.json",
                _input_sidecar(2, with_pointer=False))

    result = reconstruct(tree, include_unresolved=False)

    assert len(result.frames) == N_FRAMES - 1
    assert any(entry["frame_id"] == "000002" for entry in result.dropped)


def test_reconstruct_is_non_destructive(tree: Path) -> None:
    before = sorted(p.relative_to(tree).as_posix() for p in tree.rglob("*") if p.is_file())
    stamps = {p: p.stat().st_mtime for p in tree.rglob("*") if p.is_file()}

    reconstruct(tree)

    assert sorted(p.relative_to(tree).as_posix() for p in tree.rglob("*") if p.is_file()) == before
    assert {p: p.stat().st_mtime for p in tree.rglob("*") if p.is_file()} == stamps


def test_reconstruct_records_a_frame_whose_signal_is_unreadable(tmp_path: Path) -> None:
    root = tmp_path / "broken"
    (root / "calibrated").mkdir(parents=True)
    (root / "calibrated" / "frame_000001_mv.json").write_text("{not json", encoding="utf-8")

    result = reconstruct(root)

    assert result.counts()["canonical_frames"] == 0
    assert result.dropped or result.errors


def test_reconstruct_falls_back_to_the_session_sample_rate(tmp_path: Path) -> None:
    root = tmp_path / "no_rate"
    _write_npy(root / "calibrated" / "frame_000001_mv.npy", _signal())
    sidecar = _mv_sidecar(1, measurement_id_value="mid-1")
    sidecar.pop("sample_rate_hz")
    _write_json(root / "calibrated" / "frame_000001_mv.json", sidecar)
    _write_json(root / "session.json", {"session_id": "s1", "sampling_rate_hz": 500.0})

    result = reconstruct(root)

    assert result.frames[0].sampling_rate == 500.0
    assert any("recorded no sample rate" in w for w in result.warnings)


def test_reconstruct_warns_when_merged_arrays_disagree_in_shape(tmp_path: Path) -> None:
    root = tmp_path / "shape_clash"
    _write_npy(root / "calibrated" / "frame_000001_mv.npy", _signal())
    _write_json(root / "calibrated" / "frame_000001_mv.json",
                _mv_sidecar(1, measurement_id_value="mid-1"))
    _write_npy(root / "filtered" / "frame_000001_mv.npy", _signal()[:1000])
    _write_json(root / "filtered" / "frame_000001_mv.json",
                _mv_sidecar(1, measurement_id_value="mid-1"))

    result = reconstruct(root)

    assert len(result.frames) == 1
    assert any("array shape" in w for w in result.warnings)


def test_reconstruct_directory_is_an_alias(tree: Path) -> None:
    assert reconstruct_directory(tree).counts() == reconstruct(tree).counts()


def test_reconstruct_accepts_a_dataset_object(tree: Path) -> None:
    dataset = read_raw_dataset(tree)

    assert reconstruct(dataset).counts()["canonical_frames"] == N_FRAMES


def test_reconstruct_subjects_can_be_overridden(tree: Path) -> None:
    result = reconstruct(tree, subject_id="PAT-042")

    assert result.frames[0].subject_id == "PAT-042"
    assert result.subject_id == "PAT-042"