"""Tests for the canonical frame and the reconstruction mapping models.

The canonical frame must carry the same analysis surface for both data paths of
IDEA-REVISED.md section 10, and a :class:`FrameMapping` must always be able to
explain why a link was accepted or left unresolved.
"""

from __future__ import annotations

import numpy as np
import pytest

from ecg_eval.models import (
    MAPPING_METHOD_NONE,
    MAPPING_METHOD_RECORDED_POINTER,
    MAPPING_UNRESOLVED,
    MAPPING_VERIFIED,
    STATUS_ERROR,
    STATUS_UNRESOLVED,
    STATUS_VALID,
    STATUS_WARNING,
    ECGFrame,
    FrameMapping,
    ReconstructionResult,
    SOURCE_RAW_RECONSTRUCTED,
    SourceRef,
)


def _frame(**overrides) -> ECGFrame:
    payload = {
        "subject_id": "S01",
        "session_id": "ses1",
        "frame_id": "000001",
        "source_file": "data/raw/calibrated/frame_000001_mv.npy",
        "source_format": SOURCE_RAW_RECONSTRUCTED,
        "signal": np.zeros((2500, 3), dtype=np.float32),
    }
    payload.update(overrides)
    return ECGFrame(**payload)


# -- canonical frame ----------------------------------------------------


def test_reconstructed_frame_defaults_report_nothing_merged() -> None:
    frame = _frame()

    assert frame.model_input_available is False
    assert frame.prediction_available is False


def test_summary_surfaces_mapping_state() -> None:
    frame = _frame(
        model_input_available=True,
        prediction_available=True,
        provenance={"mapping_status": MAPPING_VERIFIED, "mapping_method": MAPPING_METHOD_RECORDED_POINTER},
    )

    summary = frame.summary()

    assert summary["model_input_available"] is True
    assert summary["prediction_available"] is True
    assert summary["mapping_status"] == MAPPING_VERIFIED
    assert summary["mapping_method"] == MAPPING_METHOD_RECORDED_POINTER


def test_provenance_view_strips_the_recording_root() -> None:
    root = "data/29-09-2026/Bryan"
    frame = _frame(
        provenance={
            "source_type": "raspberry_pi_raw",
            "raw_root": root,
            "mapping_status": MAPPING_VERIFIED,
            "mapping_method": MAPPING_METHOD_RECORDED_POINTER,
            "signal_source": "calibrated",
            "source_files": {
                "calibrated": f"{root}/calibrated/frame_000001_mv.npy",
                "model_ready": f"{root}/model_ready/frame_000001_input.npy",
            },
        }
    )

    view = frame.provenance_view()

    assert view["source_type"] == "raspberry_pi_raw"
    assert view["source_files"]["calibrated"] == "calibrated/frame_000001_mv.npy"
    assert view["source_files"]["model_ready"] == "model_ready/frame_000001_input.npy"


def test_provenance_view_leaves_external_paths_alone() -> None:
    """A firmware path from another machine must not be mangled."""
    external = "/home/pi/sessions/x/model_ready/frame_000001_input.npy"
    frame = _frame(provenance={"raw_root": "data/29-09-2026/Bryan", "source_files": {"model_ready": external}})

    assert frame.provenance_view()["source_files"]["model_ready"] == external


def test_content_hash_covers_the_samples() -> None:
    first = _frame()
    second = _frame(signal=np.ones((2500, 3), dtype=np.float32))

    assert first.content_hash() != second.content_hash()


# -- source references --------------------------------------------------


def test_source_ref_prefers_npy_then_json() -> None:
    both = SourceRef(folder="model_ready", frame_number=1, files={"json": "a.json", "npy": "a.npy"})
    json_only = SourceRef(folder="predictions", frame_number=1, files={"json": "p.json"})

    assert both.primary_file == "a.npy"
    assert json_only.primary_file == "p.json"
    assert both.frame_id == "000001"


def test_source_ref_without_a_number_has_no_frame_id() -> None:
    assert SourceRef(folder="logs").frame_id == ""


# -- frame mapping ------------------------------------------------------


def test_mapping_is_verified_when_a_pointer_was_recorded() -> None:
    mapping = FrameMapping(
        frame_id="000001",
        sources={
            "calibrated": SourceRef(folder="calibrated", frame_number=1),
            "model_ready": SourceRef(folder="model_ready", frame_number=1),
        },
        evidence={"pointer": "filtered/frame_000001_mv.npy"},
    )

    assert mapping.resolved is True
    assert mapping.has("calibrated") and mapping.has("model_ready")
    assert not mapping.has("predictions")


def test_mapping_records_why_it_is_unresolved() -> None:
    mapping = FrameMapping(
        frame_id="000007",
        status=MAPPING_UNRESOLVED,
        method=MAPPING_METHOD_NONE,
        reasons=["model_ready frame 7 records no source_file"],
    )

    assert mapping.resolved is False
    assert mapping.reasons == ["model_ready frame 7 records no source_file"]


def test_mapping_to_dict_is_json_safe() -> None:
    mapping = FrameMapping(
        frame_id="000001",
        sources={"calibrated": SourceRef(folder="calibrated", frame_number=1, files={"npy": "a.npy"})},
    )

    payload = mapping.to_dict()

    assert payload["sources"]["calibrated"]["frame_id"] == "000001"
    assert payload["status"] == MAPPING_VERIFIED


# -- reconstruction result ----------------------------------------------


def _result(**overrides) -> ReconstructionResult:
    payload = {
        "subject_id": "S01",
        "session_id": "ses1",
        "mappings": [FrameMapping(frame_id="000001"), FrameMapping(frame_id="000002")],
        "frames": [_frame(), _frame(frame_id="000002")],
    }
    payload.update(overrides)
    return ReconstructionResult(**payload)


def test_counts_summarise_merged_sources() -> None:
    result = _result(
        mappings=[
            FrameMapping(
                frame_id="000001",
                sources={
                    "calibrated": SourceRef(folder="calibrated", frame_number=1),
                    "model_ready": SourceRef(folder="model_ready", frame_number=1),
                    "predictions": SourceRef(folder="predictions", frame_number=1),
                },
            ),
            FrameMapping(frame_id="000002", sources={"calibrated": SourceRef(folder="calibrated", frame_number=2)}),
        ]
    )

    counts = result.counts()

    assert counts["canonical_frames"] == 2
    assert counts["verified"] == 2
    assert counts["with_calibrated"] == 2
    assert counts["with_model_ready"] == 1
    assert counts["with_prediction"] == 1


def test_status_is_valid_when_every_mapping_is_verified() -> None:
    assert _result().status() == STATUS_VALID


def test_status_is_unresolved_when_any_mapping_is_unresolved() -> None:
    result = _result(
        mappings=[
            FrameMapping(frame_id="000001"),
            FrameMapping(frame_id="000002", status=MAPPING_UNRESOLVED, method=MAPPING_METHOD_NONE),
        ]
    )

    assert result.status() == STATUS_UNRESOLVED
    assert [m.frame_id for m in result.unresolved()] == ["000002"]


def test_status_is_warning_for_dropped_frames() -> None:
    result = _result(dropped=[{"frame_id": "000003", "reason": "no signal"}])

    assert result.status() == STATUS_WARNING


def test_status_is_error_outranking_unresolved() -> None:
    result = _result(
        mappings=[FrameMapping(frame_id="000001", status=MAPPING_UNRESOLVED, method=MAPPING_METHOD_NONE)],
        errors=[{"reason": "session.json unreadable"}],
    )

    assert result.status() == STATUS_ERROR


def test_mapping_methods_are_tallied() -> None:
    result = _result(
        mappings=[
            FrameMapping(frame_id="000001", method=MAPPING_METHOD_RECORDED_POINTER),
            FrameMapping(frame_id="000002", method=MAPPING_METHOD_NONE, status=MAPPING_UNRESOLVED),
        ]
    )

    assert result.mapping_methods() == {MAPPING_METHOD_RECORDED_POINTER: 1, MAPPING_METHOD_NONE: 1}


def test_mapping_for_looks_up_by_frame_id() -> None:
    result = _result()

    assert result.mapping_for("000002") is not None
    assert result.mapping_for("999999") is None


def test_to_dict_can_omit_mappings_for_a_summary_view() -> None:
    result = _result()

    assert "mappings" in result.to_dict()
    assert "mappings" not in result.to_dict(include_mappings=False)