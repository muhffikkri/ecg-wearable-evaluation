"""Regression tests for annotation persistence.

The annotation page used to stage changes in session state behind a second,
co-primary "Save annotations" button. Nothing reached disk until that button was
pressed, so pressing "Assign range" alone appeared to lose the work, and a save
clicked before any assignment wrote an empty ``annotations.json`` that was
indistinguishable from a genuinely unannotated dataset.

Assignments now persist immediately, so each mutation must survive a read back
from disk and successive mutations must accumulate.
"""

from __future__ import annotations

import json

import pytest

from ecg_eval.annotation import manager, storage

SUBJECT = "ses000000000005"
SESSION = "ses000000000005"
OTHER_SUBJECT = "ses000000000013"
OTHER_SESSION = "ses000000000013"


def persisted(directory, subject_id=SUBJECT, session_id=SESSION) -> list[dict]:
    """Segments currently on disk, as plain dictionaries."""
    path = directory / storage.ANNOTATIONS_FILE
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    for record in payload["annotations"]:
        if record["subject_id"] == subject_id and record["session_id"] == session_id:
            return record["segments"]
    return []


def assign_and_save(directory, annotations, subject, session, label, start, end):
    """Mirror what the annotation page does on a single button press."""
    manager.assign(annotations, subject, session, label, start, end)
    storage.save_annotations(directory, annotations)


def test_single_assignment_is_persisted_immediately(tmp_path):
    annotations = storage.load_annotations(tmp_path)

    assign_and_save(tmp_path, annotations, SUBJECT, SESSION, "SUPINE", 0, 5)

    assert persisted(tmp_path) == [
        {"label": "SUPINE", "start_frame": 0, "end_frame": 5},
    ]


def test_optional_fields_are_written_only_when_set(tmp_path):
    annotations = storage.load_annotations(tmp_path)
    manager.assign(
        annotations, SUBJECT, SESSION, "SUPINE", 0, 5,
        start_time="00:00.0", end_time="00:02.0", note="resting",
    )
    storage.save_annotations(tmp_path, annotations)

    assert persisted(tmp_path) == [
        {
            "label": "SUPINE",
            "start_frame": 0,
            "end_frame": 5,
            "start_time": "00:00.0",
            "end_time": "00:02.0",
            "note": "resting",
        },
    ]


def test_successive_assignments_accumulate(tmp_path):
    """Choosing the next frame range and assigning again must not lose the first."""
    annotations = storage.load_annotations(tmp_path)

    assign_and_save(tmp_path, annotations, SUBJECT, SESSION, "SUPINE", 0, 5)
    assign_and_save(tmp_path, annotations, SUBJECT, SESSION, "SUPINE", 6, 11)

    segments = persisted(tmp_path)
    assert [(s["label"], s["start_frame"], s["end_frame"]) for s in segments] == [
        ("SUPINE", 0, 5),
        ("SUPINE", 6, 11),
    ]


def test_assignments_to_different_sessions_both_survive(tmp_path):
    annotations = storage.load_annotations(tmp_path)

    assign_and_save(tmp_path, annotations, SUBJECT, SESSION, "SUPINE", 0, 5)
    assign_and_save(tmp_path, annotations, OTHER_SUBJECT, OTHER_SESSION, "SITTING", 0, 3)

    assert [(s["label"], s["start_frame"], s["end_frame"]) for s in persisted(tmp_path)] == [
        ("SUPINE", 0, 5)
    ]
    assert [
        (s["label"], s["start_frame"], s["end_frame"])
        for s in persisted(tmp_path, OTHER_SUBJECT, OTHER_SESSION)
    ] == [("SITTING", 0, 3)]


def test_clearing_a_range_is_persisted(tmp_path):
    annotations = storage.load_annotations(tmp_path)
    assign_and_save(tmp_path, annotations, SUBJECT, SESSION, "SUPINE", 0, 5)
    assign_and_save(tmp_path, annotations, SUBJECT, SESSION, "STANDING", 6, 11)

    manager.clear_range(annotations, SUBJECT, SESSION, 0, 5)
    storage.save_annotations(tmp_path, annotations)

    assert [
        (s["label"], s["start_frame"], s["end_frame"]) for s in persisted(tmp_path)
    ] == [("STANDING", 6, 11)]


def test_relabelled_state_reads_back_consistently(tmp_path):
    """A fresh read must reproduce the label timeline the user just created."""
    annotations = storage.load_annotations(tmp_path)
    assign_and_save(tmp_path, annotations, SUBJECT, SESSION, "SUPINE", 0, 5)
    assign_and_save(tmp_path, annotations, SUBJECT, SESSION, "SITTING", 6, 11)

    reloaded = storage.load_annotations(tmp_path)
    labels = manager.labels_for_frames(reloaded, SUBJECT, SESSION, 12)

    assert labels[:6] == ["SUPINE"] * 6
    assert labels[6:] == ["SITTING"] * 6
    assert labels.count("UNLABELED") == 0


def test_saving_without_any_assignment_writes_an_empty_dataset(tmp_path):
    """A deliberate empty save stays valid -- it must not be confused with an error."""
    storage.save_annotations(tmp_path, storage.load_annotations(tmp_path))

    assert persisted(tmp_path) == []
    assert storage.load_annotations(tmp_path) == []


@pytest.mark.parametrize("n_segments", [0, 1, 5])
def test_manifest_tracks_the_saved_segments(tmp_path, n_segments):
    annotations = storage.load_annotations(tmp_path)
    for i in range(n_segments):
        manager.assign(annotations, SUBJECT, SESSION, "SUPINE", i * 2, i * 2 + 1)
    storage.save_annotations(tmp_path, annotations)

    manifest = storage.write_manifest(tmp_path, annotations)
    rows = manifest.read_text(encoding="utf-8").strip().splitlines()

    assert len(rows) - 1 == n_segments
    for row in rows[1:]:
        assert row.split(",")[0] == SUBJECT


def test_annotation_hash_changes_when_labels_change(tmp_path):
    annotations = storage.load_annotations(tmp_path)
    before = storage.annotation_hash(annotations)

    manager.assign(annotations, SUBJECT, SESSION, "SUPINE", 0, 5)
    storage.save_annotations(tmp_path, annotations)

    assert storage.annotation_hash(annotations) != before
    assert storage.annotation_hash(storage.load_annotations(tmp_path)) == storage.annotation_hash(
        annotations
    )