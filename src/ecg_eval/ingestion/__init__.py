"""Ingestion: discovery, parsing, validation and dataset inventory.

Two entry points coexist, and both must reach the same canonical frame
(IDEA-REVISED.md section 10):

* ``read_raw_dataset`` scans a Raspberry Pi recording folder and keeps every
  output folder separate; merging happens in :mod:`ecg_eval.reconstruction`.
* ``read_jsonl_file`` reads an existing ``*.jsonl`` web dataset.
"""

from .calibrated_reader import read_calibrated
from .jsonl_reader import JSONLReadResult, MalformedRecord, iter_jsonl_files, read_jsonl_file
from .log_reader import read_logs
from .model_ready_reader import read_model_ready, recorded_source_number
from .pipeline import Dataset, dataset_summary, ingest, load_subject_manifest
from .prediction_reader import read_predictions
from .raw_dataset_reader import (
    discover_raw_roots,
    read_raw_dataset,
    read_session_metadata,
)
from .raw_reader import RawReadResult, read_raw_directory
from .source_common import (
    collect_frame_files,
    load_npy,
    parse_frame_id,
    parse_frame_number,
    sha256_file,
)
from .validator import (
    DatasetInventory,
    SessionInventory,
    annotation_coverage,
    build_inventory,
    validate_frames,
    validate_sampling_rates,
)

__all__ = [
    "Dataset",
    "DatasetInventory",
    "JSONLReadResult",
    "MalformedRecord",
    "RawReadResult",
    "SessionInventory",
    "annotation_coverage",
    "build_inventory",
    "collect_frame_files",
    "dataset_summary",
    "discover_raw_roots",
    "ingest",
    "iter_jsonl_files",
    "load_npy",
    "load_subject_manifest",
    "parse_frame_id",
    "parse_frame_number",
    "read_calibrated",
    "read_jsonl_file",
    "read_logs",
    "read_model_ready",
    "read_predictions",
    "read_raw_dataset",
    "read_raw_directory",
    "read_session_metadata",
    "recorded_source_number",
    "sha256_file",
    "validate_frames",
    "validate_sampling_rates",
]
