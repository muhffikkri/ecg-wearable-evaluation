"""Data Reconstruction: merge Raspberry Pi folders into canonical frames.

The stage between "scan a raw dataset" and "generate JSONL"
(IDEA-REVISED.md section 5). It exists because ``calibrated/`` and
``model_ready/`` are not two interchangeable representations of one file: the
firmware emits each from the previous step, and the JSONL the web app consumes
needs information from more than one of them.

Division of labour:

* :mod:`frame_mapper` decides which files describe the same recording, from
  recorded evidence only.
* :mod:`metadata_merger` merges the descriptive fields without inventing any.
* :mod:`frame_reconstructor` assembles :class:`ECGFrame` objects and the report.
* :mod:`provenance` records where every part came from.
"""

from .frame_mapper import (
    DEFAULT_TIMESTAMP_TOLERANCE_S,
    map_frames,
    measurement_id,
)
from .frame_reconstructor import (
    SOURCE_PRECEDENCE,
    reconstruct,
    reconstruct_directory,
)
from .metadata_merger import merge_metadata, merge_prediction
from .provenance import (
    SOURCE_TYPE_JSONL,
    SOURCE_TYPE_RAW,
    build_provenance,
    content_sha256,
)

__all__ = [
    "DEFAULT_TIMESTAMP_TOLERANCE_S",
    "SOURCE_PRECEDENCE",
    "SOURCE_TYPE_JSONL",
    "SOURCE_TYPE_RAW",
    "build_provenance",
    "content_sha256",
    "map_frames",
    "measurement_id",
    "merge_metadata",
    "merge_prediction",
    "reconstruct",
    "reconstruct_directory",
]