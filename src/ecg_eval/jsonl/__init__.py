"""JSONL Generation: canonical frames to the website's dataset format.

IDEA-REVISED.md section 7 defines this stage, and makes
``templates/json-web.jsonl`` the source of truth for the schema. The order
matters and is the order of the modules:

* :mod:`jsonl_builder` maps a canonical frame onto the template fields and
  records where each field came from.
* :mod:`schema_validator` checks the result against the template before it is
  allowed to reach disk.
* :mod:`jsonl_writer` emits the file plus a manifest, refusing to write into the
  recording folder it just read.
"""

from .jsonl_builder import UNAVAILABLE_IN_RAW, BuiltRecord, build_record
from .jsonl_writer import (
    MANIFEST_SUFFIX,
    WriteResult,
    assert_output_outside_source,
    build_manifest,
    write_jsonl,
)
from .schema_validator import (
    ECG_FORMAT,
    OPTIONAL_FIELDS,
    REQUIRED_FIELDS,
    TEMPLATE_FIELDS,
    TEMPLATE_PATH,
    VALIDATION_STATUSES,
    RecordValidation,
    ValidationIssue,
    expand_elisions,
    load_template,
    template_fields,
    validate_record,
)

__all__ = [
    "ECG_FORMAT",
    "MANIFEST_SUFFIX",
    "OPTIONAL_FIELDS",
    "REQUIRED_FIELDS",
    "TEMPLATE_FIELDS",
    "TEMPLATE_PATH",
    "UNAVAILABLE_IN_RAW",
    "VALIDATION_STATUSES",
    "BuiltRecord",
    "RecordValidation",
    "ValidationIssue",
    "WriteResult",
    "assert_output_outside_source",
    "build_manifest",
    "build_record",
    "expand_elisions",
    "load_template",
    "template_fields",
    "validate_record",
    "write_jsonl",
]