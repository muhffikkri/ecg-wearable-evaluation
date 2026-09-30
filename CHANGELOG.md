# Changelog

All notable changes to this project are recorded here.

## [0.1.0] - Unreleased

First implementation of the architecture specified in `IDEA-REVISED.md`.

### Added

- Split ingestion readers, one per source format, each normalising errors
  instead of failing silently.
- Canonical `ECGFrame` model shared by every reader and writer.
- Frame reconstruction that links `calibrated`, `filtered`, `model_ready` and
  `predictions` using only the pointers the recording stores, classifying each
  frame `VERIFIED`, `WARNING`, `UNRESOLVED` or `ERROR`.
- JSONL generation following `templates/json-web.jsonl` field for field, with a
  manifest sidecar recording per-field origins and unavailable fields.
- Configuration module with provenance-annotated parameters, inheritance chains
  and a scientific fingerprint used in the analysis cache key.
- Body-position annotation with atomic writes.
- Opt-in preprocessing that never mutates the caller's array.
- Three R-peak detectors behind one interface, feeding qSQI.
- pSQI, kSQI and basSQI, each storing its spectral intermediates.
- Fuzzy comprehensive evaluation with dispatching synthesis operators.
- Analysis pipeline, descriptive statistics and repeated-measures comparison.
- Interpretation layer that blocks clinical claims and rewrites them.
- Streamlit application with Dataset, Annotation, Analysis, Interpretation and
  Reconstruction tabs.
- `scripts/build_jsonl_dataset.py`, a CLI for building datasets without opening
  the app. Accepts a recording, a date folder or a `data` folder, previews with
  `--dry-run`, emits JSON summaries with `--json`, and exits 0/1/2 for
  success/validation failure/unusable arguments.

### Fixed

- The manifest listed `unavailable_fields` twice per frame. `validate_record`
  already reports every absent template field and the builder separately reports
  what the recording cannot supply, so a plain extend counted each absence twice.
  The two lists are now merged.

### Changed

- `IDEA-REVISED.md` supersedes `IDEA.md` where they differ. Sections 7, 8 and
  11 of `IDEA.md` described a two-way conversion subsystem; they now describe
  reconstruction.
- The run label is derived from configuration provenance rather than the detector
  name, so a run on the shipped configs reports a *structured* reproduction
  rather than claiming a verified one.
- Membership values are reported as the configured synthesis operator computes
  them. Under `bounded_max_product` they do not sum to 1 and are not presented as
  shares.

### Removed

- `ecg_eval/conversion` and its test module, and the Conversion tab. The
  premise was wrong: the raw folders are complementary evidence about the same
  frames, not two representations of one recording.

### Known limitations

- No scientific parameter has been verified against the published paper, so no
  result is a verified reproduction.
- The R-peak detectors are unvalidated; the dataset has no beat annotations.
- Baseline removal is an explicit adaptation, not part of the reference method.