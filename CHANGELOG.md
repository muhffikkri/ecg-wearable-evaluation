# Changelog

All notable changes to this project are recorded here.

## [0.2.0] - Unreleased

Corrects the four signal-quality indices and re-anchors the fuzzy stage to the
values printed in Zhao & Zhang (2018), and adds the simple heuristic fusion of
the per-index acceptance criteria.

### Changed

- **Deteksi R-peak (qSQI)** now uses the article's Eq (1), `2N / (Na + Nb)`.
  Dividing by `min(Na, Nb)` -- the previous behaviour -- reports a perfect match
  whenever one detector's peaks are a subset of the other's, so a detector that
  missed half the beats still scored 1.0.
- **Distribusi Daya Spektral QRS (pSQI)** now integrates its denominator over
  5-40 Hz (Eq 3) instead of 0.5-40 Hz. Sub-5 Hz baseline wander therefore no
  longer dilutes the index, which is the separate job of the baseline relative
  power index.
- **Kurtosis Sinyal (kSQI)** now defaults to the fourth standardized moment
  (Pearson, Gaussian = 3) as in Eq (9), rather than excess kurtosis (Fisher,
  Gaussian = 0). Eq (10)'s threshold of 5 is stated on the nu4 scale.
- **Daya Relatif Baseline (basSQI)** now implements Eq (11) including its
  leading `1 - `, so a clean signal scores close to 1. The bare ratio inverted
  every acceptance band of Eq (12).
- Fuzzy membership functions replaced with the article's Eq (22)-(32), the
  weight vector with (0.4, 0.4, 0.1, 0.1), the synthesis operator with the
  bounded sum M(., +), and the decision with the weighted-membership score `v`
  of Eq (33) using the 1.50 / 2.40 thresholds of Eq (34).
- Membership families are now configured per rating level, because the article
  uses a different family for each level of the same index (qSQI is the
  increasing Cauchy half for Excellent, a centred one for Barely Acceptable and
  the decreasing half for Unacceptable).
- `Config.iter_provenance` walks annotated values inside lists, so the
  heart-rate bands of Eq (5) appear in the pending/verified report instead of
  sitting outside provenance because of their container.

### Added

- Per-index acceptance criteria (optimal / suspicious / unqualified) with the
  article's equations, including the heart-rate-dependent pSQI limits, and an
  explicit `undefined` verdict when a criterion does not apply to a frame.
- `src/ecg_eval/sqi/heuristic_fusion.py`: the simple heuristic fusion of
  Eq (13)-(16) for 2-5 indices, plus heart-rate estimation from the R peaks.
- `Acceptance` and `HeuristicFusionResult` result models, with acceptance levels,
  the fused class and the heart rate recorded on every frame.
- `acceptance_recap` / `acceptance_recap_table`: per-activity distribution of
  each index over the acceptance levels, and the fused class beside the fuzzy
  class.
- `docs/SQI_ZHAO_ZHANG.md`: the methodology, formulas and acceptance criteria
  written out from the article.
- `scripts/sqi_recap_report.py`: regenerates the per-activity recap tables,
  the acceptance decision table and the recap figures.
- `acceptance_summary`: the decisive acceptance category of every index per
  activity -- the category most frames of that activity landed in -- with the
  frame count and share behind it.
- `acceptance_decision_table` / `acceptance_coverage_table`: that verdict as a
  table with one column per index and one row per activity, plus how strong each
  verdict is.
- `acceptance_figure`: stacked bars of the acceptance categories per index and
  activity, and the recap figures written as self-contained interactive HTML.

### Notes

- Parameters the article does not print (the R-peak matching tolerance, the PSD
  estimator settings, whether kurtosis is computed before filtering) remain
  `pending_verification`, so runs are still reported as a *structured*
  reproduction rather than a strict one.
- On the recorded wearable sessions the R-peak agreement index is unqualified in
  every frame, which is what the corrected Eq (1) exposes; the recap report
  states it rather than hiding it behind a different denominator.

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
- `ingest()` no longer counts a generated dataset twice when it is kept inside
  `data/` beside the recording it came from. The raw tree is read first, and a
  JSONL file whose `session_id` the raw tree already provides is skipped and
  recorded in `dataset.derived_datasets`. Previously the JSONL reader derived the
  subject from the filename while the raw reader derived it from the folder, so
  the same 20 frames appeared as two sessions and inflated every count.

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