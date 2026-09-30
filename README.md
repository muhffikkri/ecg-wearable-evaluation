# ECG Wearable Evaluation

Signal-quality evaluation of ECG recorded by the ECGRHYTHMIA wearable prototype
under three static body positions, following the SQI methodology of Zhao & Zhang
(2018).

**This project evaluates ECG signal quality only.** It is not a diagnostic tool
and must not be used to infer arrhythmia or any clinical condition. The
interpretation layer enforces this: diagnostic and clinical-performance claims
are blocked and rewritten into scope statements.

> The current architecture is specified in [`IDEA-REVISED.md`](IDEA-REVISED.md),
> which supersedes [`IDEA.md`](IDEA.md) where the two differ. Sections 7, 8 and
> 11 of `IDEA.md` described a two-way conversion subsystem that has been deleted.

## Run it

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tabs

| Tab | Purpose |
| --- | --- |
| Dataset | Ingest and inspect frames from the raw tree or JSONL |
| Annotation | Label body position per frame |
| Analysis | Segmentation, the four SQIs, the fuzzy rating, comparisons |
| Interpretation | Prose findings, constrained to signal quality |
| Reconstruction | Assemble canonical frames from a Raspberry Pi recording and generate the website dataset |

## How a raw recording becomes a dataset

The Raspberry Pi folders are **not** two formats of one recording to convert
between. They are complementary evidence about the same frames:

```text
calibrated/    processed mV signal + calibration metadata
filtered/      baseline-corrected signal
model_ready/   the device's own analysis input + its verdict
predictions/   the device's model output
```

The reconstructor links them using the pointers the recording actually stores —
`measurement_id` between `calibrated` and `filtered`, `source_file` between
`filtered` and `model_ready`, and `source_file` between `model_ready` and
`predictions`. Frames are matched on **recorded evidence only**. A frame that
does not resolve is reported `UNRESOLVED`, never paired on a shared frame number,
because a misaligned dataset that looks complete is the worst possible outcome.

Generated JSONL follows `templates/json-web.jsonl` field for field. Where the
template asks for something the raw tree does not contain (`patient_id`,
`system`, `network`), the field is omitted and listed as unavailable rather than
filled in: a generated `system.cpu_usage_percent` would be indistinguishable from
a measurement the device never made.

The recording under `data/` is read-only. The writer refuses an output path
inside the source tree.

## Building a dataset from the command line

The Reconstruction tab drives the same pipeline the CLI does:

```bash
# Preview: validates and reports, writes nothing.
python scripts/build_jsonl_dataset.py data/29-09-2026/Bryan --dry-run

# Build one recording.
python scripts/build_jsonl_dataset.py data/29-09-2026/Bryan --output processed

# Build every recording under data/.
python scripts/build_jsonl_dataset.py data --output processed

# Machine-readable summary.
python scripts/build_jsonl_dataset.py data --output processed --json
```

Exit codes: `0` success, `1` validation failed, `2` unusable arguments. Pointing
`--output` inside the recording is refused with exit `2`.

Each run writes a `.jsonl` dataset plus a `.manifest.json` recording where every
field came from and which fields the recording simply does not have.

## Scientific status

This is **not** a verified reproduction, and the application says so on every
run. All 37 scientific parameters in `configs/zhao_zhang.yaml` are marked
`pending_verification`, so `is_strict_reproduction` is false and results are
reported as a *structured* reproduction. Confirming those constants against the
published paper is the remaining work before the label can change.

Two further limitations are stated in the code where they apply:

- The R-peak detectors are unvalidated. The dataset carries no beat
  annotations, so the wavelet detector's under-detection dominates qSQI.
- kSQI and the band power ratios assume baseline-corrected input, and the leads
  carry DC offsets of roughly -2.7 to +11.8 mV. Baseline removal is an explicit
  adaptation, not part of the reference method.

## Layout

```text
app.py                 Streamlit entry point
app_ui/                presentation only, no scientific logic
scripts/               build_jsonl_dataset.py, the CLI for dataset generation
configs/               reference, adapted and default configurations
templates/             json-web.jsonl, the dataset schema
src/ecg_eval/
  ingestion/           readers, one per source format
  reconstruction/      mapping and canonical frame assembly
  models/              ECGFrame, results, annotation, mapping
  preprocessing/       opt-in signal conditioning
  detectors/           R-peak detectors
  sqi/                 qSQI, pSQI, kSQI, basSQI, fuzzy evaluation
  analysis/            pipeline, statistics, comparison, export
  interpretation/      guarded prose findings
  jsonl/               schema validation, builder, writer
data/                  recordings; read-only, never modified
```

## Tests

```bash
python -m pytest tests/ -q
```

## Reference

Zhao, Z., & Zhang, Y. (2018). SQI Quality Evaluation Mechanism of Single-Lead
ECG Signal Based on Simple Heuristic Fusion and Fuzzy Comprehensive Evaluation.
*Frontiers in Physiology*, 9, 727. <https://doi.org/10.3389/fphys.2018.00727>