# ECGRHYTHMIA Wearable ECG Evaluation

## 1. Project Overview

Build a Python-based local web application named:

`ecg-wearable-evaluation`

The application is intended to process and evaluate ECG recordings collected from the ECGRHYTHMIA wearable prototype.

The project is **not** an ECG diagnostic application and is **not** intended to diagnose arrhythmia.

The primary objective is to evaluate the quality and acquisition performance of ECG signals obtained from the wearable under three static body positions:

1. Supine / lying
2. Sitting
3. Standing

The experimental dataset consists of approximately:

- 12 subjects
- 3 body positions per subject
- 1 minute recording per position
- 10-second analysis frames
- expected total: 216 ECG frames

The signal quality evaluation must reproduce the methodology of:

> Zhao, Z., & Zhang, Y. (2018). SQI Quality Evaluation Mechanism of Single-Lead ECG Signal Based on Simple Heuristic Fusion and Fuzzy Comprehensive Evaluation. Frontiers in Physiology, 9, 727.

The four primary SQIs are:

1. qSQI — R-peak detection matching
2. pSQI — QRS power spectrum distribution
3. kSQI — kurtosis
4. basSQI — baseline relative power

These are combined using fuzzy comprehensive evaluation into:

- Excellent (E)
- Barely Acceptable (B)
- Unacceptable (U)

The application must also provide an annotation interface so that the user can manually mark which frames correspond to:

- lying
- sitting
- standing

The application should finally provide both quantitative analysis and human-readable interpretation.

---

# 2. Important Methodological Principle

The implementation must clearly distinguish between:

### A. Reproduction

The primary evaluation should reproduce the formulation described by Zhao & Zhang (2018):

```text
ECG frame
    ↓
qSQI
pSQI
kSQI
basSQI
    ↓
membership functions
    ↓
fuzzy matrix
    ↓
weight vector
    ↓
bounded fuzzy synthesis
    ↓
weighted membership decision
    ↓
Excellent / Barely Acceptable / Unacceptable
```

Do NOT silently replace thresholds, membership functions, weights, or decision rules with arbitrary values.

All parameters originating from Zhao & Zhang must be explicitly documented in the code/configuration.

### B. Adaptation

If the original paper requires an algorithm or assumption that differs from the ECGRHYTHMIA dataset, implement it as an explicit adaptation.

Do not call an adapted implementation an exact reproduction.

For example:

```text
Zhao-Zhang qSQI
    Hilbert + adaptive threshold
              vs
    Wavelet R-peak detector
```

is the primary reproduction.

A separate experimental mode may later support:

```text
Pan-Tompkins
      vs
Wavelet detector
```

but this must be labeled as an adaptation/alternative, not the original Zhao & Zhang implementation.

---

# 3. Technology Direction

Build the application using Python.

Recommended stack:

- Python 3.11+
- Streamlit for the web UI
- NumPy
- Pandas
- SciPy
- PyWavelets
- Plotly
- JSON / JSONL standard library
- pathlib
- dataclasses or Pydantic where appropriate

The application is intended to run locally.

The initial data source is the repository's:

```text
data/
```

directory.

Do not introduce a database unless it becomes necessary.

Annotation state and generated analysis results should be stored as files inside the repository.

---

# 4. Current Data Situation

The current data directory contains recordings organized approximately as:

```text
data/
└── <date>/
    ├── <subject_01>.jsonl
    ├── <subject_02>.jsonl
    ├── ...
    └── raw/
```

There are currently:

- 11 JSONL recordings
- 1 raw recording directory caused by an MQTT recording mistake

The raw recording belongs to a subject and must be converted into the same logical representation as the JSONL recordings.

The final experiment is expected to contain 12 subjects.

Do not assume that the 11 JSONL filenames are sufficient to identify the subjects.

Use metadata and/or a configurable subject manifest where appropriate.

---

# 5. JSONL Data

Each JSONL file represents one subject/session.

Each line is one JSON object representing one frame.

The canonical JSON structure for JSONL entries is documented in:

```text
templates/json-web.jsonl
```

The implementation MUST inspect and use this template as the source of truth for the JSONL schema.

Do not invent fields that contradict the template.

The ingestion layer should:

1. discover JSONL files under `data/`
2. parse them line-by-line
3. validate each JSON object
4. retain the original metadata
5. assign an internal stable frame identifier
6. expose the data to the web UI
7. report malformed records without silently discarding them

Malformed lines must be logged and surfaced in a validation report.

---

# 6. Raw Recording Structure

The problematic raw recording is currently located under a folder named:

```text
Bryan/
```

Its structure contains:

```text
Bryan/
├── calibrated/
├── filtered/
├── logs/
├── model_ready/
├── predictions/
└── raw_adc/
```

For wearable evaluation, the primary source is:

```text
calibrated/
```

Each calibrated frame is represented by three files:

```text
frame_000020_mv.csv
frame_000020_mv.json
frame_000020_mv.npy
```

The corresponding model-ready representation is:

```text
model_ready/
├── frame_000001_input.json
└── frame_000001_input.npy
```

One model-ready frame is represented by two files.

---

# 7. Data Conversion Requirement

Implement a dedicated conversion subsystem.

The application must support conversion in both directions at the structural level:

```text
calibrated
    ↕
model_ready
```

However, the conversion must distinguish between:

### Lossless fields

Fields that can be copied or deterministically transformed without information loss.

### Derived fields

Fields generated from another representation.

### Irrecoverable fields

Information that exists in calibrated data but does not exist in model_ready data cannot be reconstructed magically.

For example:

```text
calibrated:
    CSV
    JSON
    NPY

model_ready:
    JSON
    NPY
```

If the calibrated CSV contains information not stored in model_ready JSON/NPY, reverse conversion may generate a valid CSV from the NPY but cannot guarantee that it is byte-for-byte identical to the original CSV.

Therefore:

> "Round-trip" means semantic/structural round-trip, not necessarily byte-identical round-trip.

---

# 8. Conversion Design

Create a conversion module with explicit operations:

```text
calibrated → model_ready
model_ready → calibrated
```

The converter must:

1. discover frame IDs
2. pair related files
3. validate matching frame numbers
4. validate NPY shape
5. validate JSON metadata
6. preserve provenance
7. report missing files
8. avoid overwriting source data by default
9. support dry-run mode
10. generate a conversion manifest

Example:

```text
frame_000020_mv.csv
frame_000020_mv.json
frame_000020_mv.npy

        ↓

frame_000020_input.json
frame_000020_input.npy
```

Frame numbering must be preserved whenever possible.

Do NOT silently rename frame numbers unless a mapping is explicitly recorded.

---

# 9. Canonical Internal Representation

To prevent the web application from depending directly on either JSONL or raw folder structures, create a canonical internal frame representation.

Conceptually:

```text
ECGFrame
├── subject_id
├── session_id
├── source_file
├── source_format
├── frame_id
├── timestamp
├── sampling_rate
├── signal
├── signal_shape
├── metadata
└── provenance
```

The exact fields must be adapted to the real JSONL template and raw JSON files.

The canonical representation should be used by:

- visualization
- annotation
- segmentation
- preprocessing
- R-peak detection
- SQI calculation
- fuzzy evaluation
- result export

This is critical.

The analysis layer should NOT care whether the original data came from:

```text
JSONL
```

or:

```text
calibrated/
```

---

# 10. Data Ingestion Pipeline

The application should implement:

```text
data/
   ↓
Discovery
   ↓
Format detection
   ↓
JSONL reader / raw-folder reader
   ↓
Validation
   ↓
Canonical ECGFrame
   ↓
Web application
```

The ingestion process should produce a dataset inventory.

Example:

```text
Subject | Source | Frames | Sampling Rate | Status
----------------------------------------------------
S01     | JSONL  | ...    | ...            | OK
S02     | JSONL  | ...    | ...            | OK
...
S12     | RAW    | ...    | ...            | OK
```

The UI should display this inventory before analysis.

---

# 11. Web Application Navigation

The application should have at least these major tabs/pages:

```text
1. Dataset
2. Annotation
3. Analysis
4. Interpretation
5. Data Conversion
```

The user workflow should be:

```text
Dataset
   ↓
inspect recordings
   ↓
Annotation
   ↓
assign body-position labels
   ↓
Analysis
   ↓
calculate SQIs
   ↓
fuzzy evaluation
   ↓
Interpretation
```

Data conversion should be accessible independently because it is a preprocessing/maintenance operation.

---

# 12. Dataset Tab

The first page must allow the user to inspect the available data before performing any annotation.

Display:

- dates
- subjects/sessions
- source format
- number of frames
- frame duration if available
- sampling rate
- available metadata
- validation status

The user should be able to select:

```text
Date
→ Subject
→ Frame
```

and visualize the ECG waveform.

---

# 13. ECG Visualization

For every selected frame, display:

### Raw ECG

The original signal exactly as acquired.

### Processed ECG

Only display processed ECG when preprocessing has been explicitly executed.

Do not replace raw data with processed data.

The UI should clearly label:

```text
RAW
```

and:

```text
PREPROCESSED
```

The visualization should support:

- zoom
- pan
- reset
- x-axis in seconds
- y-axis in mV where available
- grid where useful
- R-peak overlays when available

---

# 14. Annotation Tab

This is one of the most important pages.

The user already has manual notes indicating:

- when a subject started lying
- when a subject started sitting
- when a subject started standing

These annotations must be entered into the application.

The user must be able to annotate either by:

### Time

Example:

```text
start_time = ...
```

or:

### Frame

Example:

```text
frame_start = ...
frame_end = ...
```

The application should support both when the source data allows it.

---

# 15. Body Position Labels

Allowed labels:

```text
SUPINE
SITTING
STANDING
```

Optional system label:

```text
UNLABELED
```

Do not force every frame into a position before annotation is complete.

This is important because transitions between positions may contain movement artifacts.

---

# 16. Transition Handling

The application must support transition frames.

Example:

```text
SUPINE
SUPINE
SUPINE
TRANSITION
TRANSITION
SITTING
SITTING
...
```

However, for the primary 10-second evaluation frames, only frames explicitly assigned to one of:

```text
SUPINE
SITTING
STANDING
```

should enter the main position comparison.

Transition frames should either:

1. be excluded, or
2. be stored as `TRANSITION`

but must not silently become one of the three static conditions.

This prevents movement artifacts during posture changes from being incorrectly interpreted as stationary-position quality.

---

# 17. Annotation Data Format

Store annotations separately from raw ECG data.

Recommended:

```text
annotations/
├── annotations.json
└── annotation_manifest.csv
```

Conceptual structure:

```json
{
  "subject_id": "S01",
  "session_id": "...",
  "segments": [
    {
      "label": "SUPINE",
      "start_frame": 10,
      "end_frame": 15
    },
    {
      "label": "SITTING",
      "start_frame": 16,
      "end_frame": 21
    },
    {
      "label": "STANDING",
      "start_frame": 22,
      "end_frame": 27
    }
  ]
}
```

The actual implementation should support timestamps as well.

Do not hard-code the assumption that every subject has exactly 6 frames per position.

The expected value is 6, but the actual data must determine what is available.

---

# 18. Segmentation

The evaluation unit is a 10-second ECG frame.

If a source frame already represents exactly 10 seconds, use it directly.

If the source contains continuous ECG data, segment it into 10-second windows.

Do not unnecessarily resample or reconstruct a signal that is already available as the intended analysis frame.

The application should record:

```text
segment_id
subject_id
position
start_time
end_time
sample_count
sampling_rate
```

---

# 19. Primary SQI Pipeline

For every annotated 10-second ECG segment:

```text
ECG frame
    ↓
validation
    ↓
preprocessing required by SQI
    ↓
R-peak detection
    ↓
qSQI
    ↓
PSD
    ↓
pSQI
    ↓
signal statistics
    ↓
kSQI
    ↓
baseline power
    ↓
basSQI
    ↓
fuzzy comprehensive evaluation
    ↓
quality class
```

The implementation must keep intermediate results.

Do not create a black-box function that only returns:

```text
"Excellent"
```

The user must be able to inspect every component.

---

# 20. qSQI

## Primary objective

Measure the agreement between two independent R-peak detection methods.

Zhao & Zhang define qSQI using the matching degree between R-peaks detected by two different algorithms.

The original paper uses:

1. Hilbert transform + dynamic adaptive threshold
2. wavelet-transform-based R-wave detection

Therefore, the primary reproduction must implement these two detector pipelines.

Conceptually:

```text
ECG
 ├──> Hilbert + adaptive threshold ──> R peaks A
 │
 └──> Wavelet detector ──────────────> R peaks B
                                      │
                                      ↓
                                peak matching
                                      ↓
                                    qSQI
```

The matching tolerance must be implemented exactly according to the Zhao & Zhang formulation.

Do not substitute an arbitrary tolerance without documenting it.

---

# 21. Pan-Tompkins

The existing project already uses Pan-Tompkins.

Pan-Tompkins may be implemented as an additional R-peak detector.

Possible comparison:

```text
Pan-Tompkins
      vs
Wavelet detector
```

This should be available as:

```text
Experimental / Adapted qSQI
```

It must NOT replace the primary Zhao–Zhang qSQI in the main reproduction.

The UI should clearly indicate:

```text
qSQI method:
[ Zhao-Zhang reproduction ]
[ Pan-Tompkins adaptation ]
```

If implementation complexity is too high for the first version, implement Zhao–Zhang reproduction first and add Pan-Tompkins as a second scope.

---

# 22. Other R-Peak Detector Alternatives

The code architecture should allow additional detectors later.

Potential alternatives include:

- Pan-Tompkins
- Hamilton
- Christov
- Engzee
- Two Average
- Stationary Wavelet Transform

These are established classical QRS/R-peak detector families.

Do not implement all of them initially.

Use a detector interface such as:

```text
RPeakDetector
    ├── ZhaoHilbertDetector
    ├── ZhaoWaveletDetector
    ├── PanTompkinsDetector
    ├── HamiltonDetector
    └── ...
```

This will make future comparison experiments possible without rewriting qSQI.

---

# 23. qSQI Matching

For each frame:

```text
R_peaks_A
R_peaks_B
```

must be matched using the method defined by the reference.

The system must record:

```text
n_peaks_A
n_peaks_B
n_matched
qSQI
```

Also retain:

```text
peak_indices_A
peak_indices_B
matched_pairs
unmatched_A
unmatched_B
```

This enables visual debugging.

The web UI should allow:

```text
ECG waveform
    +
R peaks from detector A
    +
R peaks from detector B
```

to be displayed together.

---

# 24. pSQI

pSQI evaluates the spectral distribution associated with the QRS complex.

Implement the Zhao & Zhang formulation exactly.

The implementation must:

1. calculate the power spectral density
2. calculate power in the QRS-related band
3. calculate total ECG power according to the paper's defined denominator
4. calculate pSQI
5. store the intermediate powers

Store:

```text
qrs_band_power
total_power
pSQI
```

Do not silently substitute a different frequency range.

The frequency limits must be defined in a configuration/reference module.

---

# 25. kSQI

Calculate kurtosis according to the mathematical definition used in Zhao & Zhang.

Store:

```text
mean
std
kSQI
```

The implementation must clearly specify whether the calculation uses:

```text
Fisher kurtosis
```

or:

```text
Pearson kurtosis
```

and must use the convention required by the reference implementation.

Do not allow library defaults to silently determine the scientific definition.

---

# 26. basSQI

Calculate baseline relative power according to the Zhao & Zhang formulation.

Store:

```text
baseline_band_power
total_power
basSQI
```

The frequency ranges must be defined explicitly according to the paper.

Do not invent a new baseline definition merely because another ECG preprocessing convention is common.

---

# 27. Fuzzy Comprehensive Evaluation

Use the four factors:

```text
U = {
    qSQI,
    pSQI,
    kSQI,
    basSQI
}
```

Rating levels:

```text
V = {
    Excellent,
    Barely Acceptable,
    Unacceptable
}
```

Implement the following stages explicitly:

```text
SQI values
    ↓
membership functions
    ↓
evaluation matrix R
    ↓
weight vector W
    ↓
fuzzy synthesis
    ↓
decision
```

The implementation must reproduce the membership-function families from Zhao & Zhang:

- Cauchy distribution for qSQI
- trapezoidal distribution for pSQI
- rectangular distribution for kSQI
- Cauchy distribution for basSQI

Do not replace them with simple linear thresholds.

---

# 28. Fuzzy Weight Vector

The weight vector must follow Zhao & Zhang.

Store it in a configuration/reference module rather than hard-coding it deep inside analysis functions.

Example conceptual configuration:

```python
FUZZY_WEIGHTS = {
    "qSQI": ...,
    "pSQI": ...,
    "kSQI": ...,
    "basSQI": ...
}
```

The values must be verified directly against the original paper before implementation.

The four weights must satisfy:

```text
sum(weights) = 1
```

Add an automated validation test for this.

---

# 29. Fuzzy Decision

The final output must preserve the fuzzy membership vector:

```text
E = ...
B = ...
U = ...
```

and final class:

```text
Excellent
Barely Acceptable
Unacceptable
```

Do not discard the membership values.

Example:

```text
Excellent: 0.81
Barely Acceptable: 0.16
Unacceptable: 0.03

Final: Excellent
```

This is more informative than storing only the final label.

---

# 30. Analysis Tab

The Analysis tab should contain several levels.

## Level 1 — Individual frame

Show:

```text
Subject
Position
Frame
Waveform
qSQI
pSQI
kSQI
basSQI
Fuzzy E
Fuzzy B
Fuzzy U
Final quality
```

## Level 2 — Subject + position

For example:

```text
S01
 ├── Supine
 ├── Sitting
 └── Standing
```

Display:

- mean qSQI
- median qSQI
- mean pSQI
- median pSQI
- mean kSQI
- median kSQI
- mean basSQI
- median basSQI
- quality distribution

## Level 3 — Overall

Aggregate across all subjects.

---

# 31. Expected Dataset Structure

The application should target approximately:

```text
12 subjects
×
3 positions
×
6 frames
=
216 frames
```

But this is an expectation, not a hard-coded requirement.

The application must calculate the actual number.

Display:

```text
Expected frames: 216
Actual annotated frames: ...
Missing: ...
Excluded transitions: ...
Invalid frames: ...
```

This is important for transparent reporting.

---

# 32. Position Comparison

The primary comparison is:

```text
SUPINE
vs
SITTING
vs
STANDING
```

For each position calculate:

```text
n_frames
mean qSQI
median qSQI
SD qSQI

mean pSQI
median pSQI
SD pSQI

mean kSQI
median kSQI
SD kSQI

mean basSQI
median basSQI
SD basSQI
```

Also calculate:

```text
Excellent %
Barely Acceptable %
Unacceptable %
```

---

# 33. Subject-Level Aggregation

Do not only aggregate all 72 frames from a position.

Also calculate subject-level statistics.

For each subject:

```text
S01:
    Supine mean
    Sitting mean
    Standing mean
```

...

```text
S12:
    Supine mean
    Sitting mean
    Standing mean
```

This allows analysis of inter-subject variability.

The application should provide both:

```text
frame-level
```

and:

```text
subject-level
```

results.

---

# 34. Statistical Analysis

The application should initially focus on descriptive statistics.

Recommended outputs:

- mean
- median
- standard deviation
- minimum
- maximum
- interquartile range

For comparing the three repeated conditions within the same subjects, provide an optional statistical analysis layer.

Potential test:

```text
Friedman test
```

for non-parametric repeated-measures comparison.

If parametric assumptions are explicitly checked and satisfied, repeated-measures ANOVA may be considered.

The application should not automatically declare a position "better" or "worse".

It should report:

```text
whether a statistically detectable difference exists
```

and provide the underlying values.

---

# 35. Quality Distribution

Create visualizations such as:

### Bar chart

```text
Excellent
Barely Acceptable
Unacceptable
```

for each body position.

### Box plots

For:

```text
qSQI
pSQI
kSQI
basSQI
```

across:

```text
Supine
Sitting
Standing
```

### Per-subject heatmap

Rows:

```text
S01 ... S12
```

Columns:

```text
Supine
Sitting
Standing
```

Cell:

```text
mean quality score / quality class
```

Avoid reducing the complete result to a single score.

---

# 36. Interpretation Tab

The interpretation page should generate a structured scientific interpretation.

It must NOT simply say:

```text
The wearable is good.
```

Instead it should identify:

1. overall signal quality
2. position-specific behavior
3. SQI-specific behavior
4. problematic frames
5. possible artifact characteristics
6. inter-subject variability
7. limitations

Example structure:

```text
Overall finding
----------------
Most evaluated frames were classified as ...

Position comparison
-------------------
Supine: ...
Sitting: ...
Standing: ...

SQI interpretation
-------------------
qSQI: ...
pSQI: ...
kSQI: ...
basSQI: ...

Potential artifacts
-------------------
Frames with low basSQI suggest ...
Frames with low qSQI suggest ...

Experimental limitation
-----------------------
...
```

---

# 37. Interpretation Rules

The interpretation engine should use the SQI meaning.

For example:

### Low qSQI

Interpret as:

```text
Possible difficulty in consistent R-peak detection.
Potential causes include waveform distortion, motion artifact,
or reduced QRS clarity.
```

### Low pSQI

Interpret as:

```text
The spectral energy distribution is less concentrated in the
QRS-related band, suggesting possible high-frequency interference
or spectral distortion.
```

### Low kSQI

Interpret as:

```text
The amplitude distribution is less characteristic of a clean ECG
and may indicate increased noise contribution.
```

### Low basSQI

Interpret as:

```text
The signal contains relatively greater low-frequency/baseline
variation, suggesting baseline wander or electrode-motion effects.
```

The interpretation engine must distinguish:

```text
measurement
```

from:

```text
possible cause
```

Do not state a cause as certain when the SQI only indicates a possible artifact.

---

# 38. Interpretation of Unacceptable Frames

For an `Unacceptable` frame, inspect the four SQIs.

Generate a diagnostic explanation such as:

```text
Unacceptable quality primarily associated with:
- low qSQI
- low pSQI
```

or:

```text
Unacceptable quality primarily associated with:
- low basSQI
- reduced kSQI
```

The interpretation should explain what those SQIs represent.

Do not diagnose a medical condition from poor signal quality.

---

# 39. Important Distinction

The final interpretation concerns:

```text
ECG SIGNAL QUALITY
```

not:

```text
PATIENT HEALTH
```

Do not infer:

- arrhythmia
- cardiac disease
- patient abnormality
- clinical diagnosis

from SQI results.

A poor SQI means the recording quality may be poor.

It does not mean the subject has a cardiac abnormality.

---

# 40. Result Storage

Analysis results should be saved separately from raw data.

Recommended:

```text
results/
├── frame_results.csv
├── subject_results.csv
├── position_results.csv
├── overall_results.json
├── plots/
└── reports/
```

Frame-level output should contain at minimum:

```text
subject_id
session_id
position
frame_id
start_time
end_time

qSQI
pSQI
kSQI
basSQI

fuzzy_excellent
fuzzy_barely_acceptable
fuzzy_unacceptable

quality_class
```

Also store useful intermediate values where possible.

---

# 41. Reproducibility

Every analysis run should record:

```text
analysis timestamp
software version
configuration version
SQI method
R-peak detector
sampling rate
preprocessing configuration
fuzzy configuration
dataset source
```

A result must be reproducible from:

```text
raw data
+
annotation
+
configuration
```

Do not rely on manually modified variables inside notebooks.

---

# 42. Configuration

Create a central configuration system.

Conceptually:

```text
configs/
├── default.yaml
├── zhao_zhang.yaml
└── pan_tompkins_adapted.yaml
```

The configuration should define:

- sampling rate
- frame duration
- preprocessing parameters
- qSQI detector
- matching tolerance
- pSQI frequency ranges
- basSQI frequency ranges
- fuzzy membership parameters
- fuzzy weights

The exact Zhao & Zhang values must be verified against the paper before being committed.

---

# 43. Data Validation

Before analysis, run validation.

Check:

### Dataset

- expected subjects
- number of files
- missing files
- malformed JSON
- duplicated frames

### Signal

- NPY readable
- expected dimensions
- numeric values
- NaN
- infinite values
- empty frames

### Sampling

- sampling rate available
- sampling rate consistent
- expected sample count

### Annotation

- position assigned
- start/end valid
- overlapping annotations
- invalid frame references

Display a validation report before analysis.

---

# 44. No Silent Data Modification

The application must NEVER silently:

- overwrite raw data
- overwrite source JSONL
- modify original NPY
- modify original CSV
- change timestamps
- change frame IDs

All transformed data must be written to:

```text
processed/
```

or:

```text
results/
```

---

# 45. Recommended Repository Structure

Create the repository approximately as:

```text
ecg-wearable-evaluation/
│
├── README.md
├── IDEA.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
│
├── app.py
│
├── data/
│   └── <date>/
│       ├── *.jsonl
│       └── raw/
│
├── templates/
│   └── json-web.jsonl
│
├── annotations/
│   ├── annotations.json
│   └── annotation_manifest.csv
│
├── configs/
│   ├── default.yaml
│   └── zhao_zhang.yaml
│
├── src/
│   └── ecg_eval/
│       ├── __init__.py
│       │
│       ├── ingestion/
│       │   ├── jsonl_reader.py
│       │   ├── raw_reader.py
│       │   └── validator.py
│       │
│       ├── conversion/
│       │   ├── calibrated_to_model_ready.py
│       │   ├── model_ready_to_calibrated.py
│       │   └── manifest.py
│       │
│       ├── models/
│       │   ├── frame.py
│       │   ├── annotation.py
│       │   └── result.py
│       │
│       ├── annotation/
│       │   ├── manager.py
│       │   └── storage.py
│       │
│       ├── preprocessing/
│       │   └── pipeline.py
│       │
│       ├── detectors/
│       │   ├── base.py
│       │   ├── zhao_hilbert.py
│       │   ├── zhao_wavelet.py
│       │   └── pan_tompkins.py
│       │
│       ├── sqi/
│       │   ├── q_sqi.py
│       │   ├── p_sqi.py
│       │   ├── k_sqi.py
│       │   ├── bas_sqi.py
│       │   └── fuzzy.py
│       │
│       ├── analysis/
│       │   ├── frame_analysis.py
│       │   ├── subject_analysis.py
│       │   ├── position_analysis.py
│       │   └── statistics.py
│       │
│       ├── interpretation/
│       │   └── engine.py
│       │
│       └── visualization/
│           ├── ecg_plot.py
│           ├── sqi_plot.py
│           └── summary_plot.py
│
├── tests/
│   ├── test_ingestion.py
│   ├── test_conversion.py
│   ├── test_q_sqi.py
│   ├── test_p_sqi.py
│   ├── test_k_sqi.py
│   ├── test_bas_sqi.py
│   └── test_fuzzy.py
│
├── processed/
│
└── results/
```

The exact structure may be simplified if the agent finds a cleaner architecture, but the separation of concerns must remain.

---

# 46. UI Design

Use a simple scientific dashboard.

## Sidebar

Display:

```text
Dataset
Subject
Session
Analysis configuration
```

## Main navigation

```text
Dataset
Annotation
Analysis
Interpretation
Conversion
```

Do not create unnecessary UI complexity.

The application is primarily a research/evaluation tool.

---

# 47. Dataset Page

The first screen should immediately answer:

```text
What data do I have?
```

Show:

```text
Dates
Subjects
Frames
Sampling rates
Source formats
Validation status
```

Then allow waveform inspection.

---

# 48. Annotation Page

The annotation workflow should be:

```text
Select subject
       ↓
Show frame timeline
       ↓
Select range
       ↓
Assign:
    SUPINE
    SITTING
    STANDING
    TRANSITION
       ↓
Save annotation
```

The UI should visually distinguish annotated ranges.

The annotation must be editable.

---

# 49. Analysis Page

Provide:

```text
[Run Analysis]
```

Only annotated frames should enter the main evaluation.

After running:

```text
216 expected
XXX analyzed
XXX excluded
XXX invalid
```

Display progress while processing.

Do not rerun expensive analysis unnecessarily if results already exist and the configuration/data has not changed.

---

# 50. Analysis Cache

Use deterministic result caching.

A frame result should depend on:

```text
frame content hash
+
analysis configuration
+
annotation
```

If none changes, the application may reuse the existing result.

This is important because repeated browser interactions should not recompute all ECG frames unnecessarily.

---

# 51. Interpretation Page

Show:

### Overall

```text
Total analyzed frames
Excellent %
Barely Acceptable %
Unacceptable %
```

### By position

```text
Supine
Sitting
Standing
```

### SQI

```text
qSQI
pSQI
kSQI
basSQI
```

### Problematic frames

Allow the user to click a problematic frame and inspect:

```text
ECG
R-peaks
PSD
SQIs
fuzzy membership
```

---

# 52. Interpretation Generation

The interpretation engine should produce structured Markdown/text that can later be copied into the research report.

For example:

```text
### Signal Quality

Of the X analyzed ECG segments, ...
```

But every statement must be generated from actual calculated results.

Never hard-code conclusions.

Do not generate phrases such as:

```text
the wearable performed well
```

unless the interpretation is backed by an explicitly defined criterion.

Prefer:

```text
X% of evaluated segments were classified as Excellent,
while Y% were Barely Acceptable and Z% were Unacceptable.
```

---

# 53. Scientific Reporting

The application should eventually support exporting:

```text
CSV
JSON
PNG
Markdown
```

The Markdown report should contain:

1. dataset summary
2. annotation summary
3. analysis configuration
4. SQI methodology
5. frame-level summary
6. position-level summary
7. statistical analysis
8. interpretation
9. limitations

---

# 54. Important Experimental Limitation

The Zhao & Zhang fuzzy classifier was developed and evaluated using PhysioNet databases.

This project applies the method to ECG collected from ECGRHYTHMIA.

Therefore:

> The fuzzy evaluation should be interpreted as an application of an established signal-quality methodology to the wearable dataset, not as a newly clinically validated classifier for ECGRHYTHMIA.

The report should explicitly state this distinction.

---

# 55. Expected Scientific Interpretation

The final analysis should answer:

### Question 1

Can ECGRHYTHMIA continuously acquire ECG signals?

Use:

- number of valid frames
- missing data
- frame completeness

### Question 2

Are ECG waveforms of acceptable signal quality?

Use:

- qSQI
- pSQI
- kSQI
- basSQI
- fuzzy quality class

### Question 3

Does body position affect signal quality?

Compare:

```text
Supine
Sitting
Standing
```

### Question 4

Which signal-quality characteristics are most affected?

Inspect:

```text
qSQI
pSQI
kSQI
basSQI
```

### Question 5

Are low-quality frames associated with particular artifact characteristics?

Use the SQI-specific interpretation.

---

# 56. Do Not Overinterpret

The following conclusions must NOT be generated automatically:

```text
standing is clinically worse
```

```text
subject has arrhythmia
```

```text
wearable is clinically validated
```

```text
wearable is equivalent to hospital ECG
```

The current experiment evaluates **wearable ECG signal quality**, not clinical equivalence or diagnostic performance.

---

# 57. Development Sequence

The agent MUST develop the project incrementally.

## Scope 1 — Project foundation

Implement:

- repository structure
- Python environment
- Streamlit entry point
- configuration
- logging
- basic navigation

Then commit.

Commit example:

```text
chore: initialize ECG wearable evaluation project
```

---

## Scope 2 — Dataset ingestion

Implement:

- JSONL discovery
- JSONL validation
- raw folder discovery
- canonical ECG frame
- dataset inventory
- waveform display

Then commit.

```text
feat: add ECG dataset ingestion and visualization
```

---

## Scope 3 — Raw conversion

Implement:

- calibrated → model_ready
- model_ready → calibrated
- conversion manifest
- validation
- dry-run
- provenance

Then commit.

```text
feat: add calibrated and model-ready conversion pipeline
```

---

## Scope 4 — Annotation

Implement:

- annotation UI
- frame selection
- SUPINE
- SITTING
- STANDING
- TRANSITION
- save/load annotations

Then commit.

```text
feat: add body-position annotation workflow
```

---

## Scope 5 — Segmentation and preprocessing

Implement:

- 10-second segmentation
- signal validation
- preprocessing interface
- processed waveform visualization

Then commit.

```text
feat: add ECG segmentation and preprocessing pipeline
```

---

## Scope 6 — R-peak detection

Implement:

- detector interface
- Zhao Hilbert detector
- Zhao wavelet detector
- qSQI
- visual R-peak comparison

Then commit.

```text
feat: implement Zhao-Zhang qSQI evaluation
```

---

## Scope 7 — Remaining SQIs

Implement:

- pSQI
- kSQI
- basSQI

Then commit.

```text
feat: implement ECG signal quality indexes
```

---

## Scope 8 — Fuzzy evaluation

Implement:

- membership functions
- evaluation matrix
- weight vector
- bounded synthesis
- final quality classification

Then commit.

```text
feat: implement Zhao-Zhang fuzzy ECG quality evaluation
```

---

## Scope 9 — Analysis dashboard

Implement:

- frame-level results
- subject-level results
- position-level results
- plots
- statistics

Then commit.

```text
feat: add wearable ECG analysis dashboard
```

---

## Scope 10 — Interpretation

Implement:

- SQI interpretation
- quality distribution interpretation
- position comparison
- problematic-frame explanation
- Markdown report generation

Then commit.

```text
feat: add automated ECG quality interpretation
```

---

## Scope 11 — Validation and testing

Implement:

- unit tests
- conversion tests
- SQI tests
- fuzzy tests
- dataset validation
- reproducibility checks

Then commit.

```text
test: validate ECG evaluation pipeline
```

---

## Scope 12 — Documentation

Update:

- README
- methodology
- data format
- usage
- reproducibility
- limitations

Then commit.

```text
docs: document ECG wearable evaluation workflow
```

---

# 58. Git Discipline

The agent MUST commit periodically.

A commit must be made after completing each meaningful scope/task.

Do not wait until the entire application is complete.

Commit messages should be:

```text
chore: ...
feat: ...
fix: ...
test: ...
docs: ...
refactor: ...
```

Each commit should represent a coherent scope.

Avoid commits such as:

```text
update
changes
final
fix stuff
```

---

# 59. Git Author Requirement

IMPORTANT:

All commits must use the repository's existing user Git identity.

The agent MUST NOT add:

```text
Co-authored-by:
```

to commit messages.

Do not add any co-author.

Do not add AI attribution to Git commit trailers.

---

# 60. Development Safety

Before modifying data:

- inspect current files
- inspect schemas
- inspect representative JSONL entries
- inspect calibrated files
- inspect model_ready files

Do not assume that:

```text
frame_000020_mv.npy
```

has exactly the same semantics as:

```text
frame_000020_input.npy
```

until the files have been inspected.

The agent must first determine:

- shape
- dtype
- sampling rate
- channel structure
- units
- metadata
- scaling
- normalization
- timestamp representation

---

# 61. Data Conversion Safety

The raw source must remain untouched.

Never perform in-place conversion inside:

```text
data/
```

unless explicitly requested.

Generated conversion output should be placed in:

```text
processed/
```

or another clearly separated output directory.

---

# 62. Scientific Reproducibility Requirement

Every scientific parameter used by the implementation must be discoverable.

Do not bury values inside functions.

For example:

```python
QRS_LOW_FREQ = ...
QRS_HIGH_FREQ = ...
BASELINE_LOW_FREQ = ...
BASELINE_HIGH_FREQ = ...
```

must be configuration/reference parameters.

Similarly:

```python
FUZZY_WEIGHTS
```

must be explicit.

---

# 63. Primary Success Criterion

The application is successful when the user can perform this complete workflow without manually editing scripts:

```text
1. Start web application
        ↓
2. Inspect all recordings
        ↓
3. Inspect waveform
        ↓
4. Annotate body positions
        ↓
5. Save annotations
        ↓
6. Run analysis
        ↓
7. Calculate 4 SQIs
        ↓
8. Apply Zhao-Zhang fuzzy evaluation
        ↓
9. Inspect frame results
        ↓
10. Compare supine/sitting/standing
        ↓
11. Inspect problematic frames
        ↓
12. Generate scientific interpretation
        ↓
13. Export results
```

---

# 64. Final Design Principle

Keep the project modular.

The architecture should allow:

```text
data source
     ↓
canonical frame
     ↓
annotation
     ↓
preprocessing
     ↓
R-peak detector
     ↓
SQI
     ↓
fuzzy evaluation
     ↓
statistical analysis
     ↓
interpretation
```

Each layer must be independently testable.

The web interface should be a presentation/orchestration layer, not the place where scientific calculations are implemented.

Scientific calculations belong in `src/ecg_eval/`.

The final application should therefore remain usable both through:

```text
Streamlit UI
```

and programmatically through Python modules/tests.

---

# 65. Immediate First Task

Before implementing any SQI algorithm, the agent MUST first inspect the actual dataset.

Specifically inspect:

```text
data/
templates/json-web.jsonl
data/<date>/*.jsonl
data/<date>/raw/Bryan/calibrated/
data/<date>/raw/Bryan/model_ready/
```

Determine the actual schemas and signal representation.

Then implement the dataset inventory and waveform viewer.

Do NOT begin with qSQI/fuzzy evaluation.

The correct development order is:

```text
UNDERSTAND DATA
      ↓
NORMALIZE DATA REPRESENTATION
      ↓
ANNOTATION
      ↓
VERIFY SEGMENTATION
      ↓
IMPLEMENT SQI
      ↓
FUZZY EVALUATION
      ↓
STATISTICAL ANALYSIS
      ↓
INTERPRETATION
```

This ordering is mandatory because an incorrect assumption about the raw ECG representation would invalidate every downstream SQI calculation.
