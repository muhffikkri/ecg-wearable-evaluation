# Data architecture findings (verified against the real dataset)

All statements below were produced by inspecting the actual files in
`data/29-09-2026/Bryan/`. Nothing here is assumed.

## Verified source inventory

```text
data/29-09-2026/Bryan/
├── calibrated/   20 frames × 3 files (csv + json + npy) = 60 files
├── filtered/     20 frames × 3 files                  = 60 files
├── model_ready/  20 frames × 2 files (json + npy)     = 40 files
├── predictions/  20 frame predictions + latest_prediction.json
│                 + mqtt_publish_state.json            = 22 files
├── raw_adc/      82 files (bin + csv + json + npy per frame)
└── logs/         empty
```

Frame numbering is `000001 .. 000020` and is **complete and identical** in
`calibrated/` and `model_ready/`. There is no offset and no gap.

## Signal representation

| source | shape | dtype | fs | unit | channels |
|---|---|---|---|---|---|
| `calibrated/*.npy` | (2500, 3) | float32 | 250 Hz | mV | Lead I / II / III |
| `filtered/*.npy` | (2500, 3) | float32 | 250 Hz | mV | Lead I / II / III |
| `model_ready/*.npy` | (2500, 3) | float32 | 250 Hz | mV | Lead I / II / III |
| `*.jsonl` `ecg.samples` | 2500 rows × 3 cols | float | 250 Hz | mV | Lead I / II / III |

`ch3_derived = true` and `ch3_max_error_mV = 0.0`: Lead III is computed as
`Lead II − Lead I`, so it carries no independent information.

## Provenance chain (measured, not inferred)

`model_ready/*.json -> source_file` points at
`.../filtered/frame_000020_mv.npy`, and
`predictions/frame_000001_prediction.json -> source_file` points at
`.../model_ready/frame_000001_input.npy`.

Array comparison confirms the chain numerically:

```text
frame  max|calibrated − model_ready|   max|calibrated − filtered|   max|filtered − model_ready|
1      0.0125                          0.0125                        0.0000
5      0.0144                          0.0144                        0.0000
20     0.0148                          0.0148                        0.0000
```

So the real pipeline is:

```text
raw_adc → calibrated → filtered (50 Hz notch) → model_ready → predictions
```

`model_ready` is the *filtered* array, not the calibrated one. The ~0.01 mV
difference from `calibrated` is exactly the on-device 50 Hz notch recorded in
`source_processing.filters`.

**Consequence:** the earlier `calibrated ↔ model_ready` "conversion" framing
was wrong on two counts. They are not two interchangeable encodings of one
value, and neither is derived from the other by a reformatting. They are two
stages of one acquisition, one sample apart in time.

## Timestamps

| frame | calibrated `created_at_utc` | model_ready `created_at` |
|---|---|---|
| 1 | 2026-09-28T13:27:03.844Z | 2026-09-28T13:27:04.053Z |
| 20 | 2026-09-28T13:30:13 | 2026-09-28T13:30:14 |

model_ready is written ~0.21 s after its calibrated frame, i.e. the same
acquisition, not a different one. Timestamps are therefore a *valid*
corroborating signal for pairing, but they are second-resolution and the
frame id is exact, so **frame_id is the primary key and timestamps are the
verification signal**.

## Session identity

`calibrated/*.json -> source_metadata.session_id = session_28092026_202645`,
consistent across all 20 frames, and the same session id appears in the
absolute paths of every downstream file.

## JSONL target schema

`templates/json-web.jsonl` is the source of truth. The fields that can be
populated from the Raspberry Pi sources:

| JSONL field | source |
|---|---|
| `message_id` | derived: `{device}-raw_{session}-frame_{id}` |
| `device_id` | `source_metadata.device_id` |
| `session_id` | `source_metadata.session_id` |
| `patient_id` | **absent in raw** — left null, never invented |
| `frame_id` | frame number, zero-padded |
| `created_at` | calibrated `created_at_utc` |
| `sampling_rate_hz` | 250 from both sources (cross-checked) |
| `duration_s` | `duration_seconds` from both sources |
| `validation` | model_ready `validation` block, which is richer |
| `ecg.samples` | model_ready array (the array the AI actually consumed) |
| `prediction` | `predictions/frame_*_prediction.json` |
| `system` / `network` / `stress_test` | **absent in raw** — left null, never invented |

Absent fields are written as `null` with a note in the manifest rather than
filled with plausible-looking values. The MQTT recording in
`templates/json-mqtt.json` shows these device-telemetry blocks are produced by
the web uploader, not by the acquisition pipeline, so the Raspberry Pi source
genuinely cannot supply them.

## Field classification (replaces the old lossless/derived/irrecoverable split)

**Reconstructible from the raw dataset:** signal array, shape, dtype, unit,
channel order, sampling rate, duration, frame id, timestamps, calibration
method and scale, lead statistics, validation warnings, prediction.

**Not present in the raw dataset at all** (so a generated JSONL cannot carry
them, and does not invent them): `patient_id`, `system` CPU/memory/uptime
telemetry, `stress_test` frame counter, `network` MQTT latency/RSSI.

**Only in one source and not recoverable from the other:** the calibrated-side
`source_metadata` block (parser statistics, packet-loss counts, SHA-256
checksums, absolute producer paths) survives only in the reconstructed frame's
provenance record, never in the JSONL payload.

    ## Generated JSONL schema conformance

`processed/<subject>_<session>.jsonl` reproduces `templates/json-web.jsonl`.
Verified against the 20 Bryan frames (see git history for the run):

* 20 records, all 2500x3 `float32`, 250 Hz, 10 s, unique frame ids.
* `ecg.samples` is bit-identical to `calibrated/frame_*_mv.npy`
  (`np.array_equal` -> True).
* `prediction` matches the template block exactly: `status`, `label`,
  `confidence_percent`, `probabilities`, `threshold`, `latency_ms`, `runtime`.
* `validate_record` reports 0 errors on all 20 records.
* Top-level key set equals the existing session JSONL exactly.

### Fields the Pi recording never stored

Three template fields are absent, and this is a property of the SOURCE DATA,
not a gap in the reconstruction. The raw tree was searched: no
`patient_id`/`patient`/`name`, no `wifi`, no `cpu`, no `mqtt` anywhere in
`calibrated/`, `filtered/`, `model_ready/` or `predictions/`. Those are web
dashboard telemetry produced by the MQTT device at upload time, which the
standalone Pi recording does not have.

They are listed in the manifest under `unavailable_fields` rather than being
invented or filled with nulls.
