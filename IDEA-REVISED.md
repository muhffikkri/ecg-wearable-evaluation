Saya ingin melakukan KOREKSI FUNDAMENTAL terhadap rancangan arsitektur data pada `IDEA.md`.

## 1. KOREKSI PEMAHAMAN DATA

Jangan lagi menganggap bahwa:

```text
calibrated → model_ready
model_ready → calibrated
```

adalah proses konversi.

Itu SALAH.

Folder `raw/` adalah **data mentah/hasil pemrosesan yang tersimpan di Raspberry Pi**. Di dalamnya terdapat beberapa subfolder yang merupakan sumber informasi berbeda:

```text
data/
└── <tanggal>/
    └── raw/
        └── Bryan/
            ├── raw_adc/
            ├── calibrated/
            ├── filtered/
            ├── logs/
            ├── model_ready/
            └── predictions/
```

Untuk kebutuhan evaluasi wearable ECG, sumber utama yang perlu digunakan adalah:

```text
calibrated/
model_ready/
```

Keduanya BUKAN dua format data yang harus dikonversi bolak-balik.

Keduanya harus dibaca dan informasinya DIGABUNGKAN untuk membentuk representasi satu frame ECG yang lengkap.

---

# 2. ARSITEKTUR DATA YANG BENAR

Gunakan konsep pipeline berikut:

```text
Raspberry Pi
    │
    ▼
raw/
└── Bryan/
    ├── raw_adc/
    ├── calibrated/
    ├── filtered/
    ├── model_ready/
    ├── predictions/
    └── logs/
    │
    ▼
Data Reconstruction / Integration
    │
    ├── calibrated metadata
    ├── calibrated ECG signal
    ├── model-ready input
    ├── prediction information
    ├── timestamps
    └── frame mapping
    │
    ▼
Canonical Frame Representation
    │
    ▼
JSONL Builder
    │
    ▼
templates/json-web.jsonl
    │
    ▼
Web Application
```

Jadi JSONL adalah **format data yang digunakan oleh website sebagai representasi/canonical dataset**, sedangkan folder `raw/` adalah sumber data asli dari Raspberry Pi.

---

# 3. STRUKTUR CALIBRATED

Contoh isi:

```text
calibrated/
├── frame_000020_mv.csv
├── frame_000020_mv.json
├── frame_000020_mv.npy
├── frame_000021_mv.csv
├── frame_000021_mv.json
├── frame_000021_mv.npy
└── ...
```

Setiap frame direpresentasikan oleh:

```text
frame_<ID>_mv.csv
frame_<ID>_mv.json
frame_<ID>_mv.npy
```

Agent WAJIB memeriksa isi aktual ketiga file tersebut sebelum menentukan schema internal.

Identifikasi:

- frame ID
- timestamp
- sampling rate
- jumlah channel
- channel names
- signal shape
- dtype
- unit
- calibration/scaling
- metadata lain
- hubungan antara CSV, JSON, dan NPY

Jangan membuat asumsi jika informasi tersebut belum diperiksa dari data aktual.

---

# 4. STRUKTUR MODEL_READY

Contoh:

```text
model_ready/
├── frame_000001_input.json
├── frame_000001_input.npy
├── frame_000002_input.json
├── frame_000002_input.npy
└── ...
```

Model-ready juga bukan target konversi dari calibrated.

Ia merupakan sumber informasi lain yang dihasilkan pipeline Raspberry Pi.

Agent harus memeriksa:

- frame ID
- timestamp jika tersedia
- shape input
- dtype
- jumlah channel
- sampling rate
- preprocessing/normalization
- scaling
- informasi metadata
- hubungan frame model-ready dengan frame calibrated

Jangan menganggap:

```text
frame_000001_input
```

pasti sama dengan:

```text
frame_000001_mv
```

atau bahwa ID keduanya memiliki offset yang sederhana.

Contoh aktual dapat berupa:

```text
calibrated:
frame_000020_mv.*

model_ready:
frame_000001_input.*
```

Hal ini mengindikasikan bahwa diperlukan mekanisme **frame mapping/reconstruction**.

---

# 5. FRAME RECONSTRUCTION

Buat konsep baru bernama:

```text
Frame Reconstruction
```

Tujuannya adalah menggabungkan berbagai sumber data Raspberry Pi menjadi satu representasi frame yang dapat digunakan website.

Contoh:

```text
calibrated frame
        │
        ├── ECG signal
        ├── timestamp
        ├── frame ID
        └── metadata
        │
        └──────────────┐
                       │
model_ready frame      │
        │              │
        ├── AI input   │
        ├── shape      │
        └── metadata   │
                       │
                       ▼
              Frame Reconstruction
                       │
                       ▼
                Canonical Frame
```

Jika prediction tersedia, prediction dapat menjadi sumber tambahan:

```text
predictions/
        │
        ▼
prediction metadata
        │
        ▼
Canonical Frame
```

Namun prediction TIDAK boleh menjadi syarat agar frame ECG dapat direkonstruksi.

---

# 6. CANONICAL FRAME

Buat internal representation yang bersifat independen terhadap struktur folder Raspberry Pi.

Contoh konseptual:

```json
{
  "subject_id": "...",
  "session_id": "...",
  "frame_id": "...",
  "source": {
    "raw_root": "...",
    "calibrated": "...",
    "model_ready": "..."
  },
  "timestamp": "...",
  "sampling_rate": 250,
  "signal": {
    "channels": [],
    "shape": [],
    "unit": "mV"
  },
  "model_input": {
    "available": true,
    "shape": []
  },
  "prediction": {
    "available": true,
    "label": null,
    "confidence": null
  },
  "metadata": {}
}
```

Ini hanya contoh konsep.

Agent HARUS menyesuaikan schema berdasarkan data aktual.

Canonical frame ini menjadi jembatan antara:

```text
Raspberry Pi raw structure
```

dan:

```text
website JSONL structure
```

---

# 7. JSONL GENERATION

Setelah frame berhasil direkonstruksi:

```text
Canonical Frame
       │
       ▼
JSONL Builder
       │
       ▼
JSONL Validator
       │
       ▼
JSONL Writer
       │
       ▼
<subject/session>.jsonl
```

Struktur JSONL HARUS mengikuti:

```text
templates/json-web.jsonl
```

File tersebut merupakan **source of truth untuk schema JSONL website**.

Jangan membuat schema JSONL baru jika template sudah menentukan strukturnya.

Agent harus:

1. membaca template;
2. memahami setiap field;
3. memetakan field dari canonical frame;
4. mencatat field yang berasal dari calibrated;
5. mencatat field yang berasal dari model_ready;
6. mencatat field yang berasal dari prediction/log apabila tersedia;
7. menangani field yang tidak tersedia;
8. melakukan validation sebelum JSONL ditulis.

---

# 8. JANGAN KEHILANGAN DATA ASLI

Pipeline harus bersifat non-destructive.

JANGAN:

- memindahkan file raw;
- mengubah file calibrated;
- mengubah file model_ready;
- menghapus source data;
- overwrite source data;
- menganggap JSONL sebagai pengganti raw dataset.

Struktur ideal:

```text
data/
├── <tanggal>/
│   ├── *.jsonl
│   │
│   └── raw/
│       └── Bryan/
│           ├── calibrated/
│           ├── model_ready/
│           ├── filtered/
│           ├── predictions/
│           ├── raw_adc/
│           └── logs/
```

JSONL merupakan hasil representasi untuk web, sedangkan raw tetap dipertahankan sebagai source of truth/audit source.

---

# 9. FITUR RECONSTRUCTION DI WEB

Web harus menyediakan fitur untuk:

### A. Scan Raw Dataset

User memilih atau aplikasi mendeteksi:

```text
data/<tanggal>/raw/<session>/
```

Kemudian aplikasi membaca:

```text
calibrated/
model_ready/
predictions/
logs/
```

sesuai kebutuhan.

---

### B. Inspect

Tampilkan informasi:

```text
Session
Subject
Jumlah calibrated frame
Jumlah model-ready frame
Jumlah prediction
Sampling rate
Channel
Frame range
Timestamp range
```

---

### C. Frame Mapping

Buat proses untuk menentukan hubungan:

```text
calibrated frame
        ↕
model-ready frame
```

Prioritaskan mapping berdasarkan metadata/timestamp jika tersedia.

Jika tidak tersedia, gunakan informasi lain yang dapat dibuktikan dari dataset.

JANGAN langsung membuat asumsi:

```text
calibrated frame 20 = model_ready frame 1
```

hanya karena nomor frame berbeda.

Jika mapping tidak dapat ditentukan dengan aman, tandai sebagai:

```text
UNRESOLVED
```

dan tampilkan kepada user.

---

### D. Reconstruction Preview

Sebelum menghasilkan JSONL, tampilkan preview:

```text
Canonical Frame
────────────────────────────
Frame ID       : ...
Timestamp      : ...
Sampling Rate  : ...
Channels       : ...
Signal Shape   : ...

Calibrated     : FOUND
Model Ready    : FOUND
Prediction     : FOUND / NOT FOUND
Mapping        : VERIFIED / UNRESOLVED
```

---

### E. Generate JSONL

Setelah validasi berhasil:

```text
Raw Raspberry Pi Data
        ↓
Reconstruction
        ↓
Validation
        ↓
JSONL
```

Hasilnya harus sesuai:

```text
templates/json-web.jsonl
```

---

# 10. JSONL YANG SUDAH ADA

Website juga harus mampu membaca JSONL yang sudah tersedia.

Jadi ada dua jalur data:

```text
                    ┌── Existing JSONL
                    │
Website Data Layer ─┤
                    │
                    └── Raw Raspberry Pi
                         ↓
                    Reconstruction
                         ↓
                       JSONL
```

Dengan demikian aplikasi dapat:

### Mode 1 — Existing JSONL

```text
data/<tanggal>/*.jsonl
        ↓
JSONL Reader
        ↓
Canonical Frame
        ↓
Annotation
        ↓
Analysis
```

### Mode 2 — Raw Raspberry Pi

```text
data/<tanggal>/raw/Bryan/
        ↓
Raw Reader
        ↓
Frame Reconstruction
        ↓
JSONL Generation
        ↓
Canonical Frame
        ↓
Annotation
        ↓
Analysis
```

Kedua mode akhirnya harus menggunakan **canonical frame yang sama**, sehingga pipeline analisis tidak perlu mengetahui apakah data berasal dari JSONL existing atau hasil reconstruction.

---

# 11. UPDATE STRUKTUR SOURCE CODE

Jangan gunakan:

```text
conversion/
├── calibrated_to_model_ready.py
└── model_ready_to_calibrated.py
```

Ganti dengan:

```text
src/ecg_eval/
│
├── ingestion/
│   ├── jsonl_reader.py
│   ├── raw_dataset_reader.py
│   ├── calibrated_reader.py
│   ├── model_ready_reader.py
│   ├── prediction_reader.py
│   ├── log_reader.py
│   └── validator.py
│
├── reconstruction/
│   ├── frame_reconstructor.py
│   ├── frame_mapper.py
│   ├── metadata_merger.py
│   └── provenance.py
│
├── jsonl/
│   ├── jsonl_builder.py
│   ├── jsonl_writer.py
│   └── schema_validator.py
│
├── models/
│   ├── frame.py
│   ├── source_frame.py
│   └── reconstruction.py
```

Nama file boleh disesuaikan selama konsep arsitekturnya tetap sama.

---

# 12. PROVENANCE

Setiap canonical frame sebaiknya menyimpan provenance.

Contoh:

```json
{
  "provenance": {
    "source_type": "raspberry_pi_raw",
    "calibrated_file": "...",
    "model_ready_file": "...",
    "prediction_file": "...",
    "mapping_method": "timestamp",
    "mapping_status": "verified"
  }
}
```

Tujuannya agar kita dapat mengetahui:

> JSONL frame ini sebenarnya berasal dari file Raspberry Pi yang mana?

Ini penting untuk reproducibility dan debugging.

---

# 13. ERROR HANDLING

Pipeline harus dapat menangani:

```text
calibrated ada
model_ready tidak ada
```

atau:

```text
model_ready ada
calibrated tidak ada
```

atau:

```text
timestamp tidak cocok
```

atau:

```text
frame mapping ambigu
```

atau:

```text
file corrupt
```

atau:

```text
metadata tidak lengkap
```

Jangan diam-diam membuang data.

Setiap masalah harus masuk ke validation report:

```text
VALID
WARNING
ERROR
UNRESOLVED
```

---

# 14. PERUBAHAN TERHADAP PIPELINE UTAMA WEB

Pipeline aplikasi yang benar sekarang:

```text
                    ┌───────────────────┐
                    │ Existing JSONL    │
                    └─────────┬─────────┘
                              │
                              ▼
                       JSONL Reader
                              │
                              │
                              ▼
                    ┌───────────────────┐
                    │ Canonical Frames  │
                    └─────────┬─────────┘
                              │
                              │
Raw Raspberry Pi              │
      │                       │
      ▼                       │
Raw Dataset Reader             │
      │                       │
      ▼                       │
Calibrated + Model Ready      │
      │                       │
      ▼                       │
Frame Reconstruction           │
      │                       │
      ▼                       │
JSONL Builder                  │
      │                       │
      ▼                       │
Canonical Frames ──────────────┘
                              │
                              ▼
                         Data Viewer
                              │
                              ▼
                         Annotation
                              │
                    ┌─────────┴─────────┐
                    │                   │
                 SUPINE              SITTING
                    │                   │
                    └─────────┬─────────┘
                              │
                           STANDING
                              │
                              ▼
                     10-second segments
                              │
                              ▼
                      ECG preprocessing
                              │
                              ▼
                       R-peak detection
                              │
                              ▼
                           qSQI
                              │
                    ┌─────────┼─────────┐
                    ▼         ▼         ▼
                  pSQI      kSQI     basSQI
                    │         │         │
                    └─────────┼─────────┘
                              ▼
                    Fuzzy Comprehensive
                         Evaluation
                              │
                              ▼
                     Analysis Dashboard
                              │
                              ▼
                       Interpretation
```

---

# 15. STRUKTUR TAB WEB

Pertahankan struktur utama:

### Tab 1 — Dataset

Fungsi:

- melihat daftar JSONL;
- melihat raw Raspberry Pi;
- melihat subject/session;
- melihat jumlah frame;
- melihat metadata;
- melihat status validasi;
- melihat provenance.

### Tab 2 — Reconstruction

Khusus raw Raspberry Pi:

```text
calibrated
model_ready
predictions
logs
```

→ mapping → reconstruction → JSONL.

### Tab 3 — Annotation

Untuk menandai:

```text
SUPINE
SITTING
STANDING
```

berdasarkan frame/time range.

### Tab 4 — Analysis

Menghitung:

```text
qSQI
pSQI
kSQI
basSQI
fuzzy evaluation
```

### Tab 5 — Interpretation

Menampilkan interpretasi hasil berdasarkan data aktual.

---

# 16. HAL YANG WAJIB DILAKUKAN AGENT SEBELUM CODING

Sebelum membuat implementation:

1. Inspect `data/`.
2. Inspect seluruh struktur `raw/Bryan/`.
3. Inspect beberapa file dari `calibrated/`.
4. Inspect beberapa file dari `model_ready/`.
5. Inspect `predictions/` dan `logs/` jika relevan.
6. Inspect beberapa JSONL existing.
7. Inspect `templates/json-web.jsonl`.
8. Bandingkan frame ID.
9. Bandingkan timestamp.
10. Bandingkan jumlah frame.
11. Tentukan bagaimana hubungan calibrated ↔ model_ready sebenarnya.
12. Dokumentasikan hasilnya.
13. Baru implementasikan `Frame Reconstruction`.

Jangan mengimplementasikan mapping berdasarkan asumsi sebelum data aktual diperiksa.

---

# 17. PERUBAHAN NAMA KONSEP DI IDEA.md

Gunakan terminologi:

**Raw Raspberry Pi Dataset**

→ sumber data asli.

**Data Reconstruction**

→ proses menggabungkan data dari berbagai folder.

**Canonical Frame**

→ representasi internal standar aplikasi.

**JSONL Generation**

→ pembentukan dataset yang sesuai schema website.

**JSONL Dataset**

→ format yang digunakan web untuk processing selanjutnya.

Hindari istilah:

> calibrated-to-model-ready conversion

karena secara konsep salah.

---

# 18. GIT WORKFLOW

Tetap gunakan development incremental.

Agent WAJIB melakukan commit setelah setiap scope selesai.

Contoh:

```text
chore: initialize ECG wearable evaluation project
feat: add raw Raspberry Pi dataset inspection
feat: add calibrated and model-ready readers
feat: add frame reconstruction pipeline
feat: add JSONL generation and validation
feat: add canonical frame data model
feat: add dataset viewer
feat: add body-position annotation workflow
feat: add ECG segmentation and preprocessing
feat: implement Zhao-Zhang qSQI evaluation
feat: implement ECG signal quality indexes
feat: implement fuzzy ECG quality evaluation
feat: add analysis dashboard
feat: add interpretation module
test: validate raw reconstruction pipeline
docs: document ECG dataset architecture
```

**WAJIB:**

- commit dilakukan secara berkala;
- satu commit merepresentasikan satu scope yang selesai;
- jangan menumpuk seluruh pekerjaan dalam satu commit;
- jangan menambahkan `Co-authored-by`;
- jangan menambahkan AI attribution;
- jangan mengubah Git identity pengguna.

---

# 19. TUJUAN AKHIR

Tujuan repository ini bukan membuat converter:

```text
calibrated ↔ model_ready
```

Tetapi membuat sistem:

```text
Raspberry Pi Raw Dataset
        │
        ├── calibrated
        ├── model_ready
        ├── predictions
        ├── logs
        └── other sources
                │
                ▼
       Data Reconstruction
                │
                ▼
         Canonical Frame
                │
                ▼
       JSONL sesuai template
                │
                ▼
          Web Application
                │
                ├── Dataset Viewer
                ├── Annotation
                ├── ECG Analysis
                └── Interpretation
```

Pastikan `IDEA.md` direvisi berdasarkan arsitektur ini dan jangan lagi mendeskripsikan `calibrated` dan `model_ready` sebagai dua format yang saling dikonversi.
