# Mekanisme Evaluasi Kualitas Sinyal ECG *Single-Lead* Berdasarkan Fusi Heuristik Sederhana dan Evaluasi Fuzzy Comprehensive

**Reproduksi metode:** Zhao, Z., & Zhang, Y. (2018). *SQI Quality Evaluation Mechanism of Single-Lead ECG Signal Based on Simple Heuristic Fusion and Fuzzy Comprehensive Evaluation.* Frontiers in Physiology, 9:727.

---

## 1. Ruang Lingkup dan Kedudukan Dokumen

Dokumen ini menjelaskan secara metodologis mekanisme penilaian kualitas sinyal elektrokardiogram (ECG) *single-lead* yang digunakan pada evaluasi perekaman ECG *wearable*. Obyek yang dinilai adalah sinyal ECG berdurasi 10 detik per bingkai (*frame*) dengan laju cuplik 250 Hz, yang direkam pada tiga posisi tubuh: **SUPINE (Berbaring)**, **SITTING (Duduk)**, dan **STANDING (Berdiri)**.

Metode yang diuraikan merupakan reproduksi dari mekanisme yang diusulkan Zhao & Zhang (2018). Mekanisme tersebut terdiri atas dua modul berturut-turut: (i) **kuantifikasi indeks kualitas sinyal** (*Signal Quality Index*, SQI) dan **fusi heuristik sederhana** untuk memperoleh kombinasi indeks terbaik, serta (ii) **evaluasi fuzzy comprehensive** untuk mengklasifikasikan kualitas sinyal secara lebih halus.

Dokumen ini bersifat metodologis. Dokumen ini **tidak** melaporkan hasil validasi pada data proyek ini, dan tidak mengklaim adanya pengukuran akurasi, sensitivitas, maupun spesifisitas pada data proyek. Bilamana angka kinerja artikel disebut, angka tersebut secara eksplisit merupakan hasil artikel pada basis data PhysioNet yang digunakan artikel itu sendiri.

### Catatan penamaan

Dalam dokumen ini setiap indeks ditulis dengan **nama deskriptif berbahasa Indonesia**. Simbol singkat yang dipakai artikel hanya disebut **satu kali** pada bagian yang memperkenalkan rumusnya, semata-mata untuk keperluan penelusuran (*traceability*) ke artikel asal. Pemetaannya adalah sebagai berikut.

| Nama deskriptif dalam dokumen ini | Simbol pada artikel |
|---|---|
| Deteksi R-peak | qSQI |
| Distribusi Daya Spektral QRS | pSQI |
| Kurtosis Sinyal | kSQI |
| Daya Relatif Baseline | basSQI |

Metode yang direproduksi menggunakan **empat** indeks di atas. Artikel juga mendefinisikan indeks kelima, yaitu **variabilitas interval R-R**; indeks tersebut sengaja tidak digunakan (lihat Bagian 9, Batasan).

---

## 2. Alur Metodologi

Alur pengolahan dari sinyal mentah sampai kelas kualitas adalah sebagai berikut.

```text
Akseptasi SQI
      |
      v
Kombinasi terbaik indeks (4 metrik)
      |
      v
Ekstraksi vektor fitur U = {u1, u2, u3, u4}
      |
      v
Evaluasi Fuzzy Comprehensive
      |
      v
Kelas kualitas (E / B / U)
```

Penjelasan tiap tahap:

1. **Akseptasi SQI.** Artikel menghitung sekumpulan kandidat indeks kualitas sinyal, menilai kinerja klasifikasi tiap indeks secara individual, lalu mengeliminasi indeks yang memiliki akurasi (Acc) di bawah 75%. Tahap ini berfungsi menyaring indeks yang tidak informatif sebelum penggabungan.
2. **Penentuan kombinasi terbaik indeks.** Artikel mengevaluasi *seluruh* kemungkinan kombinasi indeks yang tersisa (pasangan, triplet, dan seterusnya) melalui validasi silang 10 lipat yang diulang 10 kali pada basis data artikel, kemudian membandingkan akurasi (Acc), sensitivitas (Se), dan spesifisitas (Sp). Perbandingan ini menghasilkan kombinasi terbaik.
3. **Ekstraksi vektor fitur U = {u1, u2, u3, u4}.** Artikel memilih kombinasi empat indeks sebagai kombinasi terbaik untuk tahap fuzzy, yaitu

   ```text
   U = { u1, u2, u3, u4 }
     = { Deteksi R-peak,
         Distribusi Daya Spektral QRS,
         Kurtosis Sinyal,
         Daya Relatif Baseline }
   ```

   Artikel mencatat bahwa penambahan indeks kelima (variabilitas interval R-R) tidak memperbaiki, bahkan menurunkan, Acc, Se, dan Sp dibandingkan kombinasi empat indeks. Karena itu kombinasi empat indeks dinilai lebih masuk akal dan dipilih sebagai vektor fitur untuk tahap evaluasi fuzzy. Keempat indeks tersebut mengukur karakteristik yang berbeda: kemampuan deteksi gelombang R, kualitas gelombang QRS pada ranah frekuensi, sifat Gaussiaan sinyal, dan kandungan daya baseline.
4. **Evaluasi Fuzzy Comprehensive.** Keempat nilai indeks dipetakan menjadi matriks keanggotaan melalui fungsi keanggotaan (Cauchy, trapesium, dan persegi), dibobot, disintesis dengan operator berbatas, lalu diputuskan menjadi satu kelas kualitas (Bagian 5).
5. **Kelas kualitas.** Keluaran akhir adalah salah satu dari tiga kelas: **E (Excellent)**, **B (Barely acceptable)**, atau **U (Unacceptable)**.

Artikel melaporkan bahwa pada basis data PhysioNet miliknya (D1 dan D2) evaluasi fuzzy comprehensive memberikan akurasi yang lebih tinggi daripada fusi heuristik sederhana dengan jumlah indeks yang sama. Angka kinerja tersebut adalah **hasil artikel pada basis data artikel**, bukan hasil pengukuran pada data proyek ini.

---

## 3. Definisi Indeks dan Kriteria Penerimaan

Setiap indeks di bawah ini memuat: rumus sebagai persamaan tampilan (*display equation*), nomor persamaan dari artikel, definisi setiap simbol, dan tabel kriteria penerimaan.

### 3.1 Deteksi R-peak

Indeks ini mengukur kemampuan pengenalan sinyal ECG melalui kesesuaian hasil deteksi puncak R oleh dua algoritma yang berbeda. Algoritma 1 menggunakan transformasi Hilbert dengan ambang adaptif dinamis, sedangkan Algoritma 2 menggunakan transformasi *wavelet* (deteksi titik singular melalui modulus maksimum koefisien *wavelet*). Karena setiap algoritma deteksi memiliki keterbatasan dan dapat menghasilkan deteksi palsu, kesesuaian kedua hasil deteksi dipakai sebagai ukuran kualitas sinyal.

Rumus (persamaan 1 pada artikel):

```text
                          2N
Deteksi R-peak  =  ----------------
                       Na + Nb
```

Definisi simbol:

- `N` — jumlah puncak R yang **cocok** (*matched*) antara kedua algoritma.
- `Na` — jumlah puncak R yang dideteksi Algoritma 1 (Hilbert + ambang adaptif dinamis).
- `Nb` — jumlah puncak R yang dideteksi Algoritma 2 (transformasi *wavelet*).

(dalam artikel disebut qSQI)

Nilai indeks berada pada rentang 0 sampai 1, atau dinyatakan dalam persen 0-100%. Pencocokan dilakukan **satu-ke-satu** di dalam jendela toleransi yang **simetris** terhadap setiap puncak; pada implementasi ini jendela toleransi tersebut adalah **150 ms**. Secara matematis bentuk di atas adalah ukuran kesesuaian bergaya **Dice** `2N / (Na + Nb)`.

Kriteria penerimaan (persamaan 2 pada artikel):

| Kondisi | Terima |
|---|---|
| optimal | nilai > 90% |
| suspicious | nilai 60% - 90% (inklusif) |
| unqualified | nilai < 60% |

### 3.2 Distribusi Daya Spektral QRS

Indeks ini mengukur kualitas gelombang QRS pada ranah frekuensi. Satu siklus denyut jantung terutama tersusun atas gelombang P, kompleks QRS, gelombang T, dan vektor penting lainnya; kompleks QRS mengakumulasi sekitar 99% energi sinyal ECG dan merupakan bagian yang paling stabil. Energi gelombang QRS terpusat pada pita frekuensi yang berpusat di sekitar **10 Hz dengan lebar 10 Hz**.

Rumus (persamaan 3 pada artikel):

```text
                              integral f=5 Hz sampai f=15 Hz  P(f) df
Distribusi Daya Spektral QRS = --------------------------------------
                              integral f=5 Hz sampai f=40 Hz  P(f) df
```

Definisi simbol:

- `P(f)` — kerapatan spektral daya (*power spectral density*) sinyal ECG.
- Pembilang — energi gelombang QRS, yaitu daya pada pita 5-15 Hz.
- Penyebut — energi keseluruhan sinyal dalam pita analisis, yaitu daya pada pita **5-40 Hz**.

(dalam artikel disebut pSQI)

Perlu ditegaskan bahwa pita penyebut menurut artikel adalah **5-40 Hz**, **bukan** 0.5-40 Hz. Jika terdapat interferensi elektromiogram (EMG), komponen frekuensi tinggi meningkat sehingga nilai indeks ini menurun.

Kriteria penerimaan (persamaan 4 dan 5 pada artikel). Batas bawah `l1`, batas atas `l2`, dan batas bawah pita *suspicious* `l3` bergantung pada denyut jantung:

| Detak jantung | l1 | l2 | l3 |
|---|---|---|---|
| 60-130 bpm | 0.5 | 0.8 | 0.4 |
| 130-160 bpm | 0.4 | 0.7 | 0.3 |

| Kondisi | Terima |
|---|---|
| optimal | nilai di dalam [l1, l2] |
| suspicious | nilai di dalam [l3, l1) |
| unqualified | nilai > l2 atau nilai < l3 |

Dua catatan penting:

1. Artikel mengkalibrasi batas-batas ini **hanya untuk denyut jantung 60-160 bpm**. Untuk denyut jantung di luar rentang tersebut kriteria ini **tidak dapat diterapkan**, sehingga implementasi harus melaporkannya sebagai **"tidak dapat diterapkan"** dan tidak boleh menebak tingkatannya.
2. Berbeda dari tiga indeks lain yang bersifat "semakin tinggi semakin baik", indeks ini bersifat menyerupai *band-pass*: nilai yang **melebihi** `l2` juga termasuk **unqualified**.

### 3.3 Kurtosis Sinyal

Indeks ini mengukur sifat Gaussiaan sinyal dan dipakai untuk menilai efektivitas peredaman tiga jenis gangguan: interferensi frekuensi daya, pergeseran baseline, dan derau acak. Artikel menyebutkan bahwa kemiringan (*skewness*) kurang kokoh terhadap derau dibandingkan kurtosis, sehingga hanya kurtosis yang dipakai pada tahap deteksi derau.

Rumus (persamaan 9 pada artikel):

```text
                                     E{ (x - mu_x)^4 }
Kurtosis Sinyal  =  mu4  =  ------------------------------
                                        sigma^4
```

Definisi simbol:

- `x` — sinyal yang dievaluasi.
- `mu_x` — rata-rata sinyal.
- `sigma` — simpangan baku sinyal.
- `E{ }` — nilai harapan (*expected value*).

(dalam artikel disebut kSQI)

Nilai ini adalah **momen terstandardisasi keempat** (konvensi **Pearson**), sehingga sinyal berdistribusi Gaussian bernilai **3**, **bukan 0**. Hal ini ditegaskan secara eksplisit karena konvensi *excess kurtosis* (Fisher) menempatkan Gaussian pada nilai 0; penggunaan konvensi Fisher akan menggeser **setiap** ambang pada indeks ini sebesar 3 dan karena itu tidak dipakai.

Kriteria penerimaan (persamaan 10 pada artikel). Kriteria ini bersifat **biner**; artikel **tidak** mendefinisikan pita *suspicious* untuk indeks ini:

| Kondisi | Terima |
|---|---|
| optimal | nilai > 5 |
| unqualified | nilai <= 5 |

Alasan yang dikemukakan artikel:

1. Untuk ECG normal berbentuk sinus yang standar dan bebas derau, nilai kurtosis adalah **> 5**.
2. Jika terdapat interferensi frekuensi daya, pergeseran baseline, atau derau acak berdistribusi Gaussian, nilainya menjadi **< 5**.
3. Jika terdapat interferensi EMG, nilainya berada **di sekitar 5**.

Selanjutnya, penugasan keanggotaan pada persamaan (29) artikel konsisten dengan kriteria biner di atas: nilai **> 5** dipetakan ke baris *Excellent*, nilai **<= 5** dipetakan ke baris *Unacceptable*, dan derajat keanggotaan *Barely Acceptable* untuk indeks ini **identik nol**.

### 3.4 Daya Relatif Baseline

Indeks ini mengukur efektivitas peredaman pergeseran baseline. Pergeseran baseline sulit disaring, tetapi keberadaannya sangat memengaruhi penilaian dan identifikasi patologis lanjutan, sehingga peredamannya perlu dinilai secara kuantitatif.

Rumus (persamaan 11 pada artikel):

```text
                                      integral f=0 Hz sampai f=1 Hz  P(f) df
Daya Relatif Baseline  =  1  -  ---------------------------------------------
                                      integral f=0 Hz sampai f=40 Hz P(f) df
```

Definisi simbol sama seperti pada Distribusi Daya Spektral QRS: `P(f)` adalah kerapatan spektral daya sinyal.

(dalam artikel disebut basSQI)

Karena terdapat faktor pengurang berawalan "1 -", interpretasinya adalah:

- Nilai yang **mendekati 1** berarti pergeseran baseline kecil (kualitas baik).
- Nilai yang **rendah** berarti daya pada pita [0, 1 Hz] **abnormal tinggi** relatif terhadap daya pada pita [0, 40 Hz], yang kemungkinan besar disebabkan pergeseran baseline yang abnormal.

Contoh terukur yang diberikan artikel: sampel ECG berkualitas tinggi menghasilkan nilai **0.966**, sedangkan sampel berkualitas rendah menghasilkan nilai **0.5**. Kedua contoh tersebut diambil artikel dari Set-a basis data PhysioNet/CinC 2011.

Kriteria penerimaan (persamaan 12 pada artikel):

| Kondisi | Terima |
|---|---|
| optimal | nilai di dalam [0.95, 1] |
| suspicious | nilai di dalam [0.9, 0.95) |
| unqualified | nilai < 0.9 |

---

## 4. Fusi Heuristik Sederhana antar Indeks

Sebelum tahap fuzzy, artikel menggabungkan tingkat (*level*) tiap indeks secara logis. Setiap indeks diklasifikasikan ke salah satu dari tiga tingkat berikut, sesuai kriteria penerimaan pada Bagian 3:

- **optimal**
- **suspicious**
- **unqualified**

Misalkan `#optimal`, `#suspicious`, dan `#unqualified` menyatakan berturut-turut banyaknya indeks pada tingkat *optimal*, *suspicious*, dan *unqualified* di dalam kombinasi yang sedang dievaluasi. Keluaran fusi adalah tiga kelas kualitas:

- **E** — *Excellent*
- **B** — *Barely acceptable*
- **U** — *Unacceptable*

Aturan evaluasi dievaluasi **secara berurutan**: kondisi **E** diuji lebih dahulu, kemudian kondisi **U**, dan sisanya jatuh ke **B** sebagai *fallback*.

### 4.1 Kombinasi empat indeks (persamaan 15) — kombinasi yang digunakan

Kombinasi inilah yang dipakai oleh implementasi ini, sehingga diberi penekanan paling besar.

```text
Empat indeks {SQI1, SQI2, SQI3, SQI4}:

  Excellent (E):
      #optimal >= 3  dan  #unqualified = 0

  Unacceptable (U):
      #unqualified >= 3
      atau (#unqualified = 2 dan #suspicious >= 1)
      atau (#unqualified = 1 dan #suspicious = 3)

  Barely acceptable (B):
      selain kondisi di atas
```

### 4.2 Kombinasi lain (persamaan 13, 14, dan 16)

Aturan untuk jumlah indeks lain disajikan dalam bentuk ringkas berikut.

```text
Dua indeks {SQI1, SQI2}  (persamaan 13):

  E: #optimal = 2
  B: #suspicious = 2  atau  (#optimal = 1 dan #suspicious = 1)
  U: selain kondisi di atas

Tiga indeks {SQI1, SQI2, SQI3}  (persamaan 14):

  E: #optimal >= 2  dan  #unqualified = 0
  U: #unqualified >= 2  atau  (#unqualified = 1 dan #suspicious = 2)
  B: selain kondisi di atas

Lima indeks {SQI1, SQI2, SQI3, SQI4, SQI5}  (persamaan 16):

  E: #optimal >= 4  dan  #unqualified = 0
  U: #unqualified >= 4
     atau (#unqualified = 3 dan #suspicious >= 1)
     atau (#unqualified = 2 dan #suspicious = 2)
     atau (#unqualified = 1 dan #suspicious = 4)
  B: selain kondisi di atas
```

### 4.3 Catatan mengenai koefisien fusi

Artikel menyatakan secara eksplisit bahwa koefisien (jumlah `optimal`, `suspicious`, dan `unqualified`) yang muncul pada persamaan 13-16 bersifat **arbitrer** dan ditetapkan **secara empiris melalui coba-coba** (*trial and error*). Artikel juga menyatakan bahwa koefisien-koefisien tersebut *dapat* dioptimalkan, tetapi kemungkinan besar logikanya tidak optimal, sehingga pencarian menyeluruh atas semua kombinasi logika dan ambang **tidak** dilakukan. Dengan demikian, aturan fusi ini harus dipandang sebagai heuristik yang praktis, bukan hasil optimasi.

---

## 5. Evaluasi Fuzzy Comprehensive

Bagian ini menguraikan tahap kedua menurut artikel, yaitu evaluasi fuzzy comprehensive, dengan urutan penyajian sesuai persamaan (20) sampai (34) artikel.

### 5.1 Himpunan faktor U

```text
U = { u1, u2, u3, u4 }

  u1 = Deteksi R-peak
  u2 = Distribusi Daya Spektral QRS
  u3 = Kurtosis Sinyal
  u4 = Daya Relatif Baseline
```

Keempat faktor ini adalah vektor fitur hasil tahap fusi heuristik sederhana pada Bagian 4.1.

### 5.2 Himpunan peringkat V

```text
V = { E, B, U }
  = { Excellent, Barely Acceptable, Unacceptable }
```

Urutan ini penting: elemen pertama matriks keanggotaan selalu bersesuaian dengan *Excellent*, elemen kedua *Barely Acceptable*, dan elemen ketiga *Unacceptable*.

### 5.3 Matriks evaluasi R

Setiap faktor dievaluasi secara tunggal (*single factor evaluation*) terhadap ketiga peringkat, menghasilkan satu baris derajat keanggotaan. Keempat baris disusun menjadi matriks evaluasi fuzzy berukuran 4 faktor x 3 level:

```text
        [ r1 ]   [ r11  r12  r13 ]
    R = [ r2 ] = [ r21  r22  r23 ]
        [ r3 ]   [ r31  r32  r33 ]
        [ r4 ]   [ r41  r42  r43 ]
```

dengan `r_ij` menyatakan derajat keanggotaan faktor ke-i terhadap peringkat ke-j. Kolom 1 adalah *Excellent*, kolom 2 *Barely Acceptable*, dan kolom 3 *Unacceptable*.

### 5.4 Fungsi keanggotaan tiap faktor

Artikel memilih bentuk fungsi keanggotaan yang berbeda untuk tiap faktor, sesuai distribusi empiris nilai faktornya. Perlu diperhatikan skala yang dipakai artikel:

- **Deteksi R-peak** dievaluasi pada **skala 0-100** (nilai persentase `q`).
- **Daya Relatif Baseline** juga dievaluasi pada **skala 0-100** (nilai `b`).
- **Distribusi Daya Spektral QRS** dievaluasi pada **skala 0-1** (nilai `x`, yaitu nilai rasio pada persamaan 3).
- **Kurtosis Sinyal** dievaluasi pada **nilai mentahnya** (tanpa normalisasi), persis seperti hasil persamaan 9.

#### 5.4.1 Deteksi R-peak — distribusi Cauchy

Misalkan `q` adalah nilai persentase kecocokan deteksi R-peak, `q` di dalam [0, 100].

*Excellent*, persamaan (22):

```text
U_qH(q) =  0                                  , 0 <= q <= 80
        =  1 / ( 1 + [ 0.3 (q - 80) ]^2 )     , 80 < q < 90
        =  1                                  , 90 <= q <= 100
```

*Unacceptable*, persamaan (24):

```text
U_qJ(q) =  1                                  , q <= 55
        =  1 / ( 1 + ( (q - 55) / 5 )^2 )     , 55 <= q <= 100
```

*Barely Acceptable*, persamaan (25):

```text
U_qI(q) =  1 / ( 1 + ( (q - 75) / 7.5 )^2 )
```

#### 5.4.2 Distribusi Daya Spektral QRS — distribusi trapesium

Misalkan `x` adalah nilai rasio daya seperti pada persamaan 3, `x` di dalam [0, 1].

*Excellent*, persamaan (26):

```text
U_pH(x) =  0                , x <= 0.25
        =  10 (x - 0.25)    , 0.25 < x < 0.35
        =  1                , x >= 0.35
```

*Unacceptable*, persamaan (27):

```text
U_pJ(x) =  1                , x < 0.15
        =  10 (0.25 - x)    , 0.15 <= x <= 0.25
        =  0                , x > 0.25
```

*Barely Acceptable*, persamaan (28):

```text
U_pI(x) =  0                , x < 0.18
        =  25 (x - 0.18)    , 0.18 <= x < 0.22
        =  1                , 0.22 <= x < 0.28
        =  25 (0.32 - x)    , 0.28 <= x < 0.32
        =  0                , x >= 0.32
```

#### 5.4.3 Kurtosis Sinyal — distribusi persegi (*rectangular*)

Artikel memilih distribusi persegi sebagai fungsi keanggotaan karena kriteria penerimaannya bersifat biner. Persamaan (29):

```text
jika nilai > 5   maka  r3 = ( 1, 0, 0 )    -> Excellent
jika nilai <= 5  maka  r3 = ( 0, 0, 1 )    -> Unacceptable
```

Derajat keanggotaan *Barely Acceptable* untuk faktor ini **identik nol** pada kedua kasus; hasil ini konsisten dengan tidak adanya pita *suspicious* pada kriteria penerimaan indeks ini (Bagian 3.3).

#### 5.4.4 Daya Relatif Baseline — distribusi Cauchy

Misalkan `b` adalah nilai Daya Relatif Baseline pada skala 0-100, `b` di dalam [0, 100].

*Excellent*, persamaan (30):

```text
U_bH(b) =  0                                       , 0 <= b <= 90
        =  1 / ( 1 + [ 0.8718 (b - 90) ]^2 )       , 90 < b < 95
        =  1                                       , 95 <= b <= 100
```

*Unacceptable*, persamaan (31):

```text
U_bJ(b) =  1                                       , b <= 85
        =  1 / ( 1 + ( (b - 85) / 5 )^2 )          , 85 < b <= 100
```

*Barely Acceptable*, persamaan (32):

```text
U_bI(b) =  1 / ( 1 + ( (b - 92) / 2.5 )^2 )
```

### 5.5 Vektor bobot W

```text
W = ( w1, w2, w3, w4 ) = ( 0.4, 0.4, 0.1, 0.1 )

dengan  sum(W) = 1
```

Bobot ini dipilih artikel setelah membandingkan beberapa set bobot pada kedua basis datanya melalui 10 ulangan validasi silang 10 lipat. Artikel melaporkan bahwa rasio `(0.4, 0.4, 0.1, 0.1)` memberikan akurasi yang relatif tinggi dengan fluktuasi minimal, baik pada basis data D1 maupun D2. Bobot terbesar diberikan pada dua faktor pertama (Deteksi R-peak dan Distribusi Daya Spektral QRS), sedangkan dua faktor terakhir (Kurtosis Sinyal dan Daya Relatif Baseline) berbobot lebih kecil.

### 5.6 Sintesis fuzzy

Artikel memilih operator **M(., +)**, yaitu **operator berbatas** (*bounded operator*), di antara empat jenis operator sintesis fuzzy yang dibahasnya. Alasannya: dari sudut pandang komprehensif, penggunaan operator berbatas dan penjumlahan memastikan seluruh informasi yang disediakan vektor fuzzy `R` dimanfaatkan sepenuhnya.

Sintesis dinyatakan sebagai:

```text
S = W o R

s_j = min( 1 ,  sum_{i=1..4} ( w_i * r_ij ) )     untuk j = 1, 2, 3

S = ( s1, s2, s3 )
```

dengan `o` menyatakan operasi sintesis fuzzy, dan `s1`, `s2`, `s3` berturut-turut adalah derajat keanggotaan agregat terhadap *Excellent*, *Barely Acceptable*, dan *Unacceptable*.

### 5.7 Keputusan

Artikel menggunakan **prinsip derajat keanggotaan berbobot** (*principle of weighted membership degree*). Nilai peringkat diberi skor `v1 = 1` untuk *Excellent* (E), `v2 = 2` untuk *Barely Acceptable* (B), dan `v3 = 3` untuk *Unacceptable* (U).

Persamaan (33):

```text
                sum_{j=1..3} ( s_j^2 * j )
        v  =  -----------------------------
                sum_{j=1..3} ( s_j^2 )
```

dengan `j` adalah indeks peringkat (1 untuk E, 2 untuk B, 3 untuk U) dan `s_j` adalah derajat keanggotaan agregat hasil sintesis pada Bagian 5.6. Pembobotan dengan `s_j^2` membuat peringkat dengan derajat keanggotaan besar lebih dominan dalam menentukan `v`.

Klasifikasi akhir (persamaan 34):

| Kondisi | Kelas |
|---|---|
| v <= 1.50 | Excellent (E) |
| 1.50 < v < 2.40 | Barely Acceptable (B) |
| v >= 2.40 | Unacceptable (U) |

Kedua ambang tersebut, `v_th1 = 1.50` dan `v_th2 = 2.40`, diperoleh artikel dengan menvariasikan `v` pada kurva ROC basis data D2 dan memilih nilai yang memberi akurasi klasifikasi terbaik.

### 5.8 Tindakan lanjutan setelah penilaian

Artikel menetapkan tiga aturan tindak lanjut berdasarkan kelas yang diperoleh.

1. **Jika E.** Kualitas sinyal ECG baik. Sinyal dapat langsung digunakan untuk identifikasi, pemantauan keamanan, atau aplikasi lain.
2. **Jika U.** Periksa keempat indeks:
   - Jika **Kurtosis Sinyal** atau **Daya Relatif Baseline** berstatus *unqualified*, berarti terdapat artefak derau. Lakukan **denoising** terlebih dahulu, kemudian nilai ulang kualitas ECG.
   - Jika **Distribusi Daya Spektral QRS** atau **Deteksi R-peak** yang berstatus *unqualified*, maka **rekam ulang** ECG subjek.
3. **Jika B.** Lakukan **penilaian ulang** terhadap kualitas ECG. Jika hasil penilaian ulang adalah E, perlakukan sinyal seperti aturan 1. Jika tidak, perlakukan seperti aturan 2.

---

## 6. Tabel Rekapitulasi per Aktivitas

Keluaran laporan disusun sebagai tabel dengan satu baris per posisi tubuh dan satu kolom per indeks, dengan bentuk sebagai berikut.

| Aktivitas | Deteksi R-peak | Distribusi Daya Spektral QRS | Kurtosis Sinyal | Daya Relatif Baseline |
|---|---|---|---|---|
| Berbaring | ... | ... | ... | ... |
| Duduk | ... | ... | ... | ... |
| Berdiri | ... | ... | ... | ... |

Ketentuan pengisian tabel:

- Setiap sel adalah **nilai rata-rata (mean)** atas seluruh bingkai yang dianalisis pada aktivitas tersebut. Nilai tidak ditampilkan di sini karena bergantung pada data yang sedang diproses.
- Tabel yang sama juga dihasilkan dengan **median** dan **simpangan baku sampel** dengan derajat kebebasan `ddof = 1`.
- Bingkai yang oleh metode ini ditandai **tidak valid** dikeluarkan dari perhitungan rata-rata, dan **jumlah**nya dilaporkan secara terpisah, bukan disisipkan sebagai nilai ke dalam sel.
- Laporan juga membawa **kelas hasil fusi** per aktivitas (E, B, atau U) di samping tabel indeks di atas.

---

## 7. Ringkasan Parameter

### 7.1 Ambang kriteria penerimaan per indeks

| Indeks | Parameter | Nilai | Persamaan |
|---|---|---|---|
| Deteksi R-peak | batas *optimal* | > 90% | (2) |
| Deteksi R-peak | batas bawah *suspicious* | 60% | (2) |
| Deteksi R-peak | batas atas *suspicious* | 90% (inklusif) | (2) |
| Deteksi R-peak | jendela toleransi pencocokan puncak (implementasi) | 150 ms, simetris | (1) |
| Distribusi Daya Spektral QRS | pita energi QRS (pembilang) | 5-15 Hz | (3) |
| Distribusi Daya Spektral QRS | pita analisis (penyebut) | 5-40 Hz | (3) |
| Distribusi Daya Spektral QRS | `l1`, `l2`, `l3` untuk 60-130 bpm | 0.5 / 0.8 / 0.4 | (5) |
| Distribusi Daya Spektral QRS | `l1`, `l2`, `l3` untuk 130-160 bpm | 0.4 / 0.7 / 0.3 | (5) |
| Distribusi Daya Spektral QRS | rentang detak jantung yang tervalidasi | 60-160 bpm | (5) |
| Kurtosis Sinyal | ambang optimal (*biner*) | > 5 | (10) |
| Daya Relatif Baseline | pita baseline (pembilang) | 0-1 Hz | (11) |
| Daya Relatif Baseline | pita total (penyebut) | 0-40 Hz | (11) |
| Daya Relatif Baseline | batas bawah *optimal* | 0.95 | (12) |
| Daya Relatif Baseline | batas bawah *suspicious* | 0.9 | (12) |

### 7.2 Parameter fungsi keanggotaan

| Faktor | Bentuk | Parameter | Persamaan |
|---|---|---|---|
| Deteksi R-peak (`q`, skala 0-100) | Cauchy | `a = 80`, `alpha = 0.3` | (22) |
| Deteksi R-peak (`q`) | Cauchy | `a = 55`, pembagi 5 (dari `beta = 2`, `gamma = 0.2`) | (24) |
| Deteksi R-peak (`q`) | Cauchy | `a = 75`, `beta = 1/7.5` | (25) |
| Distribusi Daya Spektral QRS (`x`, skala 0-1) | trapesium | 0.25 -> 0.35 (naik) | (26) |
| Distribusi Daya Spektral QRS (`x`) | trapesium | 0.25 -> 0.15 (turun) | (27) |
| Distribusi Daya Spektral QRS (`x`) | trapesium | 0.18 / 0.22 / 0.28 / 0.32 | (28) |
| Kurtosis Sinyal (nilai mentah) | persegi | ambang 5, keanggotaan B identik 0 | (29) |
| Daya Relatif Baseline (`b`, skala 0-100) | Cauchy | `a = 90`, `alpha = 0.8718` | (30) |
| Daya Relatif Baseline (`b`) | Cauchy | `a = 85`, pembagi 5 | (31) |
| Daya Relatif Baseline (`b`) | Cauchy | `a = 92`, pembagi 2.5 | (32) |

### 7.3 Bobot dan ambang keputusan

| Parameter | Nilai | Persamaan |
|---|---|---|
| Vektor bobot `W = (w1, w2, w3, w4)` | (0.4, 0.4, 0.1, 0.1); `sum(W) = 1` | setelah (32), sebelum (33) |
| Operator sintesis | `M(., +)` — operator berbatas, `s_j = min(1, sum_i w_i r_ij)` | sebelum (33) |
| Skor peringkat | `v1 = 1` (E), `v2 = 2` (B), `v3 = 3` (U) | (33) |
| Ambang kelas E/B | `v_th1 = 1.50` | (34) |
| Ambang kelas B/U | `v_th2 = 2.40` | (34) |

---

## 8. Referensi

Zhao, Z., & Zhang, Y. (2018). SQI Quality Evaluation Mechanism of Single-Lead ECG Signal Based on Simple Heuristic Fusion and Fuzzy Comprehensive Evaluation. *Frontiers in Physiology*, 9:727. https://doi.org/10.3389/fphys.2018.00727

---

## 9. Batasan

1. **Rentang detak jantung untuk Distribusi Daya Spektral QRS.** Artikel mengkalibrasi batas `l1`, `l2`, dan `l3` untuk indeks ini **hanya** pada denyut jantung 60-160 bpm. Di luar rentang tersebut kriteria penerimaan indeks ini tidak dapat diterapkan, sehingga hasilnya harus dilaporkan sebagai "tidak dapat diterapkan" dan tidak boleh ditebak.
2. **Ketidakkonsistenan internal pada Distribusi Daya Spektral QRS.** Fungsi keanggotaan *Excellent* yang diberikan artikel untuk indeks ini sudah **jenuh (bernilai 1) sejak `x >= 0.35`**, sehingga tidak sejalan dengan kriteria penerimaannya yang menetapkan *optimal* pada selang **[0.5, 0.8]**. Artikel tidak konsisten secara internal pada titik ini. Implementasi mengikuti **keduanya sebagaimana tertulis**, masing-masing pada tempatnya: kriteria penerimaan dipakai untuk menentukan tingkat per indeks, sedangkan fungsi keanggotaan dipakai pada tahap fuzzy. Keduanya **tidak** direkonsiliasi secara diam-diam.
3. **Kurtosis Sinyal tidak memiliki pita *suspicious*.** Artikel tidak mendefinisikan tingkat *suspicious* untuk indeks ini; kriterianya biner, dan konsekuensinya derajat keanggotaan *Barely Acceptable* untuk faktor ini selalu nol.
4. **Koefisien fusi tidak dioptimalkan.** Artikel menyatakan sendiri bahwa koefisien jumlah `optimal`, `suspicious`, dan `unqualified` pada aturan fusi ditetapkan secara empiris melalui coba-coba, dan pencarian menyeluruh atas kombinasi logika maupun ambang tidak dilakukan. Aturan fusi karena itu bersifat heuristik.
5. **Indeks kelima tidak digunakan.** Artikel juga mendefinisikan indeks kelima, yaitu **variabilitas interval R-R**, namun indeks tersebut sengaja tidak digunakan di sini karena artikel sendiri melaporkan bahwa penambahan indeks itu dari empat menjadi lima indeks tidak memperbaiki Acc, Se, maupun Sp, sehingga kombinasi empat indeks dinilai lebih masuk akal.
6. **Kelas kualitas menggambarkan rekaman, bukan kesehatan peserta.** Kelas E, B, atau U yang dihasilkan mekanisme ini adalah penilaian atas **kualitas rekaman sinyal ECG**, bukan penilaian atas **kondisi kesehatan peserta**. Kelas kualitas tidak boleh ditafsirkan sebagai diagnosis atau indikator klinis.
7. **Belum ada klaim validasi pada data proyek ini.** Dokumen ini menguraikan metode dan parameternya; dokumen ini tidak menyatakan bahwa metode telah divalidasi terhadap data proyek ini, dan angka kinerja apa pun yang berasal dari artikel tetap merupakan hasil artikel pada basis data PhysioNet yang digunakannya sendiri.
