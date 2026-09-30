"""Bahasa Indonesia untuk tabel, figure, legenda dan interpretasi.

Semua label yang dibaca manusia ditulis di sini, bukan di dalam plot. Nama kunci
mesin (``SUPINE``, ``qSQI``, ``Excellent``) tetap dipakai pada data, konfigurasi,
dan file keluaran; modul ini hanya menerjemahkan apa yang ditampilkan.

Interpretasi selalu dihitung dari angka yang sedang ditampilkan, tidak ada
kalimat yang ditulis manual. Karena itu tabel dan figure tidak mungkin
berbeda dengan data di atasnya.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import pandas as pd

from ..analysis.statistics import box_outliers, class_table
from ..models.result import SQI_KEYS, SQI_LABELS

#: Aktivitas rekaman. Sumbu-y figure dan baris tabel memakai nama ini.
POSITION_LABELS_ID: dict[str, str] = {
    "SUPINE": "Berbaring",
    "SITTING": "Duduk",
    "STANDING": "Berdiri",
}

#: Kelas penerimaan kualitas sinyal.
CLASS_LABELS_ID: dict[str, str] = {
    "Excellent": "Sangat Baik",
    "Barely Acceptable": "Cukup Diterima",
    "Unacceptable": "Tidak Diterima",
}

POSITION_ORDER = ("SUPINE", "SITTING", "STANDING")

#: Warna titik penyimpang. Biru tidak dipakai oleh kelas kualitas
#: (hijau/kuning/merah) maupun warna posisi, sehingga titik biru selalu terbaca
#: sebagai "ini peserta yang berbeda", bukan sebagai kategori tertentu.
OUTLIER_COLOR = "#1d4ed8"
BOX_COLOR = "#94a3b8"

_SCOPE_NOTE = (
    "Seluruh angka di bawah menggambarkan kualitas rekaman sinyal, bukan "
    "kondisi kesehatan peserta."
)


def position_label(position: str) -> str:
    """Nama aktivitas dalam bahasa Indonesia."""
    return POSITION_LABELS_ID.get(position, str(position))


def class_label(level: str) -> str:
    """Nama kelas kualitas dalam bahasa Indonesia."""
    return CLASS_LABELS_ID.get(level, str(level))


def sqi_label_id(key: str) -> str:
    """Nama indeks kualitas sinyal dalam bahasa Indonesia."""
    return SQI_LABELS.get(key, key)


def _pct(value: Any, digits: int = 1) -> str:
    """Format persentase, dengan ``-`` untuk nilai yang tidak terdefinisi."""
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(number):
        return "-"
    return f"{number:.{digits}f}%"


# ---------------------------------------------------------------------------
# Tabel: klasifikasi kualitas per aktivitas
# ---------------------------------------------------------------------------

def acceptance_table(classes: pd.DataFrame) -> pd.DataFrame:
    """Tabel ``Aktivitas`` x ``Kualitas Penerimaan``, siap ditampilkan.

    Setiap kelas punya dua kolom: jumlah frame dan persentasenya. Jumlah
    diperlukan agar persentase dapat diperiksa, dan persentase diperlukan karena
    jumlah frame tiap aktivitas tidak sama.
    """
    if classes is None or classes.empty:
        return pd.DataFrame()

    columns: dict[str, Any] = {"Aktivitas": [], "Total Frame": []}
    for level in CLASS_LABELS_ID:
        columns[f"{class_label(level)} (n)"] = []
        columns[f"{class_label(level)} (%)"] = []
    columns["Diterima (%)"] = []

    for _, row in classes.iterrows():
        columns["Aktivitas"].append(position_label(row["position"]))
        columns["Total Frame"].append(int(row["n_frames"]))
        for level in CLASS_LABELS_ID:
            columns[f"{class_label(level)} (n)"].append(int(row.get(f"count_{level}", 0)))
            columns[f"{class_label(level)} (%)"].append(_pct(row.get(f"pct_{level}")))
        columns["Diterima (%)"].append(_pct(row.get("pct_accepted")))

    return pd.DataFrame(columns)


#: Pasangan sufiks statistik pada frame analisis dengan Bahasa Indonesia-nya.
#: Dipakai oleh tabel rekapitulasi dan oleh ringkasan per aktivitas supaya
#: tidak ada dua tabel di halaman yang memakai penamaan kolom berbeda.
RECAP_STATS = (
    ("mean", "rata-rata"),
    ("median", "tengah"),
    ("sd", "simpangan baku"),
)


#: Nama tampilan untuk kolom identitas pada tabel ringkasan per aktivitas. kolom
#: ``position`` menyimpan kode mesin aktivitas, jadi pembaca harus melihat
#: ``Aktivitas`` seperti pada tabel rekapitulasi.
POSITION_SUMMARY_LABELS = {
    "position": "Aktivitas",
    "n_frames": "Total Frame",
}


def sqi_stat_display_names() -> dict[str, str]:
    """Nama tampilan untuk kolom ``<indeks>_mean``/``_median``/``_sd``.

    Frame hasil analisis tetap memakai nama mesin supaya ekspor CSV dan laporan
    bisa dicocokkan dengan kode analisis; hanya header yang dilihat pembaca yang
    diganti. Sufiksnya ditulis penuh agar setiap kolom yang terlihat di layar
    berbahasa Indonesia.
    """
    return {
        f"{key}_{suffix}": f"{sqi_label_id(key)} · {name}"
        for key in SQI_LABELS
        for suffix, name in RECAP_STATS
    }


def sqi_stat_columns() -> list[str]:
    """Urutan kolom mesin ringkasan indeks, siap dipakai ``df[columns]``."""
    return [f"{key}_{suffix}" for key in SQI_LABELS for suffix, _name in RECAP_STATS]


def activity_recap_table(recap: pd.DataFrame) -> pd.DataFrame:
    """Tabel rekapitulasi per aktivitas, siap ditampilkan.

    Gabungan kelas penerimaan dan ringkasan indeks kualitas sinyal untuk satu
    baris per aktivitas. Nilai indeks ditampilkan dengan tiga desimal karena
    indeksnya sudah dinormalisasi ke rentang 0-1, sehingga pembulatan satu
    desimal akan menghapus perbedaan yang justru sedang dicari.
    """
    if recap is None or recap.empty:
        return pd.DataFrame()

    columns: dict[str, Any] = {"Aktivitas": [], "Total Frame": []}
    for level in CLASS_LABELS_ID:
        columns[f"{class_label(level)} (n)"] = []
        columns[f"{class_label(level)} (%)"] = []
    columns["Diterima (%)"] = []
    for key in SQI_KEYS:
        if f"{key}_mean" not in recap:
            continue
        for suffix, name in RECAP_STATS:
            columns[f"{sqi_label_id(key)} · {name}"] = []

    for _, row in recap.iterrows():
        columns["Aktivitas"].append(position_label(row["position"]))
        columns["Total Frame"].append(int(row["n_frames"]))
        for level in CLASS_LABELS_ID:
            columns[f"{class_label(level)} (n)"].append(int(row.get(f"count_{level}", 0)))
            columns[f"{class_label(level)} (%)"].append(_pct(row.get(f"pct_{level}")))
        columns["Diterima (%)"].append(_pct(row.get("pct_accepted")))
        for key in SQI_KEYS:
            if f"{key}_mean" not in recap:
                continue
            for suffix, name in RECAP_STATS:
                columns[f"{sqi_label_id(key)} · {name}"].append(
                    _fixed(row.get(f"{key}_{suffix}"), 3)
                )

    return pd.DataFrame(columns)


def _fixed(value: Any, digits: int = 3) -> str:
    """Format angka desimal, dengan ``-`` bila nilainya tidak terdefinisi."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(number):
        return "-"
    return f"{number:.{digits}f}"


def activity_recap_legend() -> str:
    """Legenda tabel rekapitulasi per aktivitas."""
    stats = " · ".join(name for _suffix, name in RECAP_STATS)
    indices = " · ".join(sqi_label_id(key) for key in SQI_KEYS)
    classes = " · ".join(class_label(level) for level in CLASS_LABELS_ID)
    return (
        "**Legenda tabel.** Setiap baris menggabungkan seluruh frame dari satu "
        "aktivitas rekaman. Kolom `Total Frame` adalah jumlah frame 10 detik "
        f"yang dianalisis pada aktivitas tersebut, kolom `{classes}` adalah "
        "kelas penerimaannya, dan kolom `Diterima (%)` menggabungkan *Sangat "
        "Baik* dengan *Cukup Diterima*. Kolom setelahnya merangkum tiap indeks "
        f"kualitas sinyal ({indices}) dengan tiga ukuran: {stats}. "
        "`simpangan baku` memakai rumus contoh (ddof=1) dan tidak terdefinisi "
        "bila hanya ada satu frame. " + _SCOPE_NOTE
    )


def activity_recap_interpretation(recap: pd.DataFrame) -> str:
    """Interpretasi rekapitulasi per aktivitas, dihitung dari tabelnya sendiri."""
    if recap is None or recap.empty:
        return "Belum ada data yang cukup untuk ditafsirkan."

    ranked = recap.sort_values("pct_accepted", ascending=False)
    best = ranked.iloc[0]
    worst = ranked.iloc[-1]
    total = int(recap["n_frames"].sum())

    if float(best["pct_accepted"]) == float(worst["pct_accepted"]):
        sentences = [
            f"**Interpretasi.** Dari total {total} frame yang dianalisis, ketiga "
            "aktivitas menghasilkan persentase frame diterima yang sama "
            f"({_pct(best['pct_accepted'])}), sehingga tidak ada aktivitas yang "
            "terbanyak frame diterima."
        ]
    else:
        sentences = [
            f"**Interpretasi.** Dari total {total} frame yang dianalisis, "
            f"aktivitas **{position_label(best['position'])}** menghasilkan "
            f"persentase frame diterima tertinggi ({_pct(best['pct_accepted'])}), "
            f"sedangkan **{position_label(worst['position'])}** terendah "
            f"({_pct(worst['pct_accepted'])})."
        ]

    if len(recap) > 1:
        busiest = int(recap["n_frames"].max())
        fewest = int(recap["n_frames"].min())
        if busiest == fewest:
            sentences.append(
                f"Jumlah frame tiap aktivitas sama ({busiest} frame), sehingga "
                "perbandingan persentase antar aktivitas tidak dipengaruhi "
                "perbedaan jumlah frame."
            )
        else:
            shared = recap[recap["n_frames"] == busiest]
            where = (
                position_label(shared.iloc[0]["position"])
                if len(shared) == 1
                else " · ".join(position_label(p) for p in shared["position"])
            )
            sentences.append(
                f"Aktivitas {where} menyumbang frame terbanyak "
                f"({busiest} dari {total}), sehingga persentasenya paling "
                "bergantung pada rekaman pada aktivitas itu. Jumlah frame tiap "
                "aktivitas tidak sama, sehingga perbandingan persentase perlu "
                "dibaca bersama kolom `Total Frame`."
            )

    sentences.append(
        "Rangkuman di atas menggambarkan hasil pengukuran pada tiap aktivitas "
        "dan tidak menyatakan apa pun tentang kondisi peserta."
    )
    return " ".join(sentences)


def acceptance_table_legend() -> str:
    """Legenda tabel klasifikasi kualitas per aktivitas."""
    classes = " · ".join(class_label(level) for level in CLASS_LABELS_ID)
    return (
        f"**Legenda.** Baris adalah aktivitas rekaman. Kolom `{classes}` adalah "
        "kelas penerimaan kualitas sinyal. Kolom `(n)` adalah jumlah frame 10 "
        "detik, kolom `(%)` adalah persentasenya terhadap seluruh frame pada "
        "aktivitas tersebut. Kolom `Diterima (%)` menggabungkan *Sangat Baik* "
        "dan *Cukup Diterima*, yaitu frame yang layak dipakai. " + _SCOPE_NOTE
    )


def acceptance_table_interpretation(classes: pd.DataFrame) -> str:
    """Interpretasi tabel, dihitung dari angkanya sendiri."""
    if classes is None or classes.empty:
        return "Belum ada data yang cukup untuk ditafsirkan."

    ranked = classes.sort_values("pct_accepted", ascending=False)
    best = ranked.iloc[0]
    worst = ranked.iloc[-1]
    total = int(classes["n_frames"].sum())
    accepted_classes = [level for level in CLASS_LABELS_ID if level != "Unacceptable"]

    def accepted_count(row: Any) -> int:
        return sum(int(row.get(f"count_{level}", 0)) for level in accepted_classes)

    # When several activities share the same percentage, naming one of them as
    # the top and another as the bottom would read as a contradiction.
    tied = float(best["pct_accepted"]) == float(worst["pct_accepted"])
    if tied:
        sentences = [
            f"**Interpretasi.** Dari total {total} frame yang dianalisis, ketiga "
            "aktivitas memiliki persentase frame diterima yang sama "
            f"({_pct(best['pct_accepted'])}, masing-masing "
            f"{accepted_count(best)} frame), sehingga tidak ada aktivitas yang "
            "terbanyak atau tersedikit frame diterima."
        ]
    else:
        sentences = [
            f"**Interpretasi.** Dari total {total} frame yang dianalisis, aktivitas "
            f"**{position_label(best['position'])}** memiliki persentase frame "
            f"diterima tertinggi ({_pct(best['pct_accepted'])}, "
            f"{accepted_count(best)} frame), sedangkan "
            f"**{position_label(worst['position'])}** terendah "
            f"({_pct(worst['pct_accepted'])}, {accepted_count(worst)} frame)."
        ]

    worst_bad = classes.loc[classes["pct_Unacceptable"].idxmax()]
    if classes["pct_Unacceptable"].nunique() == 1:
        sentences.append(
            f"Proporsi *Tidak Diterima* juga sama rata di ketiga aktivitas "
            f"({_pct(worst_bad['pct_Unacceptable'])})."
        )
    else:
        sentences.append(
            f"Aktivitas **{position_label(worst_bad['position'])}** menyumbang "
            f"proporsi *Tidak Diterima* tertinggi "
            f"({_pct(worst_bad['pct_Unacceptable'])}, "
            f"{int(worst_bad.get('count_Unacceptable', 0))} frame)."
        )

    spread = float(best["pct_accepted"]) - float(worst["pct_accepted"])
    # Selisih kecil hanya berarti angka antar aktivitas mirip. Kalimatnya tidak
    # boleh menyiratkan bahwa rekamananya baik: pada 0%Accepted semua aktivitas
    # bisa saja sama-sama buruk.
    if spread < 5.0:
        sentences.append(
            f"Selisih antar aktivitas hanya {spread:.1f} poin persentase, sehingga "
            "tingkat penerimaan frame pada ketiga aktivitas relatif sama. Angka "
            "tersebut perlu dibaca bersama kolom `Diterima (%)` pada tabel, "
            "karena nilai yang sama dapat muncul pada tingkat penerimaan "
            "yang rendah maupun tinggi."
        )
    else:
        sentences.append(
            f"Selisih antar aktivitas mencapai {spread:.1f} poin persentase, "
            "sehingga kualitas rekaman berbeda cukup jelas antar aktivitas."
        )

    sentences.append(
        "Perbedaan di atas menggambarkan karakter rekaman pada tiap aktivitas dan "
        "tidak menyatakan apa pun tentang kondisi peserta."
    )
    return " ".join(sentences)


# ---------------------------------------------------------------------------
# Figure: kualitas sinyal per peserta dan aktivitas
# ---------------------------------------------------------------------------

def quality_by_activity_figure(share: pd.DataFrame, *, height: int = 540):
    """Box plot horizontal kualitas per peserta, dikelompokkan per aktivitas.

    Sumbu-x adalah persentase frame yang diterima untuk satu peserta pada satu
    aktivitas, dan sumbu-y adalah aktivitas. Setiap peserta menyumbang satu
    nilai, sehingga lebar kotak menyatakan sebaran antar peserta, bukan antar
    frame. Titik di luar pagar 1,5 x IQR digambar sebagai titik biru terpisah
    agar dapat diberi entri legenda dan warnanya sendiri.
    """
    import plotly.graph_objects as go

    from .plots import INK, INK_SOFT, _style

    figure = go.Figure()
    title = "Kualitas sinyal per peserta dan aktivitas"
    if share is None or share.empty:
        return _style(figure, title=title, height=height, showlegend=False)

    order = [p for p in POSITION_ORDER if p in set(share["position"])]

    for position in order:
        values = share[share["position"] == position]["pct_accepted"].dropna().tolist()
        summary, _flagged = box_outliers(values)
        if not summary:
            continue
        label = position_label(position)
        figure.add_trace(
            go.Box(
                x=values,
                y=[label] * len(values),
                orientation="h",
                name=label,
                showlegend=False,
                fillcolor="rgba(148, 163, 184, 0.28)",
                line=dict(color=BOX_COLOR, width=1.4),
                boxpoints=False,
                hovertemplate=(
                    f"{label}<br>persentase diterima %{{x:.1f}}%<extra></extra>"
                ),
            )
        )

    if figure.data:
        # Satu entri legenda untuk seluruh kotak: nama aktivitas sudah tercetak
        # pada sumbu-y, sehingga mengulangnya hanya menambah kepadatan.
        figure.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="markers",
                marker=dict(symbol="square", size=10, color=BOX_COLOR),
                name="Distribusi antar peserta",
                hoverinfo="skip",
            )
        )

    # Titik biru digabung dari ketiga aktivitas agar hanya ada satu entri
    # legenda, sementara sumbu-y tetap menunjukkan asal setiap titik.
    flagged_x: list[float] = []
    flagged_y: list[str] = []
    for position in order:
        _summary, flagged = box_outliers(
            share[share["position"] == position]["pct_accepted"].dropna().tolist()
        )
        flagged_x.extend(flagged)
        flagged_y.extend([position_label(position)] * len(flagged))
    if flagged_x:
        figure.add_trace(
            go.Scatter(
                x=flagged_x,
                y=flagged_y,
                mode="markers",
                marker=dict(
                    symbol="circle", size=9, color=OUTLIER_COLOR,
                    line=dict(width=1, color="#ffffff"),
                ),
                name="Penyimpang (di luar 1,5 x IQR)",
                hovertemplate=(
                    "Penyimpang<br>%{y}<br>persentase diterima %{x:.1f}%<extra></extra>"
                ),
            )
        )

    _style(
        figure,
        title=title,
        height=height,
        left=140,
        right=32,
        top=84,
        bottom=108,
        showlegend=True,
    )
    figure.update_layout(
        xaxis=dict(
            title="Persentase frame yang diterima (%)", ticksuffix="%", range=[-2, 102]
        ),
        yaxis=dict(
            title="Aktivitas",
            # Kategori pertama pada sumbu-y muncul di paling bawah, jadi
            # urutannya dibalik agar Berbaring berada di atas.
            categoryorder="array",
            categoryarray=[position_label(p) for p in reversed(order)],
            tickfont=dict(size=12, color=INK),
            title_font=dict(size=12, color=INK),
        ),
        hoverlabel=dict(font=dict(size=11, color=INK_SOFT)),
    )
    return figure


# ---------------------------------------------------------------------------
# Figure: rekapitulasi per aktivitas (sebaran frame, bukan antar peserta)
# ---------------------------------------------------------------------------

def activity_recap_figure(
    df: pd.DataFrame, *, columns: Sequence[str] = SQI_KEYS, height: int | None = None
):
    """Box plot sebaran indeks kualitas per aktivitas, satu panel per indeks.

    Berbeda dengan :func:`quality_by_activity_figure` yang satu nilai per
    peserta, figure ini menggambar seluruh frame yang dianalisis pada setiap
    aktivitas. Karena itu kotak di sini menyatakan sebaran frame, bukan variasi
    antar peserta. Tiap indeks mendapat panel sendiri dengan skala sendiri:
    nilai kurtosis dan daya baseline tidak sebanding dengan nilai deteksi R-peak
    pada satu sumbu bersama, sehingga memaksanya satu sumbu akan menekan salah
    satu kelompok Boxes menjadi tidak terbaca.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from .plots import INK, INK_SOFT, _style

    title = "Rekapitulasi sebaran kualitas sinyal per aktivitas"
    if df is None or df.empty:
        return _style(go.Figure(), title=title, height=520, showlegend=False)

    frame = df[df["valid"]] if "valid" in df else df
    present = [key for key in columns if key in frame and frame[key].notna().any()]
    if not present:
        return _style(go.Figure(), title=title, height=520, showlegend=False)

    order = [p for p in POSITION_ORDER if p in set(frame["position"])]
    labels = [position_label(p) for p in order]
    n_panels = len(present)
    n_cols = 2 if n_panels > 1 else 1
    n_rows = math.ceil(n_panels / n_cols)
    if height is None:
        # Each panel needs room for its own three activity rows, plus the
        # reserved bands for the panel title and the shared legend.
        height = 110 + n_rows * 300 + 130

    figure = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[sqi_label_id(key) for key in present],
        horizontal_spacing=0.18,
        vertical_spacing=0.16,
    )

    for index, key in enumerate(present):
        row, col = divmod(index, n_cols)
        row += 1
        col += 1
        for position in order:
            values = frame[frame["position"] == position][key].dropna().tolist()
            if not values:
                continue
            figure.add_trace(
                go.Box(
                    x=values,
                    y=[position_label(position)] * len(values),
                    orientation="h",
                    name=position_label(position),
                    showlegend=False,
                    fillcolor="rgba(148, 163, 184, 0.28)",
                    line=dict(color=BOX_COLOR, width=1.4),
                    boxpoints=False,
                    hovertemplate=(
                        f"{position_label(position)}<br>{sqi_label_id(key)} "
                        "%{x:.3f}<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )

        flagged_x: list[float] = []
        flagged_y: list[str] = []
        for position in order:
            values = frame[frame["position"] == position][key].dropna().tolist()
            _summary, flagged = box_outliers(values)
            flagged_x.extend(flagged)
            flagged_y.extend([position_label(position)] * len(flagged))
        if flagged_x:
            figure.add_trace(
                go.Scatter(
                    x=flagged_x,
                    y=flagged_y,
                    mode="markers",
                    marker=dict(
                        symbol="circle", size=7, color=OUTLIER_COLOR,
                        line=dict(width=1, color="#ffffff"),
                    ),
                    name="Penyimpang (di luar 1,5 x IQR)",
                    showlegend=(index == 0),
                    legendgroup="outliers",
                    hovertemplate=(
                        f"Penyimpang · {sqi_label_id(key)}<br>%{{y}}<br>"
                        "%{x:.3f}<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )

        figure.update_xaxes(title_text="Nilai indeks", row=row, col=col)
        figure.update_yaxes(
            title_text="Aktivitas" if col == 1 else None,
            # The first category lands at the bottom, so the order is reversed
            # to keep Berbaring at the top as in the other activity figures.
            categoryorder="array",
            categoryarray=list(reversed(labels)),
            row=row,
            col=col,
        )

    # Satu entri legenda untuk seluruh kotak, ditambahkan ke panel pertama.
    figure.add_trace(
        go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(symbol="square", size=10, color=BOX_COLOR),
            name="Sebaran seluruh frame",
            legendgroup="box",
            showlegend=True,
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    _style(figure, title=title, height=height, showlegend=True)
    figure.update_layout(
        hoverlabel=dict(font=dict(size=11, color=INK_SOFT)),
        margin=dict(l=140, r=32, t=110, b=110),
    )
    for annotation in figure.layout.annotations:
        # Panel titles are the only annotations this figure owns.
        annotation.font = dict(size=13, color=INK)
    return figure


def activity_recap_distribution_legend() -> str:
    """Legenda figure sebaran indeks kualitas per aktivitas."""
    indices = " · ".join(sqi_label_id(key) for key in SQI_KEYS)
    activities = " · ".join(position_label(p) for p in POSITION_ORDER)
    return (
        "**Legenda.** Setiap panel adalah satu indeks kualitas sinyal "
        f"({indices}) dengan skalanya sendiri, sehingga tinggi kotak antar "
        "panel tidak boleh dibandingkan langsung. Sumbu-y pada setiap panel "
        f"adalah aktivitas rekaman ({activities}) dan sumbu-x adalah nilai "
        "indeks. Kotak abu-abu menunjukkan sebaran seluruh frame pada "
        "aktivitas itu: garis tengah adalah median, tepi kotak adalah kuartil "
        "ke-1 dan ke-3, dan whisker menuju nilai terdekat dalam pagar 1,5 x "
        "IQR. Titik biru adalah frame yang nilainya berada di luar pagar "
        "tersebut. Berbeda dengan figure per peserta, di sini setiap frame "
        "menyumbang satu nilai, sehingga kotak menggambarkan variasi antar "
        "frame pada aktivitas tersebut. " + _SCOPE_NOTE
    )


def activity_recap_distribution_interpretation(
    df: pd.DataFrame, *, columns: Sequence[str] = SQI_KEYS
) -> str:
    """Interpretasi sebaran indeks per aktivitas, dihitung dari angkanya sendiri."""
    if df is None or df.empty:
        return "Belum ada data yang cukup untuk ditafsirkan."
    frame = df[df["valid"]] if "valid" in df else df
    if frame.empty:
        return "Belum ada data yang cukup untuk ditafsirkan."

    order = [p for p in POSITION_ORDER if p in set(frame["position"])]
    parts = ["**Interpretasi.** Perbandingan nilai tengah tiap indeks pada tiap aktivitas:"]
    spreads: list[tuple[str, float]] = []
    flagged_total = 0

    for key in columns:
        if key not in frame:
            continue
        medians: dict[str, float] = {}
        iqrs: dict[str, float] = {}
        for position in order:
            values = frame[frame["position"] == position][key].dropna().tolist()
            summary, flagged = box_outliers(values)
            if not summary:
                continue
            medians[position_label(position)] = summary["median"]
            iqrs[position_label(position)] = summary["q3"] - summary["q1"]
            flagged_total += len(flagged)
        if len(medians) < 2:
            continue
        best = max(medians, key=medians.get)
        worst = min(medians, key=medians.get)
        if len(set(round(v, 6) for v in medians.values())) == 1:
            detail = (
                f"nilainya sama pada ketiga aktivitas ({medians[best]:.3f}), "
                "sehingga aktivitas tidak memisahkan nilai indeks ini."
            )
        else:
            detail = (
                f"tertinggi pada **{best}** ({medians[best]:.3f}) dan terendah "
                f"pada **{worst}** ({medians[worst]:.3f})."
            )
        parts.append(f"- **{sqi_label_id(key)}**: {detail}")
        widest = max(iqrs, key=iqrs.get)
        spreads.append((sqi_label_id(key), iqrs[widest], widest))

    if len(parts) == 1:
        return "Belum ada data yang cukup untuk ditafsirkan."

    if spreads:
        label, value, where = max(spreads, key=lambda item: item[1])
        parts.append(
            f"Lebar sebaran antar frame terbesar pada **{label}** di **{where}** "
            f"(IQR {value:.3f}), sehingga rekaman pada aktivitas itu paling "
            "beragam-antara-frame untuk indeks tersebut."
        )

    if flagged_total:
        parts.append(
            f"Sebanyak **{flagged_total}** frame berada di luar pagar 1,5 x IQR. "
            "Frame seperti ini dapat berasal dari kondisi pengukuran, "
            "penempatan elektroda, atau gangguan sesaat saat perekaman. Data ini "
            "tidak dapat menentukan penyebabnya dan tidak menyatakan apa pun "
            "tentang kesehatan peserta."
        )
    else:
        parts.append(
            "Tidak ada frame yang berada di luar pagar 1,5 x IQR. Ini hanya "
            "berarti tidak ada nilai yang terlalu jauh dari kelompok frame pada "
            "aktivitas masing-masing, dan tidak otomatis berarti sebarannya "
            "sempit."
        )
    return " ".join(parts)


def quality_by_activity_legend() -> str:
    """Legenda figure box plot kualitas per aktivitas."""
    activities = " · ".join(position_label(p) for p in POSITION_ORDER)
    return (
        "**Legenda.** Sumbu-y adalah aktivitas rekaman "
        f"({activities}). Sumbu-x adalah persentase frame yang diterima untuk satu "
        "peserta pada aktivitas tersebut. Kotak abu-abu menunjukkan sebaran "
        "antar peserta: garis tengah adalah median, tepi kotak adalah kuartil "
        "ke-1 dan ke-3, dan whisker menuju nilai terdekat dalam pagar "
        "1,5 x IQR. Titik biru adalah peserta yang nilai persentasenya berada "
        "di luar pagar tersebut (penyimpang). Karena setiap peserta "
        "menyumbang satu nilai, kotak menggambarkan variasi antar peserta, "
        "bukan antar frame. " + _SCOPE_NOTE
    )


def quality_by_activity_interpretation(share: pd.DataFrame) -> str:
    """Interpretasi figure box plot, dihitung dari angkanya sendiri."""
    if share is None or share.empty:
        return "Belum ada data yang cukup untuk ditafsirkan."

    rows: list[str] = []
    spreads: list[tuple[str, float]] = []
    flagged_total = 0

    for position in POSITION_ORDER:
        values = share[share["position"] == position]["pct_accepted"].dropna().tolist()
        summary, flagged = box_outliers(values)
        if not summary:
            continue
        label = position_label(position)
        spreads.append((label, summary["q3"] - summary["q1"]))
        flagged_total += len(flagged)
        rows.append(
            f"- **{label}**: {summary['n']} peserta, median "
            f"{summary['median']:.1f}%, rentang kuartil "
            f"{summary['q1']:.1f}–{summary['q3']:.1f}%, "
            + (f"{len(flagged)} penyimpang." if flagged else "tanpa penyimpang.")
        )

    if not rows:
        return "Belum ada data yang cukup untuk ditafsirkan."

    body = [
        "**Interpretasi.** Ringkasan sebaran antar peserta untuk tiap aktivitas:",
        *rows,
    ]

    if spreads:
        widest, widest_iqr = max(spreads, key=lambda item: item[1])
        narrowest, narrowest_iqr = min(spreads, key=lambda item: item[1])
        body.append(
            f"Aktivitas **{widest}** menunjukkan sebaran antar peserta paling "
            f"lebar (IQR {widest_iqr:.1f} poin persentase), sedangkan "
            f"**{narrowest}** paling rapat (IQR {narrowest_iqr:.1f} poin)."
        )

    if flagged_total:
        body.append(
            f"Sebanyak **{flagged_total}** titik biru teridentifikasi sebagai "
            "penyimpang. Penyimpang berarti rekaman peserta tersebut berada cukup "
            "jauh dari pola umum kelompok pada aktivitas yang sama. Faktor yang "
            "mungkin berperan antara lain kondisi pengukuran, penempatan "
            "elektroda, atau gangguan saat perekaman. Data ini tidak dapat "
            "menentukan penyebabnya dan tidak menyatakan apa pun tentang "
            "kesehatan peserta."
        )
    else:
        # Tidak adanya penyimpang hanya berarti tidak ada nilai yang melewati
        # pagar 1,5 x IQR. Itu bukan bukti bahwa sebarannya sempit, karena IQR
        # yang lebar pun bisa seluruhnya berada di dalam pagar.
        body.append(
            "Tidak ada peserta yang berada di luar pagar 1,5 x IQR. Ini hanya "
            "berarti tidak ada nilai yang terlalu jauh dari kelompok pada "
            "aktivitas masing-masing, dan tidak otomatis berarti sebarannya "
            "sempit; lebar sebaran tetap ditentukan oleh rentang kuartil di "
            "atas."
        )

    return "\n\n".join(body)


# ---------------------------------------------------------------------------
# Legenda dan interpretasi figure yang sudah ada
# ---------------------------------------------------------------------------

def quality_distribution_legend() -> str:
    """Legenda figure distribusi kelas kualitas per posisi."""
    classes = " · ".join(class_label(level) for level in CLASS_LABELS_ID)
    return (
        f"**Legenda.** Sumbu-x adalah aktivitas rekaman, sumbu-y adalah persentase "
        f"frame pada tiap kelas kualitas. Warna batang: {classes}. "
        "Angka di dalam batang adalah persentasenya bila cukup besar untuk "
        "dibaca. " + _SCOPE_NOTE
    )


def quality_distribution_interpretation(df: pd.DataFrame) -> str:
    """Interpretasi figure distribusi kelas kualitas."""
    if df is None or df.empty:
        return "Belum ada data yang cukup untuk ditafsirkan."
    table = class_table(df)
    if table.empty:
        return "Belum ada data yang cukup untuk ditafsirkan."

    best = float(table["pct_Excellent"].max())
    worst = float(table["pct_Unacceptable"].max())
    best_label = position_label(
        table.loc[table["pct_Excellent"].idxmax(), "position"]
    )
    worst_label = position_label(
        table.loc[table["pct_Unacceptable"].idxmax(), "position"]
    )

    # Naming one activity as both "highest" and "lowest" reads as a
    # contradiction, so a tie is stated as such instead.
    if table["pct_Excellent"].nunique() == 1:
        first = (
            f"Setiap aktivitas memiliki porsi *Sangat Baik* yang sama "
            f"({_pct(best)}), sehingga tidak ada aktivitas yang menonjol di "
            "sisi atas."
        )
    else:
        first = (
            f"Aktivitas **{best_label}** memiliki porsi *Sangat Baik* tertinggi "
            f"({_pct(best)})."
        )

    if table["pct_Unacceptable"].nunique() == 1:
        second = (
            f"Porsi *Tidak Diterima* juga sama rata di semua aktivitas "
            f"({_pct(worst)})."
        )
    else:
        second = (
            f"Sedangkan **{worst_label}** memiliki porsi *Tidak Diterima* "
            f"tertinggi ({_pct(worst)})."
        )

    return f"**Interpretasi.** {first} {second}"


def sqi_box_legend() -> str:
    """Legenda figure box plot indeks kualitas per posisi."""
    names = " · ".join(sqi_label_id(key) for key in SQI_KEYS)
    return (
        "**Legenda.** Setiap kelompok kotak membandingkan sebaran satu indeks "
        f"kualitas sinyal ({names}) pada tiga aktivitas. Warna kotak mengikuti "
        "aktivitas. Garis di dalam kotak adalah median, tepi kotak adalah "
        "kuartil ke-1 dan ke-3, titik adalah nilai yang berada di luar pagar "
        "1,5 x IQR. Sumbu-x memuat seluruh nilai, sehingga aktivitas dengan "
        "nilai yang jauh lebih rendah akan membuat kotak aktivitas lain "
        "tertekan. Tabel di atas memuat angka yang sama dalam bentuk ringkas. "
        + _SCOPE_NOTE
    )


def sqi_box_interpretation(df: pd.DataFrame) -> str:
    """Interpretasi figure box plot indeks kualitas."""
    if df is None or df.empty or "valid" not in df:
        return "Belum ada data yang cukup untuk ditafsirkan."
    frame = df[df["valid"]]
    if frame.empty:
        return "Belum ada data yang cukup untuk ditafsirkan."

    parts = ["**Interpretasi.**"]
    for key in SQI_KEYS:
        if key not in frame:
            continue
        per_position = {
            position: frame[frame["position"] == position][key].dropna()
            for position in POSITION_ORDER
            if position in set(frame["position"])
        }
        per_position = {k: v for k, v in per_position.items() if not v.empty}
        if len(per_position) < 2:
            continue
        medians = {position: float(values.median()) for position, values in per_position.items()}
        best = max(medians, key=medians.get)
        worst = min(medians, key=medians.get)
        parts.append(
            f"- **{sqi_label_id(key)}**: nilai median tertinggi pada "
            f"**{position_label(best)}** ({medians[best]:.3f}) dan terendah pada "
            f"**{position_label(worst)}** ({medians[worst]:.3f})."
        )
    if len(parts) == 1:
        return "Belum ada data yang cukup untuk ditafsirkan."
    parts.append(
        "Perbedaan antar aktivitas di atas adalah perbedaan kualitas rekaman, "
        "bukan pernyataan mengenai kondisi peserta."
    )
    return " ".join(parts)


def sqi_trend_legend() -> str:
    """Legenda figure indeks kualitas per frame."""
    activities = " · ".join(position_label(p) for p in POSITION_ORDER)
    return (
        "**Legenda.** Setiap titik adalah satu frame yang dianalisis. Sumbu-x "
        "adalah urutan frame, sumbu-y adalah nilai indeks kualitas. Warna titik "
        f"menunjukkan aktivitas rekaman ({activities}). "
        "Kenaikan atau penurunan nilai antar frame mencerminkan variasi rekaman "
        "pada sesi tersebut, bukan perubahan kondisi peserta. " + _SCOPE_NOTE
    )


def sqi_trend_interpretation(df: pd.DataFrame) -> str:
    """Interpretasi figure indeks kualitas per frame."""
    if df is None or df.empty or "valid" not in df:
        return "Belum ada data yang cukup untuk ditafsirkan."
    frame = df[df["valid"]]
    if frame.empty:
        return "Belum ada data yang cukup untuk ditafsirkan."

    frame = frame.sort_values(["subject_id", "position", "frame_id"])
    parts = ["**Interpretasi.**"]
    for key in SQI_KEYS:
        if key not in frame:
            continue
        grouped = frame.groupby(["subject_id", "position"])[key]
        # ``first``/``last`` pada GroupBy sudah melewati nilai kosong, jadi frame
        # pertama dan terakhir yang benar-benar terisi saja yang dibandingkan.
        first = grouped.first().dropna()
        last = grouped.last().dropna()
        if first.empty or last.empty:
            continue
        first_mean = float(first.mean())
        last_mean = float(last.mean())
        if last_mean > first_mean:
            direction = "naik"
        elif last_mean < first_mean:
            direction = "turun"
        else:
            direction = "tetap"
        parts.append(
            f"- **{sqi_label_id(key)}**: nilai rata-rata frame awal "
            f"{first_mean:.3f} menjadi {last_mean:.3f} pada frame akhir, "
            f"sehingga trennya {direction} sepanjang sesi rekaman."
        )
    if len(parts) == 1:
        return "Belum ada data yang cukup untuk ditafsirkan."
    return " ".join(parts)


def membership_legend() -> str:
    """Legenda figure keanggotaan fuzzy."""
    classes = " · ".join(class_label(level) for level in CLASS_LABELS_ID)
    return (
        f"**Legenda.** Tinggi batang menunjukkan tingkat keanggotaan fuzzy frame "
        f"ini pada tiap kelas ({classes}). Nilai berada pada rentang 0 sampai 1 "
        "dan tidak selalu berjumlah tepat 1, sehingga batang dapat menunjukkan "
        "lebih dari satu kelas. " + _SCOPE_NOTE
    )


def membership_interpretation(membership: dict[str, float]) -> str:
    """Interpretasi figure keanggotaan fuzzy."""
    if not membership:
        return "Belum ada data yang cukup untuk ditafsirkan."
    ranked = sorted(membership.items(), key=lambda item: float(item[1]), reverse=True)
    top_level, top_value = ranked[0]
    return (
        "**Interpretasi.** Keanggotaan tertinggi frame ini ada pada kelas "
        f"**{class_label(top_level)}** ({float(top_value):.3f}). "
        "Nilai keanggotaan yang tinggi pada lebih dari satu kelas menandakan "
        "frame berada di antara dua kategori, bukan salah satu secara pasti."
    )


def heatmap_legend() -> str:
    """Legenda figure heatmap per peserta dan posisi."""
    return (
        "**Legenda.** Setiap baris adalah satu peserta, setiap kolom adalah "
        "aktivitas rekaman. Warna menyatakan nilai rata-rata indeks kualitas "
        "pada peserta dan aktivitas tersebut: hijau lebih tinggi, merah lebih "
        "rendah, putih di tengah. Baris atau kolom kosong berarti tidak ada "
        "frame yang dianalisis pada kombinasi itu. " + _SCOPE_NOTE
    )


def heatmap_interpretation(matrix: pd.DataFrame) -> str:
    """Interpretasi figure heatmap per peserta dan posisi."""
    if matrix is None or matrix.empty:
        return "Belum ada data yang cukup untuk ditafsirkan."
    values = matrix.to_numpy(dtype=float)
    if values.size == 0 or not bool(pd.notna(values).any()):
        return "Belum ada data yang cukup untuk ditafsirkan."

    best_position, best_subject = max(
        ((position, subject) for subject in matrix.index for position in matrix.columns
         if pd.notna(matrix.loc[subject, position])),
        key=lambda item: matrix.loc[item[1], item[0]],
    )
    worst_position, worst_subject = min(
        ((position, subject) for subject in matrix.index for position in matrix.columns
         if pd.notna(matrix.loc[subject, position])),
        key=lambda item: matrix.loc[item[1], item[0]],
    )
    return (
        "**Interpretasi.** Kombinasi peserta dan aktivitas dengan nilai tertinggi "
        f"adalah **{worst_subject} pada {position_label(best_position)}** "
        f"({matrix.loc[best_subject, best_position]:.3f}), sedangkan yang terendah "
        f"adalah **{worst_subject} pada {position_label(worst_position)}** "
        f"({matrix.loc[worst_subject, worst_position]:.3f}). "
        "Sebaran nilai antar peserta pada satu aktivitas menunjukkan seberapa "
        "seragam kualitas rekaman kelompok pada aktivitas tersebut."
    )


def psd_legend() -> str:
    """Legenda figure power spectral density."""
    return (
        "**Legenda.** Sumbu-x adalah frekuensi dalam Hertz, sumbu-y adalah "
        "kekuatan spektral. Kurva abu-abu adalah estimasi kekuatan spektral "
        "sinyal. Area hijau adalah pita QRS, area jingga adalah pita baseline; "
        "nama masing-masing pita tercetak di dalam area tersebut. "
        "Kekuatan spektral yang rendah pada pita QRS menunjukkan energi sinyal "
        "berkurang di rentang frekuensi komponen QRS. " + _SCOPE_NOTE
    )


def psd_interpretation(
    frequencies, psd, qrs_band=None, analysis_band=None
) -> str:
    """Interpretasi figure power spectral density, dihitung dari kurvanya."""
    import numpy as np

    frequencies = np.asarray(frequencies, dtype=float)
    psd = np.asarray(psd, dtype=float)
    if frequencies.size == 0 or frequencies.size != psd.size:
        return "Belum ada data yang cukup untuk ditafsirkan."
    finite = np.isfinite(psd) & np.isfinite(frequencies)
    frequencies, psd = frequencies[finite], psd[finite]
    if frequencies.size == 0:
        return "Belum ada data yang cukup untuk ditafsirkan."

    low, high = (analysis_band or (frequencies.min(), frequencies.max()))[:2]
    window = (frequencies >= low) & (frequencies <= high)
    total = float(np.sum(psd[window]))
    if total <= 0:
        return "Belum ada data yang cukup untuk ditafsirkan."

    parts = [
        f"**Interpretasi.** Puncak kekuatan spektral berada di sekitar "
        f"{float(frequencies[int(np.argmax(psd))]):.1f} Hz."
    ]
    if qrs_band:
        qrs_mask = (frequencies >= qrs_band[0]) & (frequencies <= qrs_band[1])
        if qrs_mask.any():
            share = 100.0 * float(np.sum(psd[qrs_mask])) / total
            parts.append(
                f"Porsi kekuatan spektral pada pita QRS "
                f"({qrs_band[0]:g}–{qrs_band[1]:g} Hz) adalah {share:.1f}% dari "
                f"total kekuatan pada pita analisis {low:g}–{high:g} Hz."
            )
    parts.append(
        "Nilai ini merupakan deskripsi karakter spektral rekaman dan tidak "
        "menyatakan apa pun tentang kondisi peserta."
    )
    return " ".join(parts)


def ecg_legend() -> str:
    """Legenda figure sinyal ECG."""
    return (
        "**Legenda.** Garis biru tua adalah sinyal mentah (RAW) pada lead yang "
        "ditandai, garis biru muda adalah lead lainnya. Garis hijau putus-putus "
        "adalah sinyal setelah praproses dan hanya muncul bila praproses benar-benar "
        "dijalankan. Segitiga merah dan berlian biru adalah lokasi puncak R yang "
        "ditemukan oleh dua detektor yang dibandingkan; ketidaksesuaian antara "
        "keduanya ditafsirkan sebagai indikasi kualitas rekaman, bukan sebagai "
        "diagnosis. " + _SCOPE_NOTE
    )


def multilead_legend() -> str:
    """Legenda figure multi-lead."""
    return (
        "**Legenda.** Setiap panel adalah satu lead, disembunyikan urutannya dari "
        "atas ke bawah sesuai urutan channel. Sumbu-x adalah waktu dalam detik. "
        "Perbedaan bentuk dan kestabilan antar lead membantu menentukan apakah "
        "gangguan berasal dari satu lead atau seluruh rekaman. " + _SCOPE_NOTE
    )


def fuzzy_matrix_legend() -> str:
    """Legenda figure matriks evaluasi fuzzy."""
    names = " · ".join(sqi_label_id(key) for key in SQI_KEYS)
    return (
        "**Legenda.** Baris adalah indeks kualitas sinyal yang menjadi masukan "
        f"({names}), kolom adalah kelas penilaian. Warna menyatakan bobot "
        "keanggotaan (0 sampai 1): semakin gelap, semakin besar bobotnya. "
        "Matriks ini menampilkan bagaimana tiap masukan ikut menentukan kelas "
        "penilaian. " + _SCOPE_NOTE
    )


def fuzzy_matrix_interpretation(matrix: dict[str, dict[str, float]]) -> str:
    """Interpretasi figure matriks evaluasi fuzzy."""
    if not matrix:
        return "Belum ada data yang cukup untuk ditafsirkan."
    rows = []
    for factor, weights in matrix.items():
        if not weights:
            continue
        worst = max(weights, key=lambda level: float(weights[level]))
        rows.append(
            f"- **{sqi_label_id(factor)}**: bobot tertinggi pada kelas "
            f"{class_label(worst)} ({float(weights[worst]):.3f})."
        )
    if not rows:
        return "Belum ada data yang cukup untuk ditafsirkan."
    return "**Interpretasi.** " + " ".join(rows)


__all__ = [
    "CLASS_LABELS_ID",
    "OUTLIER_COLOR",
    "POSITION_LABELS_ID",
    "POSITION_ORDER",
    "RECAP_STATS",
    "acceptance_table",
    "acceptance_table_interpretation",
    "acceptance_table_legend",
    "activity_recap_distribution_interpretation",
    "activity_recap_distribution_legend",
    "activity_recap_figure",
    "activity_recap_interpretation",
    "activity_recap_legend",
    "activity_recap_table",
    "class_label",
    "ecg_legend",
    "heatmap_interpretation",
    "heatmap_legend",
    "membership_interpretation",
    "membership_legend",
    "multilead_legend",
    "position_label",
    "psd_interpretation",
    "psd_legend",
    "quality_by_activity_figure",
    "quality_by_activity_interpretation",
    "quality_by_activity_legend",
    "quality_distribution_interpretation",
    "quality_distribution_legend",
    "sqi_box_interpretation",
    "sqi_box_legend",
    "sqi_trend_interpretation",
    "sqi_trend_legend",
    "fuzzy_matrix_interpretation",
    "fuzzy_matrix_legend",
    "sqi_label_id",
]