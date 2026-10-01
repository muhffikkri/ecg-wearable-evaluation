"""Regenerate the per-activity SQI recap tables and figures.

Writes, under ``results/``:

* ``sqi_recap.md``                    -- the recap tables and the acceptance
  decision table (one row per activity, one column per index)
* ``sqi_recap.csv``                   -- the full mean/median/SD recap as data
* ``sqi_acceptance_recap.csv``        -- per-activity acceptance distribution
* ``sqi_acceptance_decision.csv``     -- per-activity decided acceptance level
* ``figures/recap_indeks_per_aktivitas.html``  -- sebaran tiap indeks per aktivitas
* ``figures/penerimaan_per_aktivitas.html``    -- kategori penerimaan tiap indeks

The analysis runs without a cache, so nothing under ``processed/`` is touched.

Usage::

    python scripts/sqi_recap_report.py [--config zhao_zhang] [--out results]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ecg_eval.analysis import (  # noqa: E402
    ResultCache,
    acceptance_recap,
    acceptance_summary,
    activity_recap,
    results_to_frame,
    run_analysis,
)
from ecg_eval.annotation import storage  # noqa: E402
from ecg_eval.config import load_config  # noqa: E402
from ecg_eval.ingestion import ingest  # noqa: E402
from ecg_eval.models.result import SQI_KEYS  # noqa: E402
from ecg_eval.visualization import (  # noqa: E402
    FUSION_LABELS_ID,
    acceptance_coverage_table,
    acceptance_decision_legend,
    acceptance_decision_table,
    acceptance_figure,
    acceptance_recap_legend,
    acceptance_recap_table,
    activity_recap_figure,
    activity_recap_legend,
    activity_recap_table,
    position_label,
    sqi_label_id,
)


def md_table(frame: pd.DataFrame) -> str:
    """Render a DataFrame as a Markdown table without requiring ``tabulate``."""
    if frame is None or frame.empty:
        return "_Tidak ada data._\n"
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = [
        "| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
        for row in frame.itertuples(index=False)
    ]
    return "\n".join([header, rule, *body]) + "\n"


def compact_recap(recap: pd.DataFrame) -> pd.DataFrame:
    """One row per activity, one column per index: the requested table shape."""
    if recap is None or recap.empty:
        return pd.DataFrame()
    table: dict[str, Any] = {"Aktivitas": []}
    for key in SQI_KEYS:
        table[sqi_label_id(key)] = []
    table["Total Frame"] = []
    for _, row in recap.iterrows():
        table["Aktivitas"].append(position_label(row["position"]))
        for key in SQI_KEYS:
            value = row.get(f"{key}_mean")
            table[sqi_label_id(key)].append(
                "-" if value is None or not pd.notna(value) else f"{float(value):.4f}"
            )
        table["Total Frame"].append(int(row["n_frames"]))
    return pd.DataFrame(table)


def fusion_class_counts(frame: pd.DataFrame) -> pd.DataFrame:
    """How many frames each step rated into each class, per activity."""
    if frame is None or frame.empty:
        return pd.DataFrame()
    valid = frame[frame["valid"]] if "valid" in frame else frame
    if valid.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    order = {"SUPINE": 0, "SITTING": 1, "STANDING": 2}
    positions = sorted(set(valid["position"]), key=lambda name: order.get(str(name), 99))
    for position in positions:
        group = valid[valid["position"] == position]
        row: dict[str, Any] = {"Aktivitas": position_label(position), "Total Frame": len(group)}
        for column, label in (
            ("fusion_class", "Fusi heuristik"),
            ("quality_class", "Evaluasi fuzzy"),
        ):
            for level, name in FUSION_LABELS_ID.items():
                count = int((group[column] == level).sum())
                row[f"{label} · {name} (n)"] = count
                row[f"{label} · {name} (%)"] = f"{100.0 * count / len(group):.1f}%"
        rows.append(row)
    return pd.DataFrame(rows)


def write_figures(frame: pd.DataFrame, directory: Path) -> list[tuple[str, Path]]:
    """Write every recap figure as a self-contained interactive HTML file.

    HTML rather than PNG because the plotting library's static export needs an
    extra binary package that is not installed here, and a self-contained HTML
    carries its own JavaScript so it opens offline in any browser.
    """
    directory.mkdir(parents=True, exist_ok=True)
    written: list[tuple[str, Path]] = []
    builders = (
        (
            "recap_indeks_per_aktivitas",
            "Sebaran tiap indeks kualitas sinyal per aktivitas",
            activity_recap_figure(frame),
        ),
        (
            "penerimaan_per_aktivitas",
            "Kategori penerimaan tiap indeks per aktivitas",
            acceptance_figure(frame),
        ),
    )
    for name, description, figure in builders:
        path = directory / f"{name}.html"
        figure.write_html(path, include_plotlyjs=True, full_html=True)
        written.append((description, path))
        print(f"wrote {path}")
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="default", help="configuration name")
    parser.add_argument("--out", default="results", help="output directory")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = ingest(config=config)
    annotations_dir = REPO / str(config.get("paths.annotations_dir", "annotations"))
    annotations = storage.load_annotations(annotations_dir)
    if not annotations:
        print(f"no annotations in {annotations_dir}; nothing to analyse", file=sys.stderr)
        return 1

    analysis = run_analysis(
        dataset, annotations, config, cache=ResultCache(None, enabled=False)
    )
    frame = results_to_frame(analysis["results"])
    counts = analysis["counts"]
    print(
        f"analysed {counts['tasks']} frame(s): {counts['valid']} valid, "
        f"{counts['invalid']} invalid"
    )

    recap = activity_recap(frame)
    if recap.empty:
        print("no valid frames to summarise", file=sys.stderr)
        return 1
    accepted = acceptance_recap(frame)
    summary = acceptance_summary(frame)

    compact = compact_recap(recap)
    full = activity_recap_table(recap)
    decision = acceptance_decision_table(summary)
    coverage = acceptance_coverage_table(summary)
    acceptance_table = acceptance_recap_table(accepted)
    classes = fusion_class_counts(frame)
    figures = write_figures(frame, out_dir / "figures")

    pending = [path for path, _ in config.pending_parameters()]
    figure_lines = [
        f"- [{description}](figures/{path.name})" for description, path in figures
    ]
    document = "\n".join(
        [
            "# Rekapitulasi Indeks Kualitas Sinyal per Aktivitas",
            "",
            f"- Konfigurasi: `{config.name}` versi `{config.version}`",
            f"- Metodologi: {config.methodology}",
            f"- Frame dianalisis: {counts['tasks']} ({counts['valid']} valid, "
            f"{counts['invalid']} tidak valid)",
            "- Setiap baris menggabungkan seluruh frame 10 detik pada satu aktivitas.",
            "",
            "## 1. Rekapitulasi rata-rata tiap indeks",
            "",
            md_table(compact),
            "",
            "## 2. Kategori penerimaan tiap indeks",
            "",
            "Satu kolom per indeks, satu baris per aktivitas, satu kategori per sel.",
            "",
            md_table(decision),
            acceptance_decision_legend(),
            "",
            "### 2.1 Kekuatan keputusan",
            "",
            (
                "Persentase frame yang benar-benar berada pada kategori di atas. "
                "Kategori ditentukan per frame 10 detik lalu diambil yang terbanyak, "
                "bukan dengan menerapkan kriteria pada nilai rata-rata."
            ),
            "",
            md_table(coverage),
            "",
            "## 3. Rekapitulasi lengkap (rata-rata, tengah, simpangan baku)",
            "",
            md_table(full),
            activity_recap_legend(),
            "",
            "## 4. Sebaran tiap tingkat penerimaan",
            "",
            md_table(acceptance_table),
            acceptance_recap_legend(),
            "",
            "## 5. Kelas kualitas per aktivitas",
            "",
            md_table(classes),
            "",
            "## 6. Figure",
            "",
            *figure_lines,
            "",
            "## 7. Status reproduksi",
            "",
            (
                "- Seluruh rumus, kriteria penerimaan, fungsi keanggotaan, vektor bobot "
                "dan ambang keputusan mengikuti artikel Zhao & Zhang (2018) dan "
                "tercatat sebagai terverifikasi."
            ),
            (
                f"- Parameter yang belum tercetak di artikel: {', '.join(pending)}. "
                "Karena itu hasil ini dilaporkan sebagai reproduksi terstruktur, "
                "bukan reproduksi setia."
                if pending
                else "- Tidak ada parameter yang menunggu verifikasi."
            ),
            (
                "- Kategori penerimaan dan kelas kualitas menggambarkan kualitas "
                "rekaman, bukan kondisi kesehatan peserta."
            ),
            "",
        ]
    )

    (out_dir / "sqi_recap.md").write_text(document, encoding="utf-8")
    full.to_csv(out_dir / "sqi_recap.csv", index=False)
    accepted.to_csv(out_dir / "sqi_acceptance_recap.csv", index=False)
    summary.to_csv(out_dir / "sqi_acceptance_decision.csv", index=False)
    print(f"wrote {out_dir / 'sqi_recap.md'}")
    print(f"wrote {out_dir / 'sqi_recap.csv'}")
    print(f"wrote {out_dir / 'sqi_acceptance_recap.csv'}")
    print(f"wrote {out_dir / 'sqi_acceptance_decision.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
