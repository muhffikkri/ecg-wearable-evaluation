#!/usr/bin/env python3
"""Convert Raspberry Pi recordings into ``json-web.jsonl`` datasets.

This is the command-line entry point for the same pipeline the Reconstruction
tab drives. It reconstructs canonical frames from a recording directory and
writes a dataset that validates against ``templates/json-web.jsonl``, plus a
manifest recording where every field came from.

The recording is never modified. Output must resolve outside the source tree and
the writer refuses otherwise, so a mistyped ``--output`` cannot overwrite a
recording.

Examples::

    # Preview only, write nothing.
    python scripts/build_jsonl_dataset.py data/29-09-2026/Bryan --dry-run

    # Build every recording found under data/.
    python scripts/build_jsonl_dataset.py data --output processed

    # One recording, with a specific filename.
    python scripts/build_jsonl_dataset.py data/29-09-2026/Bryan \\
        --output processed --filename bryan.jsonl

The exit code is 0 on success, 1 if validation failed, and 2 if the arguments or
a recording path were unusable, so this is safe to use in a script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from ecg_eval.ingestion.raw_reader import discover_raw_roots  # noqa: E402
from ecg_eval.jsonl import build_record, write_jsonl  # noqa: E402
from ecg_eval.reconstruction import reconstruct_directory  # noqa: E402

EXIT_OK = 0
EXIT_INVALID = 1
EXIT_USAGE = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="build_jsonl_dataset.py",
        description=(
            "Reconstruct frames from a Raspberry Pi recording and write them as "
            "json-web.jsonl."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples::", 1)[-1],
    )
    parser.add_argument(
        "source",
        nargs="+",
        type=Path,
        help=(
            "Recording directories, or a date/data directory containing "
            "recordings to discover."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=REPO_ROOT / "processed",
        help="Output directory. Must be outside the source tree (default: processed).",
    )
    parser.add_argument(
        "--filename",
        help="Output filename. Defaults to the subject id, lowercased, with .jsonl.",
    )
    parser.add_argument(
        "--subject-id",
        help="Override the subject id. Defaults to the recording directory name.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report, but write no files.",
    )
    parser.add_argument(
        "--allow-invalid",
        action="store_true",
        help=(
            "Include records that fail schema validation. Off by default: a "
            "dataset published with known-bad records in it looks complete."
        ),
    )
    parser.add_argument(
        "--exclude-unresolved",
        action="store_true",
        help="Drop frames that could not be linked by recorded evidence.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the result summary as JSON instead of text.",
    )
    return parser


def is_recording(path: Path) -> bool:
    """True if ``path`` looks like a Raspberry Pi recording root."""
    return (path / "session.json").is_file() or (path / "raw_adc").is_dir() or (
        (path / "calibrated").is_dir() and (path / "model_ready").is_dir()
    )


def resolve_roots(sources: list[Path]) -> tuple[list[Path], list[str]]:
    """Turn the CLI arguments into concrete recording roots.

    Accepts a recording, a date folder holding recordings, or a ``data`` folder
    holding date folders, because all three are plausible things to point at and
    guessing wrong should not be a usage error.
    """
    roots: list[Path] = []
    errors: list[str] = []
    for source in sources:
        source = source.expanduser()
        if not source.exists():
            errors.append(f"not found: {source}")
            continue
        if source.is_file():
            errors.append(f"expected a directory, got a file: {source}")
            continue
        if is_recording(source):
            roots.append(source)
            continue
        # data/ of date folders, and a date folder of recordings, are both
        # discovered here; discover_raw_roots only understands the former.
        discovered = list(discover_raw_roots(source))
        discovered.extend(
            sorted(child for child in source.iterdir() if child.is_dir() and is_recording(child))
        )
        if discovered:
            roots.extend(discovered)
        else:
            errors.append(f"no recordings found under: {source}")
    # Preserve order but drop duplicates, so overlapping arguments are harmless.
    unique: list[Path] = []
    for root in roots:
        if root not in unique:
            unique.append(root)
    return unique, errors


def describe(root: Path, subject_id: str | None, *, dry_run: bool, allow_invalid: bool,
             exclude_unresolved: bool, filename: str | None, output: Path) -> dict:
    """Reconstruct and write one recording. Returns a summary dict."""
    result = reconstruct_directory(
        root,
        subject_id=subject_id,
        include_unresolved=not exclude_unresolved,
    )

    summary: dict = {
        "source_root": str(root),
        "subject_id": result.subject_id,
        "session_id": result.session_id,
        "frames_reconstructed": len(result.frames),
        "frames_unresolved": len(result.unresolved()),
        "reconstruction_warnings": result.warnings,
        "output": None,
        "manifest": None,
        "records_written": 0,
        "records_skipped": [],
        "unavailable_fields": [],
        "dry_run": dry_run,
    }

    if not result.frames:
        summary["error"] = "no frames could be reconstructed"
        return summary

    write_result = write_jsonl(
        result.frames,
        output,
        source_root=root,
        filename=filename,
        dry_run=dry_run,
        allow_invalid=allow_invalid,
    )
    summary["records_written"] = write_result.written
    summary["records_skipped"] = write_result.skipped
    summary["unavailable_fields"] = write_result.unavailable_fields
    summary["output"] = write_result.output_path
    summary["manifest"] = write_result.manifest_path

    if dry_run:
        # Confirm what would be written actually survives a read-back.
        statuses = {}
        for frame in result.frames:
            built = build_record(frame)
            statuses[built.validation.status] = statuses.get(built.validation.status, 0) + 1
        summary["validation_statuses"] = statuses
        sample = build_record(result.frames[0]).record
        summary["sample_record_keys"] = list(sample)
    return summary


def print_text(summary: dict) -> None:
    print(f"recording      : {summary['source_root']}")
    print(f"subject/session: {summary['subject_id']} / {summary['session_id']}")
    print(
        f"frames         : {summary['frames_reconstructed']}"
        f" reconstructed, {summary['frames_unresolved']} unresolved"
    )
    for warning in summary["reconstruction_warnings"]:
        print(f"  warning      : {warning}")
    if summary.get("error"):
        print(f"error          : {summary['error']}")
        return
    verb = "would write" if summary["dry_run"] else "wrote"
    print(f"{verb:<15}: {summary['records_written']} records -> {summary['output']}")
    if summary["manifest"]:
        print(f"manifest       : {summary['manifest']}")
    if summary["records_skipped"]:
        print(f"skipped        : {len(summary['records_skipped'])}")
        for frame_id in summary["records_skipped"][:10]:
            print(f"  - {frame_id}")
    if summary.get("validation_statuses"):
        print(f"validation     : {summary['validation_statuses']}")
    if summary["unavailable_fields"]:
        print(f"unavailable    : {', '.join(summary['unavailable_fields'])}")
        print("                  (absent from the recording; omitted, not invented)")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    roots, errors = resolve_roots(args.source)
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    if not roots:
        return EXIT_USAGE

    summaries = []
    failed = False
    for root in roots:
        subject = args.subject_id or root.name
        filename = args.filename
        if len(roots) > 1 and filename:
            print(
                f"error: --filename cannot be used with several recordings; "
                f"{len(roots)} were found",
                file=sys.stderr,
            )
            return EXIT_USAGE
        try:
            summary = describe(
                root,
                subject,
                dry_run=args.dry_run,
                allow_invalid=args.allow_invalid,
                exclude_unresolved=args.exclude_unresolved,
                filename=filename,
                output=args.output,
            )
        except ValueError as exc:
            # Raised by the writer when the output lands inside the recording.
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_USAGE
        except Exception as exc:  # noqa: BLE001
            print(f"error: {root}: {exc}", file=sys.stderr)
            failed = True
            continue
        if summary.get("error") or summary["records_written"] == 0:
            failed = True
        summaries.append(summary)

    if args.json:
        print(json.dumps(summaries, indent=2))
    else:
        for index, summary in enumerate(summaries):
            if index:
                print()
            print_text(summary)

    return EXIT_INVALID if failed else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())