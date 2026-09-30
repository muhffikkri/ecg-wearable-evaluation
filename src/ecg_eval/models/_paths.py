"""Path trimming for provenance display.

Windows and POSIX recordings of the same tree differ only in the separator, and
provenance is shown next to the recording root, so the root prefix has to be
removed under both. Kept in one place so the dataset viewer and the generated
JSONL cannot disagree about what a "relative" source path means.
"""

from __future__ import annotations

import re

_SEPARATORS = re.compile(r"[\\/]")


def normalise_separators(path: str) -> str:
    """Collapse every separator to ``/`` for comparison only."""
    return _SEPARATORS.sub("/", str(path))


def strip_root(path: str, root: str) -> str:
    """Return ``path`` relative to ``root``, or unchanged when it is outside.

    Accepts a path written with either separator and a root given with a
    trailing separator. A firmware path recorded on the Raspberry Pi is absolute
    and unrelated to the local root, so it is returned verbatim rather than
    being mangled into something that looks local.
    """
    if not path:
        return ""
    if not root:
        return path
    candidate = normalise_separators(path)
    prefix = normalise_separators(root).rstrip("/")
    if not prefix:
        return path
    if candidate == prefix:
        return ""
    if candidate.startswith(prefix + "/"):
        return candidate[len(prefix) + 1:]
    return path


__all__ = ["normalise_separators", "strip_root"]