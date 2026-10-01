"""Reading ``{value, source}`` provenance parameters.

Scientific parameters in the configuration carry their provenance next to the
number, so every reader has to unwrap that form. One implementation means a
parameter cannot be read two different ways in two modules, and an annotated
value can never be mistaken for a plain one.
"""

from __future__ import annotations

from typing import Any, Mapping


def unwrap(value: Any, default: Any = None) -> Any:
    """Return the bare value of a ``{value, source}`` parameter.

    A mapping without a ``value`` key, or an explicit ``None``, falls back to
    ``default``. Everything else is returned unchanged, so the helper is safe to
    apply to a value that was never annotated.
    """
    if isinstance(value, Mapping):
        return value.get("value", default)
    return default if value is None else value


__all__ = ["unwrap"]
