"""Central configuration system.

Every scientific parameter used anywhere in the pipeline is loaded from here.
No module is allowed to bury a numeric constant inside a function
(IDEA.md sections 42 and 62).

Values that originate from Zhao & Zhang (2018) carry a ``source`` field of
``zhao_zhang_2018`` once verified, or ``pending_verification`` while still
unconfirmed. :func:`Config.pending_parameters` exposes the unconfirmed ones so
the UI can state honestly that a run is not yet a strict reproduction.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import yaml

# Repository root: src/ecg_eval/config.py -> parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"

_SOURCE_ZHAO = "zhao_zhang_2018"
_SOURCE_PENDING = "pending_verification"

#: Sentinel distinguishing "argument omitted" from an explicit ``None`` value.
_UNSET = object()


@dataclass(frozen=True)
class Config:
    """An immutable, fully resolved configuration."""

    name: str
    data: dict[str, Any]
    path: Path
    parent_names: tuple[str, ...] = ()

    # -- access ---------------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        """Fetch a value by dotted path, e.g. ``cfg.get("p_sqi.qrs_band_hz")``."""
        node: Any = self.data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def section(self, dotted: str) -> dict[str, Any]:
        value = self.get(dotted, {})
        return value if isinstance(value, dict) else {}

    @property
    def version(self) -> str:
        return str(self.get("meta.config_version", "0.0.0"))

    @property
    def is_strict_reproduction(self) -> bool:
        return bool(self.get("meta.is_strict_reproduction", False))

    @property
    def methodology(self) -> str:
        return str(self.get("meta.methodology", "unknown"))

    @property
    def adaptation_note(self) -> str:
        return str(self.get("meta.adaptation", "") or "")

    # -- provenance -----------------------------------------------------
    def iter_provenance(self, node: Any = _UNSET, path: str = "") -> Iterator[tuple[str, str, Any]]:
        """Yield ``(dotted_path, source, value)`` for every annotated parameter.

        Uses a sentinel rather than ``None`` as the default so that a config
        value which is genuinely ``None`` (``highpass_hz: null``) is walked
        instead of resetting the traversal to the root.
        """
        if node is _UNSET:
            node = self.data
        if isinstance(node, dict):
            if "value" in node and "source" in node:
                yield path, str(node["source"]), node["value"]
                return
            for key, child in node.items():
                child_path = f"{path}.{key}" if path else str(key)
                yield from self.iter_provenance(child, child_path)

    def pending_parameters(self) -> list[tuple[str, Any]]:
        """Parameters still awaiting verification against the published paper."""
        return [(p, v) for p, s, v in self.iter_provenance() if s == _SOURCE_PENDING]

    def verified_parameters(self) -> list[tuple[str, Any]]:
        return [(p, v) for p, s, v in self.iter_provenance() if s == _SOURCE_ZHAO]

    # -- reproducibility ------------------------------------------------
    def scientific_fingerprint(self) -> str:
        """Hash of everything that can change a numeric result.

        Used as part of the analysis cache key so cached results are never
        reused across a configuration change (IDEA.md sections 41 and 50).
        Excludes purely operational keys such as paths and cache settings.
        """
        relevant = copy.deepcopy(self.data)
        for key in ("paths", "cache"):
            relevant.pop(key, None)
        relevant.get("meta", {}).pop("config_version", None)
        blob = json.dumps(relevant, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.data)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_config(name: str = "default", config_dir: Path | None = None) -> Config:
    """Load a configuration by name, resolving its ``extends`` chain recursively.

    Cycle detection is included, so a config can inherit from another config as
    well as from YAML. The innermost config is applied first and each parent is
    merged over it, so the config named in the call always wins.
    """
    config_dir = Path(config_dir) if config_dir is not None else CONFIG_DIR
    chain: list[str] = []
    seen: set[str] = set()

    current = name
    resolved: dict[str, Any] = {}
    # Innermost config first, then apply parents over it, so the child wins.
    collected: list[tuple[str, dict[str, Any]]] = []

    while True:
        if current in seen:
            raise ValueError(f"circular config inheritance: {' -> '.join(chain + [current])}")
        seen.add(current)
        path = config_dir / f"{current}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"config not found: {path}")
        raw = _load_yaml(path)
        collected.append((current, raw))
        chain.append(current)
        parent = raw.pop("extends", None)
        if not parent:
            break
        current = str(parent)

    for _, raw in reversed(collected):
        resolved = _deep_merge(resolved, raw)

    resolved.setdefault("meta", {})
    resolved["meta"]["config_name"] = name

    return Config(name=name, data=resolved, path=config_dir / f"{name}.yaml", parent_names=tuple(chain))


def available_configs(config_dir: Path | None = None) -> list[str]:
    config_dir = Path(config_dir) if config_dir is not None else CONFIG_DIR
    return sorted(p.stem for p in config_dir.glob("*.yaml"))


__all__ = [
    "Config",
    "CONFIG_DIR",
    "REPO_ROOT",
    "available_configs",
    "load_config",
]
