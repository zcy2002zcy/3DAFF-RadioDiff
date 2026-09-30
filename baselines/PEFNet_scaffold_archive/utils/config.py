"""Configuration loading with relative-path-aware callers."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load YAML and return both its mapping and resolved path.

    The checked-in YAML uses JSON-compatible YAML syntax, so a clean environment
    can run the parser with the standard library. PyYAML remains supported for
    ordinary YAML syntax once project requirements are installed.
    """

    config_path = Path(path).expanduser().resolve()
    text = config_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        data = json.loads(text)
    else:
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise TypeError(f"Configuration root must be a mapping: {config_path}")
    base_name = data.pop("base_config", None)
    if base_name is not None:
        base, _ = load_config(config_path.parent / str(base_name))
        data = _deep_merge(base, data)
    return _expand_environment(data), config_path


def _expand_environment(value: Any) -> Any:
    """Expand ${NAME} placeholders without changing non-string config values."""

    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        if "${" in expanded:
            raise EnvironmentError(
                f"Unresolved environment variable in configuration value: {value}"
            )
        return expanded
    return value


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result
