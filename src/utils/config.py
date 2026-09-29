"""Configuration loading and path resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must contain a mapping: {config_path}")
    for section in ("paths", "classes", "ssl", "detector", "probe", "pipeline", "weather"):
        if section not in config:
            raise KeyError(f"Missing configuration section: {section}")
    config["_config_path"] = str(config_path)
    return config


def configured_path(config: dict[str, Any], key: str, must_exist: bool = False) -> Path:
    try:
        value = config["paths"][key]
    except KeyError as error:
        raise KeyError(f"Missing paths.{key} in configuration") from error
    path = Path(value).expanduser().resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(path)
    return path


def workspace_path(config: dict[str, Any], *parts: str) -> Path:
    return configured_path(config, "workspace") / Path(*parts)

