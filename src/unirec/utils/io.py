from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    ensure_dir(destination.parent)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def save_yaml(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    ensure_dir(destination.parent)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=False, sort_keys=False)


def resolve_default_config(config_root: str | Path) -> dict[str, Any]:
    config_root = Path(config_root)
    config_paths = [
        config_root / "dataset" / "taac_demo.yaml",
        config_root / "features" / "taac_demo.yaml",
        config_root / "model" / "mlp_cvr.yaml",
        config_root / "train" / "default.yaml",
        config_root / "eval" / "default.yaml",
    ]
    merged: dict[str, Any] = {}
    for path in config_paths:
        merged = deep_merge(merged, load_yaml(path))
    return merged
