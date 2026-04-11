from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    kind: str


def infer_column_kind(field: pa.Field) -> str:
    if pa.types.is_list(field.type):
        value_type = field.type.value_type
        if pa.types.is_floating(value_type):
            return "list_float"
        if pa.types.is_integer(value_type):
            return "list_int"
    if pa.types.is_integer(field.type):
        return "scalar_int"
    if pa.types.is_floating(field.type):
        return "scalar_float"
    return "unknown"


def scan_parquet_schema(parquet_path: str) -> dict[str, ColumnSpec]:
    parquet_schema = pq.read_schema(parquet_path)
    return {
        field.name: ColumnSpec(name=field.name, kind=infer_column_kind(field))
        for field in parquet_schema
    }


def feature_columns_from_config(config: dict[str, Any]) -> list[str]:
    features = config["features"]
    data = config["data"]
    columns: list[str] = []
    for key in [
        "id_features",
        "user_sparse",
        "user_list_sparse",
        "user_dense_only",
        "item_sparse",
        "item_list_sparse",
    ]:
        columns.extend(features.get(key, []))
    for pair in features.get("aligned_dense_pairs", []):
        columns.append(pair["int_column"])
        columns.append(pair["dense_column"])
    for domain_cfg in features.get("sequence_domains", {}).values():
        columns.extend(domain_cfg.get("columns", []))
    columns.extend(data.get("passthrough_columns", []))
    columns.append(data["label_column"])
    columns.append(data["timestamp_column"])
    deduped: list[str] = []
    seen: set[str] = set()
    for column in columns:
        if column not in seen:
            seen.add(column)
            deduped.append(column)
    return deduped


def resolve_runtime_config(config: dict[str, Any], available_columns: set[str]) -> dict[str, Any]:
    runtime_config = copy.deepcopy(config)
    features = runtime_config["features"]

    for key in [
        "id_features",
        "user_sparse",
        "user_list_sparse",
        "user_dense_only",
        "item_sparse",
        "item_list_sparse",
    ]:
        features[key] = [column for column in features.get(key, []) if column in available_columns]

    features["aligned_dense_pairs"] = [
        pair
        for pair in features.get("aligned_dense_pairs", [])
        if pair["int_column"] in available_columns and pair["dense_column"] in available_columns
    ]

    resolved_domains = {}
    for domain, domain_cfg in features.get("sequence_domains", {}).items():
        columns = [column for column in domain_cfg.get("columns", []) if column in available_columns]
        if columns:
            resolved_domains[domain] = {**domain_cfg, "columns": columns}
    features["sequence_domains"] = resolved_domains
    return runtime_config


def validate_feature_config(config: dict[str, Any], available_columns: set[str]) -> None:
    required_columns = {
        config["data"]["label_column"],
        config["data"]["timestamp_column"],
        *config["data"].get("id_columns", []),
    }
    missing = [column for column in required_columns if column not in available_columns]
    if missing:
        preview = ", ".join(sorted(missing[:10]))
        raise ValueError(f"Missing configured columns in parquet schema: {preview}")
