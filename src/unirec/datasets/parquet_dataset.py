from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset

from unirec.features.sequences import normalize_list, pad_sequence_batch
from unirec.tasks.cvr import build_cvr_labels

from .schema import feature_columns_from_config, resolve_runtime_config, scan_parquet_schema, validate_feature_config


DEFAULT_LIST_MAX_LEN = 32


def _normalize_scalar(value: Any) -> int:
    if value is None:
        return 0
    return int(value)


class ParquetRowDataset(Dataset[dict[str, Any]]):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def load_parquet_rows(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data_config = config["data"]
    parquet_path = data_config.get("path")
    if not parquet_path:
        raise ValueError("data.path must point to a local parquet file before training starts")

    parquet_path = str(Path(parquet_path).expanduser())
    schema = scan_parquet_schema(parquet_path)
    validate_feature_config(config, set(schema))
    runtime_config = resolve_runtime_config(config, set(schema))

    table = pq.read_table(parquet_path, columns=feature_columns_from_config(runtime_config))
    max_rows = data_config.get("max_rows")
    if max_rows:
        table = table.slice(0, int(max_rows))

    raw_rows = table.to_pylist()
    allowed_labels = set(data_config.get("allowed_label_values", []))

    normalized_rows: list[dict[str, Any]] = []
    for raw_row in raw_rows:
        label_value = _normalize_scalar(raw_row[data_config["label_column"]])
        if allowed_labels and label_value not in allowed_labels:
            continue

        normalized_row: dict[str, Any] = {}
        for column, spec in schema.items():
            if column not in raw_row:
                continue
            value = raw_row[column]
            if spec.kind == "list_int":
                normalized_row[column] = normalize_list(value, dtype="int")
            elif spec.kind == "list_float":
                normalized_row[column] = normalize_list(value, dtype="float")
            elif spec.kind in {"scalar_int", "scalar_float"}:
                normalized_row[column] = _normalize_scalar(value)
            else:
                normalized_row[column] = value
        normalized_rows.append(normalized_row)
    return normalized_rows, {"schema": schema, "runtime_config": runtime_config}


def build_split_dataset(rows: list[dict[str, Any]], indices: list[int]) -> ParquetRowDataset:
    return ParquetRowDataset([rows[index] for index in indices])


def build_collate_fn(config: dict[str, Any]):
    data_config = config["data"]
    feature_config = config["features"]
    positive_values = set(int(value) for value in data_config["positive_label_values"])

    def collate_fn(samples: list[dict[str, Any]]) -> dict[str, Any]:
        batch: dict[str, Any] = {
            "labels": build_cvr_labels(
                [sample[data_config["label_column"]] for sample in samples],
                positive_values=positive_values,
            ),
            "static_sparse": {},
            "list_sparse": {},
            "dense_only": {},
            "aligned_dense_pairs": {},
            "domain_sequences": {},
            "context": {},
        }

        static_columns = (
            feature_config.get("id_features", [])
            + feature_config.get("user_sparse", [])
            + feature_config.get("item_sparse", [])
        )
        for column in static_columns:
            batch["static_sparse"][column] = torch.tensor(
                [_normalize_scalar(sample.get(column, 0)) for sample in samples],
                dtype=torch.long,
            )

        list_sparse_columns = feature_config.get("user_list_sparse", []) + feature_config.get("item_list_sparse", [])
        for column in list_sparse_columns:
            values, mask, lengths = pad_sequence_batch(
                [normalize_list(sample.get(column), dtype="int") for sample in samples],
                max_len=DEFAULT_LIST_MAX_LEN,
                pad_value=0,
                dtype=torch.long,
            )
            batch["list_sparse"][column] = {"values": values, "mask": mask, "lengths": lengths}

        for column in feature_config.get("user_dense_only", []):
            values, mask, lengths = pad_sequence_batch(
                [normalize_list(sample.get(column), dtype="float") for sample in samples],
                max_len=DEFAULT_LIST_MAX_LEN,
                pad_value=0.0,
                dtype=torch.float32,
            )
            batch["dense_only"][column] = {"values": values, "mask": mask, "lengths": lengths}

        for pair in feature_config.get("aligned_dense_pairs", []):
            ids, mask, lengths = pad_sequence_batch(
                [normalize_list(sample.get(pair["int_column"]), dtype="int") for sample in samples],
                max_len=DEFAULT_LIST_MAX_LEN,
                pad_value=0,
                dtype=torch.long,
            )
            dense_values, dense_mask, _ = pad_sequence_batch(
                [normalize_list(sample.get(pair["dense_column"]), dtype="float") for sample in samples],
                max_len=DEFAULT_LIST_MAX_LEN,
                pad_value=0.0,
                dtype=torch.float32,
            )
            batch["aligned_dense_pairs"][pair["name"]] = {
                "ids": ids,
                "weights": dense_values,
                "mask": torch.minimum(mask, dense_mask),
                "lengths": lengths,
            }

        total_sequence_lengths = torch.zeros(len(samples), dtype=torch.long)
        for domain, domain_cfg in feature_config.get("sequence_domains", {}).items():
            batch["domain_sequences"][domain] = {}
            max_len = int(domain_cfg.get("max_len", DEFAULT_LIST_MAX_LEN))
            for column in domain_cfg.get("columns", []):
                values, mask, lengths = pad_sequence_batch(
                    [normalize_list(sample.get(column), dtype="int") for sample in samples],
                    max_len=max_len,
                    pad_value=0,
                    dtype=torch.long,
                )
                total_sequence_lengths += lengths
                batch["domain_sequences"][domain][column] = {
                    "values": values,
                    "mask": mask,
                    "lengths": lengths,
                }

        batch["context"]["sequence_length_total"] = total_sequence_lengths
        for column in data_config.get("id_columns", []) + data_config.get("passthrough_columns", []):
            batch["context"][column] = torch.tensor(
                [_normalize_scalar(sample.get(column, 0)) for sample in samples],
                dtype=torch.long,
            )
        return batch

    return collate_fn


def build_dataloader(
    dataset: ParquetRowDataset,
    config: dict[str, Any],
    shuffle: bool,
) -> DataLoader:
    dataloader_config = config["data"]["dataloader"]
    return DataLoader(
        dataset,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=shuffle,
        num_workers=int(dataloader_config.get("num_workers", 0)),
        pin_memory=bool(dataloader_config.get("pin_memory", False)),
        collate_fn=build_collate_fn(config),
    )
