from __future__ import annotations

from typing import Any


def time_based_split(
    rows: list[dict[str, Any]],
    timestamp_column: str,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    order: str = "ascending",
) -> dict[str, list[int]]:
    total_ratio = train_ratio + valid_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("train/valid/test ratios must sum to 1.0")

    indexed_rows = list(enumerate(rows))
    reverse = order == "descending"
    indexed_rows.sort(key=lambda item: item[1].get(timestamp_column, 0), reverse=reverse)

    total_size = len(indexed_rows)
    train_end = int(total_size * train_ratio)
    valid_end = train_end + int(total_size * valid_ratio)

    return {
        "train": [idx for idx, _ in indexed_rows[:train_end]],
        "valid": [idx for idx, _ in indexed_rows[train_end:valid_end]],
        "test": [idx for idx, _ in indexed_rows[valid_end:]],
    }
