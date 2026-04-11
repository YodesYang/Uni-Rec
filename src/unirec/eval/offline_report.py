from __future__ import annotations

from typing import Any

from unirec.tasks.metrics import summarize_binary_metrics


def _bucket_name(left: int, right: int | None) -> str:
    if right is None:
        return f"[{left},+inf)"
    return f"[{left},{right})"


def build_offline_report(
    labels: list[float],
    preds: list[float],
    sequence_lengths: list[int],
    eval_config: dict[str, Any],
) -> dict[str, Any]:
    metrics = summarize_binary_metrics(labels, preds)
    buckets = eval_config.get("sequence_length_buckets", [])
    bucket_metrics: dict[str, dict[str, float]] = {}

    for idx, left in enumerate(buckets):
        right = buckets[idx + 1] if idx + 1 < len(buckets) else None
        bucket_indices = []
        for row_idx, seq_len in enumerate(sequence_lengths):
            if right is None:
                matched = seq_len >= left
            else:
                matched = left <= seq_len < right
            if matched:
                bucket_indices.append(row_idx)
        if not bucket_indices:
            continue
        bucket_labels = [labels[row_idx] for row_idx in bucket_indices]
        bucket_preds = [preds[row_idx] for row_idx in bucket_indices]
        bucket_metrics[_bucket_name(left, right)] = summarize_binary_metrics(bucket_labels, bucket_preds)

    return {
        "overall": metrics,
        "sequence_length_buckets": bucket_metrics,
    }
