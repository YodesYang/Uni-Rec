from __future__ import annotations

import math

import numpy as np


def auc_score(labels: list[float], preds: list[float]) -> float:
    if not labels:
        return 0.0
    labels_np = np.asarray(labels, dtype=np.float64)
    preds_np = np.asarray(preds, dtype=np.float64)

    positive = float(labels_np.sum())
    negative = float(len(labels_np) - positive)
    if positive == 0.0 or negative == 0.0:
        return 0.5

    order = np.argsort(preds_np)
    sorted_preds = preds_np[order]
    ranks = np.empty(len(preds_np), dtype=np.float64)

    start = 0
    current_rank = 1.0
    while start < len(sorted_preds):
        end = start + 1
        while end < len(sorted_preds) and sorted_preds[end] == sorted_preds[start]:
            end += 1
        average_rank = (current_rank + (current_rank + (end - start) - 1.0)) / 2.0
        ranks[order[start:end]] = average_rank
        current_rank += end - start
        start = end

    positive_rank_sum = ranks[labels_np == 1.0].sum()
    return float((positive_rank_sum - positive * (positive + 1.0) / 2.0) / (positive * negative))


def logloss_score(labels: list[float], preds: list[float]) -> float:
    if not labels:
        return 0.0
    eps = 1e-7
    total = 0.0
    for label, pred in zip(labels, preds, strict=True):
        clipped = min(max(pred, eps), 1.0 - eps)
        total += -(label * math.log(clipped) + (1.0 - label) * math.log(1.0 - clipped))
    return total / max(len(labels), 1)


def pcoc_score(labels: list[float], preds: list[float]) -> float:
    if not labels:
        return 0.0
    actual = sum(labels)
    expected = sum(preds)
    if actual <= 0.0:
        return 0.0
    return expected / actual


def summarize_binary_metrics(labels: list[float], preds: list[float]) -> dict[str, float]:
    return {
        "auc": auc_score(labels, preds),
        "logloss": logloss_score(labels, preds),
        "pcoc": pcoc_score(labels, preds),
        "avg_pred": float(sum(preds) / max(len(preds), 1)),
        "positive_rate": float(sum(labels) / max(len(labels), 1)),
    }
