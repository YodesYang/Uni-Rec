from __future__ import annotations

import torch

from .losses import binary_logloss


def build_cvr_labels(raw_labels: list[int | float], positive_values: set[int]) -> torch.Tensor:
    binary = [1.0 if int(label) in positive_values else 0.0 for label in raw_labels]
    return torch.tensor(binary, dtype=torch.float32)


def compute_cvr_loss(outputs: dict[str, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
    return binary_logloss(outputs["logits"], labels)
