from __future__ import annotations

from typing import Sequence

import torch


def normalize_list(value: object, dtype: str) -> list[int] | list[float]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        if dtype == "int":
            return [int(item) for item in value if item is not None]
        return [float(item) for item in value if item is not None]
    return []


def pad_sequence_batch(
    batch_sequences: Sequence[Sequence[int | float]],
    max_len: int,
    pad_value: int | float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = len(batch_sequences)
    values = torch.full((batch_size, max_len), pad_value, dtype=dtype)
    mask = torch.zeros((batch_size, max_len), dtype=torch.float32)
    lengths = torch.zeros(batch_size, dtype=torch.long)

    for idx, sequence in enumerate(batch_sequences):
        if not sequence:
            continue
        clipped = list(sequence[-max_len:])
        current_len = len(clipped)
        if current_len == 0:
            continue
        values[idx, :current_len] = torch.tensor(clipped, dtype=dtype)
        mask[idx, :current_len] = 1.0
        lengths[idx] = current_len
    return values, mask, lengths


def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    denom = mask.sum(dim=dim, keepdim=True).clamp(min=1.0)
    weighted = values * mask.unsqueeze(-1)
    return weighted.sum(dim=dim) / denom
