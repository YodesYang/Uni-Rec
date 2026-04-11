from __future__ import annotations

import torch
from torch import nn

from .sequences import masked_mean


def hash_ids(values: torch.Tensor, bucket_size: int) -> torch.Tensor:
    values = values.long()
    hashed = torch.zeros_like(values)
    non_padding = values != 0
    hashed[non_padding] = torch.remainder(values[non_padding].abs(), bucket_size - 1) + 1
    return hashed


class HashEmbedding(nn.Module):
    def __init__(self, bucket_size: int, embed_dim: int) -> None:
        super().__init__()
        self.bucket_size = bucket_size
        self.embedding = nn.Embedding(bucket_size, embed_dim, padding_idx=0)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hashed = hash_ids(values.long(), self.bucket_size)
        return self.embedding(hashed)


class SequenceEmbeddingEncoder(nn.Module):
    def __init__(self, bucket_size: int, embed_dim: int) -> None:
        super().__init__()
        self.embedding = HashEmbedding(bucket_size=bucket_size, embed_dim=embed_dim)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(values)
        return masked_mean(embedded, mask)


class WeightedSequenceEncoder(nn.Module):
    def __init__(self, bucket_size: int, embed_dim: int) -> None:
        super().__init__()
        self.embedding = HashEmbedding(bucket_size=bucket_size, embed_dim=embed_dim)

    def forward(self, ids: torch.Tensor, weights: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(ids)
        scaled = embedded * weights.unsqueeze(-1)
        return masked_mean(scaled, mask)


class DenseStatsEncoder(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(3, output_dim)

    def forward(self, values: torch.Tensor, mask: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        masked_values = values * mask
        denom = mask.sum(dim=1).clamp(min=1.0)
        mean = masked_values.sum(dim=1) / denom
        max_value = masked_values.max(dim=1).values
        length_feature = lengths.float()
        stacked = torch.stack([mean, max_value, length_feature], dim=1)
        return self.proj(stacked)
