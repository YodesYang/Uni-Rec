from __future__ import annotations

from typing import Iterable

import torch


def aggregate_domain_vectors(vectors: Iterable[torch.Tensor]) -> torch.Tensor:
    vectors = list(vectors)
    if not vectors:
        raise ValueError("domain vectors cannot be empty")
    if len(vectors) == 1:
        return vectors[0]
    stacked = torch.stack(vectors, dim=1)
    return stacked.mean(dim=1)
