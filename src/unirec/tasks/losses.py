from __future__ import annotations

import torch
import torch.nn.functional as F


def binary_logloss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    labels = labels.float()
    return F.binary_cross_entropy_with_logits(logits, labels)
