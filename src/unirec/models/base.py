from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseCVRModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, batch: dict[str, object]) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def count_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)
