"""Pick-feasibility verifier model (v1 MLP)."""

from __future__ import annotations

import torch
from torch import nn


class PickFeasibilityMLP(nn.Module):
    """90 -> 256 -> 128 -> 1 with ReLU + dropout. Returns logits."""

    def __init__(self, input_dim: int = 90, hidden: tuple[int, int] = (256, 128),
                 dropout: float = 0.1):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
