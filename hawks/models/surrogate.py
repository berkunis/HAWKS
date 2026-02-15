"""Minimal MLP surrogate that predicts next-step operator trust T_{t+1}."""

from __future__ import annotations

import torch
import torch.nn as nn


class TrustSurrogate(nn.Module):
    """Two-hidden-layer MLP: 4 → 32 → 16 → 1 (sigmoid output).

    Input features: [structural_risk, ai_confidence, ai_correctness, operator_trust]
    Output: predicted T_{t+1} in [0, 1].
    """

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
