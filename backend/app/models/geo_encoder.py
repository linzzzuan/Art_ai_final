"""Geometric feature encoder.

Architecture: Input(5) -> FC(32) + ReLU -> FC(64) + ReLU -> 64
"""

import torch.nn as nn


class GeoEncoder(nn.Module):
    def __init__(self, in_features: int = 5, hidden: int = 32, out_features: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.encoder(x)  # (B, 64)
