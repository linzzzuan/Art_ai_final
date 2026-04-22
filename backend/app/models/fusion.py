"""Feature fusion classifier.

Architecture: Concat(128+64)=192 -> FC(128) + ReLU + Dropout(0.5) -> FC(7)
"""

import torch
import torch.nn as nn


class FusionClassifier(nn.Module):
    def __init__(
        self,
        pixel_dim: int = 128,
        geo_dim: int = 64,
        hidden: int = 128,
        num_classes: int = 7,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(pixel_dim + geo_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, pixel_feat, geo_feat):
        x = torch.cat([pixel_feat, geo_feat], dim=1)  # (B, 192)
        return self.head(x)  # (B, 7) raw logits
