"""CNN network for pixel feature extraction (matching paper Table 2).

Architecture:
    Input(224x224x3)
    -> Conv2D(32, 3x3, p=1) + BN + ReLU + MaxPool(2x2)  -> 112x112x32
    -> Conv2D(64, 3x3, p=1) + BN + ReLU + MaxPool(2x2)  -> 56x56x64
    -> Conv2D(96, 3x3, p=1) + BN + ReLU + MaxPool(2x2)  -> 28x28x96
    -> Conv2D(128, 3x3, p=1) + BN + ReLU + MaxPool(2x2) -> 14x14x128
    -> GlobalAvgPool                                      -> 128
"""

import torch.nn as nn


class EmotionCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(96, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.features(x)            # (B, 128, 14, 14)
        x = self.gap(x)                 # (B, 128, 1, 1)
        return x.squeeze(-1).squeeze(-1) # (B, 128)
