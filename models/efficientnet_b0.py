"""
NOXIS — EfficientNet-B0 Deepfake Detector
Lightweight CNN with temporal average pooling.
"""

import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class EfficientNetB0Detector(nn.Module):
    """
    EfficientNet-B0 → temporal average pool → classifier.
    Input : (B, T, C, H, W)
    Output: (B, 1)
    """

    def __init__(self, num_frames: int = 16, dropout: float = 0.4, **kwargs):
        super().__init__()
        self.num_frames = num_frames
        self.is_temporal = False

        self.backbone = EfficientNet.from_pretrained("efficientnet-b0")

        # Freeze early blocks
        for param in self.backbone._conv_stem.parameters():
            param.requires_grad = False
        for param in self.backbone._bn0.parameters():
            param.requires_grad = False
        total_blocks = len(self.backbone._blocks)
        freeze_until = max(0, total_blocks - 3)
        for idx in range(freeze_until):
            for param in self.backbone._blocks[idx].parameters():
                param.requires_grad = False

        self.feature_dim = self.backbone._fc.in_features
        self.backbone._fc = nn.Identity()

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.feature_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.backbone(x)
        feats = feats.view(B, T, -1)
        feats = feats.mean(dim=1)
        feats = self.dropout(feats)
        return self.classifier(feats)
