"""
NOXIS — MobileNetV3-Large Deepfake Detector
Ultra-lightweight CNN with temporal average pooling.
"""

import torch
import torch.nn as nn
import torchvision.models as tvm


class MobileNetV3Detector(nn.Module):
    """
    MobileNetV3-Large → temporal average pool → classifier.
    Input : (B, T, C, H, W)
    Output: (B, 1)
    """

    def __init__(self, num_frames: int = 16, dropout: float = 0.4, **kwargs):
        super().__init__()
        self.num_frames = num_frames
        self.is_temporal = False

        backbone = tvm.mobilenet_v3_large(weights=tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V2)

        # Freeze early features (first 10 of 16 inverted residual blocks)
        for idx, layer in enumerate(backbone.features):
            if idx < 10:
                for param in layer.parameters():
                    param.requires_grad = False

        self.features = backbone.features
        self.avgpool = backbone.avgpool

        # MobileNetV3 classifier: Linear(960, 1280) → Hardswish → Dropout → Linear(1280, 1000)
        # We keep the projection but replace the final layer
        self.proj = nn.Sequential(
            nn.Linear(960, 1280),
            nn.Hardswish(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(1280, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)                    # (B*T, 960)
        x = self.proj(x)                    # (B*T, 1280)
        x = x.view(B, T, -1)
        x = x.mean(dim=1)                  # (B, 1280)
        return self.classifier(x)           # (B, 1)
