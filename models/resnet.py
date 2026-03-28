"""
NOXIS — ResNet50 Deepfake Detector
Frame-level CNN with average pooling over temporal dimension.
"""

import torch
import torch.nn as nn
import torchvision.models as tvm


class ResNet50Detector(nn.Module):
    """
    ResNet50 → temporal average pool → classifier.
    Input : (B, T, C, H, W)
    Output: (B, 1)
    """

    def __init__(self, num_frames: int = 16, dropout: float = 0.4, **kwargs):
        super().__init__()
        self.num_frames = num_frames
        self.is_temporal = False  # frame-level model

        backbone = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)

        # Freeze early layers (layer1, layer2)
        for name, param in backbone.named_parameters():
            if any(name.startswith(p) for p in ("conv1", "bn1", "layer1", "layer2")):
                param.requires_grad = False

        self.feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.feature_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.backbone(x)            # (B*T, feat_dim)
        feats = feats.view(B, T, -1)        # (B, T, feat_dim)
        feats = feats.mean(dim=1)           # (B, feat_dim)  — temporal avg pool
        feats = self.dropout(feats)
        return self.classifier(feats)       # (B, 1)
