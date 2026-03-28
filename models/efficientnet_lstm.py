"""
NOXIS — EfficientNet-B3 + BiLSTM Deepfake Detector
Spatial encoder + temporal sequence modelling.
"""

import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class EfficientNetLSTMDetector(nn.Module):
    """
    EfficientNet-B3 → BiLSTM → classifier.
    Input : (B, T, C, H, W)
    Output: (B, 1)
    """

    def __init__(
        self,
        num_frames: int = 16,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.4,
        **kwargs,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.is_temporal = True

        # ── Spatial encoder ──────────────────────────────────────────
        self.backbone = EfficientNet.from_pretrained("efficientnet-b3")

        for param in self.backbone._conv_stem.parameters():
            param.requires_grad = False
        for param in self.backbone._bn0.parameters():
            param.requires_grad = False
        total_blocks = len(self.backbone._blocks)
        freeze_until = max(0, total_blocks - 4)
        for idx in range(freeze_until):
            for param in self.backbone._blocks[idx].parameters():
                param.requires_grad = False

        self.cnn_dim = self.backbone._fc.in_features  # 1536
        self.backbone._fc = nn.Identity()

        # ── Temporal aggregator ──────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=self.cnn_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.backbone(x)             # (B*T, cnn_dim)
        feats = feats.view(B, T, -1)         # (B, T, cnn_dim)

        lstm_out, _ = self.lstm(feats)       # (B, T, hidden*2)
        final = lstm_out[:, -1, :]           # (B, hidden*2)
        final = self.dropout(final)
        return self.classifier(final)        # (B, 1)
