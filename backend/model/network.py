"""
NOXIS — DeepfakeDetector v2
EfficientNet-B3 (spatial) + FFT branch (frequency) + BiLSTM (temporal).
Outputs video-level logit and per-frame probabilities.

Designed for 4 GB VRAM: chunked frame processing, frozen early layers.
"""

import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

from backend.utils.fft_features import FFTFeatureExtractor


class DeepfakeDetector(nn.Module):
    """
    Dual-branch deepfake detector with temporal modelling.

    Spatial  : EfficientNet-B3 → 1536-d features per frame
    Frequency: FFT magnitude → lightweight CNN → 256-d features per frame
    Fused    : concat(spatial, freq) → 1792-d → BiLSTM → classifier

    Input : (B, T, C, H, W) RGB frames + (B, T, 1, H, W) FFT magnitudes
    Output: (B, 1) video-level logit
    """

    def __init__(
        self,
        num_frames: int = 32,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.4,
        fft_dim: int = 256,
        freeze_backbone: bool = True,
        chunk_size: int = 8,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.chunk_size = chunk_size

        # ── Spatial encoder ──────────────────────────────────────────
        self.backbone = EfficientNet.from_pretrained("efficientnet-b3")

        if freeze_backbone:
            self._freeze_early_layers()

        self.cnn_feature_dim = self.backbone._fc.in_features  # 1536
        self.backbone._fc = nn.Identity()

        # ── Frequency-domain branch ──────────────────────────────────
        self.fft_branch = FFTFeatureExtractor(fft_dim=fft_dim)
        self.fft_dim = fft_dim

        # ── Fusion dimension ─────────────────────────────────────────
        self.fused_dim = self.cnn_feature_dim + fft_dim  # 1536 + 256 = 1792

        # ── Per-frame classifier (for frame-level probabilities) ─────
        self.frame_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.fused_dim, 1),
        )

        # ── Temporal aggregator ──────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=self.fused_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        # ── Video-level classifier ───────────────────────────────────
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden * 2, 1)

    # -----------------------------------------------------------------
    def _freeze_early_layers(self):
        """Freeze stem + all blocks except the last 4."""
        for param in self.backbone._conv_stem.parameters():
            param.requires_grad = False
        for param in self.backbone._bn0.parameters():
            param.requires_grad = False

        total_blocks = len(self.backbone._blocks)
        freeze_until = max(0, total_blocks - 4)
        for idx in range(freeze_until):
            for param in self.backbone._blocks[idx].parameters():
                param.requires_grad = False

    # -----------------------------------------------------------------
    def _extract_spatial_chunked(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process frames through backbone in chunks to save VRAM.
        x: (N, C, H, W) → (N, cnn_feature_dim)
        """
        N = x.shape[0]
        if N <= self.chunk_size:
            return self.backbone(x)

        feats = []
        for i in range(0, N, self.chunk_size):
            chunk = x[i:i + self.chunk_size]
            feats.append(self.backbone(chunk))
        return torch.cat(feats, dim=0)

    # -----------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        fft_x: torch.Tensor = None,
        return_frame_probs: bool = False,
    ):
        """
        Args:
            x     : (B, T, C, H, W) RGB frames
            fft_x : (B, T, 1, H, W) FFT magnitude images (optional)
            return_frame_probs: if True, also return per-frame probabilities

        Returns:
            logit : (B, 1) video-level logit
            frame_probs : (B, T) per-frame probabilities (if return_frame_probs)
        """
        B, T, C, H, W = x.shape

        # ── Spatial features (chunked for VRAM safety) ───────────────
        flat = x.view(B * T, C, H, W)
        spatial = self._extract_spatial_chunked(flat)  # (B*T, 1536)
        spatial = spatial.view(B, T, -1)

        # ── FFT features ─────────────────────────────────────────────
        if fft_x is not None:
            fft_flat = fft_x.view(B * T, 1, H, W)
            freq = self.fft_branch(fft_flat)  # (B*T, fft_dim)
            freq = freq.view(B, T, -1)
        else:
            freq = torch.zeros(B, T, self.fft_dim, device=x.device)

        # ── Fuse ─────────────────────────────────────────────────────
        fused = torch.cat([spatial, freq], dim=-1)  # (B, T, 1792)

        # ── Per-frame logits ─────────────────────────────────────────
        frame_logits = self.frame_head(fused).squeeze(-1)  # (B, T)

        # ── Temporal modelling ───────────────────────────────────────
        lstm_out, _ = self.lstm(fused)  # (B, T, hidden*2)
        final = lstm_out[:, -1, :]      # (B, hidden*2)
        final = self.dropout(final)
        logit = self.classifier(final)  # (B, 1)

        if return_frame_probs:
            frame_probs = torch.sigmoid(frame_logits)  # (B, T)
            return logit, frame_probs

        return logit

    # -----------------------------------------------------------------
    def get_last_conv_layer(self):
        """Return the last convolutional layer for Grad-CAM."""
        return self.backbone._conv_head
