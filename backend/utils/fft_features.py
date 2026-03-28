"""
NOXIS — FFT Frequency-Domain Feature Extraction
Extracts magnitude spectrum from face crops for frequency-domain analysis.
"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import List


class FFTFeatureExtractor(nn.Module):
    """
    Lightweight CNN that processes FFT magnitude spectra.
    Input : (B, 1, H, W) — log-magnitude spectrum
    Output: (B, fft_dim)  — feature vector

    The FFT branch captures artifacts in the frequency domain
    that spatial CNNs may miss (e.g., GAN spectral fingerprints).
    """

    def __init__(self, input_size: int = 224, fft_dim: int = 256):
        super().__init__()
        self.fft_dim = fft_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),
        )

        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, fft_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, H, W) — FFT magnitude images
        Returns: (B, fft_dim)
        """
        return self.projector(self.encoder(x))


def compute_fft_magnitude(pil_image: Image.Image, size: int = 224) -> np.ndarray:
    """
    Compute log-magnitude FFT spectrum from a PIL image.

    Args:
        pil_image: RGB PIL Image
        size: output size (square)

    Returns:
        (size, size) float32 array — normalized log-magnitude spectrum
    """
    # Convert to grayscale
    gray = pil_image.convert("L").resize((size, size), Image.LANCZOS)
    img = np.array(gray, dtype=np.float32) / 255.0

    # 2D FFT
    fft = np.fft.fft2(img)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)

    # Log scale (avoid log(0))
    log_mag = np.log1p(magnitude)

    # Normalize to [0, 1]
    m_min, m_max = log_mag.min(), log_mag.max()
    if m_max - m_min > 1e-8:
        log_mag = (log_mag - m_min) / (m_max - m_min)
    else:
        log_mag = np.zeros_like(log_mag)

    return log_mag.astype(np.float32)


def batch_fft_magnitudes(
    pil_images: List[Image.Image],
    size: int = 224,
) -> torch.Tensor:
    """
    Compute FFT magnitudes for a batch of PIL images.

    Returns:
        (N, 1, size, size) float32 tensor
    """
    mags = [compute_fft_magnitude(img, size) for img in pil_images]
    # Stack: (N, H, W) → (N, 1, H, W)
    return torch.tensor(np.stack(mags), dtype=torch.float32).unsqueeze(1)
