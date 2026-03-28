"""
NOXIS — Grad-CAM Explainability Module
Generates class-activation heatmaps from the final convolutional layer of
EfficientNet-B3 and overlays them on original video frames.
"""

import os
import uuid
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Optional


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.

    Hooks into a target convolutional layer, captures forward activations
    and backward gradients, then produces a spatial heatmap showing which
    regions most influenced the prediction.
    """

    def __init__(self, model, target_layer):
        """
        Args:
            model        : The full DeepfakeDetector model.
            target_layer : nn.Module — typically model.get_last_conv_layer().
        """
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        # Register hooks
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    # ── Hook callbacks ───────────────────────────────────────────────
    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    # ── Core ─────────────────────────────────────────────────────────
    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap for a SINGLE image.

        Args:
            input_tensor : (1, C, H, W) normalised image tensor on device.

        Returns:
            heatmap : (H, W) float32 array in [0, 1].
        """
        self.model.eval()
        input_tensor.requires_grad_(True)

        # Forward through the backbone only (single image, not sequence)
        output = self.model.backbone(input_tensor)  # (1, feature_dim)

        # For binary classification: we want gradients for the single output
        # Use the max feature as proxy for single-image grad-cam
        score = output.sum()
        self.model.zero_grad()
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Hooks did not capture activations / gradients.")

        # Global-average-pool the gradients → channel weights
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of forward activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        # Resize to input spatial size
        cam = F.interpolate(
            cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False
        )
        cam = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32)

    # ── Cleanup ──────────────────────────────────────────────────────
    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


# =====================================================================
# High-level utility functions
# =====================================================================

def overlay_heatmap(
    original_frame: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlay a Grad-CAM heatmap on an original frame (RGB).

    Args:
        original_frame : (H, W, 3) uint8 RGB image.
        heatmap        : (H, W) float32 in [0, 1].
        alpha          : blending factor for the heatmap.

    Returns:
        (H, W, 3) uint8 RGB overlay image.
    """
    h, w = original_frame.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = np.uint8(alpha * heatmap_color + (1 - alpha) * original_frame)
    return overlay


def generate_gradcam_overlays(
    model,
    frames_pil: List[Image.Image],
    transform,
    device: torch.device,
    output_dir: str,
    max_frames: int = 8,
) -> List[str]:
    """
    Generate Grad-CAM heatmap overlays for a list of video frames.

    Args:
        model      : DeepfakeDetector model (on device, eval mode).
        frames_pil : List of PIL Images (RGB, already resized).
        transform  : torchvision transform for normalisation.
        device     : torch device.
        output_dir : Directory to save heatmap images.
        max_frames : Maximum number of frames to process.

    Returns:
        List of file paths to saved heatmap overlay images.
    """
    os.makedirs(output_dir, exist_ok=True)

    target_layer = model.get_last_conv_layer()
    grad_cam = GradCAM(model, target_layer)

    paths: List[str] = []
    selected = frames_pil[:max_frames]

    for idx, pil_frame in enumerate(selected):
        # Prepare input
        input_tensor = transform(pil_frame).unsqueeze(0).to(device)

        # Generate heatmap
        try:
            heatmap = grad_cam.generate(input_tensor)
        except RuntimeError:
            continue

        # Overlay on original frame
        original_np = np.array(pil_frame.resize((224, 224)))
        overlay = overlay_heatmap(original_np, heatmap, alpha=0.5)

        # Save
        filename = f"gradcam_{idx:03d}_{uuid.uuid4().hex[:6]}.png"
        filepath = os.path.join(output_dir, filename)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, overlay_bgr)
        paths.append(filename)  # relative filename for API response

    grad_cam.remove_hooks()
    return paths
