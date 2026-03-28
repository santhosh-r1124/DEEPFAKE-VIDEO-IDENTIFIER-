"""
NOXIS — Face Detection Module
MTCNN-based face detection and cropping for video frames.
Falls back to center-crop if no face is found.
"""

import numpy as np
from PIL import Image
from typing import List, Tuple, Optional

try:
    from facenet_pytorch import MTCNN
    _MTCNN_AVAILABLE = True
except ImportError:
    _MTCNN_AVAILABLE = False


class FaceDetector:
    """
    MTCNN face detector with fallback to center-crop.

    Usage:
        detector = FaceDetector(output_size=224)
        cropped = detector.crop_face(pil_image)
        batch   = detector.crop_faces(list_of_pil_images)
    """

    def __init__(self, output_size: int = 224, margin: int = 40, min_face_size: int = 50):
        self.output_size = output_size
        self.margin = margin

        if _MTCNN_AVAILABLE:
            self.mtcnn = MTCNN(
                image_size=output_size,
                margin=margin,
                min_face_size=min_face_size,
                thresholds=[0.6, 0.7, 0.7],
                post_process=False,
                select_largest=True,
                keep_all=False,
                device="cpu",  # CPU for detection — saves GPU VRAM
            )
        else:
            self.mtcnn = None

    def crop_face(self, pil_image: Image.Image) -> Image.Image:
        """
        Detect and crop the largest face from a PIL image.
        Falls back to center-crop if no face is found or MTCNN unavailable.
        """
        if self.mtcnn is not None:
            try:
                boxes, probs = self.mtcnn.detect(pil_image)

                if boxes is not None and len(boxes) > 0:
                    # Take highest-confidence face
                    best_idx = int(np.argmax(probs))
                    box = boxes[best_idx]

                    # Expand box with margin
                    x1, y1, x2, y2 = [int(b) for b in box]
                    w, h = pil_image.size

                    # Clamp to image bounds
                    x1 = max(0, x1 - self.margin // 2)
                    y1 = max(0, y1 - self.margin // 2)
                    x2 = min(w, x2 + self.margin // 2)
                    y2 = min(h, y2 + self.margin // 2)

                    face = pil_image.crop((x1, y1, x2, y2))
                    return face.resize((self.output_size, self.output_size), Image.LANCZOS)

            except Exception:
                pass  # Fall through to center-crop

        return self._center_crop(pil_image)

    def crop_faces(self, frames: List[Image.Image]) -> List[Image.Image]:
        """Crop faces from a list of frames."""
        return [self.crop_face(f) for f in frames]

    def _center_crop(self, pil_image: Image.Image) -> Image.Image:
        """Center-crop and resize fallback."""
        w, h = pil_image.size
        crop_size = min(w, h)
        left = (w - crop_size) // 2
        top = (h - crop_size) // 2
        cropped = pil_image.crop((left, top, left + crop_size, top + crop_size))
        return cropped.resize((self.output_size, self.output_size), Image.LANCZOS)


# ── Module-level singleton ───────────────────────────────────────────
_default_detector: Optional[FaceDetector] = None


def get_face_detector(output_size: int = 224) -> FaceDetector:
    """Get or create the default face detector singleton."""
    global _default_detector
    if _default_detector is None or _default_detector.output_size != output_size:
        _default_detector = FaceDetector(output_size=output_size)
    return _default_detector
