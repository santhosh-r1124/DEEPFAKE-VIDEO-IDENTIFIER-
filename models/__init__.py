"""
NOXIS — Model Registry
Central import point for all benchmark model architectures.
"""

from models.resnet import ResNet50Detector
from models.efficientnet_b0 import EfficientNetB0Detector
from models.efficientnet_b3 import EfficientNetB3Detector
from models.mobilenet import MobileNetV3Detector
from models.efficientnet_lstm import EfficientNetLSTMDetector

# ── Registry ─────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "ResNet50":            ResNet50Detector,
    "EfficientNet-B0":     EfficientNetB0Detector,
    "EfficientNet-B3":     EfficientNetB3Detector,
    "MobileNetV3-Large":   MobileNetV3Detector,
    "EfficientNet-B3+BiLSTM": EfficientNetLSTMDetector,
}


def get_model(name: str, **kwargs):
    """Instantiate a model by registry name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)


def list_models():
    """Return list of registered model names."""
    return list(MODEL_REGISTRY.keys())
