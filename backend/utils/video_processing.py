"""
NOXIS — Video Frame Extraction
Uniformly samples N frames from a video file using OpenCV.
Windows-compatible path handling.
"""

import os
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional


def extract_frames(
    video_path: str,
    num_frames: int = 16,
    size: Tuple[int, int] = (224, 224),
    return_pil: bool = True,
) -> List:
    """
    Extract uniformly-spaced frames from a video file.

    Args:
        video_path : Absolute path to the video file.
        num_frames : Number of frames to extract (default 16).
        size       : (width, height) to resize each frame.
        return_pil : If True, return PIL Images; otherwise numpy arrays (BGR).

    Returns:
        List of PIL.Image (RGB) or numpy arrays (BGR), length ≤ num_frames.

    Raises:
        FileNotFoundError : If the video does not exist.
        RuntimeError       : If OpenCV cannot open the video.
    """
    video_path = os.path.normpath(video_path)

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Video has 0 frames: {video_path}")

    # Compute uniform sample indices
    if total_frames <= num_frames:
        indices = list(range(total_frames))
    else:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()

    frames: List = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        if return_pil:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
        frames.append(frame)

    cap.release()
    return frames


def extract_single_frame(
    video_path: str,
    frame_idx: int = 0,
    size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Extract a single frame by index. Returns BGR numpy array.
    """
    video_path = os.path.normpath(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Cannot read frame {frame_idx} from {video_path}")

    if size is not None:
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

    return frame
