"""Deterministic preprocessing shared by evaluation, ONNX export and live inference.

This module depends only on NumPy and OpenCV so the deployed monitor does not need PyTorch.
Evaluation uses exactly this function, so offline metrics measure the deployed code path.
"""

from __future__ import annotations

import cv2
import numpy as np

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def crop_roi(rgb: np.ndarray, roi: tuple[float, float, float, float] | None) -> np.ndarray:
    if roi is None:
        return rgb
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = roi
    return rgb[
        int(y0 * h) : max(int(y1 * h), int(y0 * h) + 1),
        int(x0 * w) : max(int(x1 * w), int(x0 * w) + 1),
    ]


def center_crop_to_aspect(rgb: np.ndarray, aspect_wh: float) -> np.ndarray:
    """Largest centred crop with ``width / height == aspect_wh``."""
    h, w = rgb.shape[:2]
    if w / h > aspect_wh:
        new_w = max(1, round(h * aspect_wh))
        x0 = (w - new_w) // 2
        return rgb[:, x0 : x0 + new_w]
    new_h = max(1, round(w / aspect_wh))
    y0 = (h - new_h) // 2
    return rgb[y0 : y0 + new_h]


def prepare(
    rgb: np.ndarray,
    image_size: tuple[int, int],
    roi: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    """RGB uint8 HxWx3 -> normalised float32 3xHxW tensor matching ``image_size`` (h, w)."""
    out_h, out_w = image_size
    x = crop_roi(rgb, roi)
    x = center_crop_to_aspect(x, out_w / out_h)
    interp = cv2.INTER_AREA if x.shape[0] > out_h else cv2.INTER_LINEAR
    x = cv2.resize(x, (out_w, out_h), interpolation=interp)
    x = (x.astype(np.float32) / 255.0 - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)
