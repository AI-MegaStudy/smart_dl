from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageStat


def _load_rgb(image_path: str | Path, image_size: int = 224) -> Image.Image:
    return Image.open(image_path).convert("RGB").resize((image_size, image_size))


def calculate_color_score(image_path: str | Path, image_size: int = 224) -> float:
    """Estimate color vividness using HSV saturation and brightness.

    This is a PIL/NumPy fallback so the project can run before OpenCV is installed.
    """
    image = _load_rgb(image_path, image_size).convert("HSV")
    hsv = np.asarray(image, dtype=np.float32)
    saturation = hsv[..., 1] / 255.0
    value = hsv[..., 2] / 255.0
    score = (saturation.mean() * 0.65 + value.mean() * 0.35) * 100.0
    return round(float(np.clip(score, 0.0, 100.0)), 2)


def calculate_roundness_score(image_path: str | Path, image_size: int = 224) -> float:
    """Approximate roundness from a foreground mask.

    It assumes data is captured on a fairly simple background, which matches the
    MVP shooting guide. A true contour-based implementation can replace this
    later if cv2 becomes available.
    """
    image = _load_rgb(image_path, image_size)
    arr = np.asarray(image, dtype=np.float32)
    border = np.concatenate(
        [arr[:8].reshape(-1, 3), arr[-8:].reshape(-1, 3), arr[:, :8].reshape(-1, 3), arr[:, -8:].reshape(-1, 3)],
        axis=0,
    )
    bg = np.median(border, axis=0)
    distance = np.linalg.norm(arr - bg, axis=2)
    mask = distance > max(18.0, float(distance.mean()))
    ys, xs = np.where(mask)
    if len(xs) < 50:
        return 50.0

    width = max(1, xs.max() - xs.min() + 1)
    height = max(1, ys.max() - ys.min() + 1)
    aspect_score = min(width, height) / max(width, height)
    fill_ratio = mask.sum() / float(width * height)
    score = (aspect_score * 0.65 + min(fill_ratio / 0.78, 1.0) * 0.35) * 100.0
    return round(float(np.clip(score, 0.0, 100.0)), 2)


def calculate_bruise_probability(image_path: str | Path, image_size: int = 224) -> float:
    """Estimate dark-damage likelihood from low-brightness regions."""
    image = _load_rgb(image_path, image_size)
    hsv = np.asarray(image.convert("HSV"), dtype=np.float32)
    value = hsv[..., 2] / 255.0
    saturation = hsv[..., 1] / 255.0
    dark_ratio = np.logical_and(value < 0.33, saturation > 0.18).mean()
    probability = min(1.0, dark_ratio / 0.18)
    return round(float(np.clip(probability, 0.0, 1.0)), 4)


def extract_features(image_path: str | Path, image_size: int = 224) -> dict[str, float]:
    return {
        "color_score": calculate_color_score(image_path, image_size),
        "roundness_score": calculate_roundness_score(image_path, image_size),
        "bruise_probability": calculate_bruise_probability(image_path, image_size),
    }

