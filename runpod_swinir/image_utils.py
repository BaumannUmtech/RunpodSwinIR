from __future__ import annotations

import io
from typing import Tuple

import numpy as np
from PIL import Image


def center_crop_to_aspect(image: Image.Image, aspect_ratio: str) -> Image.Image:
    width, height = image.size
    target_width, target_height = _aspect_to_size(aspect_ratio, width, height)
    if width == target_width and height == target_height:
        return image.copy()
    left = max(0, (width - target_width) // 2)
    top = max(0, (height - target_height) // 2)
    right = min(width, left + target_width)
    bottom = min(height, top + target_height)
    return image.crop((left, top, right, bottom))


def _aspect_to_size(aspect_ratio: str, width: int, height: int) -> Tuple[int, int]:
    if aspect_ratio == "16:9":
        target_width = width
        target_height = int(round(width * 9 / 16))
        if target_height > height:
            target_height = height
            target_width = int(round(height * 16 / 9))
        return target_width, target_height
    if aspect_ratio == "9:16":
        target_height = height
        target_width = int(round(height * 9 / 16))
        if target_width > width:
            target_width = width
            target_height = int(round(width * 16 / 9))
        return target_width, target_height
    if aspect_ratio == "1:1":
        size = min(width, height)
        return size, size
    raise ValueError(aspect_ratio)


def composite_alpha_to_white(image: Image.Image) -> Image.Image:
    if image.mode in {"RGBA", "LA"}:
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, image).convert("RGB")
    return image.convert("RGB")


def pil_to_numpy(image: Image.Image) -> np.ndarray:
    return np.array(image, dtype=np.float32) / 255.0


def numpy_to_pil(image: np.ndarray) -> Image.Image:
    clipped = np.clip(image, 0.0, 1.0)
    return Image.fromarray(np.uint8(clipped * 255.0), "RGB")


def encode_jpeg(image: Image.Image, quality: int) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, subsampling=2)
    return buffer.getvalue()


def encode_png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()
