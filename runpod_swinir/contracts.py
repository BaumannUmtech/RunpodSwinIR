from __future__ import annotations

import base64
import binascii
import io
import os
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

VALID_ASPECT_RATIOS = {"16:9", "9:16", "1:1"}
TARGET_DIMENSIONS = {
    "16:9": (1920, 1080),
    "9:16": (1080, 1920),
    "1:1": (1080, 1080),
}


@dataclass
class RequestValidation:
    image_bytes: bytes
    aspect_ratio: str
    quality: int
    output_format: str


class ValidationError(Exception):
    def __init__(self, error_code: str, message: str):
        self.error_code = error_code
        self.message = message
        super().__init__(message)


def validate_request(event: dict[str, Any]) -> RequestValidation:
    payload = event.get("input") if isinstance(event, dict) else None
    if not isinstance(payload, dict):
        raise ValidationError("INVALID_REQUEST", "Die Anfrage enthält keine gültigen Nutzdaten.")

    image_b64 = payload.get("image_base64")
    if not isinstance(image_b64, str) or not image_b64.strip():
        raise ValidationError("INVALID_REQUEST", "Das Feld image_base64 ist erforderlich.")

    try:
        image_bytes = base64.b64decode(image_b64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValidationError("INVALID_IMAGE", "Das Eingabebild konnte nicht gelesen werden.") from exc

    max_bytes = int(os.getenv("MAX_INPUT_BYTES", "15728640"))
    if len(image_bytes) >= max_bytes:
        raise ValidationError("IMAGE_TOO_LARGE", "Das Eingabebild ist zu groß.")

    aspect_ratio = payload.get("aspect_ratio", "16:9")
    if aspect_ratio not in VALID_ASPECT_RATIOS:
        raise ValidationError("UNSUPPORTED_ASPECT_RATIO", "Das Seitenverhältnis wird nicht unterstützt.")

    output_format = payload.get("output_format", "jpeg")
    if output_format != "jpeg":
        raise ValidationError("INVALID_REQUEST", "Nur JPEG-Ausgaben sind in Stufe 1 unterstützt.")

    quality = payload.get("quality", 95)
    if not isinstance(quality, int) or not (80 <= quality <= 100):
        raise ValidationError("INVALID_REQUEST", "quality muss zwischen 80 und 100 liegen.")

    return RequestValidation(
        image_bytes=image_bytes,
        aspect_ratio=aspect_ratio,
        quality=quality,
        output_format=output_format,
    )


def decode_image(image_bytes: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img.load()
            if getattr(img, "is_animated", False):
                raise ValidationError("INVALID_IMAGE", "Animierte Bilder werden nicht unterstützt.")
            img = ImageOps.exif_transpose(img)
            if img.mode in {"RGBA", "LA", "P"}:
                img = img.convert("RGBA")
            else:
                img = img.convert("RGB")
            return img
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise ValidationError("INVALID_IMAGE", "Das Eingabebild konnte nicht gelesen werden.") from exc
