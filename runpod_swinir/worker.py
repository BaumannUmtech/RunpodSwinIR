from __future__ import annotations

import base64
import logging
import os
import time
from typing import Any

import numpy as np
import runpod
from PIL import Image

from .contracts import TARGET_DIMENSIONS, ValidationError, decode_image, validate_request
from .image_utils import (
    center_crop_to_aspect,
    composite_alpha_to_white,
    encode_jpeg,
    encode_png,
    pil_to_numpy,
)
from .upscaler import SwinIRUpscaler

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger("runpod_swinir")

_UPSCALER: SwinIRUpscaler | None = None
_FACE_ENHANCER = None


def _get_upscaler() -> SwinIRUpscaler:
    global _UPSCALER
    if _UPSCALER is None:
        _UPSCALER = SwinIRUpscaler()
    return _UPSCALER


def _get_face_enhancer():
    global _FACE_ENHANCER
    if _FACE_ENHANCER is None:
        from .face_enhancer import CodeFormerFaceEnhancer

        _FACE_ENHANCER = CodeFormerFaceEnhancer()
    return _FACE_ENHANCER


def _build_error(error_code: str, message: str) -> dict[str, Any]:
    return {"success": False, "error_code": error_code, "error": message}


def handler(event: dict[str, Any]) -> dict[str, Any]:
    start = time.time()
    try:
        request = validate_request(event)
        image = decode_image(request.image_bytes)
        image = composite_alpha_to_white(image)
        image = center_crop_to_aspect(image, request.aspect_ratio)

        target_width, target_height = TARGET_DIMENSIONS[request.aspect_ratio]
        upscaler = _get_upscaler()
        array = pil_to_numpy(image)
        output = upscaler.upscale(array, (target_width, target_height))

        if output.ndim != 3 or output.shape[2] != 3:
            raise RuntimeError("Ungültige Ausgabegröße")
        if output.shape[:2] != (target_height, target_width):
            resized = Image.fromarray(np.uint8(np.clip(output * 255.0, 0.0, 255.0)))
            resized = resized.resize((target_width, target_height), resample=Image.Resampling.LANCZOS)
            output = np.array(resized, dtype=np.float32) / 255.0
        if not np.isfinite(output).all():
            raise RuntimeError("Ungültige Modellausgabe")

        result_array = np.uint8(np.clip(output * 255.0, 0.0, 255.0))
        faces_restored = 0
        if request.face_enhance:
            result_array, faces_restored = _get_face_enhancer().enhance(result_array)
        result_image = Image.fromarray(result_array)
        if request.output_format == "png":
            output_bytes = encode_png(result_image)
            mime_type = "image/png"
        else:
            output_bytes = encode_jpeg(result_image, request.quality)
            mime_type = "image/jpeg"
        payload = base64.b64encode(output_bytes).decode("ascii")
        return {
            "success": True,
            "image_base64": payload,
            "mime_type": mime_type,
            "width": target_width,
            "height": target_height,
            "upscaler": "swinir_codeformer" if request.face_enhance else "swinir",
            "model": "002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2",
            "face_enhance": request.face_enhance,
            "faces_restored": faces_restored,
            "processing_ms": int((time.time() - start) * 1000),
        }
    except ValidationError as exc:
        logger.warning("Validation error: %s", exc.message)
        return _build_error(exc.error_code, exc.message)
    except RuntimeError as exc:
        logger.exception("Runtime error")
        if "CUDA" in str(exc):
            return _build_error("MODEL_UNAVAILABLE", "Das Modell ist derzeit nicht verfügbar.")
        if "NaN" in str(exc) or "leer" in str(exc) or "Ausgabegröße" in str(exc):
            return _build_error("INVALID_MODEL_OUTPUT", "Die Modellausgabe war ungültig.")
        return _build_error("UPSCALE_FAILED", "Die Hochskalierung konnte nicht abgeschlossen werden.")
    except Exception:
        logger.exception("Unhandled worker error")
        return _build_error("UPSCALE_FAILED", "Die Hochskalierung konnte nicht abgeschlossen werden.")


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
