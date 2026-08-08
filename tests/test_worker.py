import base64
import io
from unittest.mock import MagicMock

import pytest
from PIL import Image

from runpod_swinir.worker import handler


def test_handler_returns_error_contract_for_invalid_request():
    result = handler({})
    assert result["success"] is False
    assert result["error_code"] == "INVALID_REQUEST"


def test_handler_returns_success_contract(monkeypatch):
    image = Image.new("RGB", (64, 64), (10, 20, 30))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")

    class DummyUpscaler:
        def upscale(self, image, target_size):
            return image

    monkeypatch.setattr("runpod_swinir.worker._get_upscaler", lambda: DummyUpscaler())
    result = handler({"input": {"image_base64": payload, "aspect_ratio": "16:9", "output_format": "jpeg", "quality": 95}})
    assert result["success"] is True
    assert result["mime_type"] == "image/jpeg"
    assert result["width"] == 1920
    assert result["height"] == 1080
