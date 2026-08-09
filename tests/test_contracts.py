import base64
import io

import pytest
from PIL import Image

from runpod_swinir.contracts import ValidationError, decode_image, validate_request


def test_validate_request_rejects_invalid_payload():
    with pytest.raises(ValidationError) as exc:
        validate_request({})
    assert exc.value.error_code == "INVALID_REQUEST"


def test_validate_request_rejects_oversized_input(monkeypatch):
    monkeypatch.setenv("MAX_INPUT_BYTES", "10")
    payload = base64.b64encode(b"1234567890").decode("ascii")
    with pytest.raises(ValidationError) as exc:
        validate_request({"input": {"image_base64": payload}})
    assert exc.value.error_code == "IMAGE_TOO_LARGE"


def test_validate_request_accepts_face_enhance():
    image = Image.new("RGB", (4, 4))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    request = validate_request({"input": {"image_base64": payload, "face_enhance": True}})
    assert request.face_enhance is True


def test_validate_request_accepts_png_output():
    image = Image.new("RGB", (4, 4))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    request = validate_request(
        {"input": {"image_base64": payload, "output_format": "png"}}
    )
    assert request.output_format == "png"


def test_validate_request_rejects_non_boolean_face_enhance():
    payload = base64.b64encode(b"image").decode("ascii")
    with pytest.raises(ValidationError) as exc:
        validate_request({"input": {"image_base64": payload, "face_enhance": "yes"}})
    assert exc.value.error_code == "INVALID_REQUEST"


def test_decode_image_handles_png_and_exif():
    image = Image.new("RGBA", (4, 4), (255, 0, 0, 128))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    decoded = decode_image(buffer.getvalue())
    assert decoded.mode == "RGBA"
