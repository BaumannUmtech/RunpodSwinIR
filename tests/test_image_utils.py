import pytest
from PIL import Image

from runpod_swinir.image_utils import center_crop_to_aspect, composite_alpha_to_white, encode_jpeg


@pytest.mark.parametrize(
    ("aspect_ratio", "input_size", "expected_size"),
    [("16:9", (1600, 900), (1600, 900)), ("9:16", (1600, 1600), (900, 1600)), ("1:1", (1600, 900), (900, 900))],
)
def test_center_crop_to_aspect_uses_expected_size(aspect_ratio, input_size, expected_size):
    image = Image.new("RGB", input_size, (255, 0, 0))
    cropped = center_crop_to_aspect(image, aspect_ratio)
    assert cropped.size == expected_size


def test_composite_alpha_to_white_uses_rgb():
    image = Image.new("RGBA", (2, 2), (255, 0, 0, 0))
    converted = composite_alpha_to_white(image)
    assert converted.mode == "RGB"


def test_encode_jpeg_returns_bytes():
    image = Image.new("RGB", (16, 16), (10, 20, 30))
    encoded = encode_jpeg(image, quality=90)
    assert encoded.startswith(b"\xff\xd8\xff")
