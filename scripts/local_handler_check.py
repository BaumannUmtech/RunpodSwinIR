import base64
import io
from pathlib import Path

from PIL import Image

from runpod_swinir.worker import handler


def main() -> None:
    image = Image.new("RGB", (64, 64), (40, 80, 120))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    result = handler({
        "input": {
            "image_base64": payload,
            "aspect_ratio": "16:9",
            "output_format": "jpeg",
            "quality": 95,
        }
    })
    print(result)


if __name__ == "__main__":
    main()
