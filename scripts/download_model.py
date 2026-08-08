from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

MODEL_URL = "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth"
EXPECTED_SHA256 = "193b229909ca89cd8b55de9c9e7fce146ae759d59dfcd78d8feb9dd1d6fa0fd7"
MODEL_PATH = Path(os.getenv("SWINIR_MODEL_PATH", "/app/models/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth"))


def main() -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists():
        print(f"Modell bereits vorhanden: {MODEL_PATH}")
        return
    print(f"Lade Modell von {MODEL_URL}")
    with urllib.request.urlopen(MODEL_URL, timeout=600) as response:
        data = response.read()
    actual_sha = hashlib.sha256(data).hexdigest()
    if actual_sha != EXPECTED_SHA256:
        raise RuntimeError(f"SHA-256-Verifikation fehlgeschlagen: {actual_sha}")
    MODEL_PATH.write_bytes(data)
    print(f"Modell gespeichert unter {MODEL_PATH}")


if __name__ == "__main__":
    main()
