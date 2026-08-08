from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from PIL import Image
from spandrel import ModelLoader


class SwinIRUpscaler:
    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path or os.getenv(
            "SWINIR_MODEL_PATH",
            "/app/models/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth",
        )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA ist für SwinIR nicht verfügbar.")
        self.device = "cuda"
        self.model: Any | None = None
        self._loaded = False
        self._load_model()

    def _load_model(self) -> None:
        loader = ModelLoader()
        model = loader.load_from_file(self.model_path)
        if getattr(model, "scale", 1) != 2:
            raise RuntimeError("Das geladene Modell hat keinen Skalierungsfaktor 2.")
        self.model = model.to(self.device).eval()
        self._loaded = True

    def _normalize_tile_size(self) -> int:
        value = int(os.getenv("SWINIR_TILE_SIZE", "256"))
        return max(8, value - (value % 8))

    def _normalize_padding(self) -> int:
        value = int(os.getenv("SWINIR_TILE_PADDING", "16"))
        return max(8, value - (value % 8))

    def _run_tile(self, tile: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Modell ist nicht verfügbar.")
        tensor = torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).to(self.device).float()
        with torch.inference_mode():
            output = self.model(tensor)
        if isinstance(output, (list, tuple)):
            output = output[0]
        output = output[0].permute(1, 2, 0).detach().cpu().float().clamp(0.0, 1.0).numpy()
        return output

    @torch.inference_mode()
    def upscale(self, image: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
        if not self._loaded or self.model is None:
            raise RuntimeError("Modell ist nicht verfügbar.")
        if image.ndim != 3 or image.shape[2] != 3:
            raise RuntimeError("Ungültiges Bildformat für die Hochskalierung.")
        image = np.clip(image.astype(np.float32), 0.0, 1.0)
        height, width = image.shape[:2]
        tile_size = self._normalize_tile_size()
        padding = self._normalize_padding()

        padded = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode="reflect")
        output_canvas = np.zeros((height, width, 3), dtype=np.float32)
        output_weights = np.zeros((height, width), dtype=np.float32)

        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                tile = padded[y:y + tile_size + 2 * padding, x:x + tile_size + 2 * padding, :]
                tile_out = self._run_tile(tile)
                tile_h, tile_w = tile.shape[:2]
                crop_h = min(tile_size, max(0, tile_h - 2 * padding))
                crop_w = min(tile_size, max(0, tile_w - 2 * padding))
                if crop_h <= 0 or crop_w <= 0:
                    continue
                crop = tile_out[padding:padding + crop_h, padding:padding + crop_w, :]
                out_h = min(crop_h, height - y)
                out_w = min(crop_w, width - x)
                output_canvas[y:y + out_h, x:x + out_w, :] = crop[:out_h, :out_w, :]
                output_weights[y:y + out_h, x:x + out_w] = 1.0

        if not np.isfinite(output_canvas).all():
            raise RuntimeError("Ungültige Modellausgabe mit NaN/Inf-Werten.")
        if np.mean(output_canvas[output_weights > 0]) < 1e-6:
            raise RuntimeError("Die Modellausgabe war zu dunkel oder leer.")

        pil_image = Image.fromarray(np.uint8(np.clip(output_canvas * 255.0, 0.0, 255.0)), mode="RGB")
        pil_image = pil_image.resize(target_size, resample=Image.Resampling.LANCZOS)
        return np.array(pil_image, dtype=np.float32) / 255.0
