from __future__ import annotations

import numpy as np

from runpod_swinir.upscaler import SwinIRUpscaler


def test_upscaler_requires_cuda(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    try:
        SwinIRUpscaler(model_path="dummy")
    except RuntimeError as exc:
        assert "CUDA" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError when CUDA is not available")


def test_tiled_x2_output_uses_scaled_coordinates(monkeypatch):
    upscaler = SwinIRUpscaler.__new__(SwinIRUpscaler)
    upscaler.model = type("Model", (), {"scale": 2})()
    upscaler._loaded = True

    monkeypatch.setattr(upscaler, "_normalize_tile_size", lambda: 8)
    monkeypatch.setattr(upscaler, "_normalize_padding", lambda: 8)
    monkeypatch.setattr(
        upscaler,
        "_run_tile",
        lambda tile: np.repeat(np.repeat(tile, 2, axis=0), 2, axis=1),
    )

    image = np.zeros((16, 16, 3), dtype=np.float32)
    image[:8, :8] = (1.0, 0.0, 0.0)
    image[:8, 8:] = (0.0, 1.0, 0.0)
    image[8:, :8] = (0.0, 0.0, 1.0)
    image[8:, 8:] = (1.0, 1.0, 0.0)

    output = upscaler.upscale(image, (32, 32))

    assert output.shape == (32, 32, 3)
    np.testing.assert_allclose(output[:16, :16].mean(axis=(0, 1)), (1.0, 0.0, 0.0), atol=0.02)
    np.testing.assert_allclose(output[:16, 16:].mean(axis=(0, 1)), (0.0, 1.0, 0.0), atol=0.02)
    np.testing.assert_allclose(output[16:, :16].mean(axis=(0, 1)), (0.0, 0.0, 1.0), atol=0.02)
    np.testing.assert_allclose(output[16:, 16:].mean(axis=(0, 1)), (1.0, 1.0, 0.0), atol=0.02)
