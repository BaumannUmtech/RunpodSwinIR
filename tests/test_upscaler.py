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
