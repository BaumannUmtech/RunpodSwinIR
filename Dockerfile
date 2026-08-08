FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    SWINIR_MODEL_PATH=/app/models/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth \
    SWINIR_TILE_SIZE=256 \
    SWINIR_TILE_PADDING=16 \
    MAX_INPUT_BYTES=15728640 \
    LOG_LEVEL=INFO

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3-dev git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python3.10 -m pip install --no-cache-dir --upgrade pip && \
    python3.10 -m pip install --no-cache-dir -r /app/requirements.txt

COPY . /app
RUN mkdir -p /app/models && \
    python3 /app/scripts/download_model.py

CMD ["python3.10", "-m", "runpod_swinir.worker"]
