FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    SWINIR_MODEL_PATH=/app/models/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth \
    SWINIR_TILE_SIZE=256 \
    SWINIR_TILE_PADDING=16 \
    CODEFORMER_ROOT=/opt/CodeFormer \
    CODEFORMER_MODEL_PATH=/opt/CodeFormer/weights/CodeFormer/codeformer.pth \
    CODEFORMER_FIDELITY=0.7 \
    MAX_INPUT_BYTES=15728640 \
    LOG_LEVEL=INFO

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3-dev git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python3.10 -m pip install --no-cache-dir --upgrade pip && \
    python3.10 -m pip install --no-cache-dir -r /app/requirements.txt

ARG CODEFORMER_COMMIT=b33cc7d639d6545bfcccc7e0bc6ae51f24e79c2b
RUN git clone --filter=blob:none https://github.com/sczhou/CodeFormer.git /opt/CodeFormer && \
    cd /opt/CodeFormer && \
    git checkout "$CODEFORMER_COMMIT" && \
    sed -i -e '/from \.data import/d' \
           -e '/from \.losses import/d' \
           -e '/from \.metrics import/d' \
           -e '/from \.models import/d' \
           -e '/from \.ops import/d' \
           -e '/from \.train import/d' \
           /opt/CodeFormer/basicsr/__init__.py
RUN mkdir -p /opt/CodeFormer/weights/CodeFormer /opt/CodeFormer/weights/facelib && \
    curl --fail --location --retry 5 --retry-all-errors \
        --output /opt/CodeFormer/weights/CodeFormer/codeformer.pth \
        https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth && \
    curl --fail --location --retry 5 --retry-all-errors \
        --output /opt/CodeFormer/weights/facelib/detection_Resnet50_Final.pth \
        https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth && \
    curl --fail --location --retry 5 --retry-all-errors \
        --output /opt/CodeFormer/weights/facelib/parsing_parsenet.pth \
        https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth
RUN cd /opt/CodeFormer && \
    PYTHONPATH=/opt/CodeFormer python3.10 -c "from basicsr.archs import codeformer_arch; print('CodeFormer architecture import OK')"
RUN cd /opt/CodeFormer && \
    PYTHONPATH=/opt/CodeFormer python3.10 -c "from facelib.utils.face_restoration_helper import FaceRestoreHelper; print('CodeFormer face helper import OK')" && \
    rm -rf /opt/CodeFormer/.git

COPY . /app
RUN mkdir -p /app/models && \
    python3 /app/scripts/download_model.py

CMD ["python3.10", "-u", "/app/handler.py"]
