# RunPod SwinIR Worker

Dieser Ordner enthält einen eigenständigen RunPod-Serverless-Worker für SwinIR x2.

## Funktionen der ersten Stufe

- Validierung von Eingabe- und Ausgabeparametern
- Dekodierung von JPEG/PNG/WebP mit EXIF-Ausrichtung und Alpha-Komposition
- Zuschneiden auf 16:9, 9:16 oder 1:1
- GPU-gestützte Hochskalierung mit SwinIR x2
- Rückgabe eines Full-HD-JPEG im Vertrag

## Lokale Nutzung

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m compileall runpod_swinir
```

Ein lokaler Handler-Test kann mit einem kleinen Beispiel-Skript erfolgen:

```bash
python - <<'PY'
import base64, io
from PIL import Image
from runpod_swinir.worker import handler

img = Image.new('RGB', (64, 64), (40, 80, 120))
buf = io.BytesIO()
img.save(buf, format='PNG')
payload = base64.b64encode(buf.getvalue()).decode('ascii')
print(handler({'input': {'image_base64': payload, 'aspect_ratio': '16:9', 'output_format': 'jpeg', 'quality': 95}}))
PY
```

## Docker

```bash
docker build -t runpod-swinir:local .
```

Das Image lädt das offizielle SwinIR-Modell während des Builds herunter und verifiziert die SHA-256-Prüfsumme.

## Git-basierter Docker-Push

GitHub Actions baut das Image bei Pushes auf `main` und veröffentlicht es mit
dem integrierten `GITHUB_TOKEN` in der GitHub Container Registry. Zusätzliche
Docker-Hub-Secrets sind nicht erforderlich.

Veröffentlichte Image-Adresse:

```text
ghcr.io/baumannumtech/runpod-swinir:latest
```

Der Workflow liegt in [.github/workflows/docker-publish.yml](.github/workflows/docker-publish.yml).

## RunPod-Deployment

Ein echter RunPod-Endpoint wird hier noch nicht angelegt. Die Konfiguration bleibt für die erste Stufe lokal und dokumentiert:

- Queue-basierter Serverless-Endpoint
- GPU-Priorität A4000/A4500/RTX 4000
- workersMin=0, workersMax=1
- Concurrency pro Worker = 1
- Execution Timeout ca. 5 Minuten

Das geprüfte Image wurde als `0.1.0` und `latest` in GHCR veröffentlicht. Für
RunPod muss das Paket öffentlich sein oder mit Registry-Zugangsdaten verwendet
werden.
