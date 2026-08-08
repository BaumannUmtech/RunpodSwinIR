# Implementierungsanweisung: RunPod-SwinIR-Worker

## Ziel

Erstelle in diesem Ordner ein eigenständiges Python-Projekt für einen
RunPod-Serverless-GPU-Worker. Der Worker empfängt ein Bild, skaliert es mit dem
offiziellen Lightweight-SwinIR-x2-Modell hoch und gibt ein JPEG in der exakt
angeforderten Full-HD-Größe zurück.

Dieses Projekt bleibt vollständig von `TouServerDjK`, OmniVoice und dem
LTX-Video-Worker getrennt. Es soll ein eigenes Docker-Image und einen eigenen
RunPod-Endpoint erhalten.

## Abgrenzung der ersten Stufe

In dieser Stufe implementieren:

- SwinIR x2 auf NVIDIA-GPU/CUDA
- die Formate `16:9`, `9:16` und `1:1`
- exakte Zielgrößen 1920×1080, 1080×1920 und 1080×1080
- gekachelte Inferenz für begrenzten GPU-Speicher
- Qualitäts- und Plausibilitätsprüfungen
- RunPod-Serverless-Handler
- Docker-Image, Tests und Dokumentation

Noch nicht implementieren:

- CodeFormer oder GFPGAN
- Audio- oder Videofunktionen
- Credit-Abrechnung
- Änderungen im Repository `TouServerDjK`
- eine öffentliche Weboberfläche

## Modell

Verwende ausschließlich das offizielle Modell:

- Modell: `002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth`
- Quelle: `https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth`
- SHA-256: `193b229909ca89cd8b55de9c9e7fce146ae759d59dfcd78d8feb9dd1d6fa0fd7`

Das Modell ist klein und soll beim Docker-Build heruntergeladen, anhand der
Prüfsumme verifiziert und in das Image aufgenommen werden. Der Worker darf es
nicht bei jeder Anfrage erneut laden. Das geladene Modell muss zwischen
Aufträgen im Prozessspeicher verbleiben.

Als Modell-Loader kann `spandrel>=0.4,<1` verwendet werden. Prüfe nach dem Laden,
dass es sich um ein Bildmodell mit Skalierungsfaktor 2 handelt.

## Vorgeschlagene Projektstruktur

```text
RunpodSwinIR/
├── runpod_swinir/
│   ├── __init__.py
│   ├── contracts.py
│   ├── image_utils.py
│   ├── upscaler.py
│   └── worker.py
├── scripts/
│   └── download_model.py
├── models/
│   └── .gitkeep
├── tests/
│   ├── test_contracts.py
│   ├── test_image_utils.py
│   ├── test_upscaler.py
│   └── test_worker.py
├── .dockerignore
├── .gitignore
├── Dockerfile
├── README.md
├── pyproject.toml
└── requirements.txt
```

## API-Vertrag

Der RunPod-Handler liest die Nutzdaten aus `event["input"]`.

### Eingabe

```json
{
  "input": {
    "image_base64": "BASE64_OHNE_DATA_URL_PREFIX",
    "aspect_ratio": "16:9",
    "output_format": "jpeg",
    "quality": 95
  }
}
```

Regeln:

- `image_base64` ist erforderlich und darf JPEG, PNG oder WebP enthalten.
- Maximale dekodierte Eingabegröße über eine Umgebungsvariable begrenzen,
  standardmäßig 15 MiB.
- `aspect_ratio` erlaubt nur `16:9`, `9:16` oder `1:1`.
- `output_format` ist in Stufe 1 ausschließlich `jpeg`.
- `quality` liegt zwischen 80 und 100, Standardwert 95.
- EXIF-Ausrichtung vor der Verarbeitung anwenden.
- Transparenz kontrolliert auf einen weißen Hintergrund komponieren.
- Ungültige Werte mit einer verständlichen Fehlermeldung ablehnen.

### Erfolgreiche Ausgabe

```json
{
  "success": true,
  "image_base64": "BASE64_OHNE_DATA_URL_PREFIX",
  "mime_type": "image/jpeg",
  "width": 1920,
  "height": 1080,
  "upscaler": "swinir",
  "model": "002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2",
  "processing_ms": 1234
}
```

### Fehlerausgabe

```json
{
  "success": false,
  "error_code": "INVALID_IMAGE",
  "error": "Das Eingabebild konnte nicht gelesen werden."
}
```

Mindestens folgende Fehlercodes vorsehen:

- `INVALID_REQUEST`
- `INVALID_IMAGE`
- `IMAGE_TOO_LARGE`
- `UNSUPPORTED_ASPECT_RATIO`
- `MODEL_UNAVAILABLE`
- `UPSCALE_FAILED`
- `INVALID_MODEL_OUTPUT`

Keine Stacktraces, Dateipfade, Zugangsdaten oder internen RunPod-Informationen
an den Client zurückgeben. Vollständige Fehler nur serverseitig protokollieren.

## Bildverarbeitung

1. Bild sicher dekodieren und EXIF-Ausrichtung anwenden.
2. In RGB/sRGB umwandeln; Transparenz auf Weiß komponieren.
3. Mittig auf das ausgewählte Seitenverhältnis zuschneiden.
4. In einen CUDA-Float32-Tensor im Bereich 0–1 umwandeln.
5. SwinIR x2 gekachelt ausführen.
6. Kacheln ohne Lücken oder uninitialisierte Pixel zusammensetzen.
7. Das x2-Ergebnis mit hochwertigem Lanczos-Resampling exakt auf die Zielgröße
   normalisieren.
8. Als JPEG mit der angeforderten Qualität kodieren.

Konfigurierbare Umgebungsvariablen:

```text
SWINIR_MODEL_PATH=/app/models/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth
SWINIR_TILE_SIZE=256
SWINIR_TILE_PADDING=16
MAX_INPUT_BYTES=15728640
LOG_LEVEL=INFO
```

Kachelgröße auf ein Vielfaches des SwinIR-Fensters 8 normalisieren. Randkacheln
vor der Inferenz per Reflection Padding auf ein Vielfaches von 8 auffüllen und
die zusätzliche Ausgabe danach wieder abschneiden. Überlappungen korrekt
zurückschneiden, damit keine sichtbaren Nähte entstehen.

## Qualitätsprüfungen

Ein Ergebnis darf nur erfolgreich zurückgegeben werden, wenn:

- alle Tensorwerte endlich sind (`torch.isfinite`),
- die erwartete x2-Ausgabegröße vorliegt,
- jede Zielkachel vollständig geschrieben wurde,
- keine unerwartet schwarze oder leere Kachel entstanden ist,
- die mittleren Farbkanäle nicht unplausibel stark vom Ausgangsbild abweichen,
- das abschließende JPEG exakt die angeforderte Zielgröße besitzt.

Bei einem Fehler kein beschädigtes Bild als Erfolg zurückgeben. Der Worker gibt
`INVALID_MODEL_OUTPUT` oder `UPSCALE_FAILED` zurück. Der Lanczos-Fallback bleibt
Aufgabe von Toubot.

## GPU- und Laufzeitverhalten

- `torch.cuda.is_available()` beim Start prüfen.
- Ohne CUDA den Worker mit einer klaren Startfehlermeldung beenden.
- Modell einmalig beim Modul-/Workerstart laden, auf CUDA verschieben und
  `eval()` aufrufen.
- Inferenz ausschließlich unter `torch.inference_mode()` ausführen.
- Zunächst Float32 verwenden. FP16 erst nach einem nachgewiesenen Qualitäts- und
  Stabilitätstest optional aktivieren.
- Gleichzeitige Inferenz innerhalb eines Workerprozesses durch einen Lock oder
  RunPod-Concurrency 1 verhindern.
- Nach einem Auftrag keine globale `empty_cache()`-Routine ohne belegten Grund
  ausführen; Speicherlecks stattdessen durch Tests erkennen.

## RunPod-Handler

In `runpod_swinir/worker.py`:

```python
import runpod


def handler(event):
    # validieren, dekodieren, skalieren, Ergebnisvertrag zurückgeben
    ...


runpod.serverless.start({"handler": handler})
```

Der Handler darf beim Import keine Anfrage an Toubot senden. Zugangsdaten werden
nur über RunPod-Secrets beziehungsweise Umgebungsvariablen eingebunden.

## Docker

- Eine explizit versionierte CUDA-/PyTorch-Runtime verwenden, die von der
  gewählten RunPod-GPU unterstützt wird.
- Keine `latest`-Tags für produktive Abhängigkeiten.
- Python-Abhängigkeiten reproduzierbar pinnen.
- Modell während des Builds mit `scripts/download_model.py` herunterladen und
  SHA-256 verifizieren.
- Containerprozess direkt mit dem Worker starten.
- `.env`, Tests, Git-Daten, lokale Bilder und Caches über `.dockerignore`
  ausschließen.
- Keine API-Schlüssel in Image, Git oder Docker-History schreiben.

Das Image soll zunächst für eine A4000/A4500 mit 16 GB VRAM geeignet sein. Für
dieses Lightweight-Modell ist kein LTX-ähnlicher Speicherbedarf vorgesehen.

## Tests

Automatisierte Tests müssen mindestens abdecken:

- gültige und ungültige API-Eingaben,
- Größenlimit vor einer großen Speicherallokation,
- JPEG-, PNG- und WebP-Dekodierung,
- EXIF-Ausrichtung und Alpha-Komposition,
- alle drei Seitenverhältnisse und Zielgrößen,
- Kachelgrenzen und vollständige Abdeckung,
- Erkennung von NaN/Inf,
- Erkennung einer schwarzen/fehlenden Kachel,
- verständliche Fehlerverträge ohne interne Details,
- Modell wird pro Prozess nur einmal geladen,
- Worker-Erfolg mit gemocktem Modell,
- Worker-Fehler mit gemockter Modellausnahme.

Zusätzlich einen GPU-Smoke-Test bereitstellen, der nur ausgeführt wird, wenn CUDA
verfügbar ist. Er soll ein kleines Testbild mit dem echten Modell verarbeiten,
Zielgröße, endliche Pixel und korrekte Farbreihenfolge prüfen.

## Lokale und Docker-Verifikation

Dokumentiere und führe – soweit in der Umgebung möglich – aus:

```bash
python -m pytest -q
python -m compileall runpod_swinir
docker build -t runpod-swinir:local .
```

Für einen lokalen Handler-Test ein kleines Skript oder einen dokumentierten
Aufruf bereitstellen. Keine großen Binärbilder in Git einchecken.

## RunPod-Konfiguration

Nach erfolgreichem Image-Test dokumentieren:

- eigener Queue-basierter Serverless-Endpoint,
- GPU-Priorität: A4000, A4500 oder RTX 4000,
- `workersMin=0`, zunächst `workersMax=1`,
- Concurrency pro Worker zunächst 1,
- ausreichend langes Execution Timeout, beispielsweise 5 Minuten,
- API-Aufruf über `/run`, Ergebnisabfrage über `/status/{job_id}`,
- Modell bereits im Image, daher kein Netzwerk-Volume notwendig.

Noch keinen RunPod-Endpoint automatisch erstellen und kein Image pushen, solange
Registry, Image-Name und Zugangsdaten nicht ausdrücklich vorgegeben wurden.

## Spätere Toubot-Anbindung

Nur als Vertrag dokumentieren, noch nicht in diesem Projekt implementieren:

1. Toubot erzeugt das Standardbild.
2. Toubot sendet es asynchron an `/run` des SwinIR-Endpoints.
3. Toubot pollt RunPod `/status/{job_id}`.
4. Bei Erfolg prüft und speichert Toubot das Full-HD-JPEG.
5. Bei RunPod-Fehler oder Timeout verwendet Toubot Lanczos und zeigt den
   Fallback-Grund an.

Für eine spätere Optimierung kann Base64 durch kurzlebige signierte MinIO-/S3-
URLs ersetzt werden. Dabei SSRF-Schutz, erlaubte Hosts, kurze Ablaufzeiten und
maximale Downloadgröße vorsehen.

## Fertigstellungskriterien

Die Aufgabe ist abgeschlossen, wenn:

- das Projekt eigenständig installierbar ist,
- alle CPU-/Mock-Tests bestehen,
- der echte CUDA-Smoke-Test dokumentiert ist,
- das Docker-Image erfolgreich baut,
- das Modell im Image enthalten und verifiziert ist,
- Ein- und Ausgabe exakt dem Vertrag entsprechen,
- README die lokale Nutzung, Docker und RunPod-Deployment erklärt,
- keine Secrets und keine unnötigen großen Dateien in Git liegen.

Arbeite bestehende Dateien respektvoll ein, falls der Ordner inzwischen Inhalte
enthält. Implementiere die Aufgabe vollständig, teste proportional zum Risiko
und dokumentiere verbleibende Schritte oder externe Zugangsvoraussetzungen.
