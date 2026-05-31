# Theo – Production-Image (FastAPI/uvicorn + OpenCV).
FROM python:3.11-slim

# Systembibliotheken für OpenCV (headless) und Video-Dekodierung (ffmpeg).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Optional: YOLO-Detektor (zieht PyTorch -> großes Image, mehr RAM nötig).
ARG INSTALL_YOLO=false

# Abhängigkeiten zuerst (besseres Layer-Caching).
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[web]" \
    && if [ "$INSTALL_YOLO" = "true" ]; then \
         pip install --no-cache-dir "ultralytics>=8.1"; \
       fi

# Produktionseinstellungen (per docker-compose/.env überschreibbar).
ENV THEO_WORKERS=2 \
    THEO_MAX_ANALYZE_SECONDS=30 \
    THEO_MAX_UPLOAD_MB=200 \
    THEO_DEFAULT_DETECTOR=hog \
    PORT=8000

EXPOSE 8000

# uvicorn mit konfigurierbarer Worker-Zahl starten.
CMD ["sh", "-c", "uvicorn theo.web.app:app --host 0.0.0.0 --port ${PORT} --workers ${THEO_WORKERS}"]
