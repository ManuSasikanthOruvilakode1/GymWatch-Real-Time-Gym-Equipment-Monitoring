# Inference-only image: ONNX Runtime + OpenCV + FastAPI. No PyTorch, no training data.
# Build after `gymwatch export` has produced models/gymwatch.onnx:
#   docker build -t gymwatch .
#   docker run -p 8000:8000 -v "$PWD/configs:/app/configs" -v "$PWD/video:/app/video" gymwatch
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

RUN pip install --no-cache-dir \
        "numpy>=1.26" "opencv-python-headless>=4.10" "onnxruntime>=1.18" \
        "fastapi>=0.115" "uvicorn>=0.30" "python-multipart>=0.0.9" \
        "pydantic>=2.7" "pyyaml>=6.0"

COPY src/gymwatch/__init__.py src/gymwatch/config.py src/gymwatch/labels.py \
     src/gymwatch/preprocess.py src/gymwatch/predictor.py /app/gymwatch/
COPY src/gymwatch/realtime /app/gymwatch/realtime
COPY src/gymwatch/api /app/gymwatch/api
COPY scripts/serve.py /app/serve.py
COPY configs /app/configs
COPY models/gymwatch.onnx /app/models/gymwatch.onnx

RUN useradd --create-home app && mkdir -p /app/artifacts && chown -R app /app/artifacts
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')"
CMD ["python", "serve.py", "--config", "configs/default.yaml", "--host", "0.0.0.0", "--port", "8000"]
