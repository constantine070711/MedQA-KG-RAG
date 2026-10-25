FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PROJECT_ROOT=/app

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-api.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements-api.txt

COPY api ./api
COPY app ./app
COPY evaluation ./evaluation
COPY scripts ./scripts
COPY src ./src
COPY data ./data

RUN useradd --create-home --shell /usr/sbin/nologin medqa \
    && mkdir -p /app/lightrag_cache /app/models \
    && chown -R medqa:medqa /app

USER medqa

EXPOSE 8000

HEALTHCHECK --interval=20s --timeout=5s --start-period=30s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)"]

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
