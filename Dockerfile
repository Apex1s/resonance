# ---- build -----------------------------------------------------------------
FROM python:3.12-slim AS build
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip build \
 && pip wheel --no-cache-dir --wheel-dir /wheels .

# ---- runtime ---------------------------------------------------------------
FROM python:3.12-slim
LABEL org.opencontainers.image.title="Resonance" \
      org.opencontainers.image.description="Hybrid audio-feature recommendation engine" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RESONANCE_MODELS_DIR=/app/models \
    RESONANCE_REPORTS_DIR=/app/reports \
    RESONANCE_DATA_DIR=/app/data

WORKDIR /app
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels

# Pre-trained artefacts ship with the image so the container is useful on boot.
COPY models ./models
COPY reports ./reports
COPY data/raw ./data/raw

RUN useradd --create-home --uid 10001 resonance && chown -R resonance:resonance /app
USER resonance

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=4s --start-period=10s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health').status==200 else 1)"

CMD ["uvicorn", "resonance.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
