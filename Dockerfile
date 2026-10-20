# ---- build -----------------------------------------------------------------
# Base image pinned by digest; refresh it deliberately (docker pull, then
# update the digest) rather than picking up whatever 3.12-slim points to.
FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1 AS build
WORKDIR /build
COPY pyproject.toml README.md requirements.lock ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip build \
 && pip wheel --no-cache-dir --no-deps --wheel-dir /wheels .

# ---- runtime ---------------------------------------------------------------
FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1
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
COPY requirements.lock /tmp/requirements.lock
# Exact dependency versions from the lock file: the shipped model is a pickle
# and must be loaded by the scikit-learn version it was trained with.
RUN pip install --no-cache-dir -r /tmp/requirements.lock \
 && pip install --no-cache-dir --no-deps /wheels/*.whl \
 && rm -rf /wheels /tmp/requirements.lock

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
