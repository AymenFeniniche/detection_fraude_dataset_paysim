FROM python:3.11-slim

# Installer la bibliothèque système requise par XGBoost.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /code

# Installer les dépendances API séparément pour conserver cette couche en cache.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

RUN mkdir -p /code/models \
    && useradd --create-home --shell /usr/sbin/nologin api \
    && chown -R api:api /code

COPY --chown=api:api app/ app/
COPY --chown=api:api models/model_final.joblib models/model_meta.json models/

USER api

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/health', timeout=3).read()" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
