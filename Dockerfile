# Backend image. Runs the FastAPI app under uvicorn; migrations are
# applied as a separate step (entrypoint below), not baked into the
# image, so a redeploy with no schema change doesn't re-run alembic.
FROM python:3.12-slim AS base

# libgomp1: PaddlePaddle's OpenMP runtime dependency on Debian slim -
# without it, `import paddlepaddle` fails at container start, not at
# build time, which is a much worse place to discover it.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

COPY app/ app/
COPY ml_service/ ml_service/
COPY db/ db/
COPY migrations/ migrations/
COPY alembic.ini .
COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

ENTRYPOINT ["./docker-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
