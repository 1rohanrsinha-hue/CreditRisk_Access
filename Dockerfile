# Research prototype - not for credit decisions. Serves the pre-trained model in models/deploy/.
FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
# libgomp1: OpenMP runtime required by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app
WORKDIR /app
COPY requirements-docker.txt .
RUN pip install -r requirements-docker.txt
COPY app.py ./
COPY creditrisk ./creditrisk
COPY templates ./templates
COPY models/deploy ./models/deploy
COPY experiments/results ./experiments/results
COPY experiments/figures ./experiments/figures
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "60", "app:create_app()"]
