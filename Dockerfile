FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/hf_cache

WORKDIR /app

# CPU-only torch first (the default wheel pulls CUDA libraries); versions = the ones measured locally.
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt transformers==5.19.0

COPY src ./src
COPY data/kb ./data/kb

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /hf_cache /app/data/index /app/data/llm_cache /app/logs \
    && chown -R app:app /hf_cache /app/data /app/logs
USER app

EXPOSE 8000
CMD ["python", "src/serve.py", "--host", "0.0.0.0", "--port", "8000"]