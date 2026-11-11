FROM python:3.15-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# Install deps first so this layer is cached until requirements.txt changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY mcp_server ./mcp_server
COPY data/sample_docs ./data/sample_docs

# Run as a non-root user; only the index directory is writable.
RUN useradd --create-home --uid 1001 appuser && mkdir -p /app/data/index && chown -R appuser /app/data/index
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

# One worker: the FAISS index lives in process memory. Scale with more containers (Week 6-7), not --workers.
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
