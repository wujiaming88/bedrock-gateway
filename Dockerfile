FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy source
COPY bedrock_gateway/ bedrock_gateway/
COPY config.example.yaml config.example.yaml

# Non-root user
RUN useradd --create-home appuser
USER appuser

EXPOSE 4000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:4000/health')"

ENTRYPOINT ["python", "-m", "bedrock_gateway"]
