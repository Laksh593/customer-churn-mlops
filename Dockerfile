FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install minimal OS dependencies (libgomp1 required for XGBoost runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root system user
RUN useradd -m -u 1001 appuser

# Copy application code, configs, migrations, scripts, and MLflow assets
COPY app/ ./app/
COPY src/ ./src/
COPY configs/ ./configs/
COPY alembic/ ./alembic/
COPY alembic.ini .
COPY scripts/ ./scripts/
COPY mlartifacts/ ./mlartifacts/
COPY mlflow.db .

# Normalize MLflow SQLite database file URIs for Linux paths and verify model availability
RUN python scripts/normalize_mlflow_db.py --db-path /app/mlflow.db --target-base-dir /app

# Grant appuser ownership of /app
RUN chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Container health check: verifies application liveness and model readiness
HEALTHCHECK --interval=10s --timeout=5s --retries=3 --start-period=20s \
    CMD python -c "import urllib.request, json; res = urllib.request.urlopen('http://127.0.0.1:8000/health'); data = json.loads(res.read()); exit(0 if data.get('status') == 'healthy' and data.get('model_status') == 'ready' else 1)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
