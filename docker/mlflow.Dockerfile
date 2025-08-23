FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install MLflow-specific requirements
COPY requirements-mlflow.txt .
RUN pip install --no-cache-dir -r requirements-mlflow.txt

# Create mlruns directory
RUN mkdir -p /app/mlruns

# Create non-root user
RUN useradd -m -u 1000 mlflowuser && chown -R mlflowuser:mlflowuser /app
USER mlflowuser

# Expose MLflow port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Environment variables
ENV MLFLOW_BACKEND_STORE_URI=file:///app/mlruns
ENV MLFLOW_DEFAULT_ARTIFACT_ROOT=/app/mlruns

# Run MLflow server
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "file:///app/mlruns", \
     "--default-artifact-root", "/app/mlruns"]