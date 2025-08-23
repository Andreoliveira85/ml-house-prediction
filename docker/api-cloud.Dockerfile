FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and models
COPY src/ ./src/
COPY models/ ./models/

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Cloud Run uses PORT environment variable
EXPOSE 8080

# Environment variables
ENV PYTHONPATH=/app

# Use PORT environment variable for Cloud Run compatibility
CMD ["sh", "-c", "cd src && python -m uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}"]