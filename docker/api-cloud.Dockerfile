# Use Python 3.13 slim base
FROM python:3.13-slim

# ---- Runtime basics
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    MODEL_PATH=/app/models/house_price_model.joblib \
    PORT=8080

WORKDIR /app

# ---- System dependencies
# Add libgomp1 for scikit-learn OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# ---- Python deps (cache-friendly: copy requirements first)
COPY requirements.txt .
RUN python -m pip install --upgrade pip wheel && \
    pip install --no-cache-dir -r requirements.txt

# ---- App code + model
COPY src/ ./src/
COPY models/ ./models/

# Build-time debug (confirms model inside the image)
RUN echo "=== DEBUG (build): files under /app/models ===" && ls -la /app/models || true && \
    echo "=== DEBUG (build): searching for *.joblib ===" && find /app -name "*.joblib" -ls || true

# ---- Security: non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080

# Runtime check appears in Cloud Run logs before app starts
# (Optional) simplified start without 'cd'; PYTHONPATH=/app makes module import fine.
CMD sh -lc 'echo "=== DEBUG (runtime): listing /app/models"; ls -la /app/models || true; \
            python -m uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT}'