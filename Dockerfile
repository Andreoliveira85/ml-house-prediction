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

# Copy streamlit app
COPY streamlit_app/ ./streamlit_app/

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Cloud Run uses PORT environment variable
EXPOSE 8080

# Environment variables
ENV PYTHONPATH=/app

# Run streamlit on the port Cloud Run expects
CMD ["sh", "-c", "streamlit run streamlit_app/app.py --server.address 0.0.0.0 --server.port ${PORT:-8080} --server.headless true --server.fileWatcherType none --browser.gatherUsageStats false"]