# ==============================================================================
# SONIC ATLAS — Production Containerization (Phase 5 Delivery)
# Multi-stage optimized Dockerfile for Streamlit Music Intelligence Platform
# ==============================================================================

FROM python:3.9-slim

# Environment configuration
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install minimal OS dependencies for compilation and healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application assets, datasets, trained model artifacts, and source code
COPY src/ ./src/
COPY data/processed/ ./data/processed/
COPY data/sample_playlists/ ./data/sample_playlists/
COPY models/ ./models/
COPY assets/ ./assets/
COPY .streamlit/ ./.streamlit/
COPY app.py .

# Expose default Streamlit port
EXPOSE 8501

# Healthcheck to verify platform availability
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Launch the SONIC ATLAS production platform
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
