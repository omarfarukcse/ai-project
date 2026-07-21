# Dockerfile
# Production-grade Dockerfile for CDSS Healthcare

# ============================================================================
# Stage 1: Builder
# ============================================================================
FROM python:3.9-slim as builder

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
COPY requirements-dev.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements-dev.txt

# ============================================================================
# Stage 2: Production
# ============================================================================
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 cdss && chown -R cdss:cdss /app

# Copy from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application
COPY . .

# Create directories
RUN mkdir -p data/raw data/processed data/feature_store models outputs/logs outputs/reports outputs/figures \
    && chown -R cdss:cdss /app

# Switch to non-root user
USER cdss

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Entrypoint
ENTRYPOINT ["python", "-m", "uvicorn", "src.api.app:app"]

# Default command
CMD ["--host", "0.0.0.0", "--port", "8000", "--workers", "4"]