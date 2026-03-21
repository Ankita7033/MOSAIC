# ============================================================
#  MOSAIC — Dockerfile
#  Multi-stage: builder compiles C profiler, runtime is lean
#
#  Build:
#    docker build -t mosaic:latest .
#
#  Run scheduler daemon:
#    docker run -d --name mosaic \
#      --privileged \
#      -v mosaic-data:/app/data \
#      -p 7777:7777 \
#      mosaic:latest scheduler
#
#  Run benchmark:
#    docker run --rm mosaic:latest benchmark
#
#  Run dashboard:
#    docker run -d -p 7777:7777 mosaic:latest dashboard
#
#  Run demo (start + workload + status):
#    docker run --rm -it mosaic:latest demo
# ============================================================

# ── Stage 1: Build C profiler ─────────────────────────────────
FROM gcc:12-bullseye AS builder

WORKDIR /build
COPY profiler/profiler.c profiler/Makefile ./

RUN make -f Makefile profiler

# ── Stage 2: Runtime ──────────────────────────────────────────
FROM python:3.11-slim-bullseye AS runtime

LABEL maintainer="MOSAIC Project"
LABEL description="Multi-Objective Scheduler for AI-Cloud Inference Colocation"
LABEL version="1.0"

# Install minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    linux-perf \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy everything
COPY . .

# Copy built profiler
COPY --from=builder /build/profiler ./profiler/profiler

# Optional: matplotlib for chart generation
RUN pip install --no-cache-dir matplotlib 2>/dev/null || true

# Create runtime directories
RUN mkdir -p data/logs results experiments

# Initialise the fingerprint database
RUN python3 profiler/build_db.py --matrix 2>/dev/null || true

# Expose dashboard port
EXPOSE 7777

# Default env
ENV MOSAIC_POWER_CAP=85
ENV MOSAIC_LOG_LEVEL=INFO
ENV PYTHONUNBUFFERED=1

# ── Entrypoint ───────────────────────────────────────────────
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["scheduler"]

# Health check — polls /health endpoint
HEALTHCHECK --interval=10s --timeout=3s --start-period=8s --retries=3 \
    CMD curl -sf http://localhost:7777/health || exit 1
