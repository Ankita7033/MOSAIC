FROM python:3.10-slim

WORKDIR /app

# Install OS dependencies for PMU (perf) and visualization
RUN apt-get update && apt-get install -y \
    linux-perf \
    htop \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# If requirements.txt doesn't exist, we fallback safely.
RUN pip install --no-cache-dir -r requirements.txt || pip install fastapi uvicorn matplotlib

COPY . .

ENV PYTHONPATH=/app
CMD ["python", "mosaic/run_experiment.py"]
