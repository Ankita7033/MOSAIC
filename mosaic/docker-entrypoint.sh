#!/usr/bin/env bash
# MOSAIC Docker Entrypoint
set -euo pipefail

RED='\033[0;31m'; GRN='\033[0;32m'; CYN='\033[0;36m'; BLD='\033[1m'; RST='\033[0m'
info() { echo -e "${CYN}[mosaic] $*${RST}"; }
ok()   { echo -e "${GRN}[mosaic] ✓ $*${RST}"; }
err()  { echo -e "${RED}[mosaic] ✗ $*${RST}"; exit 1; }

CMD="${1:-scheduler}"

case "${CMD}" in

  scheduler)
    info "Starting MOSAIC scheduler daemon (power_cap=${MOSAIC_POWER_CAP}W)"
    # Start scheduler in background
    python3 /app/scheduler/scheduler.py \
        --power-cap "${MOSAIC_POWER_CAP}" \
        --log-level "${MOSAIC_LOG_LEVEL}" &
    SCHED_PID=$!

    # Wait for socket
    for i in $(seq 1 30); do
        [[ -S /app/data/mosaic.sock ]] && break
        sleep 0.3
    done
    [[ -S /app/data/mosaic.sock ]] || err "Scheduler failed to start"
    ok "Scheduler started (PID=${SCHED_PID})"

    # Start dashboard server
    info "Starting dashboard on port 7777"
    python3 /app/dashboard/server.py --port 7777 --host 0.0.0.0 &

    # Keep container alive, forward signals
    wait ${SCHED_PID}
    ;;

  dashboard)
    info "Starting dashboard server on port 7777"
    python3 /app/dashboard/server.py --port 7777 --host 0.0.0.0
    ;;

  benchmark)
    PATTERN="${2:-burst}"
    RATE="${3:-6}"
    DURATION="${4:-60}"
    info "Running benchmark: pattern=${PATTERN} rate=${RATE}/s duration=${DURATION}s"
    python3 /app/run_experiment.py \
        --compare all \
        --pattern "${PATTERN}" \
        --rate "${RATE}" \
        --duration "${DURATION}" \
        --no-charts
    ;;

  demo)
    info "Running MOSAIC demo"
    # Start scheduler
    python3 /app/scheduler/scheduler.py \
        --power-cap "${MOSAIC_POWER_CAP}" &
    SCHED_PID=$!

    for i in $(seq 1 30); do
        [[ -S /app/data/mosaic.sock ]] && break
        sleep 0.3
    done
    ok "Scheduler started"

    # Run workload
    info "Running disaster workload (30s)"
    python3 /app/workload-gen/workload_gen.py \
        --pattern disaster --rate 5 --duration 30 \
        --scheduler --quiet

    # Show status
    python3 /app/scheduler/client.py status

    # Stop scheduler
    kill "${SCHED_PID}" 2>/dev/null || true
    ok "Demo complete"
    ;;

  test)
    info "Running test suite"
    python3 /app/tests/test_all.py
    ;;

  workload)
    PATTERN="${2:-disaster}"
    RATE="${3:-5}"
    DURATION="${4:-60}"
    info "Running workload generator: pattern=${PATTERN}"
    python3 /app/workload-gen/workload_gen.py \
        --pattern "${PATTERN}" \
        --rate "${RATE}" \
        --duration "${DURATION}" \
        --scheduler
    ;;

  shell)
    exec /bin/bash
    ;;

  *)
    # Pass through arbitrary commands
    exec "$@"
    ;;

esac
