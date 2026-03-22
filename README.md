# MOSAIC — Multi-Objective Scheduler for AI-Cloud Inference Colocation

> Interference-aware deadline scheduler for disaster-response edge computing.
> **75% P99 latency reduction · 0% starvation · perfect Jain's Fairness Index (1.000) · +86% energy efficiency**

[![Tests](https://img.shields.io/badge/tests-63%20passing-brightgreen)](#testing)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](#requirements)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20Docker-lightgrey)](#quick-start)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)
[![Domain](https://img.shields.io/badge/domain-disaster--response%20edge-red)](#the-problem)

---

## The Problem

Edge servers at disaster sites -- wildfire command posts, flood zone EOCs, hospital surge facilities -- run dangerous mixtures of workloads simultaneously on the same physical hardware:

| Class | Example | Deadline | Miss consequence |
|---|---|---|---|
| **CRITICAL** (T1) | Drone imagery -> survivor detection (AI) | 500ms-3s | Responder misses survivor |
| **URGENT** (T2) | Field dispatch API, ambulance routing | 50-200ms | Coordination breakdown |
| **IMPORTANT** (T3) | Population heatmaps, supply chain | 2-10s | Degraded situational awareness |
| **BACKGROUND** (T4) | Incident log sync, sensor archival | 30s-5min | Delayed, not dangerous |

**When naive schedulers colocate these classes, everything fails.** A drone inference burst saturates the LLC cache and memory bus (28 GB/s, 42% miss rate), spiking the dispatch API from 40ms to 800ms. Responders lose coordination exactly when they need it most.

---

## Why Existing Schedulers Fail

| Scheduler | Failure mode |
|---|---|
| **Linux CFS** | Optimises fairness -- no concept of deadlines or cache interference |
| **Round Robin** | Preempts GPU kernels mid-inference, adding 10-50ms context-switch overhead |
| **Static Priority** | Stale priorities cause inversion -- a background job blocks a critical alert |
| **Kubernetes default** | Blind to LLC interference patterns -- only sees CPU/memory requests |

None model the **interference relationship** between workloads. MOSAIC does.

---

## Results

Benchmarked against FCFS, Round Robin, SJF, and static Priority on identical synthetic disaster-response workloads:

```
Scheduler        Hit%    P50     P95      P99   Starve%    JFI   Eff(t/Wh)
FCFS            96.9%   174ms  3,423ms 12,732ms    4.7%  0.999       637
Round Robin     96.9%   180ms  3,560ms 13,242ms    6.2%  0.999       612
SJF             96.9%   174ms  3,423ms 12,732ms    4.7%  0.999       637
Priority        96.9%   174ms  3,423ms 12,732ms    6.2%  0.999       637
MOSAIC         100.0%   282ms    992ms  3,178ms    0.0%  1.000     1,248  <--
```

| Metric | Improvement over FCFS |
|---|---|
| P99 tail latency | 12,732ms -> 3,178ms **(−75%)** |
| Starvation rate | 4.7% -> 0.0% **(eliminated)** |
| Jain's Fairness Index | 0.999 -> 1.000 **(perfect)** |
| Energy efficiency | 637 -> 1,248 tasks/Wh **(+96%)** |

---

## What Makes MOSAIC Novel

### 1. Interference matrix as a scheduling primitive
A measured 6x6 pairwise matrix of IPC degradation and latency overhead between workload classes, built via `perf_event_open()` hardware performance counters. Pairs exceeding 35% IPC degradation are **never colocated**. Updated live in SQLite as tasks complete (online learning).

### 2. Deadline-driven dynamic urgency score
```
U(t) = tier_weight x priority_weight / max(e, remaining_ms / deadline_ms)
```
Recomputed every 100ms. Past-deadline tasks get U=inf and jump to queue head. Eliminates static priority inversion entirely.

### 3. Online ML workload classifier
Nearest-centroid classification on a 4D hardware fingerprint `(IPC, LLC_miss_rate, MemBW, branch_miss_rate)` with **EWC centroid updates** -- accuracy improves continuously as tasks complete, no training data required.

```python
result = classifier.classify(ipc=1.8, llc_miss_rate=0.42, mem_bw_gbs=28.0, branch_miss_rate=0.08)
# -> inference_critical  Tier 1 CRITICAL  100% confidence
```

### 4. Intel RAPL energy feedback
Reads power counters from `/sys/class/powercap/intel-rapl/` every 500ms. When power exceeds cap, throttles the lowest-urgency task via cgroups v2 `cpu.weight` halving (software DVFS). Critical for battery-backed edge nodes.

### 5. Starvation guard + Jain's Fairness Index
Tasks waiting >3x their deadline get urgency boosted to inf -- **0% starvation** in all experiments. JFI tracked continuously across all 6 workload classes.

---

## Quick Start

### Windows
```bat
.\mosaic.bat start        :: start scheduler daemon
.\mosaic.bat dashboard    :: live dashboard at http://localhost:7777
.\mosaic.bat run          :: send live workload (makes dashboard charts move)
.\mosaic.bat benchmark    :: reproduce the results table above
.\mosaic.bat test         :: run all 63 tests
.\mosaic.bat classify 1.8 0.42 28.0 0.08   :: ML-classify a workload fingerprint
```

**Live dashboard (open 3 terminals simultaneously):**
```
Terminal 1:  .\mosaic.bat start
Terminal 2:  .\mosaic.bat dashboard
Terminal 3:  .\mosaic.bat run
Browser:     http://localhost:7777
```

### Linux / macOS
```bash
git clone https://github.com/yourname/mosaic && cd mosaic

# One-command benchmark (the headline result)
python3 run_experiment.py --compare all

# Quick 20s sanity check
python3 run_experiment.py --compare all --quick

# Full disaster scenario (calm -> onset -> crisis -> recovery)
python3 run_experiment.py --compare all --pattern disaster --rate 6 --duration 180

# ML classifier accuracy test
python3 run_experiment.py --test-classifier

# Classify from hardware counter fingerprint
python3 run_experiment.py --classify --ipc 1.8 --llc 0.42 --bw 28.0 --br 0.08

# All 63 tests
python3 tests/test_all.py
```

### Docker
```bash
docker build -t mosaic:latest .
docker run --rm -it mosaic:latest demo
docker-compose up -d    # scheduler + live dashboard on port 7777
```

### Kubernetes
```bash
kubectl apply -f k8s/mosaic-deployment.yaml   # DaemonSet: one pod per edge node
kubectl port-forward svc/mosaic-dashboard 7777:7777
```

---

## Architecture

```
mosaic/
├── scheduler/
│   ├── core_algorithm/
│   │   ├── workload_taxonomy.py   6 workload classes + 6x6 interference matrix
│   │   ├── ml_classifier.py       Nearest-centroid + online EWC updates
│   │   └── algorithms.py          Urgency, admission, JFI, starvation, DVFS
│   ├── scheduler.py               Daemon: socket server + RAPL loop + cgroups v2
│   └── client.py                  Client library (Unix socket on Linux, TCP on Windows)
├── workload-gen/
│   └── workload_gen.py            5 arrival patterns: poisson/burst/sinusoidal/step/disaster
├── benchmarking/
│   └── benchmark.py               5 schedulers x 7 metrics
├── dashboard/
│   ├── dashboard.html             Live telemetry dashboard (100% backend-driven via SSE)
│   └── server.py                  HTTP + SSE server -- all data from SQLite + scheduler
├── profiler/
│   ├── profiler.c                 Hardware counter profiler (perf_event_open)
│   └── build_db.py                SQLite interference matrix builder
├── tests/
│   └── test_all.py                63 tests, 8 test classes
├── docs/
│   ├── DESIGN.md                  Algorithm math, complexity, related work
│   ├── INTERFERENCE.md            Matrix construction methodology
│   └── mosaic_paper.tex           6-page workshop paper (USENIX HotEdge target)
├── k8s/                           Kubernetes DaemonSet + benchmark Job manifests
├── configs/                       burst_stress / disaster_scenario / quick_sanity
└── run_experiment.py              Main entry point
```

---

## Live Dashboard

The dashboard at `http://localhost:7777` is **100% backend-driven** via Server-Sent Events (SSE). Zero simulation, zero hardcoded data.

| Element | Source | Rate |
|---|---|---|
| Hit rate, P99, power, JFI, starvation KPIs | Scheduler metrics | 500ms |
| Hit rate / latency / power / queue depth charts | Scheduler metrics history | 500ms |
| Per-class hit rate chart | `metrics.class_hit_rates` | 500ms |
| Running tasks + urgency scores | Live task list | 500ms |
| Event log (ADMIT / QUEUE / COMPLETE / THROTTLE) | `metrics.jsonl` | 400ms |
| Interference matrix | SQLite `fingerprints.db` | 500ms + on each completion |

Matrix cells start dim (seeded defaults, confidence=50%) and brighten as real task completions feed observed latency data back to the DB.

---

## Algorithm Detail

### Admission control
```
Task arrives:
  1. Auto-classify via ML if class unknown
  2. If |running| >= 16 -> QUEUE (max concurrency)
  3. For each running task:
       if ipc_degradation[candidate -> running] > 0.35 -> QUEUE (interference)
       if lat_overhead * safety > 0.40 * running.deadline_remaining -> QUEUE (squeeze)
  4. If sum(lat_overhead on candidate) > 0.60 * candidate.deadline -> QUEUE
  5. ADMIT -> cgroups v2 resource partition created
```

### Urgency score (recomputed every 100ms)
```
U(t) = TIER_WEIGHTS[tier] * PRIORITY_WEIGHTS[priority]
       -------------------------------------------------------
       max(0.0001,  deadline_remaining_ms / deadline_ms)

TIER_WEIGHTS = {CRITICAL: 4.0, URGENT: 3.0, IMPORTANT: 1.5, BACKGROUND: 0.5}
Past deadline -> U = inf    |    Age > 3x deadline -> U = inf (starvation guard)
```

### Energy feedback
```
Every 500ms:
  power = RAPL_delta_energy_uJ / (1e6 * delta_t_seconds)
  if power > 0.88 * power_cap:
    target = lowest urgency running task
    target.cpu_shares //= 2   (via cgroups v2 cpu.weight)
```

---

## Requirements

**Windows:** Python 3.9+ -- all features work (cgroups and RAPL simulated)

**Linux:** Python 3.9+ -- full feature set including hardware counters, cgroups v2, Intel RAPL

**Optional:** `matplotlib` for PNG charts, GCC for hardware profiler, Docker, kubectl

No external Python packages required for the core scheduler.

---

## Testing

```bash
python3 tests/test_all.py
# -> 63/63 tests passed
```

| Test class | Count | Covers |
|---|---|---|
| TestWorkloadTaxonomy | 8 | Domain framing, matrix validity, tier weights |
| TestUrgencyScoring | 7 | Monotonicity, tier dominance, NaN safety |
| TestMLClassifier | 9 | Accuracy, online update, metadata fallback |
| TestInterferenceAdmission | 7 | Admission logic, deadline squeeze, max concurrency |
| TestFairnessAndStarvation | 9 | JFI math, starvation detection, DVFS trigger |
| TestWorkloadGenerator | 8 | Arrival patterns, weight distribution, metrics |
| TestSchedulerBenchmark | 10 | All 5 schedulers produce valid results |
| TestMOSAICQuality | 4 | Starvation, fairness, disaster pattern, tier priority |

---

## Related Work

| System | Venue | Key difference |
|---|---|---|
| Heracles | ISCA 2015 | Reactive (monitors SLO violations post-hoc) vs MOSAIC proactive (prevents via matrix) |
| Quasar | ASPLOS 2014 | Static offline matrix vs MOSAIC online EWC updates |
| Parties | ASPLOS 2019 | Intel CAT hardware partitioning vs MOSAIC scheduling-based isolation |
| Google Borg | EuroSys 2015 | No interference model, datacenter-scale vs edge-optimised |

---

## To run in docker 
docker start mosaic-scheduler
docker exec mosaic-scheduler python workload-gen/workload_gen.py --pattern disaster --rate 4 --duration 600 --scheduler --quiet

## License

MIT -- see [LICENSE](LICENSE).
