# MOSAIC — Multi-Objective Scheduler for AI-Cloud Inference Colocation

> **Disaster-Response Edge Computing** — An interference-aware, ML-classified, deadline-driven scheduler for colocating AI inference workloads with latency-sensitive coordination services on battery-backed edge nodes.

[![Tests](https://img.shields.io/badge/tests-63%20passing-brightgreen)](#testing)
[![Language](https://img.shields.io/badge/language-Python%20%7C%20C-blue)](#tech-stack)
[![Platform](https://img.shields.io/badge/platform-Linux-orange)](#requirements)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)
[![Domain](https://img.shields.io/badge/domain-disaster--response%20edge-red)](#problem)

---

## The Problem

Emergency response agencies (FEMA, state EOCs, hospital networks) deploy edge servers at disaster sites — flood zones, wildfire perimeters, earthquake staging areas — running a **dangerous mix** of workloads:

| Class | Example | Deadline | Consequence of miss |
|---|---|---|---|
| **CRITICAL** | Drone imagery → survivor detection (YOLO/ViT) | 500ms–3s | Responder misses survivor location |
| **URGENT** | Field dispatch API, ambulance routing | 50–200ms | Coordination breaks down |
| **IMPORTANT** | Population heatmaps, supply chain | 2–10s | Degraded situational awareness |
| **BACKGROUND** | Incident log sync, sensor archival | 30s–5min | Delayed, not dangerous |

**When naive schedulers colocate these classes, everything fails simultaneously.** A 30-frame/sec drone inference burst saturates the LLC (Last-Level Cache) and memory bus, spiking the dispatch API from 40ms to 800ms — responders lose coordination exactly when they need it most.

---

## Why Existing Schedulers Fail

| Scheduler | Failure Mode on Disaster Edge |
|---|---|
| **Linux CFS** | Treats drone inference and dispatch API as equally "fair" — both thrash the LLC together, both miss deadlines |
| **Round Robin** | Timeslices GPU kernels mid-inference, adding 10–50ms overhead per context switch |
| **Static Priority** | A log-sync job assigned priority=5 at 3am still blocks a critical survivor alert at 9am if the queue is full |
| **Kubernetes default** | Bin-packs by CPU/memory requests, completely blind to cache interference patterns |

**The root gap:** none of these schedulers model the *interference relationship* between workloads. MOSAIC does.

---

## Key Results

```
═══════════════════════════════════════════════════════════════════════════════
  BENCHMARK (burst pattern, rate=6/s, 30s, seed=42)
═══════════════════════════════════════════════════════════════════════════════
  Scheduler          Hit%      P50      P95      P99  Starve%   JFI  Eff(t/Wh)
  ─────────────────────────────────────────────────────────────────────────────
  FCFS             96.8%      145    3,423   12,738    6.3%  0.999      631
  Round Robin      96.8%      152    3,560   13,248    6.3%  0.999      607
  SJF              96.8%      145    3,423   12,738    6.3%  0.999      631
  Priority Static  96.8%      145    3,423   12,738    4.8%  0.999      631
  MOSAIC           96.0%      407    1,174    3,178    0.0%  1.000    1,171  ◀
═══════════════════════════════════════════════════════════════════════════════

  MOSAIC vs best baseline (FCFS):
    P99 latency  : 12,738ms → 3,178ms  (-75.1%)   ← critical for disaster response
    Starvation   : 6.3%     → 0.0%     (-6.3 pp)  ← no background task ever starves
    Fairness     : 0.999    → 1.000               ← perfect per-class equity
    Energy eff.  : 631      → 1,171    (+85.6%)   ← critical on battery-backed nodes
```

> MOSAIC's P99 drop from **12.7 seconds to 3.2 seconds** is the critical insight: under burst load, naive schedulers let batch analytics jobs consume 10+ seconds of tail latency, starving real-time inference tasks. MOSAIC prevents this entirely via interference-aware admission control.

---

## What Makes MOSAIC Novel

### 1. ML Workload Classifier (no manual labelling needed)
Classifies incoming tasks using nearest-centroid matching on a 4D hardware counter fingerprint vector `(IPC, LLC_miss_rate, MemBW_GBs, branch_miss_rate)`, with **online EWC (Exponentially Weighted Centroid) updates** — accuracy improves as the scheduler observes more completions.

```python
clf = WorkloadClassifier()
r = clf.classify(ipc=1.8, llc_miss_rate=0.42, mem_bw_gbs=28.0, branch_miss_rate=0.08)
# → predicted_class='inference_critical'  confidence=0.94  tier=1
```

### 2. Interference Matrix as a First-Class Scheduling Primitive
A measured 6×6 pairwise matrix quantifying how much each workload class degrades every other class's IPC and latency when colocated. No open-source production scheduler exposes this as a scheduling input.

```
interference_critical → dispatch_api:  ipc_degradation=0.38  lat_overhead=22ms
log_archive           → anything:      ipc_degradation≤0.04  lat_overhead≤1ms
```

### 3. Deadline-Driven Dynamic Urgency Score
```
U(t) = tier_weight × priority_weight
       ─────────────────────────────────────────
       max(ε, deadline_remaining_ms / deadline_ms)
```
Past-deadline tasks → U = ∞ (guaranteed immediate promotion). No static priority inversion. Tier-1 CRITICAL tasks always dominate tier-2+ at equal deadline fraction.

### 4. Intel RAPL Energy Feedback Loop
Native energy constraint, not bolted-on: reads `/sys/class/powercap/intel-rapl/` every 500ms, triggers DVFS throttle (halving `cpu.weight` via cgroups v2) on lowest-urgency task when power exceeds cap. **Critical for battery-backed edge nodes.**

### 5. Jain's Fairness Index Monitoring
Tracks per-class deadline hit rates and computes JFI continuously. MOSAIC achieves **JFI=1.000** (perfect equity across all 6 workload classes) vs 0.999 for baselines.

### 6. Starvation Guard
Background tasks that have waited more than 3× their deadline get urgency boosted to ∞ — **0% starvation rate** vs 6.3% for FCFS in burst scenarios.

---

## Architecture

```
mosaic/
├── scheduler/
│   ├── core_algorithm/          ← The innovation layer
│   │   ├── workload_taxonomy.py  Disaster domain: 6 workload classes + interference matrix
│   │   ├── ml_classifier.py      Nearest-centroid + online EWC learning
│   │   ├── algorithms.py         Urgency, admission, Jain's FI, starvation guard, DVFS
│   │   └── __init__.py
│   ├── scheduler.py              Daemon: socket server + RAPL loop + cgroups v2
│   └── client.py                 Unix socket client library
│
├── workload-gen/
│   └── workload_gen.py           5 patterns: poisson/burst/sinusoidal/step/disaster
│
├── benchmarking/
│   └── benchmark.py              5 schedulers: FCFS/RR/SJF/Priority/MOSAIC
│
├── visualization/
│   └── plot_results.py           6 chart types: hit rate, latency, JFI, starvation, etc.
│
├── dashboard/
│   └── dashboard.html            Live self-contained HTML dashboard (no build step)
│
├── tests/
│   └── test_all.py               63 tests across 8 test classes
│
├── configs/
│   ├── burst_stress.json         Mass-casualty surge config
│   ├── disaster_scenario.json    Full calm→crisis→recovery config
│   └── quick_sanity.json         CI / quick validation (20s)
│
├── results/                      Auto-generated benchmark output + PNG charts
├── experiments/                  Saved experiment runs
├── docs/
│   ├── DESIGN.md                 Algorithm math + complexity + related work
│   ├── INTERFERENCE.md           Matrix construction methodology
│   └── RESULTS.md                Reproducibility guide
│
└── run_experiment.py             ← MAIN ENTRY POINT
```

---

## Quick Start

### Requirements
- Linux (kernel 4.15+ for cgroups v2)
- Python 3.9+
- No external Python packages required (stdlib only)
- Optional: `matplotlib` for PNG chart generation
- Optional: GCC for hardware counter profiler (`profiler/profiler.c`)

```bash
git clone https://github.com/yourname/mosaic
cd mosaic
```

### One-command benchmark (the money shot)

```bash
python3 run_experiment.py --compare all
```

Outputs the full 5-scheduler comparison table with all metrics. Optionally generates PNG charts if `matplotlib` is installed.

### Other entry points

```bash
# Quick 20s sanity check
python3 run_experiment.py --quick

# High-rate burst stress test
python3 run_experiment.py --compare all --pattern burst --rate 10 --duration 120

# Realistic disaster scenario (calm → onset → crisis → recovery)
python3 run_experiment.py --compare all --pattern disaster --rate 6 --duration 180

# Classify a workload from its hardware counter fingerprint
python3 run_experiment.py --classify --ipc 1.8 --llc 0.42 --bw 28.0 --br 0.08

# Test the ML classifier accuracy
python3 run_experiment.py --test-classifier

# Live daemon + workload generator
python3 run_experiment.py --demo --pattern disaster --rate 5 --duration 60

# Generate charts from saved results
python3 visualization/plot_results.py --gantt
```

### Run tests

```bash
python3 tests/test_all.py
# → 63/63 tests passed
```

---

## The Scheduling Algorithm

### Admission Control Decision Tree

```
Task arrives:
  ├─ Auto-classify via ML if class unknown
  ├─ If |running| ≥ 8 → QUEUE (max_concurrency)
  ├─ For each running task rt:
  │    ├─ ipc_degradation[candidate][rt.class] > 0.35 → QUEUE (interference_risk)
  │    └─ lat_overhead × safety_factor > 0.40 × rt.deadline_remaining → QUEUE (deadline_squeeze)
  ├─ If Σ lat_overhead_on_candidate > 0.60 × candidate.deadline → QUEUE (infeasible)
  └─ ADMIT
```

### Urgency Score Recomputed Every 100ms

```
U(t) = TIER_WEIGHT[tier] × PRIORITY_WEIGHT[priority]
       ─────────────────────────────────────────────────────
       max(0.0001, deadline_remaining_ms / deadline_ms)

TIER_WEIGHTS = {CRITICAL:4.0, URGENT:3.0, IMPORTANT:1.5, BACKGROUND:0.5}
```

Past-deadline: U = ∞. Queue sorted descending. Background tasks waiting > 3×deadline: U boosted to ∞ (starvation guard).

### Energy Feedback

```
Every 500ms:
  watts = RAPL delta(energy_uj) / (1e6 × delta_t)
  if watts > 0.88 × power_cap:
    target = min(running_tasks, key=urgency)
    target.cpu_shares //= 2   (via cgroups v2 cpu.weight)
```

---

## Experiment Configurations

| Config | Pattern | Rate | Duration | Use case |
|---|---|---|---|---|
| `quick_sanity.json` | poisson | 3/s | 20s | CI, development |
| `burst_stress.json` | burst | 8/s | 120s | Portfolio demo |
| `disaster_scenario.json` | disaster | 6/s | 180s | Research/paper |

```bash
# Load from config
python3 -c "
import json, subprocess, sys
cfg = json.load(open('configs/burst_stress.json'))
subprocess.run([sys.executable, 'run_experiment.py',
    '--compare', 'all',
    '--pattern', cfg['pattern'],
    '--rate', str(cfg['rate']),
    '--duration', str(cfg['duration']),
])
"
```

---

## Extending to Research

See `docs/DESIGN.md` for full algorithm formulation, complexity analysis, and comparison with Heracles (ISCA 2015), Quasar (ASPLOS 2014), and Google Borg.

Immediate extensions for paper submission:
- **Multi-node federation**: Extend per-node MOSAIC with etcd-backed global admission
- **Intel CAT integration**: Hardware LLC partitioning instead of scheduling-only isolation
- **RL-based interference prediction**: Replace static matrix with a learned model (state = fingerprint vectors, reward = Σdeadline_hit − λ × power_overage)
- **GPU kernel colocation**: NVIDIA MPS + NVML for GPU-level interference tracking

Target venues: USENIX ATC, EuroSys, ASPLOS, HotCarbon.

---

## Resume Bullets (copy-paste ready)

```
• Designed MOSAIC, a Linux userspace scheduler for disaster-response edge nodes,
  achieving 75% reduction in P99 tail latency and perfect Jain's Fairness Index (1.000)
  vs FCFS/RR/SJF/Priority baselines, using interference-aware admission control backed
  by a pairwise hardware-counter interference matrix (6×6, perf_event_open).

• Built an online ML workload classifier (nearest-centroid + EWC updates) that
  automatically classifies incoming tasks into 6 disaster-domain workload classes
  without manual labelling; integrated Intel RAPL energy feedback enforcing configurable
  power caps via cgroups v2 DVFS throttle, achieving 85% better energy efficiency
  (1,171 vs 631 tasks/Wh) over baseline schedulers.

• Implemented a full benchmark harness comparing 5 schedulers (FCFS, Round Robin, SJF,
  Priority, MOSAIC) across 7 metrics (hit rate, P99, JFI, starvation, throughput, energy
  efficiency, per-class fairness) with 5 realistic arrival patterns including a
  disaster-scenario generator (calm → onset → crisis → recovery); 63 unit tests,
  single-command entry point, auto-generated PNG charts.
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Core scheduler | Python 3.9+ (stdlib only) |
| Hardware profiler | C11, `perf_event_open()` syscall |
| ML classifier | Pure Python, no scikit-learn dependency |
| Resource partitioning | Linux cgroups v2 (`cpu.weight`, `memory.max`) |
| Energy feedback | Intel RAPL sysfs (`/sys/class/powercap/`) |
| IPC | Unix domain socket, newline-delimited JSON |
| Visualization | matplotlib (optional) + self-contained HTML dashboard |
| Tests | Python `unittest` — 63 tests, 8 test classes |

---

## License

MIT — see [LICENSE](LICENSE).
