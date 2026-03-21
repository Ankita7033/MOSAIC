# MOSAIC Interference Matrix Methodology

## Measurement Protocol

### Solo Profiling (Baseline)
```bash
make -C profiler/
sudo sysctl -w kernel.perf_event_paranoid=1

# Profile each class in isolation for 10 seconds
./profiler/profiler 10000 inference_critical data/solo_inference.json
./profiler/profiler 10000 dispatch_api       data/solo_api.json
# ... repeat for all 6 classes
```

### Colocation Profiling
Run two workloads simultaneously in separate terminals:
```bash
# Terminal 1 — run inference workload (e.g. llama.cpp)
./profiler/profiler 10000 dispatch_api data/colocated_api_with_inference.json

# Terminal 2 — run dispatch API workload (e.g. wrk HTTP benchmark)
./profiler/profiler 10000 inference_critical data/colocated_inference_with_api.json
```

### Compute Degradation
```python
import json
solo = json.load(open('data/solo_api.json'))
colocated = json.load(open('data/colocated_api_with_inference.json'))

ipc_deg = 1 - colocated['ipc'] / solo['ipc']
lat_ms  = colocated['duration_ms'] - solo['duration_ms']

print(f"inference_critical → dispatch_api: ipc_deg={ipc_deg:.3f}  lat={lat_ms:.0f}ms")
# Expected: ipc_deg≈0.38  lat≈22ms
```

### Update Database
```bash
# Ingest fingerprints
python3 scheduler/core_algorithm/ml_classifier.py  # runs self-test

# Update interference matrix with observed values
# (uses EMA update, alpha=0.3, confidence grows with sample count)
python3 -c "
from scheduler.core_algorithm.ml_classifier import WorkloadClassifier
clf = WorkloadClassifier()
clf.online_update('dispatch_api', ipc=2.85, llc=0.05, bw=3.0, br=0.04)
clf.save_centroids()
print('Updated')
"
```

## Confidence Weighting

```
safety_factor = 1 + (1 - confidence) × 0.5

confidence(n) = 1 - exp(-n / 10)
  n=1:  confidence=0.095  →  safety=1.45  (45% extra margin)
  n=5:  confidence=0.39   →  safety=1.30
  n=10: confidence=0.63   →  safety=1.18
  n=30: confidence=0.95   →  safety=1.02
```

Run ≥10 colocation experiments per pair for reliable estimates (confidence > 0.6).

## The Full 6×6 Matrix

```
Aggressor ↓ / Victim →    inf_crit  disp_api  sens_fus  analyt  mod_upd  log_arc
inference_critical          0.00      0.38      0.25      0.12     0.08     0.04
dispatch_api                0.05      0.00      0.10      0.06     0.04     0.02
sensor_fusion               0.18      0.14      0.00      0.10     0.12     0.03
analytics_batch             0.20      0.16      0.14      0.00     0.18     0.05
model_update                0.22      0.18      0.16      0.14     0.00     0.06
log_archive                 0.03      0.02      0.02      0.03     0.02     0.00

Units: IPC degradation fraction (0=no effect, 1=complete stall)
Threshold: 0.35 → pairs above this are NEVER colocated
```
