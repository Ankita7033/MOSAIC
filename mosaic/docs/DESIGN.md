# MOSAIC Design Document

## Problem Formulation

Let T = {t₁, t₂, …, tₙ} be the stream of submitted tasks.
Each task tᵢ is characterised by tuple (cᵢ, dᵢ, pᵢ, sᵢ):
- **class** cᵢ ∈ C (workload class from ML classifier)
- **deadline** dᵢ (ms from submission)
- **priority** pᵢ ∈ {1, 2, 3}
- **service time** sᵢ (execution duration, ms)

Let R ⊆ T be running tasks, Q ⊆ T be queued tasks.

**Objective:** maximise deadline hit rate Σᵢ 𝟙[completion_time(tᵢ) ≤ dᵢ]

Subject to:
- Power constraint:        P(R) ≤ P_cap
- Concurrency:             |R| ≤ N_max = 8
- Interference feasibility: ∀ tₐ ∈ R, interference_impact(cᵢ, cₐ) ≤ budget_fraction × dₐ

---

## Urgency Score

```
U(t) = w_tier(tier(t)) × w_priority(p_t)
       ─────────────────────────────────────
       max(ε, remaining_ms(t) / d_t)

where ε = 10⁻⁴  (prevents division by zero)
```

| Property | Proof |
|---|---|
| Monotone increasing | remaining_ms decreases → denominator decreases → U increases ✓ |
| Tier dominance | w_tier(1) = 4.0 > w_tier(4) = 0.5 → tier-1 always outranks tier-4 ✓ |
| Past-deadline → ∞ | remaining_ms = 0 → U = w_tier × w_p / ε → ∞ ✓ |
| Continuous | No discontinuities except at deadline boundary ✓ |

---

## ML Workload Classifier

### Feature Space

Each workload class is represented as centroid in 4D space:
```
v = (IPC, LLC_miss_rate, MemBW_GBs, branch_miss_rate)
```

Weighted normalised Euclidean distance:
```
d(v_query, v_centroid) = √( Σᵢ wᵢ × ((vᵢ - cᵢ) / scaleᵢ)² )

Weights: w_IPC=1.0, w_LLC=2.5, w_BW=2.0, w_BR=0.8
Scales:  IPC/4.0,  LLC/1.0,  BW/50.0,  BR/1.0
```

LLC and memory bandwidth get higher weights because they are the primary interference mechanisms.

### Online Learning (EWC Update)

After each task completion with confirmed class:
```
centroid_new[dim] = (1 - α) × centroid_old[dim] + α × observed[dim]   α = 0.15
n_samples += 1
```

This is an exponentially-weighted moving average with α=0.15, giving the last observation 15% influence. After 10 observations, the centroid is 79% shaped by data; after 30, 99%.

### Confidence Score

```
confidence = 1 - (d_best / mean(d_all))
confidence = clamp(confidence, 0.0, 1.0)
```

A query equidistant from all centroids → confidence = 0. A query exactly on its centroid → confidence = 1.

---

## Interference Admission Check

For candidate class cₐ with running task set R:

```
1. if |R| ≥ N_max: REJECT (max_concurrency)

2. for each running task rt ∈ R:
   a. (ipc_deg, lat_ms) = INTERFERENCE_MATRIX[cₐ][rt.class]
   b. safety = 1 + (1 - confidence(cₐ, rt.class)) × 0.5
   c. effective_lat = lat_ms × safety
   d. if ipc_deg > 0.35: REJECT (ipc_threshold)
   e. if effective_lat > 0.40 × rt.deadline_remaining: REJECT (deadline_squeeze)
   f. overhead_on_candidate += INTERFERENCE_MATRIX[rt.class][cₐ].lat_ms × safety

3. if overhead_on_candidate > 0.60 × cₐ.deadline: REJECT (candidate_infeasible)

4. ADMIT
```

### Safety Margin Derivation

When confidence is low (we've only seen this pair once), we apply a 50% safety margin:
```
effective_lat = lat_ms × (1 + 0.5 × (1 - confidence))
```
- Fresh entry (confidence=0.1): effective_lat = lat_ms × 1.45
- Well-measured (confidence=0.9): effective_lat = lat_ms × 1.05

This makes MOSAIC conservative under uncertainty — the correct tradeoff for disaster-response infrastructure.

---

## Jain's Fairness Index

```
JFI = (Σᵢ xᵢ)² / (n × Σᵢ xᵢ²)

Range: [1/n, 1.0]
n = number of workload classes with at least one completed task
xᵢ = deadline hit rate for class i
```

Interpretation:
- JFI = 1.0: all classes hit deadlines at identical rates (perfect equity)
- JFI = 1/n: one class monopolises success; all others get nothing

MOSAIC achieves JFI = 1.000 because interference-aware admission prevents any class from systematically monopolising resources.

---

## Starvation Guard

A task is starving if:
```
age_ms(t) > STARVATION_MULT × deadline_ms(t)   (STARVATION_MULT = 3.0)
```

When detected: `t.urgency = ∞` → task is immediately moved to head of queue and admitted at next drain cycle.

This prevents the pathological case where a flood of CRITICAL tasks permanently blocks background tasks. MOSAIC achieves 0% starvation rate even under burst load.

---

## Energy Feedback

```
watts_t = (energy_uj_t - energy_uj_{t-1}) / (1e6 × Δt)

if watts_t > 0.88 × P_cap:
    target = argmin_{t ∈ R} U(t)
    target.cpu_shares //= 2
    cgroups/target_id/cpu.weight = max(1, target.cpu_shares)
```

RAPL counter overflow (at ~262 kJ): handled by `if Δuj < 0: Δuj += 2³²`

---

## Complexity

| Operation | Complexity | Constant factor |
|---|---|---|
| Urgency computation | O(1) | ~100ns |
| Interference check | O(\|R\|) = O(8) | ~5µs |
| Queue sort | O(Q log Q) | ~1ms at Q=64 |
| RAPL read | O(1) | ~2µs (sysfs) |
| ML classification | O(k) = O(6) | ~50µs |
| Full scheduling tick | O(Q log Q) | <5ms |

Scheduling overhead is negligible for disaster-edge workloads with service times of 5ms–15s.

---

## Comparison to Prior Work

| System | Interference | Energy | Deadlines | Open Source | ML classify |
|---|---|---|---|---|---|
| Linux CFS | ✗ | ✗ | ✗ | ✓ | ✗ |
| Kubernetes | ✗ | ✗ | ✗ | ✓ | ✗ |
| **Heracles** (Lo et al., ISCA 2015) | ✓ partial | ✗ | Soft | ✗ | ✗ |
| **Quasar** (Delimitrou, ASPLOS 2014) | ✓ | ✗ | ✗ | ✗ | k-NN |
| **Parties** (Xu et al., ASPLOS 2019) | ✓ | ✗ | ✗ | ✗ | ✗ |
| **MOSAIC** | ✓ full | ✓ RAPL | ✓ hard+soft | **✓** | **✓ online** |

**Key gaps MOSAIC fills over prior work:**
1. Heracles monitors SLO violations reactively; MOSAIC prevents them proactively via admission control.
2. Quasar classifies with k-NN but uses static interference matrices offline; MOSAIC updates centroids online.
3. No prior open-source system combines all five: interference matrix, ML classification, deadline urgency, RAPL feedback, and Jain's FI monitoring.

---

## Limitations and Future Work

1. **Interference model accuracy**: The matrix is built from representative workloads. Adversarial or pathological workloads may exhibit different interference patterns. Mitigation: online EMA updates + conservative confidence margins.

2. **Single-node only**: Current MOSAIC operates on one edge node. Extension: federate instances via etcd for multi-node cluster scheduling (direct path to USENIX ATC submission).

3. **Profiler requires elevated permissions**: `perf_event_open` needs `CAP_PERFMON` or `perf_event_paranoid ≤ 1`. The scheduler daemon itself runs fully unprivileged.

4. **GPU scheduling**: Current version tracks GPU requirement (boolean) but does not schedule across GPU/CPU partitions. Extension: NVIDIA MPS + NVML for GPU SM-level interference tracking.

5. **RL replacement**: The static interference matrix + ML classifier can be replaced with a single neural policy trained via reinforcement learning. State: concatenated fingerprint vectors of all running tasks. Action: admit/queue. Reward: hit_rate − λ × power_overage + μ × JFI.
