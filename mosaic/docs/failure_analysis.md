# MOSAIC Failure Analysis & Limitations

While MOSAIC demonstrates significant improvements in tail latency and system fairness during disaster-response edge colocation, any rigorous systems evaluation must explicitly document boundary conditions and failure states. This document analyzes the conditions under which MOSAIC's PMU-guided admission controller and workload classifier degrade.

## 1. PMU Misclassification (The "Ghost Signature" Problem)
MOSAIC's workload classifier relies on a hardware counter fingerprint (`ipc`, `llc_miss_rate`, `mem_bw_gbs`, `branch_miss_rate`).
- **Failure Condition:** If a new, unprofiled background workload exhibits a PMU signature identical to an `inference_critical` task (e.g., a highly optimized tensor batch processing script that mimics real-time inference), the classifier will assign it a Tier 1 priority.
- **Consequence:** MOSAIC will aggressively admit the imposter tasks, rapidly exhausting the `MAX_RUN` slot limit and starvation budgets. Genuine inference tasks will experience queue wait times exceeding their hard deadlines, leading to a localized tail latency explosion (P99 spikes > 5000ms).
- **Mitigation:** Operator-provided metadata hints (`tier_hint=1`) take precedence over PMU classification.

## 2. Extreme Overload Collapse ($\lambda \gg \text{Saturation}$)
MOSAIC is designed to shed load gracefully by rejecting tasks when `queue_wait_est + service_time + interference > deadline`. 
- **Failure Condition:** When the arrival rate ($\lambda$) reaches 10x the saturation limit (e.g., during a catastrophic sensor network retry-storm).
- **Consequence:** The predictive admission control logic is invoked so frequently that the scheduler thread itself becomes a bottleneck. Even though tasks are rejected in O(1) time, the volume of rejections consumes CPU cycles, delaying the dispatch of already-admitted tasks. 
- **Emergent Unfairness:** In this state, MOSAIC aggressively sacrifices *all* background classes (100% functional starvation) to protect inference tasks. While intentional, the rejection rate approaches 95%, essentially reducing the system to a single-tenant inference engine.

## 3. The "Deadline Squeeze" False Positive
MOSAIC utilizes a deadline squeeze check to avoid colocating tasks that would push a running critical task past its deadline.
- **Failure Condition:** If an `inference_critical` task is scheduled with a very tight deadline margin (e.g., remaining deadline = 5ms), the budget fraction check will flag almost *any* incoming task as a conflict, regardless of how light the incoming task's memory footprint is.
- **Consequence:** MOSAIC artificially starves the system, running only the single critical task and leaving the remaining `MAX_RUN - 1` slots completely empty. Overall throughput drops significantly until the critical task completes.

## 4. GPU Saturation
MOSAIC currently uses PMU counters (CPU and Memory bound) as the primary indicator of interference.
- **Failure Condition:** If multiple `model_update` and `inference_critical` tasks are admitted, they may not show severe LLC interference, but they completely saturate the GPU memory bandwidth or compute units.
- **Consequence:** Because MOSAIC's interference matrix primarily models CPU LLC misses and IPC degradation, it fails to predict GPU contention. The tasks are colocated, but their actual execution time stretches 3x-4x beyond the estimated `service_ms`, causing cascading deadline misses.
- **Future Work:** Integrating NVIDIA NVML metrics alongside Intel PMU counters for multi-dimensional interference tracking.
