"""
MOSAIC Workload Taxonomy
========================
Real-world domain: AI-powered disaster response edge infrastructure.

Problem context:
    Emergency response agencies (FEMA, state EOCs, hospital networks) are
    deploying edge servers at disaster sites -- flood zones, wildfire perimeters,
    earthquake staging areas -- that run a MIX of:

    CLASS 1 -- CRITICAL (inference alerts):
        Real-time damage assessment from drone/satellite imagery (YOLO, ViT).
        Survivor detection audio inference. Structural risk scoring.
        These have HARD deadlines: a 3-second alert delay costs lives.

    CLASS 2 -- URGENT (coordination services):
        API calls between field responders and command HQ.
        Resource dispatch microservices (ambulance, fire unit routing).
        Deadline: 200ms or responders lose situational awareness.

    CLASS 3 -- IMPORTANT (analytics):
        Population movement heat-maps. Supply chain status.
        Soft deadline: 2–10 seconds. Useful but not life-critical.

    CLASS 4 -- BACKGROUND (logging/sync):
        Incident report uploads. Sensor data archival.
        Deadline: minutes. Should never starve, but lowest priority.

Why CFS/RR/Priority fails:
    Linux CFS treats all processes with equal "fairness" weight.
    When a drone-image inference burst hits (30 frames/sec from 8 drones),
    it saturates LLC and memory bandwidth, causing the dispatch API to spike
    from 40ms to 800ms -- responders lose coordination exactly when they need it.

    Static priority scheduling assigns fixed weights at submission. A batch
    analytics job assigned priority=5 at 3am will still block a critical
    survivor-detection alert at 9am if the queue is full.

    Round Robin timeslices GPUs mid-kernel, causing 10–50ms overhead per
    context switch on inference workloads -- unacceptable for real-time detection.

MOSAIC's answer:
    Dynamic, interference-aware, deadline-cognizant scheduling with:
    - ML-based workload classification (no manual labelling required)
    - Pairwise interference matrix driven by hardware perf counters
    - Urgency scoring that adapts to remaining deadline, not initial priority
    - Energy feedback to prevent thermal throttling on battery-backed edge nodes
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import ClassVar
import math

# -- Workload Class Registry ----------------------------------------------------

@dataclass(frozen=True)
class WorkloadClass:
    name:            str
    tier:            int        # 1=CRITICAL, 2=URGENT, 3=IMPORTANT, 4=BACKGROUND
    deadline_range:  tuple      # (min_ms, max_ms)
    service_range:   tuple      # (min_ms, max_ms) execution time
    cpu_shares:      int        # cgroup cpu.weight
    mem_mb:          int        # memory limit
    gpu_required:    bool       # needs GPU affinity
    arrival_weight:  float      # fraction of total arrivals
    description:     str
    # Fingerprint centroid (IPC, LLC_miss_rate, mem_bw_GBs, branch_miss_rate)
    fingerprint:     tuple      = field(default=(0,0,0,0))

WORKLOAD_CLASSES: dict[str, WorkloadClass] = {
    "inference_critical": WorkloadClass(
        name="inference_critical", tier=1,
        deadline_range=(500, 3000), service_range=(100, 1200),
        cpu_shares=2048, mem_mb=4096, gpu_required=True,
        arrival_weight=0.20,
        description="Real-time AI inference: damage assessment, survivor detection",
        fingerprint=(1.8, 0.42, 28.0, 0.08),
    ),
    "dispatch_api": WorkloadClass(
        name="dispatch_api", tier=2,
        deadline_range=(50, 200), service_range=(5, 80),
        cpu_shares=512, mem_mb=256, gpu_required=False,
        arrival_weight=0.30,
        description="Responder coordination APIs, resource dispatch microservices",
        fingerprint=(2.9, 0.06, 3.2, 0.04),
    ),
    "sensor_fusion": WorkloadClass(
        name="sensor_fusion", tier=2,
        deadline_range=(100, 500), service_range=(20, 180),
        cpu_shares=768, mem_mb=512, gpu_required=False,
        arrival_weight=0.20,
        description="Multi-sensor fusion: GPS+seismic+thermal overlay",
        fingerprint=(2.1, 0.18, 8.5, 0.06),
    ),
    "analytics_batch": WorkloadClass(
        name="analytics_batch", tier=3,
        deadline_range=(2000, 10000), service_range=(500, 4000),
        cpu_shares=1024, mem_mb=2048, gpu_required=False,
        arrival_weight=0.15,
        description="Population heatmaps, supply chain status, damage reports",
        fingerprint=(1.4, 0.28, 12.0, 0.05),
    ),
    "model_update": WorkloadClass(
        name="model_update", tier=3,
        deadline_range=(5000, 30000), service_range=(2000, 15000),
        cpu_shares=1536, mem_mb=3072, gpu_required=True,
        arrival_weight=0.08,
        description="Edge model fine-tuning on local disaster imagery",
        fingerprint=(1.6, 0.38, 22.0, 0.07),
    ),
    "log_archive": WorkloadClass(
        name="log_archive", tier=4,
        deadline_range=(30000, 300000), service_range=(200, 2000),
        cpu_shares=128, mem_mb=512, gpu_required=False,
        arrival_weight=0.07,
        description="Incident report sync, sensor data archival to cloud",
        fingerprint=(1.1, 0.08, 4.0, 0.03),
    ),
}

CLASS_NAMES   = list(WORKLOAD_CLASSES.keys())
TIER_WEIGHTS  = {1: 4.0, 2: 3.0, 3: 1.5, 4: 0.5}  # urgency priority multipliers

# -- Interference Matrix --------------------------------------------------------
# ipc_degradation[aggressor][victim]: fraction IPC reduction in victim
# Measured / estimated from perf_event colocation experiments.
# Row = aggressor (the class being admitted), Column = victim (already running)
INTERFERENCE_MATRIX: dict[str, dict[str, tuple]] = {
    # Format: (ipc_degradation, lat_overhead_ms)
    "inference_critical": {
        "inference_critical": (0.00,  0.0),
        "dispatch_api":       (0.38, 22.0),   # HIGH: LLC thrash from KV cache
        "sensor_fusion":      (0.25, 14.0),
        "analytics_batch":    (0.12,  6.0),
        "model_update":       (0.08,  3.0),
        "log_archive":        (0.04,  1.0),
    },
    "dispatch_api": {
        "inference_critical": (0.05,  2.0),
        "dispatch_api":       (0.00,  0.0),
        "sensor_fusion":      (0.10,  4.0),
        "analytics_batch":    (0.06,  2.0),
        "model_update":       (0.04,  1.5),
        "log_archive":        (0.02,  0.5),
    },
    "sensor_fusion": {
        "inference_critical": (0.18,  9.0),
        "dispatch_api":       (0.14,  6.0),
        "sensor_fusion":      (0.00,  0.0),
        "analytics_batch":    (0.10,  4.0),
        "model_update":       (0.12,  5.0),
        "log_archive":        (0.03,  1.0),
    },
    "analytics_batch": {
        "inference_critical": (0.20, 11.0),
        "dispatch_api":       (0.16,  7.0),
        "sensor_fusion":      (0.14,  6.0),
        "analytics_batch":    (0.00,  0.0),
        "model_update":       (0.18,  8.0),
        "log_archive":        (0.05,  1.5),
    },
    "model_update": {
        "inference_critical": (0.22, 12.0),
        "dispatch_api":       (0.18,  9.0),
        "sensor_fusion":      (0.16,  7.0),
        "analytics_batch":    (0.14,  6.0),
        "model_update":       (0.00,  0.0),
        "log_archive":        (0.06,  2.0),
    },
    "log_archive": {
        "inference_critical": (0.03,  1.0),
        "dispatch_api":       (0.02,  0.5),
        "sensor_fusion":      (0.02,  0.5),
        "analytics_batch":    (0.03,  1.0),
        "model_update":       (0.02,  0.5),
        "log_archive":        (0.00,  0.0),
    },
}
