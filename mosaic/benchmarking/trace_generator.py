#!/usr/bin/env python3
"""
MOSAIC Deterministic Trace Generator
=====================================
Generates a workload trace ONCE, saves it to JSON, and replays it identically
across all schedulers. This guarantees fair comparison.

Key features:
  - Heavy-tail (lognormal) service durations for realistic queue buildup
  - Workload asymmetry: short critical jobs vs long background jobs
  - Burst cascades with correlated arrivals
  - Deterministic replay: every scheduler sees the EXACT same trace
"""

from __future__ import annotations

import sys
import json
import math
import random
import hashlib
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "scheduler" / "core_algorithm"))
from workload_taxonomy import WORKLOAD_CLASSES, CLASS_NAMES

# -- Phase weights (unchanged from workload_gen) --------------------------------

PHASE_WEIGHTS = {
    "calm":     {"inference_critical":0.10,"dispatch_api":0.40,"sensor_fusion":0.25,
                 "analytics_batch":0.15,"model_update":0.05,"log_archive":0.05},
    "crisis":   {"inference_critical":0.35,"dispatch_api":0.35,"sensor_fusion":0.20,
                 "analytics_batch":0.05,"model_update":0.03,"log_archive":0.02},
    "recovery": {"inference_critical":0.20,"dispatch_api":0.30,"sensor_fusion":0.20,
                 "analytics_batch":0.18,"model_update":0.08,"log_archive":0.04},
}


def _phase_for_pattern(pattern: str, t: float) -> str:
    if pattern == "disaster":
        if t < 20: return "calm"
        if t < 90: return "crisis"
        return "recovery"
    if pattern == "burst":
        return "crisis"
    if pattern == "sinusoidal":
        return "recovery"
    return "calm"


def _sample_class(phase: str, rng: random.Random) -> str:
    weights_d = PHASE_WEIGHTS.get(phase, PHASE_WEIGHTS["crisis"])
    classes = list(weights_d.keys())
    weights = list(weights_d.values())
    return rng.choices(classes, weights=weights, k=1)[0]


def _lognormal_service(wc, rng: random.Random) -> int:
    """
    Heavy-tail lognormal service time, clipped to [smin, smax*2].
    This creates realistic queue buildup and policy divergence.
    """
    smin, smax = wc.service_range
    mu = math.log((smin + smax) / 2.0)
    sigma = 0.6  # controls tail heaviness
    val = rng.lognormvariate(mu, sigma)
    return max(smin, min(int(val), smax * 2))


def _generate_arrivals(pattern: str, rate: float, duration: float,
                       rng: random.Random) -> list[float]:
    """Generate arrival timestamps (not inter-arrival times) deterministically."""
    timestamps = []
    t = 0.0

    if pattern == "burst":
        burst_every = 20.0
        burst_dur = 5.0
        next_burst = burst_every + rng.uniform(-4, 4)
        in_burst = False
        burst_end = 0.0
        while t < duration:
            if not in_burst and t >= next_burst:
                in_burst, burst_end = True, t + burst_dur
            if in_burst and t >= burst_end:
                in_burst = False
                next_burst = t + burst_every + rng.uniform(-4, 4)
            r = rate * (8.0 if in_burst else 1.0)
            iat = rng.expovariate(max(0.01, r))
            t += iat
            if t < duration:
                timestamps.append(t)

    elif pattern == "disaster":
        while t < duration:
            if t < 20:
                r = rate * 0.3
            elif t < 40:
                frac = (t - 20) / 20
                r = rate * (0.3 + frac * 4.7)
            elif t < 90:
                spike = rng.choices([1.0, 2.5], weights=[0.85, 0.15])[0]
                r = rate * 5.0 * spike
            else:
                r = rate * 1.5
            iat = rng.expovariate(max(0.05, r))
            t += iat
            if t < duration:
                timestamps.append(t)

    elif pattern == "sinusoidal":
        period = 60.0
        while t < duration:
            r = rate * (1.0 + 0.70 * math.sin(2 * math.pi * t / period))
            iat = rng.expovariate(max(0.05, r))
            t += iat
            if t < duration:
                timestamps.append(t)

    else:  # poisson
        while t < duration:
            iat = rng.expovariate(max(0.01, rate))
            t += iat
            if t < duration:
                timestamps.append(t)

    return timestamps


def generate_trace(pattern: str = "burst", rate: float = 6.0,
                   duration: float = 60.0, seed: int = 42,
                   enable_bursts: bool = True) -> list[dict]:
    """
    Generate a complete deterministic workload trace.

    Returns list of task dicts, each with:
      - task_id, arrival_time, class, deadline_ms, service_ms,
        priority, tier, mem_mb, gpu_required, cpu_shares, phase
    """
    rng = random.Random(seed)

    # Step 1: Generate arrival timestamps
    timestamps = _generate_arrivals(pattern, rate, duration, rng)

    # Step 2: Inject burst cascades (correlated arrivals)
    if enable_bursts and pattern in ("burst", "disaster"):
        extra = []
        for ts in list(timestamps):
            phase = _phase_for_pattern(pattern, ts)
            if phase == "crisis" and rng.random() < 0.12:
                # Correlated burst: 3-6 tasks within 200ms
                n_burst = rng.randint(3, 6)
                for j in range(n_burst):
                    bt = ts + rng.uniform(0.01, 0.2)
                    if bt < duration:
                        extra.append(bt)
        timestamps.extend(extra)
        timestamps.sort()

    # Step 3: Generate task details for each arrival
    trace = []
    for i, arr_time in enumerate(timestamps):
        phase = _phase_for_pattern(pattern, arr_time)
        cls = _sample_class(phase, rng)
        wc = WORKLOAD_CLASSES[cls]

        dmin, dmax = wc.deadline_range
        service_ms = _lognormal_service(wc, rng)

        trace.append({
            "task_id":      f"t_{i+1:06d}",
            "arrival_time": round(arr_time, 6),
            "class":        cls,
            "deadline_ms":  rng.randint(dmin, dmax),
            "service_ms":   service_ms,
            "priority":     1 if wc.tier <= 2 else 2,
            "tier":         wc.tier,
            "mem_mb":       wc.mem_mb,
            "gpu_required": wc.gpu_required,
            "cpu_shares":   wc.cpu_shares,
            "phase":        phase,
        })

    return trace


def save_trace(trace: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2)


def load_trace(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def trace_summary(trace: list[dict]) -> dict:
    """Return summary statistics about a trace."""
    by_class = {}
    for t in trace:
        c = t["class"]
        by_class[c] = by_class.get(c, 0) + 1
    return {
        "total_tasks":   len(trace),
        "duration_s":    round(trace[-1]["arrival_time"], 2) if trace else 0,
        "by_class":      by_class,
        "unique_classes": len(by_class),
    }


# -- Adversarial trace modes ---------------------------------------------------

def generate_adversarial_trace(mode: str, seed: int = 42,
                               duration: float = 60.0, rate: float = 6.0) -> list[dict]:
    """
    Generate adversarial workload traces for stress testing.

    Modes:
      cache_thrash   - Multiple inference jobs with large tensor ops
      retry_storm    - API retries causing correlated bursts
      misclassify    - PMU noise injected into service times
      overload       - Arrival rate > saturation
    """
    rng = random.Random(seed)

    if mode == "cache_thrash_extreme":
        # Simultaneous tensor inference + Redis cache pressure + burst arrivals
        trace = generate_trace("burst", rate * 2.5, duration, seed, True)
        for t in trace:
            t["class"] = rng.choices(
                ["inference_critical", "log_archive", "analytics_batch"],
                weights=[0.5, 0.4, 0.1])[0]
            wc = WORKLOAD_CLASSES[t["class"]]
            t["service_ms"] = _lognormal_service(wc, rng)
            t["tier"] = wc.tier
        return trace

    elif mode == "cache_thrash":
        # Heavy inference-only workload
        trace = generate_trace("poisson", rate * 3, duration, seed, False)
        for t in trace:
            t["class"] = rng.choices(
                ["inference_critical", "model_update", "analytics_batch"],
                weights=[0.5, 0.3, 0.2])[0]
            wc = WORKLOAD_CLASSES[t["class"]]
            t["service_ms"] = _lognormal_service(wc, rng)
            t["tier"] = wc.tier
        return trace

    elif mode == "retry_storm":
        base = generate_trace("burst", rate, duration, seed, False)
        extra = []
        for t in base:
            if t["class"] == "dispatch_api" and rng.random() < 0.3:
                for retry in range(rng.randint(2, 5)):
                    nt = dict(t)
                    nt["task_id"] = f"{t['task_id']}_r{retry}"
                    nt["arrival_time"] = t["arrival_time"] + 0.1 * (retry + 1)
                    if nt["arrival_time"] < duration:
                        extra.append(nt)
        base.extend(extra)
        base.sort(key=lambda x: x["arrival_time"])
        # Re-index
        for i, t in enumerate(base):
            t["task_id"] = f"t_{i+1:06d}"
        return base

    elif mode == "misclassify":
        trace = generate_trace("burst", rate, duration, seed)
        for t in trace:
            noise = rng.gauss(0, 0.1)
            t["service_ms"] = max(1, int(t["service_ms"] * (1 + noise)))
        return trace

    elif mode == "overload":
        return generate_trace("poisson", rate * 6, duration, seed, False)

    else:
        raise ValueError(f"Unknown adversarial mode: {mode}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="MOSAIC Trace Generator")
    p.add_argument("--pattern", default="burst",
                   choices=["poisson", "burst", "sinusoidal", "disaster"])
    p.add_argument("--rate", type=float, default=6.0)
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="")
    args = p.parse_args()

    trace = generate_trace(args.pattern, args.rate, args.duration, args.seed)
    summary = trace_summary(trace)

    print(f"\nTrace generated: {summary['total_tasks']} tasks over {summary['duration_s']}s")
    print(f"Classes: {summary['by_class']}")

    if args.output:
        save_trace(trace, Path(args.output))
        print(f"Saved to {args.output}")
    else:
        out = _ROOT / "results" / f"trace_seed_{args.seed}.json"
        save_trace(trace, out)
        print(f"Saved to {out}")
