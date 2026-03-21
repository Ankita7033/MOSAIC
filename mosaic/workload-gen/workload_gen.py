#!/usr/bin/env python3
"""
MOSAIC Disaster-Response Workload Generator
============================================
Generates realistic mixed workload arrivals simulating an edge compute node
deployed at a disaster response staging area.

import sys as _sys
if _sys.platform == "win32":
    import re as _re, builtins as _bi
    _op = _bi.print
    def _wp(*a, **k): _op(*[_re.sub("\033[^m]*m","",str(x)) for x in a], **k)
    _bi.print = _wp

Scenario:
    8 drones streaming imagery → inference_critical jobs
    200 field responders → dispatch_api calls
    5 sensor arrays → sensor_fusion jobs
    Analytics pipeline → analytics_batch
    Edge model updates → model_update
    Log sync → log_archive

Patterns:
    poisson    -- steady-state (recovery phase)
    burst      -- sudden mass-casualty event triggers
    sinusoidal -- day/night operational tempo variation
    step       -- resource saturation stress test
    disaster   -- realistic disaster scenario: quiet → sudden spike → sustained

Usage:
    python3 workload-gen/workload_gen.py --pattern disaster --rate 8 --duration 120
    python3 workload-gen/workload_gen.py --pattern burst --dry-run --duration 30
"""

from __future__ import annotations

import sys
import time
import math
import json
import random
import argparse
import itertools
import threading
from pathlib import Path
from typing import Iterator

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "scheduler" / "core_algorithm"))
from workload_taxonomy import WORKLOAD_CLASSES, CLASS_NAMES

# -- Workload arrival weights by phase -----------------------------------------

PHASE_WEIGHTS = {
    "calm":     {"inference_critical":0.10,"dispatch_api":0.40,"sensor_fusion":0.25,
                 "analytics_batch":0.15,"model_update":0.05,"log_archive":0.05},
    "crisis":   {"inference_critical":0.35,"dispatch_api":0.35,"sensor_fusion":0.20,
                 "analytics_batch":0.05,"model_update":0.03,"log_archive":0.02},
    "recovery": {"inference_critical":0.20,"dispatch_api":0.30,"sensor_fusion":0.20,
                 "analytics_batch":0.18,"model_update":0.08,"log_archive":0.04},
}


def sample_class(phase: str = "crisis") -> str:
    weights_d = PHASE_WEIGHTS.get(phase, PHASE_WEIGHTS["crisis"])
    classes   = list(weights_d.keys())
    weights   = list(weights_d.values())
    return random.choices(classes, weights=weights, k=1)[0]


def sample_task(cls: str, task_id: str) -> dict:
    wc    = WORKLOAD_CLASSES[cls]
    dmin, dmax = wc.deadline_range
    smin, smax = wc.service_range
    return {
        "task_id":        task_id,
        "class":          cls,
        "deadline_ms":    random.randint(dmin, dmax),
        "priority":       2 if wc.tier >= 3 else 1,
        "mem_mb":         wc.mem_mb,
        "gpu_required":   wc.gpu_required,
        "cpu_shares":     wc.cpu_shares,
        "tier":           wc.tier,
        "service_time_ms": random.randint(smin, smax),
    }


# -- Arrival Patterns ----------------------------------------------------------

def poisson_arrivals(rate: float) -> Iterator[float]:
    while True:
        yield random.expovariate(rate)


def burst_arrivals(base_rate: float, burst_mult: float = 8.0,
                   burst_every: float = 20.0, burst_dur: float = 5.0) -> Iterator[float]:
    t = 0.0
    next_burst = burst_every + random.uniform(-4, 4)
    in_burst, burst_end = False, 0.0
    while True:
        if not in_burst and t >= next_burst:
            in_burst, burst_end = True, t + burst_dur
        if in_burst and t >= burst_end:
            in_burst   = False
            next_burst = t + burst_every + random.uniform(-4, 4)
        rate = base_rate * (burst_mult if in_burst else 1.0)
        iat  = random.expovariate(max(0.01, rate))
        t   += iat
        yield iat


def sinusoidal_arrivals(mean_rate: float, period: float = 60.0) -> Iterator[float]:
    t = 0.0
    while True:
        rate = mean_rate * (1.0 + 0.70 * math.sin(2 * math.pi * t / period))
        iat  = random.expovariate(max(0.05, rate))
        t   += iat
        yield iat


def step_arrivals(rates: list[float], step_every: float) -> Iterator[float]:
    t = 0.0
    for rate in itertools.cycle(rates):
        step_end = t + step_every
        while t < step_end:
            iat = random.expovariate(max(0.01, rate))
            t  += iat
            yield iat


def disaster_arrivals(base_rate: float) -> Iterator[float]:
    """
    Realistic disaster scenario:
    0–20s:  calm (pre-event, base_rate × 0.3)
    20–40s: ramp-up (event onset)
    40–90s: crisis (base_rate × 5, burst spikes)
    90s+:   recovery (base_rate × 1.5)
    """
    t = 0.0
    while True:
        if t < 20:
            rate = base_rate * 0.3
        elif t < 40:
            # Linear ramp
            frac = (t - 20) / 20
            rate = base_rate * (0.3 + frac * 4.7)
        elif t < 90:
            # Crisis: high rate with random spikes
            spike = random.choices([1.0, 2.5], weights=[0.85, 0.15])[0]
            rate = base_rate * 5.0 * spike
        else:
            rate = base_rate * 1.5
        iat = random.expovariate(max(0.05, rate))
        t  += iat
        yield iat


def get_arrivals(pattern: str, rate: float) -> tuple[Iterator[float], callable]:
    """Returns (arrival_iterator, phase_fn). phase_fn(elapsed_s) → phase name."""
    if pattern == "poisson":
        return poisson_arrivals(rate), lambda t: "calm"
    elif pattern == "burst":
        return burst_arrivals(rate), lambda t: "crisis"
    elif pattern == "sinusoidal":
        return sinusoidal_arrivals(rate), lambda t: "recovery"
    elif pattern == "step":
        return step_arrivals([rate*0.5, rate, rate*2.0, rate*3.0], 25.0), lambda t: "crisis"
    elif pattern == "disaster":
        return disaster_arrivals(rate), lambda t: (
            "calm" if t < 20 else "crisis" if t < 90 else "recovery"
        )
    raise ValueError(f"Unknown pattern: {pattern}")


# -- Metrics Collector ---------------------------------------------------------

class MetricsCollector:
    def __init__(self):
        self._lock      = threading.Lock()
        self._latencies: list[float] = []
        self._hits = 0; self._misses = 0
        self._by_class: dict[str, dict] = {
            c: {"hits":0,"misses":0,"latencies":[]} for c in CLASS_NAMES
        }

    def record(self, actual_ms: float, deadline_ms: float,
               hit: bool, cls: str):
        with self._lock:
            self._latencies.append(actual_ms)
            if hit: self._hits += 1
            else:   self._misses += 1
            if cls in self._by_class:
                self._by_class[cls]["hits" if hit else "misses"] += 1
                self._by_class[cls]["latencies"].append(actual_ms)

    def summary(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            def pct(arr, p):
                if not arr: return 0.0
                s = sorted(arr)
                return s[max(0, int(p/100*len(s))-1)]
            by_class = {}
            for cls, d in self._by_class.items():
                t = d["hits"]+d["misses"]
                by_class[cls] = {
                    "hit_rate": d["hits"]/t if t else 0.0,
                    "total":    t,
                    "p50_ms":   pct(d["latencies"],50),
                    "p95_ms":   pct(d["latencies"],95),
                }
            return {
                "total": total, "hits": self._hits, "misses": self._misses,
                "hit_rate": self._hits/total if total else 0.0,
                "p50_ms":  pct(self._latencies,50),
                "p95_ms":  pct(self._latencies,95),
                "p99_ms":  pct(self._latencies,99),
                "by_class": by_class,
            }


# -- Task Executor -------------------------------------------------------------

class TaskExecutor:
    """Simulates task execution; in production wraps real workload processes."""

    def __init__(self, client, metrics: MetricsCollector):
        self._c = client  # kept for reference only
        self._m = metrics
        self._lock = __import__("threading").Lock()

    def execute(self, task: dict, interference_overhead_ms: float = 0.0):
        svc_ms = task["service_time_ms"]
        jitter = random.gauss(0, svc_ms * 0.08)
        actual = max(1.0, svc_ms + jitter + interference_overhead_ms)
        time.sleep(actual / 1000.0)
        actual_ms = actual
        hit = actual_ms <= task["deadline_ms"]
        # Use a fresh per-thread connection to avoid race conditions
        try:
            from client import MOSAICClient
            with MOSAICClient() as c:
                r = c.complete(task["task_id"], actual_ms)
                hit = r.get("deadline_hit", hit)
        except Exception:
            pass
        self._m.record(actual_ms, task["deadline_ms"], hit, task["class"])


# -- Generator Runner ----------------------------------------------------------

def run(pattern: str, rate: float, duration: float,
        dry_run: bool, quiet: bool, output: str) -> dict:

    arrivals, phase_fn = get_arrivals(pattern, rate)
    metrics   = MetricsCollector()
    counter   = itertools.count(1)
    threads: list[threading.Thread] = []
    client    = None

    if not dry_run:
        # Import client only when connecting
        sys.path.insert(0, str(_ROOT / "scheduler"))
        from client import MOSAICClient
        client = MOSAICClient()
        client.connect()

    executor = TaskExecutor(client, metrics) if client else None

    print(f"\n[gen] MOSAIC Workload Generator")
    print(f"      Pattern={pattern}  rate={rate:.1f}/s  duration={duration:.0f}s  "
          f"{'DRY-RUN' if dry_run else 'LIVE → scheduler'}")
    print()

    start = time.monotonic()
    try:
        for iat in arrivals:
            time.sleep(iat)
            elapsed = time.monotonic() - start
            if elapsed >= duration:
                break
            phase = phase_fn(elapsed)
            cls   = sample_class(phase)
            tid   = f"t{next(counter):06d}"
            task  = sample_task(cls, tid)

            if dry_run:
                if not quiet:
                    print(f"  [dry] {tid} cls={cls:<22} deadline={task['deadline_ms']:>6}ms "
                          f"phase={phase}")
                metrics.record(task["service_time_ms"], task["deadline_ms"], True, cls)
                continue

            try:
                result = client.submit(tid, cls, task["deadline_ms"],
                                       task["priority"], task["cpu_shares"], task["mem_mb"])
                res = result.get("result","?")
            except Exception as e:
                if not quiet: print(f"  [err] {tid}: {e}")
                continue

            if not quiet:
                print(f"  [{res:>8}] {tid} cls={cls:<22} deadline={task['deadline_ms']:>6}ms "
                      f"urgency={result.get('urgency',0):.3f}")

            if res == "admitted" and executor:
                t = threading.Thread(target=executor.execute, args=(task,), daemon=True)
                t.start(); threads.append(t)

    except KeyboardInterrupt:
        print("\n[gen] Interrupted")
    finally:
        for t in threads:
            t.join(timeout=30.0)
        if client:
            client.close()

    s = metrics.summary()
    print(f"\n{'='*58}")
    print(f"  Workload Generator Results -- {pattern.upper()}")
    print(f"{'='*58}")
    print(f"  Total tasks    : {s['total']}")
    print(f"  Deadline hits  : {s['hits']}")
    print(f"  Deadline misses: {s['misses']}")
    print(f"  Hit rate       : {s['hit_rate']:.1%}")
    print(f"  Mean latency   : {s.get('p50_ms',0):.0f} ms (P50)")
    print(f"  P95 latency    : {s['p95_ms']:.0f} ms")
    print(f"  P99 latency    : {s['p99_ms']:.0f} ms")
    print(f"\n  Per-class breakdown:")
    for cls, d in sorted(s["by_class"].items(), key=lambda x: x[1]["total"], reverse=True):
        if d["total"] > 0:
            print(f"    {cls:<25} hit={d['hit_rate']:.1%}  n={d['total']:>4}  p95={d['p95_ms']:.0f}ms")
    print(f"{'='*58}\n")

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output,"w") as f:
            json.dump({"config":{"pattern":pattern,"rate":rate,"duration":duration},
                       "summary":s}, f, indent=2)
        print(f"[gen] Saved to {output}")

    return s


def main():
    p = argparse.ArgumentParser(description="MOSAIC Workload Generator")
    p.add_argument("--pattern",   default="disaster",
                   choices=["poisson","burst","sinusoidal","step","disaster"])
    p.add_argument("--rate",      type=float, default=5.0,
                   help="Mean arrival rate (tasks/sec)")
    p.add_argument("--duration",  type=float, default=120.0,
                   help="Duration in seconds")
    p.add_argument("--scheduler", action="store_true",
                   help="Submit to live MOSAIC daemon")
    p.add_argument("--dry-run",   action="store_true")
    p.add_argument("--quiet",     action="store_true")
    p.add_argument("--output",    default="",
                   help="Save results JSON to path")
    args = p.parse_args()
    dry  = args.dry_run or not args.scheduler
    run(args.pattern, args.rate, args.duration, dry, args.quiet, args.output)


if __name__ == "__main__":
    main()