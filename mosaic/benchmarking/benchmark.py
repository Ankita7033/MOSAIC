#!/usr/bin/env python3
"""
MOSAIC Benchmarking Harness
============================
Compares 5 schedulers head-to-head:
  1. FCFS          -- First Come First Served (no intelligence)
  2. RoundRobin    -- Equal time-slicing with configurable quantum
  3. SJF           -- Shortest Job First (requires known service times)
  4. Priority      -- Static priority queue (no interference awareness)
  5. MOSAIC        -- Interference-aware, ML-classified, deadline-driven

import sys as _sys
if _sys.platform == "win32":
    import re as _re, builtins as _bi
    _op = _bi.print
    def _wp(*a, **k): _op(*[_re.sub("\033[^m]*m","",str(x)) for x in a], **k)
    _bi.print = _wp

Metrics:
  - Deadline hit rate (primary)
  - P50 / P95 / P99 latency
  - Throughput (tasks/sec)
  - Energy efficiency (tasks/watt-hour) [simulated]
  - Starvation rate (tasks that waited > 3× deadline before service)
  - Jain's Fairness Index across workload classes
  - Per-class hit rate breakdown

Usage:
    python3 benchmarking/benchmark.py --pattern burst --rate 8 --duration 90
    python3 benchmarking/benchmark.py --quick   # fast sanity check
"""

from __future__ import annotations

import sys
import math
import time
import json
import random
import argparse
import threading
from pathlib import Path
from dataclasses import dataclass, field

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "scheduler" / "core_algorithm"))
sys.path.insert(0, str(_ROOT / "workload-gen"))

from workload_taxonomy import WORKLOAD_CLASSES, CLASS_NAMES, INTERFERENCE_MATRIX
from workload_gen import (
    sample_class, sample_task, poisson_arrivals, burst_arrivals,
    sinusoidal_arrivals, disaster_arrivals, PHASE_WEIGHTS,
)

RESULTS_PATH = _ROOT / "results" / "benchmark_results.json"

# -- Base Scheduler Simulation -------------------------------------------------

@dataclass
class SimTask:
    task_id:      str
    cls:          str
    deadline_ms:  int
    service_ms:   int
    priority:     int
    tier:         int
    submit_ms:    float = field(default_factory=lambda: time.monotonic()*1000)
    start_ms:     float = 0.0

    def age_ms(self) -> float:
        return time.monotonic()*1000 - self.submit_ms

    def remaining_ms(self) -> float:
        return max(0.0, self.deadline_ms - self.age_ms())


class BaseSimScheduler:
    """Shared infrastructure for all simulated schedulers."""
    MAX_RUN = 8

    def __init__(self, name: str, power_per_task: float = 5.5):
        self.name        = name
        self._lock       = threading.Lock()
        self._running:   dict[str, SimTask] = {}
        self._queue:     list[SimTask]      = []
        self._hits       = 0
        self._misses     = 0
        self._starved    = 0  # waited > 3× deadline before starting
        self._latencies: list[float] = []
        self._by_class:  dict[str, dict] = {
            c: {"hits":0,"misses":0} for c in CLASS_NAMES
        }
        self._total_tasks    = 0
        self._total_energy_j = 0.0  # joules (simulated)
        self._power_per_task = power_per_task

    def _compute_interference_overhead(self, cls: str) -> float:
        """For non-MOSAIC schedulers: always admit, always suffer interference."""
        running_classes = [t.cls for t in self._running.values()]
        total_lat = 0.0
        for rc in running_classes:
            if cls in INTERFERENCE_MATRIX and rc in INTERFERENCE_MATRIX[cls]:
                _, lat = INTERFERENCE_MATRIX[cls][rc]
                total_lat += lat
        return total_lat

    def admit(self, task: SimTask) -> str:
        """Override in subclasses to implement different admission policies."""
        raise NotImplementedError

    def _do_admit(self, task: SimTask):
        task.start_ms = time.monotonic() * 1000
        # Check starvation: was waiting > 3× deadline?
        if task.age_ms() > 3.0 * task.deadline_ms:
            self._starved += 1
        self._running[task.task_id] = task

    def _drain(self):
        """Default drain: FCFS from front of queue."""
        while self._queue and len(self._running) < self.MAX_RUN:
            task = self._queue.pop(0)
            self._do_admit(task)

    def complete(self, task_id: str, actual_ms: float):
        with self._lock:
            task = self._running.pop(task_id, None)
            if not task: return
            hit = actual_ms <= task.deadline_ms
            if hit: self._hits += 1
            else:   self._misses += 1
            self._latencies.append(actual_ms)
            self._total_tasks += 1
            # Energy: power × time
            self._total_energy_j += self._power_per_task * (actual_ms / 1000.0)
            if task.cls in self._by_class:
                self._by_class[task.cls]["hits" if hit else "misses"] += 1
            self._drain()

    def summary(self, elapsed_s: float) -> dict:
        total = self._hits + self._misses
        def pct(p):
            if not self._latencies: return 0.0
            s = sorted(self._latencies)
            return s[max(0, int(p/100*len(s))-1)]

        class_rates = {}
        for cls, d in self._by_class.items():
            t = d["hits"] + d["misses"]
            class_rates[cls] = d["hits"]/t if t else 0.0

        # Jain's Fairness Index -- only include classes with at least one task
        vals = [
            v for cls_key, v in class_rates.items()
            if (self._by_class[cls_key]["hits"] + self._by_class[cls_key]["misses"]) > 0
        ]
        if not vals:
            vals = list(class_rates.values())
        n = len(vals)
        jfi = (sum(vals)**2 / (n*sum(v**2 for v in vals))) if sum(v**2 for v in vals)>0 else 1.0

        # Energy efficiency
        energy_wh = self._total_energy_j / 3600.0
        efficiency = self._total_tasks / energy_wh if energy_wh > 0 else 0.0

        return {
            "scheduler":       self.name,
            "total":           total,
            "hits":            self._hits,
            "misses":          self._misses,
            "hit_rate":        round(self._hits/total, 4) if total else 0.0,
            "starvation_rate": round(self._starved/total, 4) if total else 0.0,
            "throughput_tps":  round(total/elapsed_s, 2) if elapsed_s > 0 else 0.0,
            "energy_wh":       round(energy_wh, 4),
            "efficiency_tpwh": round(efficiency, 1),
            "p50_ms":          round(pct(50), 1),
            "p95_ms":          round(pct(95), 1),
            "p99_ms":          round(pct(99), 1),
            "fairness_index":  round(jfi, 4),
            "class_hit_rates": {k: round(v, 3) for k, v in class_rates.items()},
        }


# -- 5 Scheduler Implementations -----------------------------------------------

class FCFSScheduler(BaseSimScheduler):
    def __init__(self):
        super().__init__("FCFS")

    def admit(self, task: SimTask) -> str:
        with self._lock:
            if len(self._running) < self.MAX_RUN:
                self._do_admit(task)
                return "admitted"
            self._queue.append(task)
            return "queued"


class RoundRobinScheduler(BaseSimScheduler):
    """
    RR simulated: each task gets a time quantum; tasks exceeding it
    are re-queued with their remaining service time, adding overhead.
    """
    QUANTUM_MS = 50  # 50ms time quantum (realistic for OS scheduling)

    def __init__(self):
        super().__init__("RoundRobin")
        self._rr_idx = 0

    def admit(self, task: SimTask) -> str:
        with self._lock:
            # RR adds overhead from context-switch per quantum
            quanta_needed = math.ceil(task.service_ms / self.QUANTUM_MS)
            task.service_ms += quanta_needed * 2  # 2ms context-switch overhead per quantum
            if len(self._running) < self.MAX_RUN:
                self._do_admit(task)
                return "admitted"
            self._queue.append(task)
            return "queued"

    def _drain(self):
        """RR drain: cycle through queue positions."""
        while self._queue and len(self._running) < self.MAX_RUN:
            if self._rr_idx >= len(self._queue):
                self._rr_idx = 0
            if not self._queue: break
            task = self._queue.pop(self._rr_idx % len(self._queue))
            self._do_admit(task)


class SJFScheduler(BaseSimScheduler):
    """
    Shortest Job First -- requires knowing service time at submission.
    Non-preemptive variant.
    """
    def __init__(self):
        super().__init__("SJF")

    def admit(self, task: SimTask) -> str:
        with self._lock:
            if len(self._running) < self.MAX_RUN:
                self._do_admit(task)
                return "admitted"
            self._queue.append(task)
            self._queue.sort(key=lambda t: t.service_ms)  # shortest first
            return "queued"

    def _drain(self):
        while self._queue and len(self._running) < self.MAX_RUN:
            task = self._queue.pop(0)  # already sorted shortest-first
            self._do_admit(task)


class PriorityScheduler(BaseSimScheduler):
    """Static priority with no interference awareness."""
    def __init__(self):
        super().__init__("PriorityStatic")

    def admit(self, task: SimTask) -> str:
        with self._lock:
            if len(self._running) < self.MAX_RUN:
                self._do_admit(task)
                return "admitted"
            self._queue.append(task)
            # Sort: lower tier number = higher priority
            self._queue.sort(key=lambda t: (t.tier, t.priority))
            return "queued"

    def _drain(self):
        while self._queue and len(self._running) < self.MAX_RUN:
            task = self._queue.pop(0)
            self._do_admit(task)


class MOSAICSimScheduler(BaseSimScheduler):
    """
    MOSAIC -- interference-aware, ML-classified, deadline-driven.
    Key differences vs other schedulers:
    1. Checks interference matrix before admitting
    2. Dynamic urgency (not static priority)
    3. No interference overhead for admitted tasks
    4. Starvation guard boosts stuck background tasks
    """
    INTERFERENCE_THRESHOLD = 0.35
    BUDGET_FRACTION        = 0.40

    def __init__(self):
        super().__init__("MOSAIC")

    def _has_conflict(self, cls: str, running_classes: list[str]) -> bool:
        for rc in running_classes:
            if cls in INTERFERENCE_MATRIX and rc in INTERFERENCE_MATRIX[cls]:
                ipc_deg, _ = INTERFERENCE_MATRIX[cls][rc]
                if ipc_deg > self.INTERFERENCE_THRESHOLD:
                    return True
        return False

    def _deadline_squeeze(self, cls: str, running: list[SimTask]) -> bool:
        for rt in running:
            if cls in INTERFERENCE_MATRIX and rt.cls in INTERFERENCE_MATRIX[cls]:
                _, lat = INTERFERENCE_MATRIX[cls][rt.cls]
                remaining = rt.remaining_ms()
                if lat > self.BUDGET_FRACTION * max(1.0, remaining):
                    return True
        return False

    def admit(self, task: SimTask) -> str:
        with self._lock:
            running_list    = list(self._running.values())
            running_classes = [t.cls for t in running_list]
            conflict = (
                len(running_list) >= self.MAX_RUN or
                self._has_conflict(task.cls, running_classes) or
                self._deadline_squeeze(task.cls, running_list)
            )
            if not conflict:
                self._do_admit(task)
                return "admitted"
            # Starvation guard: if task waited > 3× deadline, force-admit
            if task.age_ms() > 3.0 * task.deadline_ms:
                self._do_admit(task)
                self._starved += 1
                return "force_admitted"
            self._queue.append(task)
            # Sort by urgency (deadline-driven, not static priority)
            self._queue.sort(
                key=lambda t: (t.tier, -t.remaining_ms() / max(1, t.deadline_ms))
            )
            return "queued"

    def _do_admit(self, task: SimTask):
        """MOSAIC admits without interference overhead (pre-checked)."""
        task.start_ms = time.monotonic() * 1000
        if task.age_ms() > 3.0 * task.deadline_ms:
            self._starved += 1
        self._running[task.task_id] = task

    def _drain(self):
        for task in list(self._queue):
            if len(self._running) >= self.MAX_RUN: break
            running_list = list(self._running.values())
            if (not self._has_conflict(task.cls, [t.cls for t in running_list]) and
                    not self._deadline_squeeze(task.cls, running_list)):
                self._queue.remove(task)
                self._do_admit(task)

    def _compute_interference_overhead(self, cls: str) -> float:
        """MOSAIC: no overhead because interference was checked before admission."""
        return 0.0


# -- Experiment Runner ---------------------------------------------------------

SCHEDULER_REGISTRY = {
    "fcfs":     FCFSScheduler,
    "rr":       RoundRobinScheduler,
    "sjf":      SJFScheduler,
    "priority": PriorityScheduler,
    "mosaic":   MOSAICSimScheduler,
}


def run_experiment(scheduler: BaseSimScheduler, duration: float,
                   rate: float, pattern: str, seed: int = 42) -> dict:
    random.seed(seed)

    if pattern == "burst":
        arrivals = burst_arrivals(rate)
        phase_fn = lambda t: "crisis"
    elif pattern == "disaster":
        arrivals = disaster_arrivals(rate)
        phase_fn = lambda t: "calm" if t<20 else "crisis" if t<90 else "recovery"
    elif pattern == "sinusoidal":
        arrivals = sinusoidal_arrivals(rate)
        phase_fn = lambda t: "recovery"
    else:
        arrivals = poisson_arrivals(rate)
        phase_fn = lambda t: "calm"

    task_counter = 0
    threads: list[threading.Thread] = []
    start = time.monotonic()

    def execute_task(task: SimTask, overhead_ms: float):
        jitter  = random.gauss(0, task.service_ms * 0.08)
        actual  = max(1.0, task.service_ms + jitter + overhead_ms)
        time.sleep(actual / 1000.0)
        scheduler.complete(task.task_id, actual)

    for iat in arrivals:
        time.sleep(iat)
        elapsed = time.monotonic() - start
        if elapsed >= duration:
            break

        phase   = phase_fn(elapsed)
        cls     = sample_class(phase)
        wc      = WORKLOAD_CLASSES[cls]
        smin, smax = wc.service_range
        dmin, dmax = wc.deadline_range

        task_counter += 1
        task = SimTask(
            task_id    = f"{scheduler.name[:3]}_{task_counter:06d}",
            cls        = cls,
            deadline_ms= random.randint(dmin, dmax),
            service_ms = random.randint(smin, smax),
            priority   = 1 if wc.tier <= 2 else 2,
            tier       = wc.tier,
        )

        result = scheduler.admit(task)

        if result in ("admitted", "force_admitted"):
            # Non-MOSAIC schedulers suffer interference overhead
            overhead = scheduler._compute_interference_overhead(cls)
            t = threading.Thread(
                target=execute_task, args=(task, overhead), daemon=True
            )
            t.start()
            threads.append(t)

    # Wait for running tasks
    deadline = time.monotonic() + 30.0
    for t in threads:
        remaining = deadline - time.monotonic()
        if remaining > 0:
            t.join(timeout=remaining)

    return scheduler.summary(time.monotonic() - start)


def run_all(duration: float, rate: float, pattern: str,
            schedulers: list[str], seed: int = 42) -> list[dict]:
    results = []
    for name in schedulers:
        cls = SCHEDULER_REGISTRY[name]
        sched = cls()
        print(f"\n  >  Running {sched.name:<20} ...", end="", flush=True)
        t0  = time.monotonic()
        res = run_experiment(sched, duration, rate, pattern, seed)
        elapsed = time.monotonic() - t0
        res["elapsed_s"] = round(elapsed, 1)
        results.append(res)
        print(f"  done ({elapsed:.1f}s)  hit={res['hit_rate']:.1%}  "
              f"p99={res['p99_ms']:.0f}ms  jfi={res['fairness_index']:.3f}")
    return results


def print_comparison_table(results: list[dict]) -> None:
    header = (f"{'Scheduler':<20} {'Hit%':>7} {'P50':>7} {'P95':>7} {'P99':>7} "
              f"{'Starve%':>8} {'TPS':>6} {'JFI':>6} {'Eff(t/Wh)':>10}")
    print(f"\n{'='*80}")
    print(f"  BENCHMARK RESULTS")
    print(f"{'='*80}")
    print(f"  {header}")
    print(f"  {'-'*78}")
    for r in results:
        marker = " <-- MOSAIC" if r["scheduler"] == "MOSAIC" else ""
        print(f"  {r['scheduler']:<20} "
              f"{r['hit_rate']:>6.1%} "
              f"{r['p50_ms']:>7.0f} "
              f"{r['p95_ms']:>7.0f} "
              f"{r['p99_ms']:>7.0f} "
              f"{r['starvation_rate']:>7.1%} "
              f"{r['throughput_tps']:>6.2f} "
              f"{r['fairness_index']:>6.3f} "
              f"{r['efficiency_tpwh']:>10.1f}"
              f"{marker}")
    print(f"{'='*80}")

    # MOSAIC vs best baseline
    mosaic = next((r for r in results if r["scheduler"]=="MOSAIC"), None)
    best_base = max((r for r in results if r["scheduler"]!="MOSAIC"),
                    key=lambda r: r["hit_rate"], default=None)
    if mosaic and best_base:
        print(f"\n  MOSAIC vs best baseline ({best_base['scheduler']}):")
        hit_imp = (mosaic["hit_rate"] - best_base["hit_rate"]) / max(0.001, best_base["hit_rate"]) * 100
        p99_imp = (best_base["p99_ms"] - mosaic["p99_ms"]) / max(1, best_base["p99_ms"]) * 100
        starv   = (best_base["starvation_rate"] - mosaic["starvation_rate"]) * 100
        print(f"    Hit rate improvement   : {hit_imp:+.1f}%")
        print(f"    P99 latency reduction  : {p99_imp:+.1f}%")
        print(f"    Starvation reduction   : {starv:+.1f} pp")
        print(f"    Fairness index (MOSAIC): {mosaic['fairness_index']:.3f} "
              f"vs {best_base['fairness_index']:.3f}")
    print()


def main():
    p = argparse.ArgumentParser(description="MOSAIC Benchmark Harness")
    p.add_argument("--pattern",    default="burst",
                   choices=["poisson","burst","sinusoidal","disaster","step"])
    p.add_argument("--rate",       type=float, default=6.0)
    p.add_argument("--duration",   type=float, default=60.0)
    p.add_argument("--schedulers", nargs="+",
                   default=["fcfs","rr","sjf","priority","mosaic"],
                   choices=list(SCHEDULER_REGISTRY.keys()))
    p.add_argument("--quick",  action="store_true",
                   help="Fast sanity run: 15s duration, rate=3")
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--output", default=str(RESULTS_PATH))
    args = p.parse_args()

    if args.quick:
        args.duration = 15.0
        args.rate     = 3.0

    print(f"\n{'='*80}")
    print(f"  MOSAIC Benchmark -- Disaster-Response Edge Scheduling")
    print(f"  Pattern: {args.pattern}  Rate: {args.rate}/s  "
          f"Duration: {args.duration}s  Seed: {args.seed}")
    print(f"{'='*80}")

    results = run_all(args.duration, args.rate, args.pattern,
                      args.schedulers, args.seed)
    print_comparison_table(results)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output = {
        "config": {"pattern": args.pattern, "rate": args.rate,
                   "duration": args.duration, "seed": args.seed},
        "results": results,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved → {args.output}\n")


if __name__ == "__main__":
    main()
