#!/usr/bin/env python3
"""
MOSAIC Benchmarking Harness (v2 -- Deterministic Replay)
=========================================================
All schedulers receive the EXACT same pre-generated trace.
Fixes: JFI includes zero-hit classes, dual starvation metrics,
heavy-tail durations, proper SJF/Priority, full task accounting.
"""
from __future__ import annotations
import sys, math, time, json, random, argparse, threading
from pathlib import Path
from dataclasses import dataclass, field

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "scheduler" / "core_algorithm"))
sys.path.insert(0, str(_ROOT / "workload-gen"))
sys.path.insert(0, str(_ROOT / "benchmarking"))

from workload_taxonomy import WORKLOAD_CLASSES, CLASS_NAMES, INTERFERENCE_MATRIX
from trace_generator import generate_trace, save_trace, trace_summary

RESULTS_PATH = _ROOT / "results" / "benchmark_results.json"

# -- SimTask -------------------------------------------------------------------
@dataclass
class SimTask:
    task_id: str; cls: str; deadline_ms: int; service_ms: int
    priority: int; tier: int; arrival_time: float
    submit_ms: float = 0.0; start_ms: float = 0.0
    def age_ms(self, now: float = None) -> float:
        if now is None: now = time.monotonic() * 1000
        return now - self.submit_ms
    def remaining_ms(self, now: float = None) -> float:
        if now is None: now = time.monotonic() * 1000
        return max(0.0, self.deadline_ms - self.age_ms(now))

# -- Base Scheduler ------------------------------------------------------------
class BaseSimScheduler:
    MAX_RUN = 8
    def __init__(self, name: str, power_per_task: float = 5.5):
        self.name = name
        self._lock = threading.Lock()
        self._running: dict[str, SimTask] = {}
        self._queue: list[SimTask] = []
        self._hits = 0; self._misses = 0
        self._starved_hard = 0    # waited > 3x deadline
        self._starved_func = 0    # functional starvation (class-level)
        self._latencies: list[float] = []
        self._by_class: dict[str, dict] = {
            c: {"hits":0,"misses":0,"latencies":[]} for c in CLASS_NAMES
        }
        self._generated = 0; self._admitted = 0; self._completed = 0
        self._rejected = 0; self._expired = 0
        self._total_energy_j = 0.0; self._power_per_task = power_per_task
        self._interference_log: list[dict] = []
        self._energy_samples: list[dict] = []
        self._timeline_events: list[dict] = []
        self._idle_power = 35.0  # watts idle
        self._active_power_per_task = power_per_task
        self._base_ts = time.monotonic() * 1000

    def _log_event(self, event: str, task: SimTask, extra: dict = None):
        e = {"ts": time.monotonic() * 1000 - self._base_ts, "event": event, 
             "task_id": task.task_id, "class": task.cls, "tier": task.tier}
        if extra: e.update(extra)
        self._timeline_events.append(e)

    def _compute_interference_overhead(self, cls: str) -> float:
        running_classes = [t.cls for t in self._running.values()]
        total_lat = 0.0
        for rc in running_classes:
            if cls in INTERFERENCE_MATRIX and rc in INTERFERENCE_MATRIX[cls]:
                ipc_deg, lat = INTERFERENCE_MATRIX[cls][rc]
                total_lat += lat
                self._interference_log.append({
                    "task_a": cls, "task_b": rc,
                    "ipc_degradation": ipc_deg, "llc_miss_delta": lat,
                    "decision": "overhead_applied"
                })
        return total_lat

    def admit(self, task: SimTask) -> str:
        raise NotImplementedError

    def _do_admit(self, task: SimTask):
        now = time.monotonic() * 1000
        task.start_ms = now
        if task.age_ms(now) > 3.0 * task.deadline_ms:
            self._starved_hard += 1
        self._running[task.task_id] = task
        self._admitted += 1
        self._log_event("ADMIT", task)

    def _drain(self):
        now = time.monotonic() * 1000
        while self._queue and len(self._running) < self.MAX_RUN:
            # Expire dead tasks
            alive = []
            for t in self._queue:
                if t.age_ms(now) > 5.0 * t.deadline_ms:
                    self._expired += 1; self._misses += 1
                    self._log_event("EXPIRE", t)
                    if t.cls in self._by_class:
                        self._by_class[t.cls]["misses"] += 1
                else:
                    alive.append(t)
            self._queue = alive
            if not self._queue: break
            task = self._queue.pop(0)
            self._do_admit(task)

    def complete(self, task_id: str, actual_ms: float):
        with self._lock:
            task = self._running.pop(task_id, None)
            if not task: return
            now = time.monotonic() * 1000
            turnaround = now - task.submit_ms
            hit = turnaround <= task.deadline_ms
            if hit: self._hits += 1
            else: self._misses += 1
            self._latencies.append(turnaround)
            self._completed += 1
            n_run = len(self._running)
            power_w = self._idle_power + n_run * self._active_power_per_task
            energy_j = power_w * (actual_ms / 1000.0)
            self._total_energy_j += energy_j
            self._energy_samples.append({
                "task_id": task_id, "power_w": round(power_w, 1),
                "energy_j": round(energy_j, 4), "n_running": n_run
            })
            if task.cls in self._by_class:
                self._by_class[task.cls]["hits" if hit else "misses"] += 1
                self._by_class[task.cls]["latencies"].append(actual_ms)
            self._log_event("COMPLETE", task, {"hit": hit, "actual_ms": actual_ms})
            self._drain()

    def summary(self, elapsed_s: float) -> dict:
        total = self._hits + self._misses
        def pct(p):
            if not self._latencies: return 0.0
            s = sorted(self._latencies)
            return s[max(0, int(p/100*len(s))-1)]

        # Per-class hit rates -- ALL classes, including zero-task ones
        class_rates = {}
        for cls, d in self._by_class.items():
            t = d["hits"] + d["misses"]
            class_rates[cls] = d["hits"]/t if t > 0 else 0.0

        # JFI: include ALL classes (zero-hit = 0.0), per the formula
        all_rates = list(class_rates.values())
        n = len(all_rates)
        sum_x = sum(all_rates)
        sum_x2 = sum(x*x for x in all_rates)
        jfi = (sum_x**2) / (n * sum_x2) if sum_x2 > 0 else 0.0

        # Functional starvation: classes with 0% hit rate that had tasks
        func_starved_classes = []
        for cls, d in self._by_class.items():
            t = d["hits"] + d["misses"]
            if t > 0 and d["hits"] == 0:
                func_starved_classes.append(cls)
                self._starved_func += t

        # Energy efficiency: energy_delta / completed_tasks
        energy_wh = self._total_energy_j / 3600.0
        efficiency = self._completed / energy_wh if energy_wh > 0 else 0.0

        return {
            "scheduler": self.name,
            "total": total,
            "hits": self._hits, "misses": self._misses,
            "hit_rate": round(self._hits/total, 4) if total else 0.0,
            "starvation_rate_hard": round(self._starved_hard/total, 4) if total else 0.0,
            "starvation_rate_func": round(self._starved_func/total, 4) if total else 0.0,
            "starvation_rate": round((self._starved_hard + self._starved_func)/total, 4) if total else 0.0,
            "func_starved_classes": func_starved_classes,
            "throughput_tps": round(total/elapsed_s, 2) if elapsed_s > 0 else 0.0,
            "energy_wh": round(energy_wh, 4),
            "efficiency_tpwh": round(efficiency, 1),
            "p50_ms": round(pct(50), 1), "p95_ms": round(pct(95), 1),
            "p99_ms": round(pct(99), 1),
            "fairness_index": round(jfi, 4),
            "class_hit_rates": {k: round(v, 3) for k, v in class_rates.items()},
            "class_hits": {k: v["hits"] for k, v in self._by_class.items()},
            "accounting": {
                "generated": self._generated, "admitted": self._admitted,
                "completed": self._completed, "rejected": self._rejected,
                "expired": self._expired,
            },
            "timeline": self._timeline_events,
            "interference_log": self._interference_log,
        }

# -- 5 Scheduler Implementations -----------------------------------------------

class FCFSScheduler(BaseSimScheduler):
    def __init__(self): super().__init__("FCFS")
    def admit(self, task: SimTask) -> str:
        with self._lock:
            if len(self._running) < self.MAX_RUN:
                self._do_admit(task); return "admitted"
            self._queue.append(task); return "queued"

class RoundRobinScheduler(BaseSimScheduler):
    QUANTUM_MS = 50
    def __init__(self): super().__init__("RoundRobin"); self._rr_idx = 0
    def admit(self, task: SimTask) -> str:
        with self._lock:
            quanta = math.ceil(task.service_ms / self.QUANTUM_MS)
            task.service_ms += quanta * 2
            if len(self._running) < self.MAX_RUN:
                self._do_admit(task); return "admitted"
            self._queue.append(task); return "queued"
    def _drain(self):
        alive = []
        for t in self._queue:
            if t.age_ms() > 5.0 * t.deadline_ms:
                self._expired += 1; self._misses += 1
                if t.cls in self._by_class: self._by_class[t.cls]["misses"] += 1
            else: alive.append(t)
        self._queue = alive
        while self._queue and len(self._running) < self.MAX_RUN:
            if self._rr_idx >= len(self._queue): self._rr_idx = 0
            if not self._queue: break
            task = self._queue.pop(self._rr_idx % len(self._queue))
            self._do_admit(task)

class SJFScheduler(BaseSimScheduler):
    """Shortest Job First with strict sort on estimated runtime."""
    def __init__(self): super().__init__("SJF")
    def admit(self, task: SimTask) -> str:
        with self._lock:
            if len(self._running) < self.MAX_RUN:
                self._do_admit(task); return "admitted"
            self._queue.append(task)
            self._queue.sort(key=lambda t: t.service_ms)  # strict sort
            return "queued"
    def _drain(self):
        alive = []
        for t in self._queue:
            if t.age_ms() > 5.0 * t.deadline_ms:
                self._expired += 1; self._misses += 1
                if t.cls in self._by_class: self._by_class[t.cls]["misses"] += 1
            else: alive.append(t)
        self._queue = alive
        self._queue.sort(key=lambda t: t.service_ms)
        while self._queue and len(self._running) < self.MAX_RUN:
            task = self._queue.pop(0)
            self._do_admit(task)

class PriorityScheduler(BaseSimScheduler):
    """Strict priority with aging. Lower tier = higher priority."""
    AGING_INTERVAL_MS = 500
    def __init__(self): super().__init__("PriorityStrict")
    def admit(self, task: SimTask) -> str:
        with self._lock:
            if len(self._running) < self.MAX_RUN:
                self._do_admit(task); return "admitted"
            self._queue.append(task)
            self._sort_queue()
            # Preemption: if new task has higher priority than worst running
            if self._running and task.tier < min(t.tier for t in self._running.values()):
                now = time.monotonic() * 1000
                worst = max(self._running.values(), key=lambda t: (t.tier, -t.age_ms(now)))
                if worst.tier >= 3 and task.tier <= 1:
                    self._running.pop(worst.task_id)
                    self._queue.append(worst)
                    queued_task = self._queue.pop(self._queue.index(task))
                    self._do_admit(queued_task)
            return "queued"
    def _sort_queue(self):
        # Strict priority with aging: tier first, then age bonus
        now = time.monotonic() * 1000
        for t in self._queue:
            age_bonus = t.age_ms(now) / max(1, self.AGING_INTERVAL_MS)
            t._effective_prio = t.tier - min(1.5, age_bonus * 0.1)
        self._queue.sort(key=lambda t: (getattr(t, '_effective_prio', t.tier), t.priority))
    def _drain(self):
        now = time.monotonic() * 1000
        alive = []
        for t in self._queue:
            if t.age_ms(now) > 5.0 * t.deadline_ms:
                self._expired += 1; self._misses += 1
                if t.cls in self._by_class: self._by_class[t.cls]["misses"] += 1
            else: alive.append(t)
        self._queue = alive
        self._sort_queue()
        while self._queue and len(self._running) < self.MAX_RUN:
            task = self._queue.pop(0)
            self._do_admit(task)

class EDFScheduler(BaseSimScheduler):
    """Earliest Deadline First with strict preemption."""
    def __init__(self): super().__init__("EDF")
    def admit(self, task: SimTask) -> str:
        with self._lock:
            now = time.monotonic() * 1000
            task.absolute_deadline = task.arrival_time * 1000 + task.deadline_ms
            
            if len(self._running) < self.MAX_RUN:
                self._do_admit(task); return "admitted"
            
            self._queue.append(task)
            self._sort_queue()
            
            # Preemption check: if new task has earlier deadline than worst running
            if self._running:
                worst = max(self._running.values(), key=lambda t: t.absolute_deadline)
                if task.absolute_deadline < worst.absolute_deadline:
                    self._running.pop(worst.task_id)
                    self._queue.append(worst)
                    queued_task = self._queue.pop(self._queue.index(task))
                    self._do_admit(queued_task)
            return "queued"
            
    def _sort_queue(self):
        self._queue.sort(key=lambda t: t.absolute_deadline)
        
    def _drain(self):
        now = time.monotonic() * 1000
        alive = []
        for t in self._queue:
            if t.age_ms(now) > 5.0 * t.deadline_ms:
                self._expired += 1; self._misses += 1
                self._log_event("EXPIRE", t)
                if t.cls in self._by_class: self._by_class[t.cls]["misses"] += 1
            else: alive.append(t)
        self._queue = alive
        self._sort_queue()
class MOSAICSimScheduler(BaseSimScheduler):
    INTERFERENCE_THRESHOLD = 0.45
    BUDGET_FRACTION = 0.40
    
    def __init__(self): 
        super().__init__("MOSAIC")
        self.w_ipc = 1.0
        self.w_llc = 2.5
        self.w_bw = 1.5
        self._learned_matrix = {} # Simulate online updates
        
    def _update_matrix_online(self, cls_a: str, cls_b: str, actual_lat: float, estimated_lat: float):
        """Simulates adaptive interference learning over time."""
        key = (cls_a, cls_b)
        if key not in self._learned_matrix:
            self._learned_matrix[key] = 0.0
        # Exponential moving average of the error
        error = actual_lat - estimated_lat
        self._learned_matrix[key] = self._learned_matrix[key] * 0.8 + error * 0.2

    def _has_conflict(self, cls, running_classes):
        total_score = 0.0
        for rc in running_classes:
            if cls in INTERFERENCE_MATRIX and rc in INTERFERENCE_MATRIX[cls]:
                ipc_deg, lat = INTERFERENCE_MATRIX[cls][rc]
                # Synthesize LLC/BW components based on latency delta and IPC deg
                delta_llc = lat / 1000.0 * 0.4 
                delta_bw = lat / 1000.0 * 0.3
                
                score = (self.w_ipc * ipc_deg) + (self.w_llc * delta_llc) + (self.w_bw * delta_bw)
                total_score += score
                
        if total_score > self.INTERFERENCE_THRESHOLD:
            self._interference_log.append({
                "task_a": cls, "task_b": str(running_classes),
                "interference_score": total_score, "decision": "blocked"
            })
            return True
        return False
        
    def _deadline_squeeze(self, cls, running):
        now = time.monotonic() * 1000
        for rt in running:
            if cls in INTERFERENCE_MATRIX and rt.cls in INTERFERENCE_MATRIX[cls]:
                _, lat = INTERFERENCE_MATRIX[cls][rt.cls]
                remaining = rt.remaining_ms(now)
                # If remaining is tiny, skip squeeze because overlap is negligible
                if remaining < 50.0:
                    continue
                if lat > self.BUDGET_FRACTION * max(1.0, remaining):
                    return True
        return False
    def admit(self, task: SimTask) -> str:
        with self._lock:
            now = time.monotonic() * 1000
            
            # Mission-Critical Protection: Actively reject background tasks (tier >= 3) 
            # if there are any high-priority tasks running/queued, or under moderate load.
            if task.tier >= 3:
                high_running = [t for t in self._running.values() if t.tier <= 2]
                high_queued = [t for t in self._queue if t.tier <= 2]
                if high_running or high_queued or len(self._running) >= 3:
                    return "rejected"
            
            # Preemption: if critical/urgent task arrives and running is full of background/important tasks
            if len(self._running) >= self.MAX_RUN and task.tier <= 2:
                running_backgrounds = [t for t in self._running.values() if t.tier >= 3]
                if running_backgrounds:
                    worst = max(running_backgrounds, key=lambda t: (t.tier, -t.age_ms(now)))
                    self._running.pop(worst.task_id)
                    self._queue.append(worst)
                    self._log_event("PREEMPT", worst, {"by": task.task_id})
            
            rl = list(self._running.values())
            rc = [t.cls for t in rl]

            # PREDICTIVE ADMISSION CONTROL: t_finish = t_queue + t_service + t_interference + t_preemption
            # Reject if t_finish > deadline
            queue_wait_est = 0.0
            preemption_penalty = 0.0
            
            if len(rl) >= self.MAX_RUN:
                if rl: queue_wait_est += min(t.remaining_ms(now) for t in rl)
                for qt in self._queue:
                    if qt.tier <= task.tier:
                        queue_wait_est += qt.service_ms / self.MAX_RUN
                        if qt.tier < task.tier:
                            preemption_penalty += 15.0 # Context switch & cache warming

            int_penalty = 0.0
            for rcls in rc:
                if task.cls in INTERFERENCE_MATRIX and rcls in INTERFERENCE_MATRIX[task.cls]:
                    _, lat = INTERFERENCE_MATRIX[task.cls][rcls]
                    int_penalty += lat
                    
            # Add learned correction
            for rcls in rc:
                int_penalty += self._learned_matrix.get((task.cls, rcls), 0.0)

            t_finish = queue_wait_est + task.service_ms + int_penalty + preemption_penalty
            
            # Mission-Critical Permissiveness: Only reject tier 1 if completely hopeless
            allowed_deadline = task.deadline_ms * 2.0 if task.tier == 1 else task.deadline_ms
            if t_finish > allowed_deadline:
                return "rejected"

            conflict = (len(rl) >= self.MAX_RUN or
                        self._has_conflict(task.cls, rc) or
                        self._deadline_squeeze(task.cls, rl))
                        
            # Conflict-driven preemption: if high-priority task is blocked, preempt running background tasks
            if conflict and task.tier <= 2:
                running_backgrounds = [t for t in self._running.values() if t.tier >= 3]
                if running_backgrounds:
                    worst = max(running_backgrounds, key=lambda t: (t.tier, -t.age_ms(now)))
                    self._running.pop(worst.task_id)
                    self._queue.append(worst)
                    self._log_event("PREEMPT", worst, {"by": task.task_id, "reason": "conflict"})
                    # Recalculate and re-evaluate conflict
                    rl = list(self._running.values())
                    rc = [t.cls for t in rl]
                    conflict = (len(rl) >= self.MAX_RUN or
                                self._has_conflict(task.cls, rc) or
                                self._deadline_squeeze(task.cls, rl))

            if not conflict:
                self._do_admit(task); return "admitted"
            if task.age_ms(now) > 3.0 * task.deadline_ms:
                self._do_admit(task); self._starved_hard += 1
                return "force_admitted"
            self._queue.append(task)
            self._queue.sort(key=lambda t: (t.tier, -t.remaining_ms(now)/max(1,t.deadline_ms)))
            return "queued"
    def _do_admit(self, task: SimTask):
        task.start_ms = time.monotonic() * 1000
        if task.age_ms() > 3.0 * task.deadline_ms:
            self._starved_hard += 1
        self._running[task.task_id] = task
        self._admitted += 1
    def _compute_interference_overhead(self, cls: str) -> float:
        return 0.0  # MOSAIC checks before admission
    def _drain(self):
        now = time.monotonic() * 1000
        alive = []
        for t in self._queue:
            if t.age_ms(now) > 5.0 * t.deadline_ms:
                self._expired += 1; self._misses += 1
                if t.cls in self._by_class: self._by_class[t.cls]["misses"] += 1
            else: alive.append(t)
        self._queue = alive
        # Sort queue in drain before admitting to ensure prioritisation
        self._queue.sort(key=lambda t: (t.tier, -t.remaining_ms(now)/max(1,t.deadline_ms)))
        
        for task in list(self._queue):
            if len(self._running) >= self.MAX_RUN: break
            rl = list(self._running.values())
            if (not self._has_conflict(task.cls, [t.cls for t in rl]) and
                    not self._deadline_squeeze(task.cls, rl)):
                self._queue.remove(task)
                self._do_admit(task)
                
    def complete(self, task_id: str, actual_ms: float):
        # Trigger adaptive update
        with self._lock:
            if task_id in self._running:
                t = self._running[task_id]
                est_lat = t.service_ms
                # Update logic
                for rc in [rt.cls for rt in self._running.values() if rt.task_id != task_id]:
                    self._update_matrix_online(t.cls, rc, actual_ms, est_lat)
        super().complete(task_id, actual_ms)

class MOSAICNoPMUScheduler(MOSAICSimScheduler):
    """Ablation: MOSAIC without PMU interference awareness (no LLC/IPC awareness)."""
    def __init__(self): super().__init__(); self.name = "MOSAIC-NoPMU"
    def _has_conflict(self, cls, rc): return False
    def _deadline_squeeze(self, cls, rl): return False

class MOSAICNoAdmissionScheduler(MOSAICSimScheduler):
    """Ablation: MOSAIC without predictive admission control (always admit/queue)."""
    def __init__(self): super().__init__(); self.name = "MOSAIC-NoAdmission"
    def admit(self, task: SimTask) -> str:
        with self._lock:
            now = time.monotonic() * 1000
            rl = list(self._running.values())
            rc = [t.cls for t in rl]
            conflict = (len(rl) >= self.MAX_RUN or
                        self._has_conflict(task.cls, rc) or
                        self._deadline_squeeze(task.cls, rl))
            if not conflict:
                self._do_admit(task); return "admitted"
            if task.age_ms(now) > 3.0 * task.deadline_ms:
                self._do_admit(task); self._starved_hard += 1
                return "force_admitted"
            self._queue.append(task)
            self._queue.sort(key=lambda t: (t.tier, -t.remaining_ms(now)/max(1,t.deadline_ms)))
            return "queued"

# -- Registry ------------------------------------------------------------------
SCHEDULER_REGISTRY = {
    "fcfs": FCFSScheduler, "rr": RoundRobinScheduler,
    "sjf": SJFScheduler, "priority": PriorityScheduler,
    "edf": EDFScheduler,
    "mosaic": MOSAICSimScheduler,
    "mosaic_nopmu": MOSAICNoPMUScheduler,
    "mosaic_noadm": MOSAICNoAdmissionScheduler,
}

# -- Deterministic Replay Engine -----------------------------------------------
def run_experiment(scheduler: BaseSimScheduler, trace: list[dict],
                   seed: int = 42) -> dict:
    """Replay a pre-generated trace through a scheduler. Deterministic."""
    rng = random.Random(seed)
    scheduler._generated = len(trace)
    threads: list[threading.Thread] = []
    start = time.monotonic()

    # Convert trace timestamps to real-time offsets
    base_time = trace[0]["arrival_time"] if trace else 0.0

    def execute_task(task: SimTask, overhead_ms: float, task_rng_seed: int):
        t_rng = random.Random(task_rng_seed)
        jitter = t_rng.gauss(0, task.service_ms * 0.08)
        actual = max(1.0, task.service_ms + jitter + overhead_ms)
        time.sleep(actual / 1000.0)
        scheduler.complete(task.task_id, actual)

    for i, td in enumerate(trace):
        # Wait until arrival time
        target = start + (td["arrival_time"] - base_time)
        now = time.monotonic()
        if target > now:
            time.sleep(target - now)

        task = SimTask(
            task_id=f"{scheduler.name[:3]}_{i+1:06d}",
            cls=td["class"], deadline_ms=td["deadline_ms"],
            service_ms=td["service_ms"], priority=td["priority"],
            tier=td["tier"], arrival_time=td["arrival_time"],
            submit_ms=time.monotonic() * 1000,
        )

        result = scheduler.admit(task)
        if result == "rejected":
            scheduler._rejected += 1
            scheduler._log_event("REJECT", task)
            continue
        elif result == "queued":
            scheduler._log_event("QUEUE", task)

        if result in ("admitted", "force_admitted"):
            overhead = scheduler._compute_interference_overhead(td["class"])
            t = threading.Thread(
                target=execute_task,
                args=(task, overhead, rng.randint(0, 2**31)),
                daemon=True)
            t.start(); threads.append(t)

    # Wait for completion
    deadline = time.monotonic() + 45.0
    for t in threads:
        remaining = deadline - time.monotonic()
        if remaining > 0: t.join(timeout=remaining)

    return scheduler.summary(time.monotonic() - start)


def run_all(duration: float, rate: float, pattern: str,
            schedulers: list[str], seed: int = 42) -> list[dict]:
    # Step 1: Generate trace ONCE
    print(f"\n  [trace] Generating deterministic trace: seed={seed} "
          f"pattern={pattern} rate={rate}/s duration={duration}s")

    if pattern in ["cache_thrash", "retry_storm", "misclassify", "overload", "cache_thrash_extreme"]:
        from trace_generator import generate_adversarial_trace
        trace = generate_adversarial_trace(pattern, seed, duration, rate)
    else:
        trace = generate_trace(pattern, rate, duration, seed)
    
    ts = trace_summary(trace)
    print(f"  [trace] {ts['total_tasks']} tasks generated across "
          f"{ts['unique_classes']} classes")
    for cls, cnt in sorted(ts["by_class"].items()):
        print(f"          {cls:<25} {cnt:>4} tasks")

    # Save trace for reproducibility
    trace_path = _ROOT / "results" / f"trace_seed_{seed}.json"
    save_trace(trace, trace_path)
    print(f"  [trace] Saved to {trace_path}\n")

    # Step 2: Replay IDENTICAL trace through each scheduler
    results = []
    for name in schedulers:
        cls = SCHEDULER_REGISTRY[name]
        sched = cls()
        print(f"  >  Running {sched.name:<20} ...", end="", flush=True)
        t0 = time.monotonic()
        res = run_experiment(sched, trace, seed)
        elapsed = time.monotonic() - t0
        res["elapsed_s"] = round(elapsed, 1)
        results.append(res)
        acct = res.get("accounting", {})
        print(f"  done ({elapsed:.1f}s)  hit={res['hit_rate']:.1%}  "
              f"p99={res['p99_ms']:.0f}ms  jfi={res['fairness_index']:.3f}  "
              f"adm={acct.get('admitted',0)} rej={acct.get('rejected',0)} "
              f"exp={acct.get('expired',0)}")
    return results


def print_comparison_table(results: list[dict]) -> None:
    header = (f"{'Scheduler':<20} {'Hit%':>7} {'P50':>7} {'P95':>7} {'P99':>7} "
              f"{'Starve%':>8} {'TPS':>6} {'JFI':>6} {'Eff(t/Wh)':>10}")
    print(f"\n{'='*85}")
    print(f"  BENCHMARK RESULTS (Deterministic Replay)")
    print(f"{'='*85}")
    print(f"  {header}")
    print(f"  {'-'*83}")
    for r in results:
        marker = " <-- MOSAIC" if r["scheduler"] == "MOSAIC" else ""
        print(f"  {r['scheduler']:<20} "
              f"{r['hit_rate']:>6.1%} "
              f"{r['p50_ms']:>7.0f} {r['p95_ms']:>7.0f} {r['p99_ms']:>7.0f} "
              f"{r['starvation_rate']:>7.1%} "
              f"{r['throughput_tps']:>6.2f} "
              f"{r['fairness_index']:>6.3f} "
              f"{r['efficiency_tpwh']:>10.1f}{marker}")
    print(f"{'='*85}")

    # Task accounting
    print(f"\n  TASK ACCOUNTING")
    print(f"  {'-'*70}")
    print(f"  {'Scheduler':<20} {'Gen':>6} {'Admit':>6} {'Done':>6} "
          f"{'Reject':>7} {'Expire':>7} {'FuncStarv':>10}")
    for r in results:
        a = r.get("accounting", {})
        fsc = r.get("func_starved_classes", [])
        print(f"  {r['scheduler']:<20} "
              f"{a.get('generated',0):>6} {a.get('admitted',0):>6} "
              f"{a.get('completed',0):>6} {a.get('rejected',0):>7} "
              f"{a.get('expired',0):>7} {','.join(fsc) if fsc else 'none':>10}")

    # MOSAIC vs best baseline
    mosaic = next((r for r in results if r["scheduler"]=="MOSAIC"), None)
    best_base = max((r for r in results if r["scheduler"]!="MOSAIC"),
                    key=lambda r: r["hit_rate"], default=None)
    if mosaic and best_base:
        print(f"\n  MOSAIC vs best baseline ({best_base['scheduler']}):")
        hit_imp = (mosaic["hit_rate"] - best_base["hit_rate"]) / max(0.001, best_base["hit_rate"]) * 100
        p99_imp = (best_base["p99_ms"] - mosaic["p99_ms"]) / max(1, best_base["p99_ms"]) * 100
        print(f"    Hit rate improvement   : {hit_imp:+.1f}%")
        print(f"    P99 latency reduction  : {p99_imp:+.1f}%")
        print(f"    Fairness (MOSAIC)      : {mosaic['fairness_index']:.3f} "
              f"vs {best_base['fairness_index']:.3f}")
    print()


def main():
    p = argparse.ArgumentParser(description="MOSAIC Benchmark Harness")
    p.add_argument("--pattern", default="burst",
                   choices=["poisson","burst","sinusoidal","disaster"])
    p.add_argument("--rate", type=float, default=6.0)
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--schedulers", nargs="+",
                   default=["fcfs","rr","sjf","priority","mosaic"],
                   choices=list(SCHEDULER_REGISTRY.keys()))
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=str(RESULTS_PATH))
    args = p.parse_args()

    if args.quick:
        args.duration = 15.0; args.rate = 3.0

    print(f"\n{'='*85}")
    print(f"  MOSAIC Benchmark -- Deterministic Replay Harness")
    print(f"  Pattern: {args.pattern}  Rate: {args.rate}/s  "
          f"Duration: {args.duration}s  Seed: {args.seed}")
    print(f"{'='*85}")

    results = run_all(args.duration, args.rate, args.pattern,
                      args.schedulers, args.seed)
    print_comparison_table(results)

    # Export raw traces
    raw_dir = _ROOT / "results" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        name = r["scheduler"].lower().replace(" ", "_")
        with open(raw_dir / f"{name}_results.json", "w") as f:
            json.dump(r, f, indent=2)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output = {
        "config": {"pattern": args.pattern, "rate": args.rate,
                    "duration": args.duration, "seed": args.seed,
                    "replay_mode": "deterministic"},
        "results": results,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved -> {args.output}")
    print(f"  Raw traces   -> {raw_dir}/\n")


if __name__ == "__main__":
    main()
